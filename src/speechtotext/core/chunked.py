"""Transcripción por trozos: durabilidad (checkpoint/resume) + paralelismo."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from speechtotext.core.finder import _home
from speechtotext.core.segments import native_signals


@dataclass
class TimedWord:
    start: float
    end: float
    word: str


@dataclass
class TimedSegment:
    start: float
    end: float
    text: str
    words: list[TimedWord] | None = None
    # Señales nativas de faster-whisper (Fase 2, G5), al final, mismo patrón que words.
    no_speech: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None


def shift_segments(segments, offset: float) -> list[TimedSegment]:
    """Copia segmentos aplicando `offset` a start/end del segmento y de cada palabra.
    Los Segment de faster-whisper son inmutables; devolvemos TimedSegment nuevos."""
    out: list[TimedSegment] = []
    for s in segments:
        words = getattr(s, "words", None)
        tw = (
            [TimedWord(w.start + offset, w.end + offset, w.word) for w in words]
            if words
            else None
        )
        no_speech, avg_logprob, compression_ratio = native_signals(s)
        out.append(TimedSegment(s.start + offset, s.end + offset, s.text, tw,
                                no_speech=no_speech, avg_logprob=avg_logprob,
                                compression_ratio=compression_ratio))
    return out


def clip_to_end(segs: list[TimedSegment], end: float) -> list[TimedSegment]:
    """Whisper rellena la última ventana del trozo a 30 s con ceros y puede emitir un segmento
    sobre el relleno (medido 2026-09-11: "Gracias por ver el video." en 598.6-628.6 sobre un
    trozo que acababa en 598.7) — y ese segmento cae ENCIMA del trozo siguiente. Un segmento
    con más relleno que audio se descarta; uno que apenas sobresale se recorta a `end`, y sus
    palabras igual. Va en tiempo global y también sobre lo que sale de un checkpoint: los
    escritos antes de este recorte traen el fantasma, y limpiarlos al leer no invalida la
    caché de nadie."""
    out: list[TimedSegment] = []
    for s in segs:
        if s.end - end > end - s.start:
            continue
        s.end = min(s.end, end)
        if s.words:
            s.words = [w for w in s.words if w.start < end] or None
            for w in s.words or ():
                w.end = min(w.end, end)
        out.append(s)
    return out


_SIL_START = re.compile(r"silence_start:\s*([0-9.]+)")
_SIL_END = re.compile(r"silence_end:\s*([0-9.]+)")


def parse_silences(stderr: str) -> list[tuple[float, float]]:
    starts = [float(m.group(1)) for m in _SIL_START.finditer(stderr)]
    ends = [float(m.group(1)) for m in _SIL_END.finditer(stderr)]
    return list(zip(starts, ends))  # zip corta el start final sin end


def pick_cuts(
    silences: list[tuple[float, float]],
    duration: float,
    target_len: float = 600.0,
    search: float = 60.0,
) -> list[tuple[float, float]]:
    mids = [(s + e) / 2 for s, e in silences]
    cuts: list[float] = []
    prev = 0.0
    boundary = target_len
    while boundary < duration - 1.0:
        near = [m for m in mids if abs(m - boundary) <= search and m > prev + 1.0]
        cut = min(near, key=lambda m: abs(m - boundary)) if near else boundary
        cuts.append(cut)
        prev = cut
        boundary = cut + target_len
    bounds = [0.0, *cuts, duration]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def plan_chunks(audio: Path, duration: float, target_len: float = 600.0) -> list[tuple[float, float]]:
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(audio),
        "-af", "silencedetect=noise=-30dB:d=0.5", "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True)
        silences = parse_silences(proc.stderr.decode("utf-8", errors="ignore"))
    except (FileNotFoundError, OSError):
        silences = []  # sin ffmpeg -> cortes fijos
    return pick_cuts(silences, duration, target_len)


def chunk_path(identity: str, start: float, end: float) -> Path:
    """Checkpoint de un trozo. `identity` la arma core.transcribe: archivo (ruta, tamano,
    mtime), motor, modelo, cuantizacion, device y la peticion EFECTIVA — asi faster-whisper
    int8 y whispercpp q5_0 sobre el mismo audio jamas comparten digest, y --vad/--no-vad
    bajo whispercpp si lo comparten (la peticion efectiva ya viene sin VAD)."""
    digest = hashlib.sha1(f"{identity}|{start}|{end}".encode("utf-8")).hexdigest()[:16]
    d = _home() / "chunks"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{digest}.json"


def seg_to_dict(seg: TimedSegment) -> dict:
    d = {"start": seg.start, "end": seg.end, "text": seg.text}
    if seg.words is not None:
        d["words"] = [{"start": w.start, "end": w.end, "word": w.word} for w in seg.words]
    if seg.no_speech is not None:
        d["no_speech"] = seg.no_speech
    if seg.avg_logprob is not None:
        d["avg_logprob"] = seg.avg_logprob
    if seg.compression_ratio is not None:
        d["compression_ratio"] = seg.compression_ratio
    return d


def seg_from_dict(d: dict) -> TimedSegment:
    words = d.get("words")
    tw = [TimedWord(w["start"], w["end"], w["word"]) for w in words] if words is not None else None
    return TimedSegment(d["start"], d["end"], d["text"], tw, no_speech=d.get("no_speech"),
                        avg_logprob=d.get("avg_logprob"), compression_ratio=d.get("compression_ratio"))


CHUNK_THRESHOLD = 1200.0  # s (20 min): por encima, auto-trocea


def should_chunk(duration: float, chunk_flag: bool | None, threshold: float = CHUNK_THRESHOLD) -> bool:
    if chunk_flag is not None:
        return chunk_flag
    return duration > threshold
