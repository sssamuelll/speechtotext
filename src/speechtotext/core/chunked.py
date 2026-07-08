"""Transcripción por trozos: durabilidad (checkpoint/resume) + paralelismo."""
from __future__ import annotations

import av
import hashlib
import json
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from faster_whisper import WhisperModel

from speechtotext.core.finder import _home


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
        out.append(TimedSegment(s.start + offset, s.end + offset, s.text, tw))
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


def chunk_path(audio: Path, opts: dict, model: str, start: float, end: float) -> Path:
    st = audio.stat()
    key = "|".join(str(x) for x in (
        audio.resolve(), st.st_size, int(st.st_mtime), model,
        opts["language"], opts["beam_size"], opts["vad_filter"],
        opts["hotwords"], opts["word_timestamps"], start, end,
    ))
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    d = _home() / "chunks"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{digest}.json"


def seg_to_dict(seg: TimedSegment) -> dict:
    d = {"start": seg.start, "end": seg.end, "text": seg.text}
    if seg.words is not None:
        d["words"] = [{"start": w.start, "end": w.end, "word": w.word} for w in seg.words]
    return d


def seg_from_dict(d: dict) -> TimedSegment:
    words = d.get("words")
    tw = [TimedWord(w["start"], w["end"], w["word"]) for w in words] if words is not None else None
    return TimedSegment(d["start"], d["end"], d["text"], tw)


def transcribe_chunk(audio, start, end, opts, model, model_name):
    """Devuelve (segmentos_globales, from_cache, idioma_detectado). El idioma se guarda
    en el checkpoint para que run_chunked reporte el real bajo --language auto."""
    path = chunk_path(audio, opts, model_name, start, end)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return [seg_from_dict(d) for d in data["segments"]], True, data.get("language")
        except (json.JSONDecodeError, KeyError):
            pass  # checkpoint corrupto -> recomputar

    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(start), "-t", str(end - start), "-i", str(audio),
            "-ar", "16000", "-ac", "1", tmp,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        segments_iter, info = model.transcribe(tmp, **opts)
        segs = shift_segments(list(segments_iter), start)
        lang = getattr(info, "language", None)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    path.write_text(
        json.dumps({"language": lang, "segments": [seg_to_dict(s) for s in segs]}, ensure_ascii=False),
        encoding="utf-8",
    )
    return segs, False, lang


def probe_duration(audio: Path) -> float:
    container = av.open(str(audio))
    if container.duration:
        return container.duration / 1_000_000.0
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nk=1:nw=1", str(audio)],
        capture_output=True, text=True,
    )
    return float(proc.stdout.strip())


def _mmss(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def run_chunked(audio, opts, jobs, model_name, device, compute_type, log=print):
    duration = probe_duration(audio)
    chunks = plan_chunks(audio, duration)
    jobs = max(1, min(jobs, len(chunks)))
    model = WhisperModel(
        model_name, device=device, compute_type=compute_type,
        cpu_threads=max(1, (os.cpu_count() or 1) // jobs), num_workers=jobs,
    )
    results: list = [None] * len(chunks)
    langs: list = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {
            pool.submit(transcribe_chunk, audio, s, e, opts, model, model_name): i
            for i, (s, e) in enumerate(chunks)
        }
        for done, fut in enumerate(as_completed(futs), start=1):
            i = futs[fut]
            results[i], from_cache, langs[i] = fut.result()
            s, e = chunks[i]
            tag = "cache" if from_cache else "nuevo"
            log(f"[{done}/{len(chunks)}] {_mmss(s)}-{_mmss(e)} OK ({tag})")
    segments = [seg for chunk_segs in results for seg in chunk_segs]
    # Bajo --language auto (opts language=None) reporta el idioma REAL que detectó el
    # primer trozo, no un "es" hardcodeado; si se forzó idioma, ese manda.
    detected = next((l for l in langs if l), None)
    info = SimpleNamespace(language=opts.get("language") or detected or "es",
                           language_probability=1.0, duration=duration)
    return segments, info


CHUNK_THRESHOLD = 1200.0  # s (20 min): por encima, auto-trocea


def should_chunk(duration: float, chunk_flag: bool | None, threshold: float = CHUNK_THRESHOLD) -> bool:
    if chunk_flag is not None:
        return chunk_flag
    return duration > threshold
