"""Transcripción por trozos: durabilidad (checkpoint/resume) + paralelismo."""
from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

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
