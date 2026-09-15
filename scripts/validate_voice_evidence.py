"""Validates `compute_voice_evidence` where it matters: over the EXACT windows
in which whisper produced text, inside stretches a voice segmenter already
accepted. Not over whole regimes -- that was the methodological error of the
first attempt (measuring the centers of silent gaps that production never
sends to the ASR).

Usage:
    python scripts/validate_voice_evidence.py <speech.pcm> <empty_room.pcm> [output.json]

The `.pcm` files are s16le mono at 16 kHz (`ffmpeg -ac 1 -ar 16000 -f s16le`).
Two clips with a known regime: one of real conversation, another of a nearly
empty room where whisper hallucinates. The audio from the original validation
is private and is NOT in the repo; it is identified by hash in the output so
the run is comparable:

    speech.pcm  9,600,000 bytes (5 min)   nearly empty room: empty_room.pcm 28,800,000 bytes (15 min)
    S21, 2026-09-08/09, 16 kHz -- hashes in PR #22.

What it prints: per measure, median and IQR in each regime, and the IQR
overlap (0% = fully separated). And a threshold sweep over `voice_band_ratio`
with what it cuts on each side. Everything is deterministic except whisper's
SEGMENTATION, which is not bit-for-bit stable between runs on CPU: the number
of windows can vary +-10%.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps

from speechtotext.audio import compute_voice_evidence

SR = 16000
MERGE_S, PAD_S = 0.8, 0.4              # hangover and padding a segmenter uses to build stretches
MIN_WINDOW_SAMPLES = 1024              # one analysis window (64 ms)


def _voice_regions(x: np.ndarray) -> list[tuple[float, float]]:
    options = VadOptions(min_silence_duration_ms=300, speech_pad_ms=200, min_speech_duration_ms=100)
    return [(r["start"] / SR, r["end"] / SR) for r in get_speech_timestamps(x, options, sampling_rate=SR)]


def _stretches(x: np.ndarray) -> list[tuple[float, float]]:
    dur = len(x) / SR
    stretches: list[list[float]] = []
    for a, b in _voice_regions(x):
        a, b = max(0.0, a - PAD_S), min(dur, b + PAD_S)
        if stretches and a - stretches[-1][1] <= MERGE_S:
            stretches[-1][1] = b
        else:
            stretches.append([a, b])
    return [(a, b) for a, b in stretches if b - a >= 1.0]


def windows_with_text(pcm: Path, model: WhisperModel) -> list[dict]:
    x = np.fromfile(pcm, dtype="<i2").astype(np.float32) / 32768.0
    rows = []
    for ta, tb in _stretches(x)[:60]:
        seg = x[int(ta * SR):int(tb * SR)]
        if len(seg) < SR // 2:
            continue
        for s in model.transcribe(seg, language="es", beam_size=5, vad_filter=False)[0]:
            text = s.text.strip()
            window = seg[int(s.start * SR):int(s.end * SR)]
            if not text or len(window) < MIN_WINDOW_SAMPLES:
                continue
            e = compute_voice_evidence(window, SR)
            rows.append({
                "t_s": round(ta + s.start, 2), "dur_s": round(s.end - s.start, 2),
                "no_speech": float(s.no_speech_prob), "text": text,
                "band": e.voice_band_ratio, "voiced": e.voiced_ratio,
                "f0": e.f0_median_hz, "flat": e.spectral_flatness,
            })
    return rows


def _column(rows: list[dict], key: str) -> np.ndarray:
    return np.array([f[key] for f in rows if f[key] is not None], dtype=float)


def summarize(speech: list[dict], empty_room: list[dict]) -> None:
    print(f"\nwindows with text: speech={len(speech)}  empty room={len(empty_room)}\n")
    print(f"{'measure':>10} | {'SPEECH med [IQR]':>26} | {'EMPTY ROOM med [IQR]':>26} | IQR overlap")
    print("-" * 84)
    for key in ("band", "voiced", "flat", "no_speech"):
        a, b = _column(speech, key), _column(empty_room, key)
        a25, a75 = np.percentile(a, [25, 75])
        b25, b75 = np.percentile(b, [25, 75])
        overlap = max(0.0, min(a75, b75) - max(a25, b25)) / max(1e-9, max(a75, b75) - min(a25, b25))
        print(f"{key:>10} | {np.median(a):6.3f} [{a25:6.3f}, {a75:6.3f}]      | "
              f"{np.median(b):6.3f} [{b25:6.3f}, {b75:6.3f}]      | {overlap * 100:3.0f} %")
    a, b = _column(speech, "band"), _column(empty_room, "band")
    print("\nsweep over voice_band_ratio (cuts if band < u)")
    print(f"{'u':>5} | cuts from EMPTY ROOM | cuts from SPEECH")
    for u in (0.3, 0.4, 0.5, 0.6):
        print(f"{u:>5} | {int((b < u).sum()):3d}/{len(b)} ({(b < u).mean() * 100:3.0f} %)"
              f"           | {int((a < u).sum()):3d}/{len(a)} ({(a < u).mean() * 100:3.0f} %)")


def main(argv: list[str]) -> None:
    if len(argv) < 3:
        sys.exit(__doc__)
    speech_pcm, empty_room_pcm = Path(argv[1]), Path(argv[2])
    for p in (speech_pcm, empty_room_pcm):
        print(f"{hashlib.sha256(p.read_bytes()).hexdigest()[:16]}  {p.name}  {p.stat().st_size} bytes")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    speech, empty_room = windows_with_text(speech_pcm, model), windows_with_text(empty_room_pcm, model)
    summarize(speech, empty_room)
    if len(argv) > 3:
        Path(argv[3]).write_text(json.dumps({"speech": speech, "empty_room": empty_room},
                                            ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv)
