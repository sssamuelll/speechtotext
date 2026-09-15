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

    conversa.pcm  9,600,000 bytes (5 min)   nearly empty room: mudo.pcm 28,800,000 bytes (15 min)
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
UNIR_S, RELLENO_S = 0.8, 0.4          # hangover and padding a segmenter uses to build stretches
VENTANA_MINIMA = 1024                 # one analysis window (64 ms)


def _voice_regions(x: np.ndarray) -> list[tuple[float, float]]:
    opciones = VadOptions(min_silence_duration_ms=300, speech_pad_ms=200, min_speech_duration_ms=100)
    return [(r["start"] / SR, r["end"] / SR) for r in get_speech_timestamps(x, opciones, sampling_rate=SR)]


def _tramos(x: np.ndarray) -> list[tuple[float, float]]:
    dur = len(x) / SR
    tramos: list[list[float]] = []
    for a, b in _voice_regions(x):
        a, b = max(0.0, a - RELLENO_S), min(dur, b + RELLENO_S)
        if tramos and a - tramos[-1][1] <= UNIR_S:
            tramos[-1][1] = b
        else:
            tramos.append([a, b])
    return [(a, b) for a, b in tramos if b - a >= 1.0]


def windows_with_text(pcm: Path, model: WhisperModel) -> list[dict]:
    x = np.fromfile(pcm, dtype="<i2").astype(np.float32) / 32768.0
    filas = []
    for ta, tb in _tramos(x)[:60]:
        seg = x[int(ta * SR):int(tb * SR)]
        if len(seg) < SR // 2:
            continue
        for s in model.transcribe(seg, language="es", beam_size=5, vad_filter=False)[0]:
            text = s.text.strip()
            ventana = seg[int(s.start * SR):int(s.end * SR)]
            if not text or len(ventana) < VENTANA_MINIMA:
                continue
            e = compute_voice_evidence(ventana, SR)
            filas.append({
                "t_s": round(ta + s.start, 2), "dur_s": round(s.end - s.start, 2),
                "no_speech": float(s.no_speech_prob), "text": text,
                "band": e.voice_band_ratio, "voiced": e.voiced_ratio,
                "f0": e.f0_median_hz, "flat": e.spectral_flatness,
            })
    return filas


def _columna(filas: list[dict], clave: str) -> np.ndarray:
    return np.array([f[clave] for f in filas if f[clave] is not None], dtype=float)


def resumen(speech: list[dict], vacio: list[dict]) -> None:
    print(f"\nwindows with text: speech={len(speech)}  empty room={len(vacio)}\n")
    print(f"{'measure':>10} | {'SPEECH med [IQR]':>26} | {'EMPTY ROOM med [IQR]':>26} | IQR overlap")
    print("-" * 84)
    for clave in ("band", "voiced", "flat", "no_speech"):
        a, b = _columna(speech, clave), _columna(vacio, clave)
        a25, a75 = np.percentile(a, [25, 75])
        b25, b75 = np.percentile(b, [25, 75])
        solape = max(0.0, min(a75, b75) - max(a25, b25)) / max(1e-9, max(a75, b75) - min(a25, b25))
        print(f"{clave:>10} | {np.median(a):6.3f} [{a25:6.3f}, {a75:6.3f}]      | "
              f"{np.median(b):6.3f} [{b25:6.3f}, {b75:6.3f}]      | {solape * 100:3.0f} %")
    a, b = _columna(speech, "band"), _columna(vacio, "band")
    print("\nsweep over voice_band_ratio (cuts if band < u)")
    print(f"{'u':>5} | cuts from EMPTY ROOM | cuts from SPEECH")
    for u in (0.3, 0.4, 0.5, 0.6):
        print(f"{u:>5} | {int((b < u).sum()):3d}/{len(b)} ({(b < u).mean() * 100:3.0f} %)"
              f"           | {int((a < u).sum()):3d}/{len(a)} ({(a < u).mean() * 100:3.0f} %)")


def main(argv: list[str]) -> None:
    if len(argv) < 3:
        sys.exit(__doc__)
    speech_pcm, vacio_pcm = Path(argv[1]), Path(argv[2])
    for p in (speech_pcm, vacio_pcm):
        print(f"{hashlib.sha256(p.read_bytes()).hexdigest()[:16]}  {p.name}  {p.stat().st_size} bytes")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    speech, vacio = windows_with_text(speech_pcm, model), windows_with_text(vacio_pcm, model)
    resumen(speech, vacio)
    if len(argv) > 3:
        Path(argv[3]).write_text(json.dumps({"speech": speech, "cuarto_vacio": vacio},
                                            ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv)
