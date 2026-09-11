"""Valida `compute_voice_evidence` donde importa: sobre las ventanas EXACTAS en
las que whisper produjo texto, dentro de tramos que un segmentador por voz ya
acepto. No sobre regimenes enteros — ese fue el error metodologico del primer
intento (medir centros de huecos mudos que produccion nunca manda al ASR).

Uso:
    python scripts/validar_evidencia_de_voz.py <habla.pcm> <cuarto_vacio.pcm> [salida.json]

Los `.pcm` son s16le mono a 16 kHz (`ffmpeg -ac 1 -ar 16000 -f s16le`). Dos
cortes con regimen conocido: uno de conversacion real, otro de cuarto casi
vacio donde whisper inventa. El audio de la validacion original es privado y
NO esta en el repo; se identifica por hash en la salida para que la corrida
sea comparable:

    conversa.pcm  9 600 000 bytes (5 min)   cuarto casi vacio: mudo.pcm 28 800 000 bytes (15 min)
    S21, 2026-09-08/09, 16 kHz — hashes en el PR #22.

Que imprime: por medida, mediana e IQR en cada regimen y el solape de IQR (0 %
= separa). Y un barrido de umbral sobre `voice_band_ratio` con lo que corta en
cada lado. Todo determinista salvo la SEGMENTACION de whisper, que no es bit a
bit estable entre corridas en CPU: el numero de ventanas puede variar +-10 %.
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
UNIR_S, RELLENO_S = 0.8, 0.4          # hangover y relleno con que un segmentador arma tramos
VENTANA_MINIMA = 1024                 # un tramo de analisis (64 ms)


def _regiones_de_voz(x: np.ndarray) -> list[tuple[float, float]]:
    opciones = VadOptions(min_silence_duration_ms=300, speech_pad_ms=200, min_speech_duration_ms=100)
    return [(r["start"] / SR, r["end"] / SR) for r in get_speech_timestamps(x, opciones, sampling_rate=SR)]


def _tramos(x: np.ndarray) -> list[tuple[float, float]]:
    dur = len(x) / SR
    tramos: list[list[float]] = []
    for a, b in _regiones_de_voz(x):
        a, b = max(0.0, a - RELLENO_S), min(dur, b + RELLENO_S)
        if tramos and a - tramos[-1][1] <= UNIR_S:
            tramos[-1][1] = b
        else:
            tramos.append([a, b])
    return [(a, b) for a, b in tramos if b - a >= 1.0]


def ventanas_con_texto(pcm: Path, modelo: WhisperModel) -> list[dict]:
    x = np.fromfile(pcm, dtype="<i2").astype(np.float32) / 32768.0
    filas = []
    for ta, tb in _tramos(x)[:60]:
        seg = x[int(ta * SR):int(tb * SR)]
        if len(seg) < SR // 2:
            continue
        for s in modelo.transcribe(seg, language="es", beam_size=5, vad_filter=False)[0]:
            texto = s.text.strip()
            ventana = seg[int(s.start * SR):int(s.end * SR)]
            if not texto or len(ventana) < VENTANA_MINIMA:
                continue
            e = compute_voice_evidence(ventana, SR)
            filas.append({
                "t_s": round(ta + s.start, 2), "dur_s": round(s.end - s.start, 2),
                "no_speech": float(s.no_speech_prob), "texto": texto,
                "band": e.voice_band_ratio, "voiced": e.voiced_ratio,
                "f0": e.f0_median_hz, "flat": e.spectral_flatness,
            })
    return filas


def _columna(filas: list[dict], clave: str) -> np.ndarray:
    return np.array([f[clave] for f in filas if f[clave] is not None], dtype=float)


def resumen(habla: list[dict], vacio: list[dict]) -> None:
    print(f"\nventanas con texto: habla={len(habla)}  cuarto vacio={len(vacio)}\n")
    print(f"{'medida':>10} | {'HABLA med [IQR]':>26} | {'CUARTO VACIO med [IQR]':>26} | solape IQR")
    print("-" * 84)
    for clave in ("band", "voiced", "flat", "no_speech"):
        a, b = _columna(habla, clave), _columna(vacio, clave)
        a25, a75 = np.percentile(a, [25, 75])
        b25, b75 = np.percentile(b, [25, 75])
        solape = max(0.0, min(a75, b75) - max(a25, b25)) / max(1e-9, max(a75, b75) - min(a25, b25))
        print(f"{clave:>10} | {np.median(a):6.3f} [{a25:6.3f}, {a75:6.3f}]      | "
              f"{np.median(b):6.3f} [{b25:6.3f}, {b75:6.3f}]      | {solape * 100:3.0f} %")
    a, b = _columna(habla, "band"), _columna(vacio, "band")
    print("\nbarrido sobre voice_band_ratio (corta si band < u)")
    print(f"{'u':>5} | corta del CUARTO VACIO | corta del HABLA")
    for u in (0.3, 0.4, 0.5, 0.6):
        print(f"{u:>5} | {int((b < u).sum()):3d}/{len(b)} ({(b < u).mean() * 100:3.0f} %)"
              f"           | {int((a < u).sum()):3d}/{len(a)} ({(a < u).mean() * 100:3.0f} %)")


def main(argv: list[str]) -> None:
    if len(argv) < 3:
        sys.exit(__doc__)
    habla_pcm, vacio_pcm = Path(argv[1]), Path(argv[2])
    for p in (habla_pcm, vacio_pcm):
        print(f"{hashlib.sha256(p.read_bytes()).hexdigest()[:16]}  {p.name}  {p.stat().st_size} bytes")
    modelo = WhisperModel("base", device="cpu", compute_type="int8")
    habla, vacio = ventanas_con_texto(habla_pcm, modelo), ventanas_con_texto(vacio_pcm, modelo)
    resumen(habla, vacio)
    if len(argv) > 3:
        Path(argv[3]).write_text(json.dumps({"habla": habla, "cuarto_vacio": vacio},
                                            ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv)
