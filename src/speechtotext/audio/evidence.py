"""Evidencia de voz por DSP determinista.

Ninguna medida de aqui es la opinion de un modelo: son descriptores de senal
con nombre propio y literatura detras. Mismo audio, mismos numeros, siempre.
Devuelve MEDIDAS, no veredictos — el umbral y la decision son de quien conoce
el contexto (el que llama), no de esta libreria.

Por que existe (aurelius, 2026-09-11, audio real del telefono): el `no_speech`
de whisper salio con el signo al reves — cortaba habla real y dejaba pasar
frases inventadas sobre silencio. La energia en banda de voz y la fraccion de
tramos con F0 separaron conversacion de cuarto vacio con 0 % de solape.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VOICE_BAND_HZ = (300.0, 3400.0)   # banda telefonica: donde vive la inteligibilidad
F0_RANGE_HZ = (70.0, 350.0)       # de voz grave adulta a voz aguda
FRAME_S = 0.064                   # >= 4 periodos de 70 Hz: la autocorrelacion resuelve el F0 mas grave
HOP_S = 0.016
VOICING_THRESHOLD = 0.5           # autocorrelacion normalizada minima para llamar sonoro a un tramo
OCTAVE_COST = 0.05                # por octava de lag: entre dos picos casi iguales gana el periodo corto (Praat)
SILENCE_RMS = 1e-4                # -80 dBFS: por debajo no hay periodicidad que medir
_EPS = 1e-12


@dataclass(frozen=True)
class VoiceEvidence:
    voice_band_ratio: float | None   # energia 300-3400 Hz / total; None si no hay energia
    voiced_ratio: float              # fraccion de tramos con F0 en 70-350 Hz
    f0_median_hz: float | None       # mediana de F0 sobre los tramos sonoros; None si ninguno
    spectral_flatness: float | None  # 0 = tonal/picudo, 1 = ruido blanco; None si no hay energia
    frames: int                      # tramos analizados; 0 = audio mas corto que un tramo


def _frames(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def _normalized_autocorrelation(frames: np.ndarray) -> np.ndarray:
    """r[L] / sqrt(E(x[0:N-L]) * E(x[L:N])) por tramo — la normalizacion de Praat.

    Dividir por r[0] a secas castiga los lags largos (el solape encoge con L) y
    sesga el F0 hacia arriba. Con la energia de cada mitad, un tramo periodico
    da ~1 en su periodo sea cual sea el lag.
    """
    n = frames.shape[1]
    nfft = 2 * n
    spec = np.fft.rfft(frames, n=nfft, axis=1)
    ac = np.fft.irfft(spec * np.conj(spec), n=nfft, axis=1)[:, :n]
    cum = np.cumsum(np.square(frames), axis=1)
    total = cum[:, -1:]
    lags = np.arange(n)
    head = cum[:, n - 1 - lags]                      # energia de x[0 : N-L]
    tail = total - np.concatenate([np.zeros((len(frames), 1)), cum[:, :-1]], axis=1)  # x[L : N]
    return ac / np.sqrt(np.maximum(head * tail, _EPS))


def compute_voice_evidence(samples: np.ndarray, sample_rate: int) -> VoiceEvidence:
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("samples debe ser mono (1-D)")
    if sample_rate <= 0 or not np.isfinite(x).all():
        raise ValueError("audio y sample_rate deben ser validos")
    frame = round(sample_rate * FRAME_S)
    hop = max(1, round(sample_rate * HOP_S))
    if len(x) < frame:
        return VoiceEvidence(None, 0.0, None, None, 0)

    frames = _frames(x, frame, hop)
    frames = frames - frames.mean(axis=1, keepdims=True)   # sin DC: el offset no es voz
    power = np.abs(np.fft.rfft(frames * np.hanning(frame), axis=1)) ** 2
    freqs = np.fft.rfftfreq(frame, 1.0 / sample_rate)

    energy = power.sum(axis=1)
    total = float(energy.sum())
    band = (freqs >= VOICE_BAND_HZ[0]) & (freqs <= VOICE_BAND_HZ[1])
    voice_band_ratio = float(power[:, band].sum() / total) if total > _EPS else None

    active = energy > _EPS
    if active.any():
        p = power[active] + _EPS
        flatness = np.exp(np.mean(np.log(p), axis=1)) / np.mean(p, axis=1)
        spectral_flatness = float(np.mean(flatness))
    else:
        spectral_flatness = None

    # F0: pico LOCAL de la autocorrelacion normalizada dentro del rango de voz.
    # Exigir pico local (y no solo valor alto) es lo que deja fuera un zumbido
    # de 50 Hz: su autocorrelacion es alta en lags cortos pero baja en cuesta,
    # sin cima dentro del rango.
    lag_min = int(np.floor(sample_rate / F0_RANGE_HZ[1]))
    lag_max = min(int(np.ceil(sample_rate / F0_RANGE_HZ[0])), frame - 2)
    r = _normalized_autocorrelation(frames)
    peak = (r[:, 1:-1] > r[:, :-2]) & (r[:, 1:-1] >= r[:, 2:])
    in_range = np.zeros(frame - 2, dtype=bool)
    in_range[lag_min - 1:lag_max] = True                     # indice i <-> lag i+1
    # Una serie armonica pura vale ~1 en su periodo Y en sus multiplos: sin coste
    # de octava el argmax cae en el subarmonico por ruido de redondeo.
    lags = np.arange(1, frame - 1)
    octave_penalty = OCTAVE_COST * np.log2(lags / max(lag_min, 1))
    candidates = np.where(peak & in_range, r[:, 1:-1] - octave_penalty, -np.inf)
    best_idx = np.argmax(candidates, axis=1)
    best_r = r[:, 1:-1][np.arange(len(frames)), best_idx]
    best_r = np.where(np.isfinite(candidates[np.arange(len(frames)), best_idx]), best_r, -np.inf)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    voiced = (best_r >= VOICING_THRESHOLD) & (rms > SILENCE_RMS)

    voiced_ratio = float(np.mean(voiced))
    if voiced.any():
        f0_median_hz = float(np.median(sample_rate / (best_idx[voiced] + 1)))
    else:
        f0_median_hz = None

    return VoiceEvidence(
        voice_band_ratio=voice_band_ratio,
        voiced_ratio=voiced_ratio,
        f0_median_hz=f0_median_hz,
        spectral_flatness=spectral_flatness,
        frames=int(len(frames)),
    )
