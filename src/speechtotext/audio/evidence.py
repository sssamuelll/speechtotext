"""Evidencia de voz por DSP determinista.

Ninguna medida de aqui es la opinion de un modelo: son descriptores de senal
con nombre propio y literatura detras. Mismo audio, mismos numeros, siempre.
Devuelve MEDIDAS, no veredictos — el umbral y la decision son de quien conoce
el contexto (el que llama), no de esta libreria.

Por que existe (medido 2026-09-11 sobre audio real de telefono): el `no_speech`
de whisper salio con el signo al reves — cortaba habla real y dejaba pasar
frases inventadas sobre silencio. La energia en banda de voz separo
conversacion de cuarto vacio con 0 % de solape por ventana.

Limites declarados (medidos en la revision adversarial del mismo dia):

- Un tono periodico DENTRO o por encima de la banda (timbre, alarma, acople a
  1 kHz) da `voice_band_ratio` ~1 y `voiced_ratio` ~1 con un F0 subarmonico:
  la autocorrelacion tiene picos en todos los multiplos del periodo y el rango
  70-350 Hz los recoge. Lo que lo delata es `spectral_flatness` (una sola
  linea espectral). El que llama combina las tres medidas.
- F0 a resolucion de lag entero (~1 Hz a 140 Hz, ~3 Hz a 350 Hz), sin
  interpolacion ni seguimiento de candidatos entre tramos: una fuente con
  SOLO armonicos pares se reporta una octava arriba, porque ese ES su periodo.
- Tramo de 64 ms y salto de 16 ms fijos; una tasa de muestreo por llamada.
- Memoria acotada por lotes de tramos: el pico no depende de la duracion.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

VOICE_BAND_HZ = (300.0, 3400.0)   # banda telefonica: donde vive la inteligibilidad
F0_RANGE_HZ = (70.0, 350.0)       # de voz grave adulta a voz aguda
FRAME_S = 0.064                   # >= 4 periodos de 70 Hz: la autocorrelacion resuelve el F0 mas grave
HOP_S = 0.016
VOICING_THRESHOLD = 0.5           # autocorrelacion normalizada minima para llamar sonoro a un tramo
OCTAVE_COST = 0.05                # por octava de lag: entre dos picos casi iguales gana el periodo corto (Praat: 0.01)
SILENCE_RMS = 1e-4                # -80 dBFS: por debajo un tramo no se MIDE (ni voz ni no-voz)
BATCH_FRAMES = 512                # ~8 s a 16 kHz por lote: decenas de MB de pico, no gigas
_FLOOR = 1e-12                    # suelo RELATIVO al maximo de cada espectro, no absoluto


@dataclass(frozen=True)
class VoiceEvidence:
    """Todas las medidas son `None` cuando ningun tramo supera `SILENCE_RMS`:
    «no pude medir» y «no hay voz» son cosas distintas, y `0.0` solo significa
    la segunda."""

    voice_band_ratio: float | None   # energia 300-3400 Hz / total, ponderada por tramo
    voiced_ratio: float | None       # fraccion de TODOS los tramos con F0 en 70-350 Hz
    f0_median_hz: float | None       # mediana de F0 sobre los tramos sonoros; None si ninguno
    spectral_flatness: float | None  # 0 = tonal/picudo, 1 = ruido blanco; ponderada por energia
    frames: int                      # tramos analizados; 0 = audio mas corto que un tramo

    def __post_init__(self) -> None:
        for nombre in ("voice_band_ratio", "voiced_ratio", "spectral_flatness"):
            v = getattr(self, nombre)
            if v is not None and not (math.isfinite(v) and 0.0 <= v <= 1.0):
                raise ValueError(f"{nombre} must be in [0, 1] or None")
        if self.f0_median_hz is not None and not (
            math.isfinite(self.f0_median_hz) and self.f0_median_hz > 0.0
        ):
            raise ValueError("f0_median_hz must be positive and finite, or None")
        if type(self.frames) is not int or self.frames < 0:
            raise ValueError("frames must be a non-negative integer")


def _normalized_autocorrelation(frames: np.ndarray) -> np.ndarray:
    """r[L] / sqrt(E(x[0:N-L]) * E(x[L:N])) por tramo — la normalizacion de Praat.

    Dividir por r[0] a secas castiga los lags largos (el solape encoge con L) y
    sesga el F0 hacia arriba. Con la energia de cada mitad, un tramo periodico
    da ~1 en su periodo sea cual sea el lag. Sin ventana a proposito: esta
    normalizacion ES la compensacion que una ventana obligaria a deshacer.
    """
    n = frames.shape[1]
    nfft = 2 * n
    spec = np.fft.rfft(frames, n=nfft, axis=1)
    ac = np.fft.irfft(spec * np.conj(spec), n=nfft, axis=1)[:, :n]
    cum = np.cumsum(np.square(frames), axis=1)
    total = cum[:, -1:]
    lags = np.arange(n)
    head = cum[:, n - 1 - lags]                                              # x[0 : N-L]
    tail = total - np.concatenate([np.zeros((len(frames), 1)), cum[:, :-1]], axis=1)  # x[L : N]
    return ac / np.sqrt(np.maximum(head * tail, _FLOOR * _FLOOR))


def compute_voice_evidence(samples: np.ndarray, sample_rate: int) -> VoiceEvidence:
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("samples must be mono (1-D)")
    if sample_rate <= 0 or not np.isfinite(x).all():
        raise ValueError("audio and sample_rate must be valid")
    frame = round(sample_rate * FRAME_S)
    hop = max(1, round(sample_rate * HOP_S))
    lag_min = int(np.floor(sample_rate / F0_RANGE_HZ[1]))
    lag_max = int(np.ceil(sample_rate / F0_RANGE_HZ[0]))
    if lag_min < 1 or lag_max + 2 > frame:
        raise ValueError("sample_rate too low to resolve F0 in 70-350 Hz")
    if len(x) < frame:
        return VoiceEvidence(None, None, None, None, 0)

    vista = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop]   # sin copia
    window = np.hanning(frame)
    freqs = np.fft.rfftfreq(frame, 1.0 / sample_rate)
    band = (freqs >= VOICE_BAND_HZ[0]) & (freqs <= VOICE_BAND_HZ[1])
    lags = np.arange(1, frame - 1)
    in_range = (lags >= lag_min) & (lags <= lag_max)
    # Una serie armonica pura vale ~1 en su periodo Y en sus multiplos: sin coste
    # de octava el argmax cae en el subarmonico por ruido de redondeo.
    octave_penalty = OCTAVE_COST * np.log2(lags / lag_min)

    energia_total = energia_banda = planitud_ponderada = 0.0
    medibles = sonoros = 0
    lags_sonoros: list[np.ndarray] = []

    for inicio in range(0, len(vista), BATCH_FRAMES):
        lote = vista[inicio:inicio + BATCH_FRAMES]
        lote = lote - lote.mean(axis=1, keepdims=True)         # sin DC: el offset no es voz
        rms = np.sqrt(np.mean(np.square(lote), axis=1))
        medible = rms > SILENCE_RMS
        if not medible.any():
            continue
        lote = lote[medible]
        medibles += int(len(lote))

        power = np.abs(np.fft.rfft(lote * window, axis=1)) ** 2
        energia = power.sum(axis=1)
        energia_total += float(energia.sum())
        energia_banda += float(power[:, band].sum())
        # Suelo relativo al maximo de CADA espectro: con un EPS absoluto la media
        # geometrica dependia del volumen (4e9x entre 0 y -40 dB). Y se pondera
        # por energia: una media plana seguia la fraccion de pausa, no el espectro.
        p = np.maximum(power, power.max(axis=1, keepdims=True) * _FLOOR)
        planitud = np.exp(np.mean(np.log(p), axis=1)) / np.mean(p, axis=1)
        planitud_ponderada += float((planitud * energia).sum())

        # F0: pico LOCAL de la autocorrelacion normalizada dentro del rango de voz.
        # Exigir pico local (y no solo valor alto) es lo que deja fuera un zumbido
        # de 50 Hz: alto en lags cortos pero bajando en cuesta, sin cima en rango.
        r = _normalized_autocorrelation(lote)
        peak = (r[:, 1:-1] > r[:, :-2]) & (r[:, 1:-1] >= r[:, 2:])
        candidates = np.where(peak & in_range, r[:, 1:-1] - octave_penalty, -np.inf)
        best_idx = np.argmax(candidates, axis=1)
        filas = np.arange(len(lote))
        hay_pico = np.isfinite(candidates[filas, best_idx])
        voiced = hay_pico & (r[:, 1:-1][filas, best_idx] >= VOICING_THRESHOLD)
        sonoros += int(voiced.sum())
        if voiced.any():
            lags_sonoros.append(best_idx[voiced] + 1)

    frames = int(len(vista))
    if medibles == 0:
        return VoiceEvidence(None, None, None, None, frames)
    f0 = None
    if lags_sonoros:
        f0 = float(np.median(sample_rate / np.concatenate(lags_sonoros)))
    return VoiceEvidence(
        voice_band_ratio=energia_banda / energia_total,
        voiced_ratio=sonoros / frames,
        f0_median_hz=f0,
        spectral_flatness=planitud_ponderada / energia_total,
        frames=frames,
    )
