"""Deterministic DSP voice evidence.

None of the measurements here is a model's opinion: they are named signal
descriptors backed by literature. Same audio, same numbers, every time.
Returns MEASUREMENTS, not verdicts — the threshold and decision belong to whoever
knows the context (the caller), not this library.

Why it exists (measured 2026-09-11 on real phone audio): Whisper's `no_speech`
came out with the sign reversed — it cut real speech and let invented phrases
through on silence. Energy in the voice band separated conversation from an
empty room with 0 % overlap per window.

Declared limitations (measured in the adversarial review that same day):

- A periodic tone WITHIN or above the band (ringtone, alarm, feedback at 1 kHz)
  yields `voice_band_ratio` ~1 and `voiced_ratio` ~1 with a subharmonic F0:
  autocorrelation has peaks at every multiple of the period, and the 70-350 Hz
  range captures them. `spectral_flatness` (a single spectral line) gives it
  away. The caller combines all three measurements.
- F0 at integer lag resolution (~1 Hz at 140 Hz, ~3 Hz at 350 Hz), without
  interpolation or candidate tracking between frames: a source with ONLY even
  harmonics is reported an octave higher, because that IS its period.
- Fixed 64 ms frame and 16 ms hop; one sample rate per call.
- Memory bounded by batches of frames: peak memory does not depend on duration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

VOICE_BAND_HZ = (300.0, 3400.0)   # telephone band: where intelligibility lives
F0_RANGE_HZ = (70.0, 350.0)       # from a low adult voice to a high voice
FRAME_S = 0.064                   # >= 4 periods of 70 Hz: autocorrelation resolves the lowest F0
HOP_S = 0.016
VOICING_THRESHOLD = 0.5           # minimum normalized autocorrelation for calling a frame voiced
OCTAVE_COST = 0.05                # per lag octave: between two nearly equal peaks, the short period wins (Praat: 0.01)
SILENCE_RMS = 1e-4                # -80 dBFS: below this a frame is not MEASURED (neither voice nor non-voice)
BATCH_FRAMES = 512                # ~8 s at 16 kHz per batch: tens of MB peak, not gigabytes
_FLOOR = 1e-12                    # floor RELATIVE to each spectrum's maximum, not absolute


@dataclass(frozen=True)
class VoiceEvidence:
    """All measurements are `None` when no frame exceeds `SILENCE_RMS`:
    "could not measure" and "there is no voice" are different things, and `0.0`
    only means the latter."""

    voice_band_ratio: float | None   # 300-3400 Hz energy / total, weighted by frame
    voiced_ratio: float | None       # fraction of ALL frames with F0 in 70-350 Hz
    f0_median_hz: float | None       # median F0 over voiced frames; None if there are none
    spectral_flatness: float | None  # 0 = tonal/peaked, 1 = white noise; energy-weighted
    frames: int                      # frames analyzed; 0 = audio shorter than one frame

    def __post_init__(self) -> None:
        for name in ("voice_band_ratio", "voiced_ratio", "spectral_flatness"):
            v = getattr(self, name)
            if v is not None and not (math.isfinite(v) and 0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0, 1] or None")
        if self.f0_median_hz is not None and not (
            math.isfinite(self.f0_median_hz) and self.f0_median_hz > 0.0
        ):
            raise ValueError("f0_median_hz must be positive and finite, or None")
        if type(self.frames) is not int or self.frames < 0:
            raise ValueError("frames must be a non-negative integer")


def _normalized_autocorrelation(frames: np.ndarray) -> np.ndarray:
    """r[L] / sqrt(E(x[0:N-L]) * E(x[L:N])) per frame — Praat normalization.

    Dividing by r[0] alone penalizes long lags (overlap shrinks with L) and
    biases F0 upward. With the energy of each half, a periodic frame yields ~1
    at its period regardless of lag. No window on purpose: this normalization
    IS the compensation that a window would force us to undo.
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

    view = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop]   # no copy
    window = np.hanning(frame)
    freqs = np.fft.rfftfreq(frame, 1.0 / sample_rate)
    band = (freqs >= VOICE_BAND_HZ[0]) & (freqs <= VOICE_BAND_HZ[1])
    lags = np.arange(1, frame - 1)
    in_range = (lags >= lag_min) & (lags <= lag_max)
    # A pure harmonic series yields ~1 at its period AND its multiples: without
    # octave cost, argmax lands on the subharmonic due to rounding noise.
    octave_penalty = OCTAVE_COST * np.log2(lags / lag_min)

    total_energy = band_energy = weighted_flatness = 0.0
    measurable_count = voiced_count = 0
    voiced_lags: list[np.ndarray] = []

    for start in range(0, len(view), BATCH_FRAMES):
        batch = view[start:start + BATCH_FRAMES]
        batch = batch - batch.mean(axis=1, keepdims=True)      # no DC: the offset is not voice
        rms = np.sqrt(np.mean(np.square(batch), axis=1))
        measurable = rms > SILENCE_RMS
        if not measurable.any():
            continue
        batch = batch[measurable]
        measurable_count += int(len(batch))

        power = np.abs(np.fft.rfft(batch * window, axis=1)) ** 2
        energy = power.sum(axis=1)
        total_energy += float(energy.sum())
        band_energy += float(power[:, band].sum())
        # Floor relative to the maximum of EACH spectrum: with an absolute EPS,
        # the geometric mean depended on volume (4e9x between 0 and -40 dB). And it
        # is energy-weighted: a plain mean tracked the pause fraction, not the spectrum.
        p = np.maximum(power, power.max(axis=1, keepdims=True) * _FLOOR)
        flatness = np.exp(np.mean(np.log(p), axis=1)) / np.mean(p, axis=1)
        weighted_flatness += float((flatness * energy).sum())

        # F0: LOCAL peak of normalized autocorrelation within the voice range.
        # Requiring a local peak (not just a high value) is what excludes a 50 Hz hum:
        # high at short lags but sloping downward, with no summit in range.
        r = _normalized_autocorrelation(batch)
        peak = (r[:, 1:-1] > r[:, :-2]) & (r[:, 1:-1] >= r[:, 2:])
        candidates = np.where(peak & in_range, r[:, 1:-1] - octave_penalty, -np.inf)
        best_idx = np.argmax(candidates, axis=1)
        rows = np.arange(len(batch))
        has_peak = np.isfinite(candidates[rows, best_idx])
        voiced = has_peak & (r[:, 1:-1][rows, best_idx] >= VOICING_THRESHOLD)
        voiced_count += int(voiced.sum())
        if voiced.any():
            voiced_lags.append(best_idx[voiced] + 1)

    frames = int(len(view))
    if measurable_count == 0:
        return VoiceEvidence(None, None, None, None, frames)
    f0 = None
    if voiced_lags:
        f0 = float(np.median(sample_rate / np.concatenate(voiced_lags)))
    return VoiceEvidence(
        voice_band_ratio=band_energy / total_energy,
        voiced_ratio=voiced_count / frames,
        f0_median_hz=f0,
        spectral_flatness=weighted_flatness / total_energy,
        frames=frames,
    )
