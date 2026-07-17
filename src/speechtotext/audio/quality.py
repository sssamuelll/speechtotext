from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from speechtotext.audio.types import AudioQualityReport, SpeechRegion

DBFS_FLOOR = -120.0
CLIP_LEVEL = 0.999


def _dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return DBFS_FLOOR
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return DBFS_FLOOR if rms <= 1e-6 else max(DBFS_FLOOR, 20.0 * math.log10(rms))


def _merge_regions(
    regions: Sequence[SpeechRegion], duration_s: float
) -> tuple[SpeechRegion, ...]:
    ordered = sorted(regions)
    merged: list[SpeechRegion] = []
    for region in ordered:
        if region.end_s > duration_s:
            raise ValueError("speech region fuera de la duracion del audio")
        if merged and region.start_s < merged[-1].end_s:
            raise ValueError("speech regions no pueden solaparse")
        merged.append(region)
    return tuple(merged)


def _region_samples(
    samples: np.ndarray,
    sample_rate: int,
    regions: Sequence[SpeechRegion],
) -> np.ndarray:
    chunks = [
        samples[round(region.start_s * sample_rate):round(region.end_s * sample_rate)]
        for region in regions
    ]
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)


def _non_speech_samples(
    samples: np.ndarray,
    sample_rate: int,
    regions: Sequence[SpeechRegion],
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    cursor = 0
    for region in regions:
        start = round(region.start_s * sample_rate)
        end = round(region.end_s * sample_rate)
        chunks.append(samples[cursor:start])
        cursor = end
    chunks.append(samples[cursor:])
    nonempty = [chunk for chunk in chunks if chunk.size]
    return np.concatenate(nonempty) if nonempty else np.empty(0, dtype=np.float32)


def _noise_floor(samples: np.ndarray, sample_rate: int) -> float:
    frame_size = max(1, round(sample_rate * 0.020))
    complete = len(samples) // frame_size
    if complete == 0:
        return _dbfs(samples)
    frames = samples[:complete * frame_size].reshape(complete, frame_size)
    levels = np.asarray([_dbfs(frame) for frame in frames], dtype=np.float64)
    return float(np.percentile(levels, 20.0))


def compute_audio_quality(
    capture: np.ndarray,
    processed: np.ndarray,
    sample_rate: int,
    speech_regions: Sequence[SpeechRegion],
    *,
    requested_gain_db: float,
    applied_gain_db: float,
    dropped_frames: int = 0,
    discontinuities: int = 0,
) -> AudioQualityReport:
    raw = np.asarray(capture, dtype=np.float32)
    cooked = np.asarray(processed, dtype=np.float32)
    if raw.ndim != 1 or cooked.ndim != 1 or len(raw) != len(cooked):
        raise ValueError("capture y processed deben ser mono y tener igual longitud")
    if sample_rate <= 0 or not np.isfinite(raw).all() or not np.isfinite(cooked).all():
        raise ValueError("audio y sample_rate deben ser validos")
    if dropped_frames < 0 or discontinuities < 0:
        raise ValueError("contadores de transporte no pueden ser negativos")
    duration_s = len(raw) / sample_rate
    regions = _merge_regions(speech_regions, duration_s)
    speech = _region_samples(cooked, sample_rate, regions)
    noise = _non_speech_samples(cooked, sample_rate, regions)
    noise_floor = _noise_floor(noise, sample_rate) if noise.size else None
    speech_level = _dbfs(speech) if speech.size else None
    snr = (
        speech_level - noise_floor
        if speech_level is not None and noise_floor is not None
        else None
    )
    peak = float(np.max(np.abs(cooked))) if cooked.size else 0.0
    warnings: list[str] = []
    if not regions:
        warnings.append("no_speech_regions")
    if applied_gain_db < requested_gain_db - 1e-9:
        warnings.append("gain_limited")
    return AudioQualityReport(
        duration_ms=round(duration_s * 1000),
        effective_voice_ms=round(
            sum(region.end_s - region.start_s for region in regions) * 1000
        ),
        input_rms_dbfs=_dbfs(raw),
        processed_rms_dbfs=_dbfs(cooked),
        peak_dbfs=DBFS_FLOOR if peak <= 1e-6 else 20.0 * math.log10(peak),
        clipping_ratio=float(np.mean(np.abs(cooked) >= CLIP_LEVEL)) if cooked.size else 0.0,
        noise_floor_dbfs=noise_floor,
        snr_db=snr,
        requested_gain_db=requested_gain_db,
        applied_gain_db=applied_gain_db,
        dropped_frames=dropped_frames,
        discontinuities=discontinuities,
        warnings=tuple(warnings),
    )
