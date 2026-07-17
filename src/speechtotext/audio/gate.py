from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from speechtotext.audio.types import AudioQualityReport

QualityReason = Literal[
    "silence",
    "too_short",
    "level_too_low",
    "snr_unavailable",
    "snr_too_low",
    "clipping",
    "dropped_audio",
    "discontinuous_audio",
]


@dataclass(frozen=True)
class QualityThresholds:
    min_effective_voice_ms: int
    min_processed_rms_dbfs: float
    min_snr_db: float
    max_clipping_ratio: float
    max_dropped_frames: int = 0
    max_discontinuities: int = 0

    def __post_init__(self) -> None:
        if self.min_effective_voice_ms <= 0:
            raise ValueError("min_effective_voice_ms debe ser positivo")
        if not math.isfinite(self.min_processed_rms_dbfs):
            raise ValueError("min_processed_rms_dbfs debe ser finito")
        if not math.isfinite(self.min_snr_db):
            raise ValueError("min_snr_db debe ser finito")
        if not math.isfinite(self.max_clipping_ratio):
            raise ValueError("max_clipping_ratio debe ser finito")
        if not 0.0 <= self.max_clipping_ratio <= 1.0:
            raise ValueError("max_clipping_ratio debe estar entre 0 y 1")
        if self.max_dropped_frames < 0 or self.max_discontinuities < 0:
            raise ValueError("los maximos de transporte no pueden ser negativos")


@dataclass(frozen=True)
class PreInferenceDecision:
    eligible: bool
    reason_codes: tuple[QualityReason, ...]


def evaluate_pre_inference(
    report: AudioQualityReport,
    thresholds: QualityThresholds,
) -> PreInferenceDecision:
    reasons: list[QualityReason] = []
    if report.effective_voice_ms == 0:
        reasons.append("silence")
    elif report.effective_voice_ms < thresholds.min_effective_voice_ms:
        reasons.append("too_short")
    if report.processed_rms_dbfs < thresholds.min_processed_rms_dbfs:
        reasons.append("level_too_low")
    if report.effective_voice_ms > 0:
        if report.snr_db is None:
            reasons.append("snr_unavailable")
        elif report.snr_db < thresholds.min_snr_db:
            reasons.append("snr_too_low")
    if report.clipping_ratio > thresholds.max_clipping_ratio:
        reasons.append("clipping")
    if report.dropped_frames > thresholds.max_dropped_frames:
        reasons.append("dropped_audio")
    if report.discontinuities > thresholds.max_discontinuities:
        reasons.append("discontinuous_audio")
    return PreInferenceDecision(eligible=not reasons, reason_codes=tuple(reasons))
