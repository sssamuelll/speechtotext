from __future__ import annotations

import math
import re

from speechtotext.asr.types import TranscriptionResult
from speechtotext.audio.types import AudioQualityReport

ASR_FEATURE_NAMES = (
    "log_duration_s",
    "effective_voice_s",
    "processed_rms_dbfs",
    "snr_db",
    "clipping_ratio",
    "no_speech",
    "avg_logprob",
    "compression_ratio",
    "language_probability",
    "token_repetition_ratio",
    "language_matches",
    "missing_snr",
    "missing_native",
)


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def extract_asr_features(
    result: TranscriptionResult,
    quality: AudioQualityReport,
    *,
    expected_language: str,
) -> dict[str, float]:
    native = result.native_signals
    native_values = (
        native.no_speech,
        native.avg_logprob,
        native.compression_ratio,
        native.language_probability,
    )
    missing_native = any(value is None for value in native_values)
    tokens = _tokens(result.text)
    repetition = 1.0 - len(set(tokens)) / len(tokens) if tokens else 0.0
    values = {
        "log_duration_s": math.log1p(max(0.0, quality.duration_ms / 1000.0)),
        "effective_voice_s": max(0.0, quality.effective_voice_ms / 1000.0),
        "processed_rms_dbfs": quality.processed_rms_dbfs,
        "snr_db": quality.snr_db if quality.snr_db is not None else 0.0,
        "clipping_ratio": quality.clipping_ratio,
        "no_speech": native.no_speech if native.no_speech is not None else 0.0,
        "avg_logprob": native.avg_logprob if native.avg_logprob is not None else 0.0,
        "compression_ratio": (
            native.compression_ratio if native.compression_ratio is not None else 0.0
        ),
        "language_probability": (
            native.language_probability
            if native.language_probability is not None
            else 0.0
        ),
        "token_repetition_ratio": repetition,
        "language_matches": float(result.language == expected_language),
        "missing_snr": float(quality.snr_db is None),
        "missing_native": float(missing_native),
    }
    ordered = {name: float(values[name]) for name in ASR_FEATURE_NAMES}
    if not all(math.isfinite(value) for value in ordered.values()):
        raise ValueError("features ASR contienen valores no finitos")
    return ordered
