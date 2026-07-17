from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Literal

ConfidenceTarget = Literal["segment_usable"]


@dataclass(frozen=True)
class TranscriptionRequest:
    language: str = "es"
    hotwords: tuple[str, ...] = ()
    word_timestamps: bool = True
    beam_size: int = 5
    context: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.language, str)
            or not self.language.strip()
            or len(self.language) > 32
        ):
            raise ValueError("language es obligatorio")
        if (
            not isinstance(self.hotwords, tuple)
            or len(self.hotwords) > 128
            or any(
                not isinstance(item, str)
                or not item.strip()
                or len(item) > 256
                for item in self.hotwords
            )
            or len(set(self.hotwords)) != len(self.hotwords)
        ):
            raise TypeError("hotwords debe ser tuple de strings unicos y acotados")
        if type(self.word_timestamps) is not bool:
            raise TypeError("word_timestamps debe ser bool")
        if type(self.beam_size) is not int or not 1 <= self.beam_size <= 100:
            raise TypeError("beam_size debe ser entero entre 1 y 100")
        if self.context is not None and (
            not isinstance(self.context, str)
            or not self.context.strip()
            or len(self.context) > 4096
        ):
            raise TypeError("context debe ser string no vacio y acotado")

    @property
    def fingerprint(self) -> str:
        payload = self.to_dict()
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "beam_size": self.beam_size,
            "context": self.context,
            "hotwords": list(self.hotwords),
            "language": self.language,
            "word_timestamps": self.word_timestamps,
        }

    def to_fingerprint_dict(self) -> dict[str, object]:
        return {
            "schema_version": "speechtotext.transcription-request/v1",
            "parameters": self.to_dict(),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class TranscriptionWord:
    text: str
    start: float
    end: float
    confidence: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text de palabra debe ser string")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in (self.start, self.end)
        ):
            raise TypeError("timestamps de palabra deben ser numericos")
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError("timestamps de palabra deben ser finitos")
        if self.start < 0.0 or self.end < self.start:
            raise ValueError("timestamps de palabra invalidos")
        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(
                self.confidence, (int, float)
            ):
                raise TypeError("confidence de palabra debe ser numerica")
            if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
                raise ValueError("confidence de palabra debe estar entre 0 y 1")


def _validate_native_signals(
    no_speech: float | None,
    avg_logprob: float | None,
    compression_ratio: float | None,
    language_probability: float | None,
) -> None:
    values = (no_speech, avg_logprob, compression_ratio, language_probability)
    if any(
        value is not None
        and (isinstance(value, bool) or not isinstance(value, (int, float)))
        for value in values
    ):
        raise TypeError("senales nativas deben ser numericas o None")
    probabilities = (no_speech, language_probability)
    if any(
        value is not None
        and (not math.isfinite(value) or not 0.0 <= value <= 1.0)
        for value in probabilities
    ):
        raise ValueError("probabilidad nativa fuera de rango")
    if avg_logprob is not None and not math.isfinite(avg_logprob):
        raise ValueError("avg_logprob debe ser finito")
    if compression_ratio is not None and (
        not math.isfinite(compression_ratio) or compression_ratio <= 0.0
    ):
        raise ValueError("compression_ratio debe ser finito y positivo")


@dataclass(frozen=True)
class SegmentNativeSignals:
    no_speech: float | None
    avg_logprob: float | None
    compression_ratio: float | None

    def __post_init__(self) -> None:
        _validate_native_signals(
            self.no_speech,
            self.avg_logprob,
            self.compression_ratio,
            None,
        )


@dataclass(frozen=True)
class TranscriptionSegment:
    start: float
    end: float
    text: str
    words: tuple[TranscriptionWord, ...]
    native_signals: SegmentNativeSignals

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in (self.start, self.end)
        ):
            raise TypeError("timestamps de segmento deben ser numericos")
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError("timestamps de segmento deben ser finitos")
        if self.start < 0.0 or self.end < self.start:
            raise ValueError("timestamps de segmento invalidos")
        if not isinstance(self.text, str):
            raise TypeError("text de segmento debe ser string")
        if not isinstance(self.words, tuple) or any(
            not isinstance(word, TranscriptionWord) for word in self.words
        ):
            raise TypeError("words de segmento debe ser tuple de TranscriptionWord")
        if not isinstance(self.native_signals, SegmentNativeSignals):
            raise TypeError("native_signals de segmento invalido")


@dataclass(frozen=True)
class NativeSignals:
    no_speech: float | None
    avg_logprob: float | None
    compression_ratio: float | None
    language_probability: float | None

    def __post_init__(self) -> None:
        _validate_native_signals(
            self.no_speech,
            self.avg_logprob,
            self.compression_ratio,
            self.language_probability,
        )


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str
    words: tuple[TranscriptionWord, ...]
    segments: tuple[TranscriptionSegment, ...]
    backend: str
    model: str
    model_version: str
    latency_ms: int
    native_signals: NativeSignals
    confidence_target: ConfidenceTarget
    calibrated_confidence: float | None
    calibrator_version: str | None
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text ASR debe ser string")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (
                self.language,
                self.backend,
                self.model,
                self.model_version,
            )
        ):
            raise ValueError("lenguaje e identidad ASR son obligatorios")
        if (
            isinstance(self.latency_ms, bool)
            or not isinstance(self.latency_ms, int)
            or self.latency_ms < 0
        ):
            raise ValueError("latency_ms debe ser entero no negativo")
        if not isinstance(self.native_signals, NativeSignals):
            raise TypeError("native_signals invalido")
        if self.confidence_target != "segment_usable":
            raise ValueError("confidence_target incompatible")
        if not isinstance(self.words, tuple) or any(
            not isinstance(word, TranscriptionWord) for word in self.words
        ):
            raise TypeError("words debe ser tuple de TranscriptionWord")
        if not isinstance(self.segments, tuple) or any(
            not isinstance(segment, TranscriptionSegment) for segment in self.segments
        ):
            raise TypeError("segments debe ser tuple de TranscriptionSegment")
        if (
            not isinstance(self.warnings, tuple)
            or any(
                not isinstance(warning, str) or not warning.strip()
                for warning in self.warnings
            )
            or len(set(self.warnings)) != len(self.warnings)
        ):
            raise TypeError("warnings debe ser tuple de strings unicos")
        if self.calibrated_confidence is not None:
            if (
                isinstance(self.calibrated_confidence, bool)
                or not isinstance(self.calibrated_confidence, (int, float))
                or not math.isfinite(self.calibrated_confidence)
                or not 0.0 <= self.calibrated_confidence <= 1.0
            ):
                raise ValueError("calibrated_confidence debe estar entre 0 y 1")
            if (
                not isinstance(self.calibrator_version, str)
                or len(self.calibrator_version) != 64
                or any(
                    char not in "0123456789abcdef"
                    for char in self.calibrator_version
                )
            ):
                raise ValueError("calibrator_version es obligatorio con confianza")
        if self.calibrated_confidence is None and self.calibrator_version is not None:
            raise ValueError("calibrator_version exige calibrated_confidence")
