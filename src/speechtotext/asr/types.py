from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptionRequest:
    language: str = "es"
    hotwords: tuple[str, ...] = ()
    word_timestamps: bool = True
    beam_size: int = 5
    context: str | None = None
    vad: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.language, str)
            or not self.language.strip()
            or len(self.language) > 32
        ):
            raise ValueError("language is required")
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
            raise TypeError("hotwords must be a tuple of unique, bounded strings")
        if type(self.word_timestamps) is not bool:
            raise TypeError("word_timestamps must be bool")
        if type(self.beam_size) is not int or not 1 <= self.beam_size <= 100:
            raise TypeError("beam_size must be an integer between 1 and 100")
        if self.context is not None and (
            not isinstance(self.context, str)
            or not self.context.strip()
            or len(self.context) > 4096
        ):
            raise TypeError("context must be a non-empty, bounded string")
        if type(self.vad) is not bool:
            raise TypeError("vad must be bool")

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
            "vad": self.vad,
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
            raise TypeError("word text must be a string")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in (self.start, self.end)
        ):
            raise TypeError("word timestamps must be numeric")
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError("word timestamps must be finite")
        if self.start < 0.0 or self.end < self.start:
            raise ValueError("invalid word timestamps")
        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(
                self.confidence, (int, float)
            ):
                raise TypeError("word confidence must be numeric")
            if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
                raise ValueError("word confidence must be between 0 and 1")


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
        raise TypeError("native signals must be numeric or None")
    probabilities = (no_speech, language_probability)
    if any(
        value is not None
        and (not math.isfinite(value) or not 0.0 <= value <= 1.0)
        for value in probabilities
    ):
        raise ValueError("native probability out of range")
    if avg_logprob is not None and not math.isfinite(avg_logprob):
        raise ValueError("avg_logprob must be finite")
    if compression_ratio is not None and (
        not math.isfinite(compression_ratio) or compression_ratio <= 0.0
    ):
        raise ValueError("compression_ratio must be finite and positive")


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
            raise TypeError("segment timestamps must be numeric")
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError("segment timestamps must be finite")
        if self.start < 0.0 or self.end < self.start:
            raise ValueError("invalid segment timestamps")
        if not isinstance(self.text, str):
            raise TypeError("segment text must be a string")
        if not isinstance(self.words, tuple) or any(
            not isinstance(word, TranscriptionWord) for word in self.words
        ):
            raise TypeError("segment words must be a tuple of TranscriptionWord")
        if not isinstance(self.native_signals, SegmentNativeSignals):
            raise TypeError("invalid segment native_signals")


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
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("ASR text must be a string")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (
                self.language,
                self.backend,
                self.model,
                self.model_version,
            )
        ):
            raise ValueError("ASR language and identity are required")
        if (
            isinstance(self.latency_ms, bool)
            or not isinstance(self.latency_ms, int)
            or self.latency_ms < 0
        ):
            raise ValueError("latency_ms must be a non-negative integer")
        if not isinstance(self.native_signals, NativeSignals):
            raise TypeError("invalid native_signals")
        if not isinstance(self.words, tuple) or any(
            not isinstance(word, TranscriptionWord) for word in self.words
        ):
            raise TypeError("words must be a tuple of TranscriptionWord")
        if not isinstance(self.segments, tuple) or any(
            not isinstance(segment, TranscriptionSegment) for segment in self.segments
        ):
            raise TypeError("segments must be a tuple of TranscriptionSegment")
        if (
            not isinstance(self.warnings, tuple)
            or any(
                not isinstance(warning, str) or not warning.strip()
                for warning in self.warnings
            )
            or len(set(self.warnings)) != len(self.warnings)
        ):
            raise TypeError("warnings must be a tuple of unique strings")
