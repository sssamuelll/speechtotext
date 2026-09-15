from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
from typing import Callable

import numpy as np

from speechtotext.asr.base import AsrError, Caps
from speechtotext.asr.types import (
    NativeSignals,
    SegmentNativeSignals,
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
)


@dataclass(frozen=True)
class FasterWhisperConfig:
    device: str = "cpu"
    compute_type: str = "int8"
    cpu_threads: int = 0
    num_workers: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.device, str)
            or not self.device.strip()
            or not isinstance(self.compute_type, str)
            or not self.compute_type.strip()
        ):
            raise ValueError("device/compute_type are required")
        if (
            type(self.cpu_threads) is not int
            or self.cpu_threads < 0
            or type(self.num_workers) is not int
            or self.num_workers < 1
        ):
            raise ValueError("cpu_threads/num_workers invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "speechtotext.faster-whisper-config/v1",
            "device": self.device,
            "compute_type": self.compute_type,
            "cpu_threads": self.cpu_threads,
            "num_workers": self.num_workers,
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=True, allow_nan=False,
            separators=(",", ":"), sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class FasterWhisperBackend:
    backend_id = "faster-whisper"
    caps = Caps(hotwords="honored", vad="honored", word_timestamps="honored")

    def __init__(
        self,
        model: str | Path,
        config: FasterWhisperConfig | None = None,
        *,
        model_version: str = "unpinned",
        model_factory: Callable | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        # Un nombre lo resuelve faster-whisper (cache de HF Hub); una ruta es un
        # directorio CTranslate2 y se carga solo local. Este backend no verifica pesos:
        # quien lo haga por manifiesto lo envuelve, pasa la ruta verificada y su
        # revision como model_version.
        if isinstance(model, Path):
            self._model_path: Path | None = model
            self._model_id = model.name
        elif isinstance(model, str):
            if not model.strip():
                raise ValueError("model cannot be empty")
            self._model_path = None
            self._model_id = model
        else:
            raise TypeError("model must be a name (str) or a path (Path)")
        if not isinstance(model_version, str) or not model_version.strip():
            raise ValueError("model_version cannot be empty")
        self.config = config if config is not None else FasterWhisperConfig()
        self._model_version = model_version
        self._model_factory = model_factory
        self._clock = clock
        self._model = None

    @property
    def config_fingerprint(self) -> str:
        return self.config.fingerprint

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def engine_version(self) -> str:
        try:
            return f"faster-whisper {_pkg_version('faster-whisper')}"
        except PackageNotFoundError:
            return "faster-whisper"

    @property
    def quant(self) -> str:
        return self.config.compute_type

    @property
    def device(self) -> str:
        return self.config.device

    def warm(self) -> None:
        if self._model is not None:
            return
        factory = self._model_factory
        if factory is None:
            from faster_whisper import WhisperModel

            factory = WhisperModel
        self._model = factory(
            str(self._model_path) if self._model_path is not None else self._model_id,
            device=self.config.device,
            compute_type=self.config.compute_type,
            cpu_threads=self.config.cpu_threads,
            num_workers=self.config.num_workers,
            local_files_only=self._model_path is not None,
        )

    def transcribe(
        self,
        samples: np.ndarray,
        request: TranscriptionRequest,
    ) -> TranscriptionResult:
        self.warm()
        started = self._clock()
        try:
            raw_segments, info = self._model.transcribe(
                samples,
                language=None if request.language == "auto" else request.language,
                beam_size=request.beam_size,
                vad_filter=request.vad,
                hotwords=", ".join(request.hotwords) or None,
                initial_prompt=request.context,
                condition_on_previous_text=False,
                word_timestamps=request.word_timestamps,
            )
            raw_segments = list(raw_segments)
        except AsrError:
            raise
        except Exception as exc:
            raise AsrError("backend_failed", True, str(exc)) from exc
        elapsed_ms = round((self._clock() - started) * 1000)
        segments: list[TranscriptionSegment] = []
        all_words: list[TranscriptionWord] = []
        weights: list[float] = []
        no_speech: list[float] = []
        logprobs: list[tuple[float, float]] = []
        compression: list[float] = []
        for raw in raw_segments:
            words = tuple(
                TranscriptionWord(
                    text=word.word,
                    start=float(word.start),
                    end=float(word.end),
                    confidence=(
                        float(word.probability)
                        if getattr(word, "probability", None) is not None
                        else None
                    ),
                )
                for word in (getattr(raw, "words", None) or ())
            )
            signals = SegmentNativeSignals(
                no_speech=(
                    float(raw.no_speech_prob)
                    if getattr(raw, "no_speech_prob", None) is not None
                    else None
                ),
                avg_logprob=(
                    float(raw.avg_logprob)
                    if getattr(raw, "avg_logprob", None) is not None
                    else None
                ),
                compression_ratio=(
                    float(raw.compression_ratio)
                    if getattr(raw, "compression_ratio", None) is not None
                    else None
                ),
            )
            segment = TranscriptionSegment(
                float(raw.start),
                float(raw.end),
                raw.text,
                words,
                signals,
            )
            segments.append(segment)
            all_words.extend(words)
            weight = max(0.001, segment.end - segment.start)
            weights.append(weight)
            if signals.no_speech is not None:
                no_speech.append(signals.no_speech)
            if signals.avg_logprob is not None:
                logprobs.append((signals.avg_logprob, weight))
            if signals.compression_ratio is not None:
                compression.append(signals.compression_ratio)
        avg_logprob = (
            sum(value * weight for value, weight in logprobs)
            / sum(weight for _, weight in logprobs)
            if logprobs
            else None
        )
        text = "".join(raw.text for raw in raw_segments).strip()
        warnings: list[str] = []
        if not text:
            warnings.append("empty_transcript")
        language = str(getattr(info, "language", None) or request.language)
        if request.language != "auto" and language != request.language:
            warnings.append("language_mismatch")
        return TranscriptionResult(
            text=text,
            language=language,
            words=tuple(all_words),
            segments=tuple(segments),
            backend=self.backend_id,
            model=self._model_id,
            model_version=self._model_version,
            latency_ms=elapsed_ms,
            native_signals=NativeSignals(
                no_speech=max(no_speech) if no_speech else None,
                avg_logprob=avg_logprob,
                compression_ratio=max(compression) if compression else None,
                language_probability=(
                    float(info.language_probability)
                    if getattr(info, "language_probability", None) is not None
                    else None
                ),
            ),
            warnings=tuple(warnings),
        )
