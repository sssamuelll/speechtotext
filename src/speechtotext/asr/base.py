from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Literal, Protocol, runtime_checkable

import numpy as np

from speechtotext.asr.types import TranscriptionRequest, TranscriptionResult, TranscriptionSegment

# Capability contract per engine. Guiding rule: degrade with a warning when the result
# is still what was requested with less precision; reject when the knob would be inert;
# never silence or substitution.
Cap = Literal["honored", "degraded", "rejected"]


@dataclass(frozen=True)
class Caps:
    hotwords: Cap
    vad: Cap
    word_timestamps: Cap


class AsrError(RuntimeError):
    def __init__(self, code: str, recoverable: bool, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.recoverable = recoverable


def raise_if_cancelled(cancel: threading.Event | None) -> None:
    """The one way to stop: the same code and message wherever the check happens."""
    if cancel is not None and cancel.is_set():
        raise AsrError("cancelled", True, "transcription cancelled")


@runtime_checkable
class AsrBackend(Protocol):
    """A speech-to-text engine. It accepts float32 mono at 16 kHz; nothing else. The caller
    resamples. The object is the model cache: warm() loads it once."""

    backend_id: str
    caps: Caps

    @property
    def model_id(self) -> str: ...

    @property
    def model_version(self) -> str: ...

    @property
    def engine_version(self) -> str: ...

    @property
    def quant(self) -> str: ...

    @property
    def device(self) -> str: ...

    def warm(self) -> None:
        ...

    def transcribe(
        self,
        samples: np.ndarray,
        request: TranscriptionRequest,
        *,
        on_segment: Callable[[TranscriptionSegment], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> TranscriptionResult:
        """Each segment goes to `on_segment` as soon as the engine has it, in order, with
        times local to `samples`. `cancel` is checked between segments: once it is set,
        this raises AsrError("cancelled") and returns nothing partial."""
        ...
