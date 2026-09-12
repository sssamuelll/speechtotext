from __future__ import annotations

from typing import Protocol, runtime_checkable

from speechtotext.audio.types import AudioClip
from speechtotext.asr.types import TranscriptionRequest, TranscriptionResult


class AsrError(RuntimeError):
    def __init__(self, code: str, recoverable: bool, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.recoverable = recoverable


@runtime_checkable
class AsrBackend(Protocol):
    backend_id: str

    @property
    def model_id(self) -> str: ...

    @property
    def model_version(self) -> str: ...

    def warm(self) -> None:
        ...

    def transcribe(
        self,
        clip: AudioClip,
        request: TranscriptionRequest,
    ) -> TranscriptionResult:
        ...
