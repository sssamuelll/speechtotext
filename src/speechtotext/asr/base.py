from __future__ import annotations

from typing import Protocol, runtime_checkable

from speechtotext.audio.types import AudioClip
from speechtotext.asr.types import TranscriptionRequest, TranscriptionResult
from speechtotext.models import VerifiedModelArtifact


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


@runtime_checkable
class CalibratedAsrBackend(AsrBackend, Protocol):
    @property
    def backend_artifact_kind(self) -> str: ...

    @property
    def backend_artifact_fingerprint(self) -> str: ...

    @property
    def config_fingerprint(self) -> str: ...


@runtime_checkable
class VerifiedLocalAsrBackend(CalibratedAsrBackend, Protocol):
    @property
    def model_artifact(self) -> VerifiedModelArtifact: ...
