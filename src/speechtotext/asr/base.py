from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from speechtotext.asr.types import TranscriptionRequest, TranscriptionResult

# Contrato de capacidades por motor. Regla madre: degradar con aviso cuando el resultado
# sigue siendo lo pedido con menos precision; rechazar cuando el knob seria inerte;
# jamas silencio ni sustitucion.
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


@runtime_checkable
class AsrBackend(Protocol):
    """Un motor de voz a texto. Entra float32 mono a 16 kHz; nada mas. Quien llama
    resamplea. El objeto es la cache del modelo: warm() carga una vez."""

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
    ) -> TranscriptionResult:
        ...
