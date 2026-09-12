"""Un archivo entra, una transcripcion sale. Una sola decodificacion, un solo backend.

El nucleo NUNCA imprime: progreso por callback, avisos en Transcript.warnings, errores
como AsrError con codigo. El CLI, el MCP y la desktop pintan cada uno lo suyo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from speechtotext.asr.base import AsrBackend, AsrError
from speechtotext.asr.types import TranscriptionRequest
from speechtotext.core.segments import LabeledSegment

SAMPLE_RATE = 16000
ENGINE_FASTER = "faster-whisper"
ENGINE_WHISPERCPP = "whispercpp"
ENGINES = (ENGINE_FASTER, ENGINE_WHISPERCPP)

Stage = Literal["decode", "load", "transcribe", "diarize"]


@dataclass(frozen=True)
class Progress:
    stage: Stage
    done: float
    total: float | None      # None = indeterminado
    detail: str


ProgressCallback = Callable[[Progress], None]


@dataclass(frozen=True)
class Route:
    engine: str
    device: str
    compute_type: str
    reason: str              # una frase para imprimir; vacia si no hay nada que avisar


def resolve_route(engine: str = "auto", device: str = "auto", compute_type: str = "auto",
                  model: str = "large-v3") -> Route:
    """Resuelve engine/device/compute_type explicitos o 'auto'. ponytail: sin sondeo de la
    maquina todavia — 'auto' es faster-whisper en CPU int8; el sondeo llega en el plan 2b."""
    engine = ENGINE_FASTER if engine == "auto" else engine
    if engine not in ENGINES:
        raise ValueError(f"engine {engine!r} no existe; disponibles: {', '.join(ENGINES)}")
    if engine == ENGINE_WHISPERCPP:
        from speechtotext.core.enginepin import _MODEL_ALIAS

        if model not in _MODEL_ALIAS:
            raise ValueError(
                f"modelo {model!r} no está pinneado para whispercpp; disponibles: "
                f"{', '.join(sorted(_MODEL_ALIAS))}"
            )
        if compute_type not in ("auto", "q5_0"):
            raise ValueError(
                f"compute_type={compute_type!r} no soportado con whispercpp; usa 'auto' o 'q5_0'. "
                "Motivo: fp16 = 0.53x tiempo real por paging WDDM en la 980 (medido 2026-07-27)."
            )
        # El binario pinneado es build CUDA y corre en la GPU SIEMPRE (medido en el smoke).
        # Etiquetar cpu seria mentir en el header, la llave y el JSON: se declara cuda y
        # el remapeo se AVISA — pisar un -d cpu en silencio seria la sustitucion callada.
        reason = "" if device == "cuda" else "whisper.cpp (build CUDA) corre en la GPU; device=cuda"
        return Route(ENGINE_WHISPERCPP, "cuda", "q5_0", reason)
    device = "cpu" if device == "auto" else device
    if compute_type == "auto":
        compute_type = "int8" if device == "cpu" else "float16"
    return Route(ENGINE_FASTER, device, compute_type, "")


def make_backend(engine: str, model: str, device: str, compute_type: str) -> AsrBackend:
    """Los DOS motores se construyen aqui y solo aqui. Imports perezosos: no pagar
    faster_whisper si el motor es whisper.cpp, ni el pin si es faster-whisper."""
    if engine == ENGINE_FASTER:
        from speechtotext.asr.faster_whisper import FasterWhisperBackend, FasterWhisperConfig

        return FasterWhisperBackend(model, FasterWhisperConfig(device=device, compute_type=compute_type))
    if engine == ENGINE_WHISPERCPP:
        from speechtotext.asr.whispercpp import WhisperCppBackend

        return WhisperCppBackend(model)
    raise ValueError(f"engine desconocido: {engine!r}; disponibles: {ENGINES}")


def load_audio(path: Path) -> np.ndarray:
    """Decodifica UNA vez con PyAV a float32 mono 16 kHz. Lanza AudioDecodeError."""
    from speechtotext.audio.io import decode_audio

    with open(path, "rb") as stream:
        return decode_audio(stream, sample_rate=SAMPLE_RATE).samples


@dataclass(frozen=True)
class EngineInfo:
    name: str
    version: str
    model: str
    quant: str
    device: str
    selection: str = "explicit"
    diarization: str | None = None

    def to_dict(self) -> dict:
        d = {"name": self.name, "version": self.version, "model": self.model,
             "quant": self.quant, "device": self.device, "selection": self.selection}
        if self.diarization is not None:
            d["diarization"] = self.diarization
        return d


@dataclass(frozen=True)
class DiarizationReport:
    speakers: int
    unattributed_pct: int
    identified: int
    enrolled: int
    best_score: float | None   # el mejor coseno cuando nadie alcanzo el umbral
    auto: bool                 # True si el numero de hablantes no lo fijo el usuario


@dataclass
class Transcript:
    segments: list[LabeledSegment]
    language: str
    language_probability: float | None
    duration: float
    speech_s: float
    gaps: list[list[float]]
    engine: EngineInfo
    request: TranscriptionRequest        # la efectiva, tras CAPS
    warnings: tuple[str, ...] = ()
    diarization: DiarizationReport | None = None
