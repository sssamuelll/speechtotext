"""Segmento de transcripción con hablante opcional."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LabeledSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None
    # Extensión del segmento que el ASR emitió, antes de que la diarización recomprima
    # el span a sus palabras (C-13). Al final y con default: todas las construcciones
    # existentes son posicionales y no se rompe ninguna. None = "soy el span original".
    src_dur: float | None = None
