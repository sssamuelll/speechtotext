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
    # Señales nativas de faster-whisper (Fase 2, G5): whisper.cpp no las emite -> None
    # de punta a punta. Al final, mismo patrón que src_dur.
    no_speech: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None


def native_signals(seg) -> tuple[float | None, float | None, float | None]:
    """Lee (no_speech, avg_logprob, compression_ratio) de un Segment, en sus dos
    dialectos: el crudo de faster-whisper (no_speech_prob) y el nuestro (no_speech).
    Ausente en ambos -> None (G5: señal que el motor no emite se omite, no se rellena)."""
    no_speech = getattr(seg, "no_speech_prob", None)
    if no_speech is None:
        no_speech = getattr(seg, "no_speech", None)
    return no_speech, getattr(seg, "avg_logprob", None), getattr(seg, "compression_ratio", None)
