"""Segmento de transcripción con hablante opcional."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LabeledSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None
