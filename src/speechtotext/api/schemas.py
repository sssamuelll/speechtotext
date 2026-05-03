"""Esquemas pydantic de las respuestas del API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    azure_configured: bool
    azure_region: str


class PhonemeScore(BaseModel):
    phoneme: str
    accuracy_score: float = Field(..., description="0–100, calidad del fonema.")


class WordScore(BaseModel):
    word: str
    accuracy_score: float
    error_type: str = Field(..., description="None | Mispronunciation | Omission | Insertion")
    phonemes: list[PhonemeScore] = []


class PronunciationScores(BaseModel):
    accuracy: float
    fluency: float
    completeness: float
    pronunciation: float


class ScoreResponse(BaseModel):
    recognized_text: str
    reference_text: str
    language: str
    scores: PronunciationScores
    words: list[WordScore]
