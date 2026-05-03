"""Health check del servicio."""
from __future__ import annotations

from fastapi import APIRouter

from speechtotext.api.config import settings
from speechtotext.api.schemas import HealthResponse

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        azure_configured=settings.azure_configured,
        azure_region=settings.azure_speech_region,
    )
