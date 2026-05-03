"""Servicio HTTP de evaluación de pronunciación con Azure AI Speech.

Endpoint principal:
    POST /score
        multipart/form-data:
            audio:           archivo de audio (cualquier formato soportado por ffmpeg)
            reference_text:  texto que el usuario debía pronunciar
            language:        BCP-47, por defecto "de-DE"

Variables de entorno:
    AZURE_SPEECH_KEY     clave del recurso Azure Speech (requerida)
    AZURE_SPEECH_REGION  región del recurso (default: westeurope)
    CORS_ORIGINS         orígenes permitidos separados por coma (default: *)

Arrancar en local:
    pip install -e ".[api]"
    export AZURE_SPEECH_KEY=...
    export AZURE_SPEECH_REGION=westeurope
    uvicorn speechtotext.api.app:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from speechtotext.api.azure_client import AzureSpeechError, score_pronunciation
from speechtotext.api.config import settings
from speechtotext.api.schemas import HealthResponse, ScoreResponse
from speechtotext.core.audio import FfmpegMissingError, TranscodeError, transcode_to_wav

app = FastAPI(
    title="Pronunciation Scoring API",
    version="0.1.0",
    description="Wrapper sobre Azure AI Speech Pronunciation Assessment.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        azure_configured=settings.azure_configured,
        azure_region=settings.azure_speech_region,
    )


@app.post("/score", response_model=ScoreResponse)
async def score(
    audio: UploadFile = File(..., description="Audio del usuario (webm/ogg/wav/mp3/m4a)."),
    reference_text: str = Form(..., description="Texto esperado que el usuario debía pronunciar."),
    language: str = Form("de-DE", description="Código BCP-47, p.ej. de-DE, en-US, es-ES."),
) -> ScoreResponse:
    if not settings.azure_configured:
        raise HTTPException(500, "AZURE_SPEECH_KEY no configurada en el servidor.")
    if not reference_text.strip():
        raise HTTPException(400, "reference_text no puede estar vacío.")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "El archivo de audio está vacío.")

    try:
        wav_path = transcode_to_wav(audio_bytes)
    except TranscodeError as e:
        raise HTTPException(400, f"No se pudo decodificar el audio: {e}")
    except FfmpegMissingError:
        raise HTTPException(500, "ffmpeg no está instalado en el servidor.")

    try:
        return score_pronunciation(
            wav_path,
            reference_text,
            language,
            azure_key=settings.azure_speech_key,
            azure_region=settings.azure_speech_region,
        )
    except AzureSpeechError as e:
        raise HTTPException(422 if e.recoverable else 502, str(e))
    finally:
        wav_path.unlink(missing_ok=True)
