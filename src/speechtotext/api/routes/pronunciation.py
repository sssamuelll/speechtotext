"""Endpoint POST /score: evalúa la pronunciación de un audio contra un texto de referencia."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from speechtotext.api.azure_client import AzureSpeechError, score_pronunciation
from speechtotext.api.config import settings
from speechtotext.api.schemas import ScoreResponse
from speechtotext.core.audio import FfmpegMissingError, TranscodeError, transcode_to_wav

router = APIRouter(tags=["pronunciation"])


@router.post("/score", response_model=ScoreResponse)
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
