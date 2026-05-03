"""Servicio HTTP de evaluación de pronunciación con Azure AI Speech.

Endpoint principal:
    POST /score
        multipart/form-data:
            audio:           archivo de audio (cualquier formato soportado por ffmpeg)
            reference_text:  texto que el usuario debía pronunciar
            language:        BCP-47, por defecto "de-DE"

Variables de entorno requeridas:
    AZURE_SPEECH_KEY     clave del recurso Azure Speech
    AZURE_SPEECH_REGION  región del recurso (p.ej. "westeurope")

Arrancar en local:
    pip install -e ".[api]"
    export AZURE_SPEECH_KEY=...
    export AZURE_SPEECH_REGION=westeurope
    uvicorn api:app --reload --port 8000 --app-dir src
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import azure.cognitiveservices.speech as speechsdk
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "westeurope")

app = FastAPI(
    title="German Pronunciation Scoring API",
    version="0.1.0",
    description="Wrapper sobre Azure AI Speech Pronunciation Assessment.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


def _transcode_to_wav(src_bytes: bytes) -> Path:
    src = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    src.write(src_bytes)
    src.close()
    dst = Path(src.name).with_suffix(".wav")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", src.name,
                "-ar", "16000", "-ac", "1", "-f", "wav", str(dst),
            ],
            check=True, capture_output=True,
        )
    finally:
        Path(src.name).unlink(missing_ok=True)
    return dst


def _score_with_azure(wav_path: Path, reference_text: str, language: str) -> dict:
    speech_config = speechsdk.SpeechConfig(
        subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION
    )
    speech_config.speech_recognition_language = language

    audio_config = speechsdk.audio.AudioConfig(filename=str(wav_path))

    pronunciation_config = speechsdk.PronunciationAssessmentConfig(
        reference_text=reference_text,
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
        enable_miscue=True,
    )

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )
    pronunciation_config.apply_to(recognizer)

    result = recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.NoMatch:
        raise HTTPException(422, "No se detectó voz en el audio.")
    if result.reason == speechsdk.ResultReason.Canceled:
        details = speechsdk.CancellationDetails(result)
        raise HTTPException(502, f"Azure canceló la petición: {details.reason} — {details.error_details}")
    if result.reason != speechsdk.ResultReason.RecognizedSpeech:
        raise HTTPException(502, f"Resultado inesperado: {result.reason}")

    pa = speechsdk.PronunciationAssessmentResult(result)

    words = []
    for w in pa.words:
        phonemes = [
            {"phoneme": p.phoneme, "accuracy_score": p.accuracy_score}
            for p in (w.phonemes or [])
        ]
        words.append({
            "word": w.word,
            "accuracy_score": w.accuracy_score,
            "error_type": w.error_type,
            "phonemes": phonemes,
        })

    return {
        "recognized_text": result.text,
        "reference_text": reference_text,
        "language": language,
        "scores": {
            "accuracy": pa.accuracy_score,
            "fluency": pa.fluency_score,
            "completeness": pa.completeness_score,
            "pronunciation": pa.pronunciation_score,
        },
        "words": words,
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "azure_configured": bool(AZURE_SPEECH_KEY),
        "azure_region": AZURE_SPEECH_REGION,
    }


@app.post("/score")
async def score_pronunciation(
    audio: UploadFile = File(..., description="Audio del usuario (webm/ogg/wav/mp3/m4a)."),
    reference_text: str = Form(..., description="Texto esperado que el usuario debía pronunciar."),
    language: str = Form("de-DE", description="Código BCP-47, p.ej. de-DE, en-US, es-ES."),
) -> dict:
    if not AZURE_SPEECH_KEY:
        raise HTTPException(500, "AZURE_SPEECH_KEY no configurada en el servidor.")
    if not reference_text.strip():
        raise HTTPException(400, "reference_text no puede estar vacío.")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "El archivo de audio está vacío.")

    try:
        wav_path = _transcode_to_wav(audio_bytes)
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode(errors="ignore")[:300] if e.stderr else "ffmpeg error"
        raise HTTPException(400, f"No se pudo decodificar el audio: {msg}")
    except FileNotFoundError:
        raise HTTPException(500, "ffmpeg no está instalado en el servidor.")

    try:
        return _score_with_azure(wav_path, reference_text, language)
    finally:
        wav_path.unlink(missing_ok=True)
