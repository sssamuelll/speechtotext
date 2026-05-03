"""Servicio HTTP de evaluación de pronunciación con Azure AI Speech.

Endpoints registrados desde speechtotext.api.routes:
    GET  /health
    POST /score    (multipart/form-data: audio, reference_text, language)

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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from speechtotext.api.config import settings
from speechtotext.api.routes import api_router


def create_app() -> FastAPI:
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
    app.include_router(api_router)
    return app


app = create_app()
