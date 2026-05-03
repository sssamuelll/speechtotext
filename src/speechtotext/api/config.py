"""Configuración del servicio HTTP, leída de variables de entorno."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    azure_speech_key: str
    azure_speech_region: str
    cors_origins: tuple[str, ...]

    @property
    def azure_configured(self) -> bool:
        return bool(self.azure_speech_key)


def load_settings() -> Settings:
    raw_origins = os.getenv("CORS_ORIGINS", "*")
    origins = tuple(o.strip() for o in raw_origins.split(",") if o.strip())
    return Settings(
        azure_speech_key=os.getenv("AZURE_SPEECH_KEY", ""),
        azure_speech_region=os.getenv("AZURE_SPEECH_REGION", "westeurope"),
        cors_origins=origins or ("*",),
    )


settings = load_settings()
