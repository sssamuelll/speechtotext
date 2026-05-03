"""Utilidades de audio compartidas: transcoding a WAV PCM 16 kHz mono via ffmpeg."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


class TranscodeError(RuntimeError):
    """ffmpeg no pudo decodificar el audio."""


class FfmpegMissingError(RuntimeError):
    """ffmpeg no está disponible en el PATH del sistema."""


def transcode_to_wav(src_bytes: bytes, *, sample_rate: int = 16_000) -> Path:
    """Convierte cualquier audio (webm/ogg/mp3/m4a/...) a WAV PCM mono.

    Devuelve la ruta a un archivo temporal; el llamante es responsable de borrarlo.
    """
    src = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    src.write(src_bytes)
    src.close()
    dst = Path(src.name).with_suffix(".wav")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", src.name,
                "-ar", str(sample_rate), "-ac", "1", "-f", "wav", str(dst),
            ],
            check=True, capture_output=True,
        )
    except FileNotFoundError as e:
        Path(src.name).unlink(missing_ok=True)
        raise FfmpegMissingError("ffmpeg no está instalado en el PATH del sistema.") from e
    except subprocess.CalledProcessError as e:
        Path(src.name).unlink(missing_ok=True)
        msg = e.stderr.decode(errors="ignore")[:300] if e.stderr else "ffmpeg error"
        raise TranscodeError(msg) from e
    finally:
        Path(src.name).unlink(missing_ok=True)
    return dst
