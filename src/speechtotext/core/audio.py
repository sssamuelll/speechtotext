"""Shared audio utilities: transcoding to 16 kHz mono PCM WAV via ffmpeg."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


class TranscodeError(RuntimeError):
    """ffmpeg could not decode the audio."""


class FfmpegMissingError(RuntimeError):
    """ffmpeg is not available on the system PATH."""


def transcode_to_wav(src_bytes: bytes, *, sample_rate: int = 16_000) -> Path:
    """Convert any audio (webm/ogg/mp3/m4a/...) to mono PCM WAV.

    Return the path to a temporary file; the caller is responsible for deleting it.
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
        raise FfmpegMissingError("ffmpeg is not installed on the system PATH.") from e
    except subprocess.CalledProcessError as e:
        Path(src.name).unlink(missing_ok=True)
        msg = e.stderr.decode(errors="ignore")[:300] if e.stderr else "ffmpeg error"
        raise TranscodeError(msg) from e
    finally:
        Path(src.name).unlink(missing_ok=True)
    return dst
