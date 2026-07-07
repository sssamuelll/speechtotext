"""Test de integración de la diarización con pyannote.

Se salta salvo que haya HF_TOKEN (y los modelos gated aceptados). Verifica que el
camino real (importar pyannote perezosamente, cargar audio en memoria, correr el
pipeline y desempaquetar la salida 4.x) no explota y devuelve los tipos esperados.
"""
import os
import subprocess

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"),
    reason="requiere HF_TOKEN + modelos gated de pyannote aceptados",
)


def test_diarize_returns_turns_and_embeddings(tmp_path):
    from speechtotext.speakers.diarization import diarize

    wav = tmp_path / "tone.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=200:duration=3",
            "-ar", "16000", "-ac", "1", str(wav),
        ],
        check=True,
        capture_output=True,
    )
    turns, embeddings = diarize(str(wav))
    assert isinstance(turns, list)
    assert isinstance(embeddings, dict)
