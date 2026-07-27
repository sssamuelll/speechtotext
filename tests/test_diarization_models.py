"""Tests de la diarización con pyannote.

El de integración se salta salvo que haya HF_TOKEN (y los modelos gated aceptados):
verifica que el camino real (importar pyannote perezosamente, cargar audio en memoria,
correr el pipeline y desempaquetar la salida 4.x) no explota y devuelve los tipos
esperados. El de configuración corre siempre, con un doble en sys.modules.
"""
import os
import subprocess
import sys
import types

import pytest


@pytest.mark.skipif(
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


def test_get_pipeline_baja_los_batch_sizes(monkeypatch):
    """El pico de RAM de pyannote lo manda el batch, y el checkpoint trae 32 en su
    config.yaml. Si alguien borra las dos asignaciones el pico se duplica en silencio:
    no falla ningún test de salida, sólo se come 1.2 GB más. Este test es el guardián."""
    from speechtotext.speakers import diarization

    creado = []

    class _PipelineDoble:
        @staticmethod
        def from_pretrained(name, token=None):
            # El doble no trae los atributos: si _get_pipeline no los asigna, el assert
            # de abajo revienta con AttributeError en vez de pasar por casualidad.
            obj = types.SimpleNamespace()
            creado.append((name, token))
            return obj

    fake = types.ModuleType("pyannote.audio")
    fake.Pipeline = _PipelineDoble
    # El import de pyannote.audio es perezoso (dentro de _get_pipeline), así que sembrar
    # sys.modules antes de la llamada evita cargar torch/pyannote de verdad.
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake)
    # El singleton es global: monkeypatch lo restaura y no contamina los demás tests.
    monkeypatch.setattr(diarization, "_PIPELINE", None)

    pipe = diarization._get_pipeline()

    assert creado == [(diarization._PIPELINE_NAME, os.environ.get("HF_TOKEN"))]
    assert pipe.embedding_batch_size == diarization._BATCH == 8
    assert pipe.segmentation_batch_size == diarization._BATCH == 8
    assert diarization._get_pipeline() is pipe  # sigue siendo singleton
    assert len(creado) == 1
