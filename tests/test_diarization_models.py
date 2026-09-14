"""Tests de la diarización con pyannote.

El de integración se salta salvo que haya HF_TOKEN (y los modelos gated aceptados):
verifica que el camino real (importar pyannote perezosamente, cargar audio en memoria,
correr el pipeline y desempaquetar la salida 4.x) no explota y devuelve los tipos
esperados. El de configuración corre siempre, con un doble en sys.modules.
"""
import importlib.util
import os
import subprocess
import sys
import types

import pytest


# La guarda pide las DOS cosas. Con solo HF_TOKEN el test arrancaba en un entorno sin el
# extra [diarize] y moría importando pyannote: un fallo que no dice nada del código.
@pytest.mark.skipif(
    not os.environ.get("HF_TOKEN")
    or importlib.util.find_spec("pyannote.audio") is None,
    reason="requiere HF_TOKEN + modelos gated aceptados + el extra [diarize]",
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
    from speechtotext.speakers.diarization import read_wav

    samples, rate = read_wav(wav)
    turns, embeddings = diarize(samples, rate)
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

    assert creado == [(diarization.EMBEDDING_MODEL, os.environ.get("HF_TOKEN"))]
    assert pipe.embedding_batch_size == diarization._BATCH == 8
    assert pipe.segmentation_batch_size == diarization._BATCH == 8
    assert diarization._get_pipeline() is pipe  # sigue siendo singleton
    assert len(creado) == 1


def test_read_wav_promedia_canales_y_normaliza(tmp_path):
    import wave

    import numpy as np

    from speechtotext.speakers.diarization import read_wav

    izq = np.full(100, 16384, dtype="<i2")
    der = np.full(100, -16384, dtype="<i2")
    estereo = np.column_stack([izq, der]).reshape(-1)
    wav = tmp_path / "st.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(estereo.tobytes())
    samples, rate = read_wav(wav)
    assert rate == 8000 and samples.dtype == np.float32 and samples.shape == (100,)
    assert float(np.abs(samples).max()) == 0.0  # la media de +0.5 y -0.5


def test_waveform_para_pyannote_es_un_tensor_1xN():
    torch = pytest.importorskip("torch")
    import numpy as np

    from speechtotext.speakers.diarization import _waveform

    wf = _waveform(np.zeros(160, dtype=np.float32), 16000)
    assert wf["sample_rate"] == 16000
    assert isinstance(wf["waveform"], torch.Tensor) and tuple(wf["waveform"].shape) == (1, 160)
