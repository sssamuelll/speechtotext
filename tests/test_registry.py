import json

import numpy as np
import pytest

from speechtotext.speakers import registry

PYANNOTE = "pyannote/speaker-diarization-community-1"
SHERPA = "sherpa-onnx/3dspeaker-eres2net"


def test_enroll_list_get_remove_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    registry.enroll("Samuel", vec, seconds=12.0, model=PYANNOTE)

    voices = registry.list_voices()
    assert [v["name"] for v in voices] == ["Samuel"]
    assert voices[0]["seconds"] == 12.0
    assert voices[0]["model"] == PYANNOTE

    got = registry.get_embeddings(PYANNOTE)
    assert np.allclose(got["Samuel"], vec)

    assert registry.remove("Samuel") is True
    assert registry.list_voices() == []
    assert registry.remove("Samuel") is False


def test_home_respects_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    assert registry.home() == tmp_path


def test_get_embeddings_skips_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    registry.enroll("Samuel", np.array([1.0, 2.0], dtype=np.float32), seconds=10.0, model=PYANNOTE)
    for npy in tmp_path.rglob("*.npy"):
        npy.unlink()
    assert registry.get_embeddings(PYANNOTE) == {}


def test_get_embeddings_solo_devuelve_las_voces_de_ese_modelo(tmp_path, monkeypatch):
    """Un embedding de sherpa-onnx y uno de pyannote viven en espacios vectoriales
    distintos: el coseno entre ellos no significa nada, y si las dimensiones coinciden
    ni siquiera falla — puntúa basura. El registro no puede devolverlos mezclados."""
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    registry.enroll("Samuel", np.array([1.0, 0.0], dtype=np.float32), seconds=10.0, model=PYANNOTE)
    registry.enroll("Ale", np.array([0.0, 1.0], dtype=np.float32), seconds=10.0, model=SHERPA)

    assert list(registry.get_embeddings(PYANNOTE)) == ["Samuel"]
    assert list(registry.get_embeddings(SHERPA)) == ["Ale"]
    assert registry.get_embeddings("otro/modelo") == {}


def test_get_embeddings_exige_el_modelo(tmp_path, monkeypatch):
    """Si el modelo vuelve a ser opcional, vuelve el bug: quien llama compara a ciegas."""
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    with pytest.raises(TypeError):
        registry.get_embeddings()


def test_la_misma_persona_en_dos_modelos_no_se_pisa(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    pyannote_vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    sherpa_vec = np.array([9.0, 8.0], dtype=np.float32)

    registry.enroll("Samuel", pyannote_vec, seconds=10.0, model=PYANNOTE)
    registry.enroll("Samuel", sherpa_vec, seconds=30.0, model=SHERPA)

    assert np.allclose(registry.get_embeddings(PYANNOTE)["Samuel"], pyannote_vec)
    assert np.allclose(registry.get_embeddings(SHERPA)["Samuel"], sherpa_vec)


def test_list_voices_filtra_por_modelo(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    registry.enroll("Samuel", np.array([1.0], dtype=np.float32), seconds=10.0, model=PYANNOTE)
    registry.enroll("Ale", np.array([1.0], dtype=np.float32), seconds=10.0, model=SHERPA)

    assert [v["name"] for v in registry.list_voices()] == ["Ale", "Samuel"]
    assert [v["name"] for v in registry.list_voices(SHERPA)] == ["Ale"]


def test_remove_borra_solo_el_modelo_pedido(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    registry.enroll("Samuel", np.array([1.0], dtype=np.float32), seconds=10.0, model=PYANNOTE)
    registry.enroll("Samuel", np.array([2.0], dtype=np.float32), seconds=10.0, model=SHERPA)

    assert registry.remove("Samuel", model=SHERPA) is True
    assert registry.get_embeddings(SHERPA) == {}
    assert list(registry.get_embeddings(PYANNOTE)) == ["Samuel"]


def test_lee_el_manifiesto_plano_de_v0_4(tmp_path, monkeypatch):
    """Las voces ya registradas con v0.4 no se pierden al actualizar: el formato plano
    se lee bajo el modelo que cada entrada ya declaraba."""
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    voices = tmp_path / "voices"
    voices.mkdir(parents=True)
    np.save(voices / "Samuel.npy", np.array([1.0, 2.0, 3.0], dtype=np.float32))
    (voices / "manifest.json").write_text(
        json.dumps(
            {
                "Samuel": {
                    "file": "Samuel.npy",
                    "seconds": 12.0,
                    "model": PYANNOTE,
                    "enrolled_at": "2026-07-01T10:00:00",
                }
            }
        ),
        encoding="utf-8",
    )

    assert np.allclose(registry.get_embeddings(PYANNOTE)["Samuel"], [1.0, 2.0, 3.0])
    assert registry.get_embeddings(SHERPA) == {}
    assert registry.list_voices()[0]["model"] == PYANNOTE
