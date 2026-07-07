import numpy as np

from speechtotext.speakers import registry


def test_enroll_list_get_remove_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    registry.enroll("Samuel", vec, seconds=12.0, model="pyannote/embedding")

    voices = registry.list_voices()
    assert [v["name"] for v in voices] == ["Samuel"]
    assert voices[0]["seconds"] == 12.0

    got = registry.get_embeddings()
    assert np.allclose(got["Samuel"], vec)

    assert registry.remove("Samuel") is True
    assert registry.list_voices() == []
    assert registry.remove("Samuel") is False


def test_home_respects_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    assert registry.home() == tmp_path


def test_get_embeddings_skips_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    registry.enroll("Samuel", np.array([1.0, 2.0], dtype=np.float32), seconds=10.0, model="m")
    (tmp_path / "voices" / "Samuel.npy").unlink()
    assert registry.get_embeddings() == {}
