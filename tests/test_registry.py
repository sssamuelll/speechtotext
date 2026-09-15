import json

import numpy as np
import pytest

from speechtotext.speakers import registry

PYANNOTE = "pyannote/speaker-diarization-community-1"
SHERPA = "sherpa-onnx/3dspeaker-eres2net"


def test_enroll_list_get_remove_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    registry.enroll("Alice", vec, seconds=12.0, model=PYANNOTE)

    voices = registry.list_voices()
    assert [v["name"] for v in voices] == ["Alice"]
    assert voices[0]["seconds"] == 12.0
    assert voices[0]["model"] == PYANNOTE

    got = registry.get_embeddings(PYANNOTE)
    assert np.allclose(got["Alice"], vec)

    assert registry.remove("Alice") is True
    assert registry.list_voices() == []
    assert registry.remove("Alice") is False


def test_home_respects_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    assert registry.home() == tmp_path


def test_get_embeddings_skips_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    registry.enroll("Alice", np.array([1.0, 2.0], dtype=np.float32), seconds=10.0, model=PYANNOTE)
    for npy in tmp_path.rglob("*.npy"):
        npy.unlink()
    assert registry.get_embeddings(PYANNOTE) == {}


def test_get_embeddings_only_returns_voices_for_that_model(tmp_path, monkeypatch):
    """A sherpa-onnx embedding and a pyannote embedding live in different vector spaces:
    the cosine between them is meaningless, and if their dimensions match it does not
    even fail—it produces a garbage score. The registry cannot return them mixed."""
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    registry.enroll("Alice", np.array([1.0, 0.0], dtype=np.float32), seconds=10.0, model=PYANNOTE)
    registry.enroll("Bob", np.array([0.0, 1.0], dtype=np.float32), seconds=10.0, model=SHERPA)

    assert list(registry.get_embeddings(PYANNOTE)) == ["Alice"]
    assert list(registry.get_embeddings(SHERPA)) == ["Bob"]
    assert registry.get_embeddings("other/model") == {}


def test_get_embeddings_requires_the_model(tmp_path, monkeypatch):
    """If the model becomes optional again, the bug returns: callers compare blindly."""
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    with pytest.raises(TypeError):
        registry.get_embeddings()


def test_enrolling_the_same_voice_in_two_models_preserves_both_embeddings(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    pyannote_vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    sherpa_vec = np.array([9.0, 8.0], dtype=np.float32)

    registry.enroll("Alice", pyannote_vec, seconds=10.0, model=PYANNOTE)
    registry.enroll("Alice", sherpa_vec, seconds=30.0, model=SHERPA)

    assert np.allclose(registry.get_embeddings(PYANNOTE)["Alice"], pyannote_vec)
    assert np.allclose(registry.get_embeddings(SHERPA)["Alice"], sherpa_vec)


def test_list_voices_filters_by_model(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    registry.enroll("Alice", np.array([1.0], dtype=np.float32), seconds=10.0, model=PYANNOTE)
    registry.enroll("Bob", np.array([1.0], dtype=np.float32), seconds=10.0, model=SHERPA)

    assert [v["name"] for v in registry.list_voices()] == ["Alice", "Bob"]
    assert [v["name"] for v in registry.list_voices(SHERPA)] == ["Bob"]


def test_remove_deletes_only_the_enrollment_for_the_requested_model(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    registry.enroll("Alice", np.array([1.0], dtype=np.float32), seconds=10.0, model=PYANNOTE)
    registry.enroll("Alice", np.array([2.0], dtype=np.float32), seconds=10.0, model=SHERPA)

    assert registry.remove("Alice", model=SHERPA) is True
    assert registry.get_embeddings(SHERPA) == {}
    assert list(registry.get_embeddings(PYANNOTE)) == ["Alice"]


def test_reads_the_flat_v0_4_manifest(tmp_path, monkeypatch):
    """Voices already enrolled with v0.4 are not lost during an upgrade: the flat format
    is read under the model already declared by each entry."""
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    voices = tmp_path / "voices"
    voices.mkdir(parents=True)
    np.save(voices / "Alice.npy", np.array([1.0, 2.0, 3.0], dtype=np.float32))
    (voices / "manifest.json").write_text(
        json.dumps(
            {
                "Alice": {
                    "file": "Alice.npy",
                    "seconds": 12.0,
                    "model": PYANNOTE,
                    "enrolled_at": "2026-07-01T10:00:00",
                }
            }
        ),
        encoding="utf-8",
    )

    assert np.allclose(registry.get_embeddings(PYANNOTE)["Alice"], [1.0, 2.0, 3.0])
    assert registry.get_embeddings(SHERPA) == {}
    assert registry.list_voices()[0]["model"] == PYANNOTE
