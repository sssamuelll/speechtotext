import json

from speechtotext.core.finder import index_path, load_or_build_index


def test_index_path_deterministic_and_model_sensitive(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x" * 100)
    p1 = index_path(audio, "tiny")
    p2 = index_path(audio, "tiny")
    p3 = index_path(audio, "base")
    assert p1 == p2
    assert p1 != p3


def test_load_uses_seeded_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x" * 100)
    seed = {"segments": [{"start": 0.0, "end": 1.0, "text": "hola"}]}
    index_path(audio, "tiny").write_text(json.dumps(seed), encoding="utf-8")

    segments, cached = load_or_build_index(audio, "tiny", rebuild=False)
    assert cached is True
    assert segments == [{"start": 0.0, "end": 1.0, "text": "hola"}]
