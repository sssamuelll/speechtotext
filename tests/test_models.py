"""core.models: per-system paths, inventory, download, and deletion. huggingface_hub is
never actually imported: stubs in sys.modules (same pattern as tests/test_enginepin.py)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from speechtotext.core import enginepin, models


# --- data_dir -----------------------------------------------------------------------------

def test_data_dir_honors_speechtotext_home(monkeypatch, tmp_path):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path / "casa"))
    assert models.data_dir() == tmp_path / "casa"


@pytest.mark.parametrize("platform, expected", [
    ("win32", Path("C:/local") / "speechtotext"),
    ("darwin", "~/Library/Application Support/speechtotext"),
    ("linux", "~/.local/share/speechtotext"),
])
def test_data_dir_depends_on_the_operating_system(monkeypatch, tmp_path, platform, expected):
    monkeypatch.delenv("SPEECHTOTEXT_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", "C:/local")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(models.sys, "platform", platform)
    expected = expected if isinstance(expected, Path) else tmp_path / expected[2:]
    assert models.data_dir() == expected


def test_data_dir_on_linux_honors_xdg(monkeypatch, tmp_path):
    monkeypatch.delenv("SPEECHTOTEXT_HOME", raising=False)
    monkeypatch.setattr(models.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert models.data_dir() == tmp_path / "xdg" / "speechtotext"


def test_install_root_is_under_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    assert enginepin.install_root() == tmp_path / "whisper-cpp" / enginepin.ENGINE_PIN["version"]


# --- installed ----------------------------------------------------------------------------

def _hf_stub(monkeypatch, repos):
    """Stub huggingface_hub.scan_cache_dir with the given repos [(repo_id, path, size)];
    repos=None simulates a nonexistent cache (CacheNotFound)."""
    class CacheNotFound(Exception):
        pass

    def scan_cache_dir():
        if repos is None:
            raise CacheNotFound("no cache")
        return SimpleNamespace(repos=[
            SimpleNamespace(repo_id=r, repo_path=p, size_on_disk=s) for r, p, s in repos])

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(scan_cache_dir=scan_cache_dir))
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", SimpleNamespace(CacheNotFound=CacheNotFound))


def test_installed_lists_faster_whisper_models_from_the_hf_cache(monkeypatch, tmp_path):
    _hf_stub(monkeypatch, [
        ("Systran/faster-whisper-small", tmp_path / "models--Systran--faster-whisper-small", 486_212_372),
        ("pyannote/segmentation-3.0", tmp_path / "otro", 5_905_440),      # not one of our models
        ("Systran/faster-whisper-large-v3", tmp_path / "models--Systran--faster-whisper-large-v3", 3_090_839_273),
    ])
    got = models.installed("faster-whisper")
    assert [(m.name, m.size_bytes, m.verified) for m in got] == [
        ("large-v3", 3_090_839_273, False), ("small", 486_212_372, False)]
    assert got[0].path == tmp_path / "models--Systran--faster-whisper-large-v3"
    assert got[0].engine == "faster-whisper"


def test_installed_without_an_hf_cache_is_an_empty_list(monkeypatch):
    _hf_stub(monkeypatch, None)
    assert models.installed("faster-whisper") == []


def test_installed_lists_whispercpp_models_with_their_verification_status(monkeypatch):
    _hf_stub(monkeypatch, [])
    root = enginepin.install_root() / "models"
    root.mkdir(parents=True)
    small = enginepin.MODELS_PIN["small"]
    (root / small["filename"]).write_bytes(b"ggml")
    (root / (small["filename"] + ".verified")).write_text(small["sha256"], encoding="utf-8")
    (root / enginepin.MODELS_PIN["large-v3-q5_0"]["filename"]).write_bytes(b"ggml-grande")
    got = models.installed()
    assert [(m.engine, m.name, m.size_bytes, m.verified) for m in got] == [
        ("whispercpp", "large-v3", 11, False), ("whispercpp", "small", 4, True)]


def test_installed_rejects_an_unknown_engine():
    with pytest.raises(ValueError, match="does not exist"):
        models.installed("chatgpt")


# --- ensure -------------------------------------------------------------------------------

def test_ensure_faster_whisper_downloads_the_snapshot_and_emits_progress(monkeypatch, tmp_path):
    calls = []

    def snapshot_download(repo_id, allow_patterns=None):
        calls.append((repo_id, allow_patterns))
        return str(tmp_path / "snap")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    events = []
    got = models.ensure("faster-whisper", "small", on_progress=events.append)
    assert got == tmp_path / "snap"
    assert calls == [("Systran/faster-whisper-small", models._FW_FILES)]
    assert [(e.stage, e.done, e.total) for e in events] == [("download", 0, None), ("download", 1, 1)]


def test_ensure_whispercpp_delegates_to_enginepin(monkeypatch, tmp_path):
    monkeypatch.setattr(enginepin, "ensure_model", lambda name: tmp_path / f"{name}.bin")
    events = []
    assert models.ensure("whispercpp", "large-v3", on_progress=events.append) == tmp_path / "large-v3.bin"
    assert events[0].detail == "ggml-large-v3-q5_0.bin" and events[-1].done == 1


def test_ensure_with_an_unknown_name_lists_the_available_models():
    with pytest.raises(ValueError, match="available") as ei:
        models.ensure("faster-whisper", "gigante")
    assert "large-v3" in str(ei.value)
    with pytest.raises(ValueError, match="available"):
        models.ensure("whispercpp", "medium")
    with pytest.raises(ValueError, match="does not exist"):
        models.ensure("chatgpt", "small")


# --- remote_size --------------------------------------------------------------------------

def test_remote_size_for_whispercpp_comes_from_the_pin():
    assert models.remote_size("whispercpp", "small") == enginepin.MODELS_PIN["small"]["size_bytes"] == 487_601_967
    assert models.remote_size("whispercpp", "large-v3") == 1_081_140_203


def test_remote_size_for_faster_whisper_sums_only_what_gets_downloaded(monkeypatch):
    siblings = [
        SimpleNamespace(rfilename="README.md", size=2052),
        SimpleNamespace(rfilename="model.bin", size=3_087_284_237),
        SimpleNamespace(rfilename="vocabulary.json", size=1_068_114),
        SimpleNamespace(rfilename=".gitattributes", size=1519),
    ]

    class HfApi:
        def model_info(self, repo, files_metadata=False):
            assert (repo, files_metadata) == ("Systran/faster-whisper-large-v3", True)
            return SimpleNamespace(siblings=siblings)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=HfApi))
    assert models.remote_size("faster-whisper", "large-v3") == 3_087_284_237 + 1_068_114


def test_remote_size_without_a_network_is_none(monkeypatch):
    class HfApi:
        def model_info(self, *a, **k):
            raise ConnectionError("sin red")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=HfApi))
    assert models.remote_size("faster-whisper", "small") is None


# --- remove -------------------------------------------------------------------------------

def test_remove_faster_whisper_deletes_the_repository_from_the_cache(monkeypatch, tmp_path):
    repo = tmp_path / "models--Systran--faster-whisper-small"
    (repo / "snapshots").mkdir(parents=True)
    _hf_stub(monkeypatch, [("Systran/faster-whisper-small", repo, 10)])
    models.remove("faster-whisper", "small")
    assert not repo.exists()


def test_remove_whispercpp_deletes_the_binary_and_marker(monkeypatch):
    _hf_stub(monkeypatch, [])
    root = enginepin.install_root() / "models"
    root.mkdir(parents=True)
    fn = enginepin.MODELS_PIN["small"]["filename"]
    (root / fn).write_bytes(b"x")
    (root / (fn + ".verified")).write_text("sha", encoding="utf-8")
    models.remove("whispercpp", "small")
    assert not (root / fn).exists() and not (root / (fn + ".verified")).exists()


def test_remove_for_a_missing_model_fails_loudly_with_its_name(monkeypatch):
    _hf_stub(monkeypatch, [])
    with pytest.raises(FileNotFoundError, match="small"):
        models.remove("whispercpp", "small")
