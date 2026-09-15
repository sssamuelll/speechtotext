import hashlib
import io
import shutil
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import speechtotext.core.enginepin as enginepin
from speechtotext.core.enginepin import (
    ENGINE_PIN,
    MODELS_PIN,
    ensure_engine,
    ensure_model,
    install_root,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- Literal contract pins --------------------------------------------------------

def test_engine_pin_literal():
    assert ENGINE_PIN["version"] == "v1.9.1"
    assert ENGINE_PIN["url"].endswith("v1.9.1/whisper-cublas-12.4.0-bin-x64.zip")
    assert ENGINE_PIN["zip_sha256"] == "106a2030eff8998e4ef320fe72e263a78449e9040386ee27c41ea80b001b601b"
    assert ENGINE_PIN["exe_relpath"] == "Release/whisper-cli.exe"
    assert ENGINE_PIN["exe_sha256"] == "789fddb0f05c0c28043b3c4f3bcf15a0ae839df24292c60f90c4edb8d02a5ab5"
    assert ENGINE_PIN["zip_bytes"] == 677_887_125


def test_models_pin_literal():
    lv3 = MODELS_PIN["large-v3-q5_0"]
    assert (lv3["repo"], lv3["filename"]) == ("ggerganov/whisper.cpp", "ggml-large-v3-q5_0.bin")
    assert lv3["sha256"] == "d75795ecff3f83b5faa89d1900604ad8c780abd5739fae406de19f23ecd98ad1"
    assert MODELS_PIN["large-v3-q5_0"]["size_bytes"] == 1_081_140_203
    small = MODELS_PIN["small"]
    assert (small["repo"], small["filename"]) == ("ggerganov/whisper.cpp", "ggml-small.bin")
    assert small["sha256"] == "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b"
    assert small["size_bytes"] == 487_601_967


def test_install_root_win32_without_home_falls_back_to_localappdata(monkeypatch, tmp_path):
    monkeypatch.delenv("SPEECHTOTEXT_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "win32")
    assert install_root() == tmp_path / "speechtotext" / "whisper-cpp" / "v1.9.1"


# --- ensure_engine ----------------------------------------------------------------

EXE = b"soy whisper-cli"


def _pin_engine(monkeypatch, zip_bytes=None):
    pin = dict(ENGINE_PIN, exe_sha256=_sha(EXE))
    if zip_bytes is not None:
        pin["zip_sha256"] = _sha(zip_bytes)
    monkeypatch.setattr(enginepin, "ENGINE_PIN", pin)
    return pin


def _install_exe(root: Path, content: bytes = EXE) -> Path:
    exe = root / "Release" / "whisper-cli.exe"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_bytes(content)
    return exe


def test_ensure_engine_adopts_an_existing_installation_and_marks_it(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    _pin_engine(monkeypatch)
    exe = _install_exe(tmp_path)
    assert ensure_engine(root=tmp_path) == exe
    marker = Path(str(exe) + ".verified")
    assert marker.read_text(encoding="utf-8").strip() == _sha(EXE)


def test_ensure_engine_marker_avoids_rehashing(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    _pin_engine(monkeypatch)
    exe = _install_exe(tmp_path)
    ensure_engine(root=tmp_path)
    # Verification is per installation, not per run: a valid marker avoids rehashing.
    exe.write_bytes(b"cambiado despues de verificar")
    assert ensure_engine(root=tmp_path) == exe


def test_ensure_engine_mismatched_sha_fails_without_marking(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    _pin_engine(monkeypatch)
    exe = _install_exe(tmp_path, b"impostor")
    with pytest.raises(RuntimeError, match="sha256"):
        ensure_engine(root=tmp_path)
    assert not Path(str(exe) + ".verified").exists()


def _zip_with_exe(exe_bytes: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Release/whisper-cli.exe", exe_bytes)
        zf.writestr("Release/ggml-cuda.dll", b"dll de mentira")
    return buf.getvalue()


def test_ensure_engine_downloads_verifies_the_zip_and_extracts_it(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    zip_bytes = _zip_with_exe(EXE)
    pin = _pin_engine(monkeypatch, zip_bytes=zip_bytes)
    urls = []
    monkeypatch.setattr(
        enginepin.urllib.request, "urlopen",
        lambda url: (urls.append(url), io.BytesIO(zip_bytes))[1],
    )
    got = ensure_engine(root=tmp_path)
    assert got == tmp_path / "Release" / "whisper-cli.exe"
    assert got.read_bytes() == EXE
    assert urls == [pin["url"]]  # Download from the pinned URL.
    assert (tmp_path / "Release" / "ggml-cuda.dll").exists()  # Release/ prefix preserved.
    assert not list(tmp_path.glob("*.zip"))  # The temporary zip was removed.


def test_ensure_engine_with_a_bad_zip_sha_extracts_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    zip_bytes = _zip_with_exe(EXE)
    pin = dict(ENGINE_PIN, exe_sha256=_sha(EXE), zip_sha256="0" * 64)
    monkeypatch.setattr(enginepin, "ENGINE_PIN", pin)
    monkeypatch.setattr(enginepin.urllib.request, "urlopen", lambda url: io.BytesIO(zip_bytes))
    with pytest.raises(RuntimeError, match="zip"):
        ensure_engine(root=tmp_path)
    assert not (tmp_path / "Release").exists()  # Verification happens BEFORE extraction.


def test_ensure_engine_fails_when_the_exe_in_the_zip_is_corrupt(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    # The zip matches but the exe inside does not match exe_sha256: still fail closed.
    zip_bytes = _zip_with_exe(b"exe troyano")
    pin = dict(ENGINE_PIN, exe_sha256=_sha(EXE), zip_sha256=_sha(zip_bytes))
    monkeypatch.setattr(enginepin, "ENGINE_PIN", pin)
    monkeypatch.setattr(enginepin.urllib.request, "urlopen", lambda url: io.BytesIO(zip_bytes))
    with pytest.raises(RuntimeError, match="whisper-cli.exe"):
        ensure_engine(root=tmp_path)


def test_ensure_engine_outside_win32_uses_the_binary_on_path(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", lambda name: "/opt/homebrew/bin/whisper-cli")
    assert ensure_engine(root=tmp_path) == Path("/opt/homebrew/bin/whisper-cli")
    assert not any(tmp_path.iterdir())   # It neither downloads nor extracts anything.


def test_ensure_engine_outside_win32_without_a_binary_fails_with_instructions(monkeypatch, tmp_path):
    from speechtotext.asr.base import AsrError

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(AsrError) as ei:
        ensure_engine(root=tmp_path)
    assert ei.value.code == "backend_failed" and ei.value.recoverable is False
    assert "brew install whisper-cpp" in str(ei.value) and "--engine faster-whisper" in str(ei.value)


# --- ensure_model -----------------------------------------------------------------

GGML = b"soy un ggml chiquito"


def _pin_models(monkeypatch):
    pins = {
        name: dict(entry, sha256=_sha(GGML))
        for name, entry in MODELS_PIN.items()
    }
    monkeypatch.setattr(enginepin, "MODELS_PIN", pins)


def test_ensure_model_verifies_marks_and_maps_a_local_alias(monkeypatch, tmp_path):
    _pin_models(monkeypatch)
    dest = tmp_path / "models" / "ggml-large-v3-q5_0.bin"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(GGML)
    # "large-v3" resolves to the q5_0 binary: the effective quantization prevails.
    assert ensure_model("large-v3", root=tmp_path) == dest
    assert Path(str(dest) + ".verified").read_text(encoding="utf-8").strip() == _sha(GGML)


def test_ensure_model_downloads_via_hf_and_copies(monkeypatch, tmp_path):
    _pin_models(monkeypatch)
    src = tmp_path / "cache-hf.bin"
    src.write_bytes(GGML)
    calls = []

    def fake_download(repo, filename):
        calls.append((repo, filename))
        return str(src)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=fake_download))
    got = ensure_model("small", root=tmp_path)
    assert got == tmp_path / "models" / "ggml-small.bin"
    assert got.read_bytes() == GGML  # A real COPY, not a reference to the HF cache.
    assert src.exists()
    assert calls == [("ggerganov/whisper.cpp", "ggml-small.bin")]


def test_ensure_model_mismatched_sha_fails(monkeypatch, tmp_path):
    _pin_models(monkeypatch)
    dest = tmp_path / "models" / "ggml-small.bin"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"impostor")
    with pytest.raises(RuntimeError, match="sha256"):
        ensure_model("small", root=tmp_path)


def test_ensure_model_when_unpinned_lists_the_available_models(tmp_path):
    with pytest.raises(RuntimeError, match="is not pinned") as ei:
        ensure_model("medium", root=tmp_path)
    assert "large-v3" in str(ei.value) and "small" in str(ei.value)
