import hashlib
import io
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


# --- pines literales del contrato -------------------------------------------------

def test_engine_pin_literal():
    assert ENGINE_PIN["version"] == "v1.9.1"
    assert ENGINE_PIN["url"].endswith("v1.9.1/whisper-cublas-12.4.0-bin-x64.zip")
    assert ENGINE_PIN["zip_sha256"] == "106a2030eff8998e4ef320fe72e263a78449e9040386ee27c41ea80b001b601b"
    assert ENGINE_PIN["exe_relpath"] == "Release/whisper-cli.exe"
    assert ENGINE_PIN["exe_sha256"] == "789fddb0f05c0c28043b3c4f3bcf15a0ae839df24292c60f90c4edb8d02a5ab5"


def test_models_pin_literal():
    lv3 = MODELS_PIN["large-v3-q5_0"]
    assert (lv3["repo"], lv3["filename"]) == ("ggerganov/whisper.cpp", "ggml-large-v3-q5_0.bin")
    assert lv3["sha256"] == "d75795ecff3f83b5faa89d1900604ad8c780abd5739fae406de19f23ecd98ad1"
    small = MODELS_PIN["small"]
    assert (small["repo"], small["filename"]) == ("ggerganov/whisper.cpp", "ggml-small.bin")
    assert small["sha256"] == "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b"


def test_install_root_bajo_localappdata(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert install_root() == tmp_path / "speechtotext" / "whisper-cpp" / "v1.9.1"


# --- ensure_engine ----------------------------------------------------------------

EXE = b"soy whisper-cli"


def _pin_engine(monkeypatch, zip_bytes=None):
    pin = dict(ENGINE_PIN, exe_sha256=_sha(EXE))
    if zip_bytes is not None:
        pin["zip_sha256"] = _sha(zip_bytes)
    monkeypatch.setattr(enginepin, "ENGINE_PIN", pin)
    return pin


def _instala_exe(root: Path, content: bytes = EXE) -> Path:
    exe = root / "Release" / "whisper-cli.exe"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_bytes(content)
    return exe


def test_ensure_engine_adopta_instalacion_existente_y_marca(monkeypatch, tmp_path):
    _pin_engine(monkeypatch)
    exe = _instala_exe(tmp_path)
    assert ensure_engine(root=tmp_path) == exe
    marker = Path(str(exe) + ".verified")
    assert marker.read_text(encoding="utf-8").strip() == _sha(EXE)


def test_ensure_engine_marcador_evita_rehash(monkeypatch, tmp_path):
    _pin_engine(monkeypatch)
    exe = _instala_exe(tmp_path)
    ensure_engine(root=tmp_path)
    # verificacion por instalacion, no por corrida: con marcador valido no se rehashea
    exe.write_bytes(b"cambiado despues de verificar")
    assert ensure_engine(root=tmp_path) == exe


def test_ensure_engine_sha_que_no_cuadra_revienta_sin_marcar(monkeypatch, tmp_path):
    _pin_engine(monkeypatch)
    exe = _instala_exe(tmp_path, b"impostor")
    with pytest.raises(RuntimeError, match="sha256"):
        ensure_engine(root=tmp_path)
    assert not Path(str(exe) + ".verified").exists()


def _zip_con_exe(exe_bytes: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Release/whisper-cli.exe", exe_bytes)
        zf.writestr("Release/ggml-cuda.dll", b"dll de mentira")
    return buf.getvalue()


def test_ensure_engine_descarga_verifica_zip_y_extrae(monkeypatch, tmp_path):
    zip_bytes = _zip_con_exe(EXE)
    pin = _pin_engine(monkeypatch, zip_bytes=zip_bytes)
    urls = []
    monkeypatch.setattr(
        enginepin.urllib.request, "urlopen",
        lambda url: (urls.append(url), io.BytesIO(zip_bytes))[1],
    )
    got = ensure_engine(root=tmp_path)
    assert got == tmp_path / "Release" / "whisper-cli.exe"
    assert got.read_bytes() == EXE
    assert urls == [pin["url"]]  # descarga desde la URL pinneada
    assert (tmp_path / "Release" / "ggml-cuda.dll").exists()  # prefijo Release/ conservado
    assert not list(tmp_path.glob("*.zip"))  # el zip temporal se limpio


def test_ensure_engine_zip_sha_malo_no_extrae_nada(monkeypatch, tmp_path):
    zip_bytes = _zip_con_exe(EXE)
    pin = dict(ENGINE_PIN, exe_sha256=_sha(EXE), zip_sha256="0" * 64)
    monkeypatch.setattr(enginepin, "ENGINE_PIN", pin)
    monkeypatch.setattr(enginepin.urllib.request, "urlopen", lambda url: io.BytesIO(zip_bytes))
    with pytest.raises(RuntimeError, match="zip"):
        ensure_engine(root=tmp_path)
    assert not (tmp_path / "Release").exists()  # verificacion ANTES de extraer


def test_ensure_engine_exe_del_zip_corrupto_revienta(monkeypatch, tmp_path):
    # el zip cuadra pero el exe adentro no cuadra con exe_sha256: fail-closed igual
    zip_bytes = _zip_con_exe(b"exe troyano")
    pin = dict(ENGINE_PIN, exe_sha256=_sha(EXE), zip_sha256=_sha(zip_bytes))
    monkeypatch.setattr(enginepin, "ENGINE_PIN", pin)
    monkeypatch.setattr(enginepin.urllib.request, "urlopen", lambda url: io.BytesIO(zip_bytes))
    with pytest.raises(RuntimeError, match="whisper-cli.exe"):
        ensure_engine(root=tmp_path)


# --- ensure_model -----------------------------------------------------------------

GGML = b"soy un ggml chiquito"


def _pin_models(monkeypatch):
    pins = {
        name: dict(entry, sha256=_sha(GGML))
        for name, entry in MODELS_PIN.items()
    }
    monkeypatch.setattr(enginepin, "MODELS_PIN", pins)


def test_ensure_model_local_verifica_marca_y_mapea_alias(monkeypatch, tmp_path):
    _pin_models(monkeypatch)
    dest = tmp_path / "models" / "ggml-large-v3-q5_0.bin"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(GGML)
    # "large-v3" resuelve al bin q5_0: la quant efectiva manda
    assert ensure_model("large-v3", root=tmp_path) == dest
    assert Path(str(dest) + ".verified").read_text(encoding="utf-8").strip() == _sha(GGML)


def test_ensure_model_descarga_via_hf_y_copia(monkeypatch, tmp_path):
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
    assert got.read_bytes() == GGML  # COPIA real, no referencia al cache HF
    assert src.exists()
    assert calls == [("ggerganov/whisper.cpp", "ggml-small.bin")]


def test_ensure_model_sha_que_no_cuadra_revienta(monkeypatch, tmp_path):
    _pin_models(monkeypatch)
    dest = tmp_path / "models" / "ggml-small.bin"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"impostor")
    with pytest.raises(RuntimeError, match="sha256"):
        ensure_model("small", root=tmp_path)


def test_ensure_model_no_pinneado_lista_disponibles(tmp_path):
    with pytest.raises(RuntimeError, match="no esta pinneado") as ei:
        ensure_model("medium", root=tmp_path)
    assert "large-v3" in str(ei.value) and "small" in str(ei.value)
