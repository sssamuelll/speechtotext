"""Pin duro del binario whisper.cpp y de los modelos ggml (plan 2.9).

Fail-closed: sha que no cuadra = RuntimeError con causa, JAMAS fallback silencioso a
otro motor. La verificacion es por instalacion (marcador .verified junto al artefacto),
no por corrida. La tabla pinneada en codigo ES el manifest: una cadena de custodia por manifiesto
verificado (DACL+lease) es incompatible con el cache HF y no se reutiliza.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# Subir el pin = commit consciente con re-benchmark en la 980 (riesgo PTX 500 sin
# garantia futura). El zip es autocontenido (DLLs CUDA propias, solo exige driver).
ENGINE_PIN = {
    "version": "v1.9.1",
    "url": "https://github.com/ggml-org/whisper.cpp/releases/download/v1.9.1/whisper-cublas-12.4.0-bin-x64.zip",
    "zip_sha256": "106a2030eff8998e4ef320fe72e263a78449e9040386ee27c41ea80b001b601b",
    "exe_relpath": "Release/whisper-cli.exe",
    "exe_sha256": "789fddb0f05c0c28043b3c4f3bcf15a0ae839df24292c60f90c4edb8d02a5ab5",
}

MODELS_PIN = {
    "large-v3-q5_0": {
        "repo": "ggerganov/whisper.cpp",
        "filename": "ggml-large-v3-q5_0.bin",
        "sha256": "d75795ecff3f83b5faa89d1900604ad8c780abd5739fae406de19f23ecd98ad1",
    },
    "small": {
        "repo": "ggerganov/whisper.cpp",
        "filename": "ggml-small.bin",
        "sha256": "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b",
    },
}

# large-v3 resuelve al bin q5_0 porque la quant efectiva bajo whispercpp es q5_0 (plan 2.4).
_MODEL_ALIAS = {"large-v3": "large-v3-q5_0", "small": "small"}


def install_root() -> Path:
    return Path(os.environ["LOCALAPPDATA"]) / "speechtotext" / "whisper-cpp" / ENGINE_PIN["version"]


def installed_exe() -> Path | None:
    """Binario ya presente, sin descargar ni verificar: el pinneado (win32) o `whisper-cli`
    en el PATH (macOS/Linux: brew o compilado). None si no hay ninguno."""
    if sys.platform == "win32":
        try:
            exe = install_root() / ENGINE_PIN["exe_relpath"]
        except KeyError:            # LOCALAPPDATA ausente: no hay pinneado posible
            return None
        return exe if exe.exists() else None
    found = shutil.which("whisper-cli")
    return Path(found) if found else None


def _sha256_file(path: Path) -> str:
    # Mismo patron de bloques que models/manifest._sha256_stream; sin importar su cadena.
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified(path: Path, expected: str, what: str) -> Path:
    """Devuelve path si su sha256 cuadra con el pin; cachea el veredicto en <path>.verified.

    El marcador guarda el sha esperado: si el pin sube de version, el marcador viejo deja
    de cuadrar y se re-verifica solo.
    """
    marker = Path(str(path) + ".verified")
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == expected:
        return path
    actual = _sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"sha256 de {what} no cuadra con el pin: esperado {expected}, obtenido {actual} ({path})"
        )
    marker.write_text(expected, encoding="utf-8")
    return path


def _download_and_extract(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".zip", dir=root)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(ENGINE_PIN["url"]) as resp:
            shutil.copyfileobj(resp, out)
        actual = _sha256_file(Path(tmp))
        if actual != ENGINE_PIN["zip_sha256"]:
            raise RuntimeError(
                f"sha256 del zip de whisper.cpp no cuadra con el pin: esperado "
                f"{ENGINE_PIN['zip_sha256']}, obtenido {actual}; no se extrae nada"
            )
        # El zip ya esta verificado; se extrae conservando el prefijo Release/ tal cual
        # (decision de layout: cero codigo de aplanado, el pin apunta adentro).
        with zipfile.ZipFile(tmp) as zf:
            zf.extractall(root)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def ensure_engine(root: Path | None = None) -> Path:
    """Path del whisper-cli.exe verificado; descarga y extrae si falta. Fail-closed."""
    root = Path(root) if root is not None else install_root()
    exe = root / ENGINE_PIN["exe_relpath"]
    if not exe.exists():
        _download_and_extract(root)
    # Instalacion ya existente sin marcador (la adopcion de hoy): se verifica el exe
    # contra el pin y se escribe el marcador. Con marcador que cuadra: cero rehash.
    return _verified(exe, ENGINE_PIN["exe_sha256"], "whisper-cli.exe")


def ensure_model(name: str, root: Path | None = None) -> Path:
    """Path del .bin ggml verificado en install_root()/models/; descarga de HF si falta."""
    key = _MODEL_ALIAS.get(name, name if name in MODELS_PIN else "")
    pin = MODELS_PIN.get(key)
    if pin is None:
        raise RuntimeError(
            f"modelo {name!r} no esta pinneado para whispercpp; disponibles: "
            f"{', '.join(sorted(_MODEL_ALIAS))}"
        )
    root = Path(root) if root is not None else install_root()
    dest = root / "models" / pin["filename"]
    if not dest.exists():
        # Perezoso: solo se paga el import si de verdad hay que descargar. Sin xet el
        # snapshot HF solo valida tamano, por eso el sha lo verificamos NOSOTROS abajo.
        from huggingface_hub import hf_hub_download

        dest.parent.mkdir(parents=True, exist_ok=True)
        src = hf_hub_download(pin["repo"], pin["filename"])
        shutil.copyfile(src, dest)
    return _verified(dest, pin["sha256"], pin["filename"])
