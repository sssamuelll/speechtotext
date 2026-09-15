"""Hard pin for the whisper.cpp binary and ggml models (plan 2.9).

Fail-closed: mismatched sha = RuntimeError with cause, NEVER a silent fallback to another
engine. Verification is per installation (.verified marker next to the artifact), not per
run. The table pinned in code IS the manifest: a chain of custody through a verified manifest
(DACL+lease) is incompatible with the HF cache and is not reused.
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

from speechtotext.asr.base import AsrError

# Raising the pin = deliberate commit with re-benchmarking on the 980 (PTX 500 risk with
# no future guarantee). The zip is self-contained (its own CUDA DLLs, only requires a driver).
ENGINE_PIN = {
    "version": "v1.9.1",
    "url": "https://github.com/ggml-org/whisper.cpp/releases/download/v1.9.1/whisper-cublas-12.4.0-bin-x64.zip",
    "zip_sha256": "106a2030eff8998e4ef320fe72e263a78449e9040386ee27c41ea80b001b601b",
    "zip_bytes": 677_887_125,
    "exe_relpath": "Release/whisper-cli.exe",
    "exe_sha256": "789fddb0f05c0c28043b3c4f3bcf15a0ae839df24292c60f90c4edb8d02a5ab5",
}

MODELS_PIN = {
    "large-v3-q5_0": {
        "repo": "ggerganov/whisper.cpp",
        "filename": "ggml-large-v3-q5_0.bin",
        "sha256": "d75795ecff3f83b5faa89d1900604ad8c780abd5739fae406de19f23ecd98ad1",
        "size_bytes": 1_081_140_203,
    },
    "small": {
        "repo": "ggerganov/whisper.cpp",
        "filename": "ggml-small.bin",
        "sha256": "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b",
        "size_bytes": 487_601_967,
    },
}

# large-v3 resolves to the q5_0 binary because effective quantization under whispercpp is q5_0 (plan 2.4).
_MODEL_ALIAS = {"large-v3": "large-v3-q5_0", "small": "small"}


def install_root() -> Path:
    from speechtotext.core.models import data_dir   # lazy: models imports enginepin

    return data_dir() / "whisper-cpp" / ENGINE_PIN["version"]


def installed_exe() -> Path | None:
    """Binary already present, without downloading or verifying: the pinned one (win32) or
    `whisper-cli` on the PATH (macOS/Linux: brew or compiled). None if neither exists."""
    if sys.platform == "win32":
        exe = install_root() / ENGINE_PIN["exe_relpath"]
        return exe if exe.exists() else None
    found = shutil.which("whisper-cli")
    return Path(found) if found else None


def _sha256_file(path: Path) -> str:
    # Same block pattern as models/manifest._sha256_stream; without importing its chain.
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified(path: Path, expected: str, what: str) -> Path:
    """Return path if its sha256 matches the pin; cache the verdict in <path>.verified.

    The marker stores the expected sha: if the pin moves to a new version, the old marker no
    longer matches and is automatically reverified.
    """
    marker = Path(str(path) + ".verified")
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == expected:
        return path
    actual = _sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"sha256 of {what} does not match the pin: expected {expected}, got {actual} ({path})"
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
                f"sha256 of the whisper.cpp zip does not match the pin: expected "
                f"{ENGINE_PIN['zip_sha256']}, got {actual}; extracting nothing"
            )
        # The zip is already verified; extract it while preserving the Release/ prefix as-is
        # (layout decision: zero flattening code, the pin points inside).
        with zipfile.ZipFile(tmp) as zf:
            zf.extractall(root)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def ensure_engine(root: Path | None = None) -> Path:
    """Path to the verified whisper-cli. win32: the pinned zip (download and extract if missing,
    fail-closed by sha256). macOS/Linux: `whisper-cli` from the PATH (brew or compiled); without
    an official release for those systems, nothing is downloaded or compiled."""
    if sys.platform != "win32":
        found = shutil.which("whisper-cli")
        if not found:
            raise AsrError(
                "backend_failed", False,
                "whisper-cli is not on the PATH. Install it: brew install whisper-cpp (macOS) or "
                "build it from https://github.com/ggml-org/whisper.cpp (Linux); "
                "or use --engine faster-whisper",
            )
        return Path(found)
    root = Path(root) if root is not None else install_root()
    exe = root / ENGINE_PIN["exe_relpath"]
    if not exe.exists():
        _download_and_extract(root)
    # Existing installation without a marker (today's adoption): verify the executable
    # against the pin and write the marker. With a matching marker: zero rehashing.
    return _verified(exe, ENGINE_PIN["exe_sha256"], "whisper-cli.exe")


def ensure_model(name: str, root: Path | None = None) -> Path:
    """Path to the verified ggml .bin in install_root()/models/; download from HF if missing."""
    key = _MODEL_ALIAS.get(name, name if name in MODELS_PIN else "")
    pin = MODELS_PIN.get(key)
    if pin is None:
        raise RuntimeError(
            f"model {name!r} is not pinned for whispercpp; available: "
            f"{', '.join(sorted(_MODEL_ALIAS))}"
        )
    root = Path(root) if root is not None else install_root()
    dest = root / "models" / pin["filename"]
    if not dest.exists():
        # Lazy: pay for the import only if a download is actually needed. Without xet, the
        # HF snapshot only validates size, so WE verify the sha below.
        from huggingface_hub import hf_hub_download

        dest.parent.mkdir(parents=True, exist_ok=True)
        src = hf_hub_download(pin["repo"], pin["filename"])
        shutil.copyfile(src, dest)
    return _verified(dest, pin["sha256"], pin["filename"])
