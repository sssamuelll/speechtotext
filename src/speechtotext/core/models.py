"""Modelos como API: dónde viven, cuáles hay, bajar y borrar. Sin dependencias nuevas:
huggingface_hub ya viene con faster-whisper; whisper.cpp lo lleva enginepin."""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from speechtotext.core import enginepin
from speechtotext.core.probe import ENGINE_FASTER, ENGINE_WHISPERCPP, ENGINES
from speechtotext.core.transcribe import Progress, ProgressCallback

APP_NAME = "speechtotext"   # el nombre del paquete manda; se renombra con él (spec §9.1)

# Repos de HF por nombre corto (los seis que documenta el CLI). Tabla propia y no
# faster_whisper.utils._MODELS: es privada y traería el paquete entero al listar.
_FW_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
}
# Lo que faster-whisper baja de un repo (faster_whisper/utils.py, download_model): sin
# README ni .gitattributes. Sirve para bajar y para sumar el tamaño antes.
_FW_FILES = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"]


@dataclass(frozen=True)
class ModelInfo:
    engine: str
    name: str
    path: Path
    size_bytes: int
    verified: bool      # True solo si el sha256 se comprobó contra el pin (whispercpp)


def data_dir() -> Path:
    """Raíz de datos de la app (modelos y binarios). SPEECHTOTEXT_HOME manda si está puesto
    (un solo directorio para todo, y la suite no toca la máquina); si no, la convención de
    cada sistema."""
    env = os.environ.get("SPEECHTOTEXT_HOME")
    if env:
        return Path(env)
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / APP_NAME


def _check_engine(engine: str) -> None:
    if engine not in ENGINES:
        raise ValueError(f"engine {engine!r} no existe; disponibles: {', '.join(ENGINES)}")


def _fw_repo(name: str) -> str:
    try:
        return _FW_REPOS[name]
    except KeyError:
        raise ValueError(f"modelo {name!r} no está en la tabla de faster-whisper; disponibles: "
                         f"{', '.join(_FW_REPOS)}") from None


def _wcpp_key(name: str) -> str:
    try:
        return enginepin._MODEL_ALIAS[name]
    except KeyError:
        raise ValueError(f"modelo {name!r} no está pinneado para whispercpp; disponibles: "
                         f"{', '.join(sorted(enginepin._MODEL_ALIAS))}") from None


def _fw_installed() -> list[ModelInfo]:
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound

    by_repo = {repo: name for name, repo in _FW_REPOS.items()}
    try:
        info = scan_cache_dir()
    except CacheNotFound:
        return []
    return [ModelInfo(ENGINE_FASTER, by_repo[r.repo_id], Path(r.repo_path), r.size_on_disk, False)
            for r in sorted(info.repos, key=lambda r: r.repo_id) if r.repo_id in by_repo]


def _wcpp_installed() -> list[ModelInfo]:
    root = enginepin.install_root() / "models"
    out: list[ModelInfo] = []
    for name, key in sorted(enginepin._MODEL_ALIAS.items()):
        pin = enginepin.MODELS_PIN[key]
        path = root / pin["filename"]
        if not path.exists():
            continue
        marker = Path(str(path) + ".verified")
        verified = marker.exists() and marker.read_text(encoding="utf-8").strip() == pin["sha256"]
        out.append(ModelInfo(ENGINE_WHISPERCPP, name, path, path.stat().st_size, verified))
    return out


def installed(engine: str | None = None) -> list[ModelInfo]:
    """Modelos locales: faster-whisper desde la caché de HF, whispercpp desde data_dir()."""
    if engine is not None:
        _check_engine(engine)
    out: list[ModelInfo] = []
    if engine in (None, ENGINE_FASTER):
        out += _fw_installed()
    if engine in (None, ENGINE_WHISPERCPP):
        out += _wcpp_installed()
    return out


def remote_size(engine: str, name: str) -> int | None:
    """Bytes que bajaría ensure(); None si no se pudo consultar (sin red). Una consulta a HF."""
    _check_engine(engine)
    if engine == ENGINE_WHISPERCPP:
        return enginepin.MODELS_PIN[_wcpp_key(name)]["size_bytes"]
    repo = _fw_repo(name)
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(repo, files_metadata=True)
    except Exception:   # sin red, 5xx, repo movido: el tamaño es cortesía, no requisito
        return None
    return sum(s.size or 0 for s in info.siblings if any(fnmatch(s.rfilename, p) for p in _FW_FILES))


def ensure(engine: str, name: str, on_progress: ProgressCallback | None = None) -> Path:
    """Descarga si falta y devuelve la ruta local. Progreso: un evento "download" al
    empezar (total=None) y otro al terminar; los bytes los pinta huggingface_hub en su
    propia barra (stderr). ponytail: sin bytes por callback — hf no expone un hook por
    bytes estable; si algún día lo hace, entra aquí."""
    _check_engine(engine)
    emit = on_progress or (lambda p: None)
    if engine == ENGINE_WHISPERCPP:
        emit(Progress("download", 0, None, enginepin.MODELS_PIN[_wcpp_key(name)]["filename"]))
        path = enginepin.ensure_model(name)
    else:
        repo = _fw_repo(name)
        from huggingface_hub import snapshot_download

        emit(Progress("download", 0, None, repo))
        path = Path(snapshot_download(repo, allow_patterns=_FW_FILES))
    emit(Progress("download", 1, 1, name))
    return path


def remove(engine: str, name: str) -> None:
    """Borra el modelo local. FileNotFoundError si no está."""
    for mi in installed(engine):
        if mi.name != name:
            continue
        if mi.path.is_dir():
            shutil.rmtree(mi.path)
        else:
            mi.path.unlink()
            Path(str(mi.path) + ".verified").unlink(missing_ok=True)
        return
    raise FileNotFoundError(f"{name} ({engine}) no está instalado")
