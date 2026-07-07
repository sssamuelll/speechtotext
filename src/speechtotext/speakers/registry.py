"""Registro de voces para identificación: guarda un embedding por persona."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np


def home() -> Path:
    env = os.environ.get("SPEECHTOTEXT_HOME")
    return Path(env) if env else Path.home() / ".speechtotext"


def _voices_dir() -> Path:
    d = home() / "voices"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest_path() -> Path:
    return _voices_dir() / "manifest.json"


def _load_manifest() -> dict:
    p = _manifest_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save_manifest(m: dict) -> None:
    _manifest_path().write_text(
        json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _slug(name: str) -> str:
    return re.sub(r"[^\w.-]", "_", name)


def enroll(name: str, embedding: np.ndarray, *, seconds: float, model: str) -> None:
    fname = f"{_slug(name)}.npy"
    np.save(_voices_dir() / fname, np.asarray(embedding, dtype=np.float32))
    m = _load_manifest()
    m[name] = {
        "file": fname,
        "seconds": round(float(seconds), 1),
        "model": model,
        "enrolled_at": datetime.now().isoformat(timespec="seconds"),
    }
    _save_manifest(m)


def list_voices() -> list[dict]:
    return [{"name": k, **v} for k, v in sorted(_load_manifest().items())]


def get_embeddings() -> dict[str, np.ndarray]:
    d = _voices_dir()
    return {name: np.load(d / meta["file"]) for name, meta in _load_manifest().items()}


def remove(name: str) -> bool:
    m = _load_manifest()
    if name not in m:
        return False
    (_voices_dir() / m[name]["file"]).unlink(missing_ok=True)
    del m[name]
    _save_manifest(m)
    return True
