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
    """Carga el manifiesto anidado por modelo: {modelo: {nombre: meta}}.

    Reagrupa transparentemente el formato plano de v0.4 ({nombre: meta}) usando el
    campo "model" de cada entrada. El discriminador es isinstance(valor, str) sobre
    "file": en el formato plano entrada["file"] es un str; en el anidado,
    manifiesto[modelo]["file"] sería el dict de meta de una persona llamada "file"."""
    p = _manifest_path()
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not raw:
        return {}
    primera = next(iter(raw.values()))
    if isinstance(primera.get("file"), str):
        # formato plano v0.4: cada entrada trae su propio modelo
        nested: dict = {}
        for name, meta in raw.items():
            nested.setdefault(meta["model"], {})[name] = meta
        return nested
    return raw


def _save_manifest(m: dict) -> None:
    _manifest_path().write_text(
        json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _slug(name: str) -> str:
    return re.sub(r"[^\w.-]", "_", name)


def enroll(name: str, embedding: np.ndarray, *, seconds: float, model: str) -> None:
    # Ruta relativa a voices/, con "/" fijo (no os.sep) para que el manifiesto sea
    # portable entre plataformas.
    rel = f"{_slug(model)}/{_slug(name)}.npy"
    dest = _voices_dir() / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.save(dest, np.asarray(embedding, dtype=np.float32))
    m = _load_manifest()
    m.setdefault(model, {})[name] = {
        "file": rel,
        "seconds": round(float(seconds), 1),
        "enrolled_at": datetime.now().isoformat(timespec="seconds"),
    }
    _save_manifest(m)


def list_voices(model: str | None = None) -> list[dict]:
    m = _load_manifest()
    modelos = [model] if model is not None else list(m)
    rows = [
        {"name": name, "model": mod, **meta}
        for mod in modelos
        for name, meta in m.get(mod, {}).items()
    ]
    return sorted(rows, key=lambda r: r["name"])


def get_embeddings(model: str) -> dict[str, np.ndarray]:
    d = _voices_dir()
    out: dict[str, np.ndarray] = {}
    for name, meta in _load_manifest().get(model, {}).items():
        path = d / meta["file"]
        if path.exists():
            out[name] = np.load(path)
    return out


def remove(name: str, *, model: str | None = None) -> bool:
    m = _load_manifest()
    modelos = [model] if model is not None else list(m)
    borrado = False
    for mod in modelos:
        voces = m.get(mod, {})
        if name in voces:
            (_voices_dir() / voces[name]["file"]).unlink(missing_ok=True)
            del voces[name]
            borrado = True
    if borrado:
        _save_manifest(m)
    return borrado
