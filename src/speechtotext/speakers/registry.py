"""Voice registry for identification: stores one embedding per person."""
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
    """Load the manifest nested by model: {model: {name: meta}}.

    Transparently regroup the flat v0.4 format ({name: meta}) using each entry's
    "model" field. The discriminator is isinstance(value, str) on "file": in the
    flat format entry["file"] is a str; in the nested one, manifest[model]["file"]
    would be the meta dict for a person named "file"."""
    p = _manifest_path()
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not raw:
        return {}
    first = next(iter(raw.values()))
    if isinstance(first.get("file"), str):
        # flat v0.4 format: each entry carries its own model
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
    # Path relative to voices/, with a fixed "/" (not os.sep) so the manifest is
    # portable across platforms.
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
    models = [model] if model is not None else list(m)
    rows = [
        {"name": name, "model": mod, **meta}
        for mod in models
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
    models = [model] if model is not None else list(m)
    deleted = False
    for mod in models:
        voices = m.get(mod, {})
        if name in voices:
            (_voices_dir() / voices[name]["file"]).unlink(missing_ok=True)
            del voices[name]
            deleted = True
    if deleted:
        _save_manifest(m)
    return deleted
