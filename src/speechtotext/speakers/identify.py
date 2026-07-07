"""Identificación de hablantes: comparar embeddings contra voces registradas."""
from __future__ import annotations

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def assign_names(
    clusters: dict[str, np.ndarray],
    enrolled: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, str]:
    """Asigna cada speaker_id anónimo a un nombre registrado (greedy por score)."""
    if not clusters or not enrolled:
        return {}
    candidates = [
        (cosine(vec, ref), sid, name)
        for sid, vec in clusters.items()
        for name, ref in enrolled.items()
    ]
    candidates.sort(reverse=True)  # mayor score primero
    result: dict[str, str] = {}
    used_names: set[str] = set()
    for score, sid, name in candidates:
        if score < threshold:
            break
        if sid in result or name in used_names:
            continue
        result[sid] = name
        used_names.add(name)
    return result
