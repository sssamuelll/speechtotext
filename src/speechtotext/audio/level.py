from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GainResult:
    samples: np.ndarray
    requested_gain_db: float
    applied_gain_db: float
    limited: bool


def apply_fixed_gain(
    samples: np.ndarray,
    gain_db: float,
    *,
    max_gain_db: float = 18.0,
    peak_limit_dbfs: float = -1.0,
) -> GainResult:
    if not math.isfinite(gain_db) or not math.isfinite(max_gain_db):
        raise ValueError("gain_db y max_gain_db deben ser finitos")
    if gain_db > max_gain_db:
        raise ValueError("gain_db supera max_gain_db")
    if peak_limit_dbfs > 0.0:
        raise ValueError("peak_limit_dbfs debe ser <= 0")
    source = np.asarray(samples, dtype=np.float32)
    if source.ndim != 1 or not np.isfinite(source).all():
        raise ValueError("samples debe ser audio mono finito")
    peak = float(np.max(np.abs(source))) if source.size else 0.0
    applied = gain_db
    if peak > 0.0:
        peak_limit = 10.0 ** (peak_limit_dbfs / 20.0)
        headroom_db = 20.0 * math.log10(peak_limit / peak)
        applied = min(gain_db, headroom_db)
    scale = 10.0 ** (applied / 20.0)
    output = np.array(source * scale, dtype=np.float32, order="C", copy=True)
    output.flags.writeable = False
    return GainResult(
        samples=output,
        requested_gain_db=gain_db,
        applied_gain_db=applied,
        limited=applied < gain_db - 1e-9,
    )
