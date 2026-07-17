from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Sequence

from speechtotext.evaluation.manifest import CorpusEntry, CorpusManifest


@dataclass(frozen=True, init=False)
class DatasetSplit:
    manifest_fingerprint: str
    development_ids: tuple[str, ...]
    calibration_ids: tuple[str, ...]
    holdout_ids: tuple[str, ...]
    seed: int

    @classmethod
    def create(
        cls,
        manifest: CorpusManifest,
        development: Sequence[CorpusEntry],
        calibration: Sequence[CorpusEntry],
        holdout: Sequence[CorpusEntry],
        seed: int,
    ) -> "DatasetSplit":
        canonical = {entry.clip_id: entry for entry in manifest.entries}
        session_days: dict[str, set[date]] = defaultdict(set)
        for entry in manifest.entries:
            session_days[entry.session_id].add(entry.recorded_on)
        if any(len(days) > 1 for days in session_days.values()):
            raise ValueError("una sesion abarca mas de una fecha")

        def ids(entries: Sequence[CorpusEntry]) -> tuple[str, ...]:
            values = tuple(entry.clip_id for entry in entries)
            if len(values) != len(set(values)):
                raise ValueError("split contiene clip_id duplicado")
            if any(canonical.get(entry.clip_id) != entry for entry in entries):
                raise ValueError("split contiene una entry que no coincide con el manifest")
            return values

        partition_ids = (ids(development), ids(calibration), ids(holdout))
        flattened = tuple(item for part in partition_ids for item in part)
        if len(flattened) != len(set(flattened)):
            raise ValueError("un clip no puede aparecer en dos particiones")
        if set(flattened) != set(canonical):
            raise ValueError("el split debe cubrir exactamente el manifest")
        partitions = (tuple(development), tuple(calibration), tuple(holdout))
        for left_index, left in enumerate(partitions):
            left_days = {entry.recorded_on for entry in left}
            left_sessions = {entry.session_id for entry in left}
            for right in partitions[left_index + 1:]:
                if left_days & {entry.recorded_on for entry in right}:
                    raise ValueError("las particiones deben ser disjuntas por fecha")
                if left_sessions & {entry.session_id for entry in right}:
                    raise ValueError("las particiones deben ser disjuntas por sesion")
        instance = object.__new__(cls)
        object.__setattr__(instance, "manifest_fingerprint", manifest.version)
        object.__setattr__(instance, "development_ids", partition_ids[0])
        object.__setattr__(instance, "calibration_ids", partition_ids[1])
        object.__setattr__(instance, "holdout_ids", partition_ids[2])
        object.__setattr__(instance, "seed", int(seed))
        return instance

    def partition(
        self,
        name: str,
        manifest: CorpusManifest,
    ) -> tuple[CorpusEntry, ...]:
        if name not in {"development", "calibration", "holdout"}:
            raise ValueError(f"particion invalida: {name}")
        if manifest.version != self.manifest_fingerprint:
            raise ValueError("manifest no coincide con el split verificado")
        canonical = {entry.clip_id: entry for entry in manifest.entries}
        selected = getattr(self, f"{name}_ids")
        try:
            return tuple(canonical[clip_id] for clip_id in selected)
        except KeyError as exc:
            raise ValueError("split referencia un clip ausente del manifest") from exc

    @property
    def fingerprint(self) -> str:
        payload = {
            "schema_version": "speechtotext.split/v1",
            "seed": self.seed,
            "manifest_fingerprint": self.manifest_fingerprint,
            "development": list(self.development_ids),
            "calibration": list(self.calibration_ids),
            "holdout": list(self.holdout_ids),
        }
        return hashlib.sha256(
            json.dumps(
                payload, ensure_ascii=True, allow_nan=False,
                separators=(",", ":"), sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()


def _count(total: int, fraction: float) -> int:
    return max(3, round(total * fraction))


def split_by_recording_day(
    manifest: CorpusManifest,
    *,
    seed: int = 20260716,
    calibration_fraction: float = 0.2,
    holdout_fraction: float = 0.2,
) -> DatasetSplit:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction debe estar en (0, 1)")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction debe estar en (0, 1)")
    groups: dict[date, list[CorpusEntry]] = defaultdict(list)
    for entry in manifest.entries:
        groups[entry.recorded_on].append(entry)
    days = sorted(groups)
    if len(days) < 9:
        raise ValueError("se requieren al menos nueve fechas de grabacion")
    rng = random.Random(seed)
    rng.shuffle(days)
    calibration_count = _count(len(days), calibration_fraction)
    holdout_count = _count(len(days), holdout_fraction)
    while calibration_count + holdout_count > len(days) - 3:
        if calibration_count >= holdout_count and calibration_count > 3:
            calibration_count -= 1
        elif holdout_count > 3:
            holdout_count -= 1
        else:
            raise ValueError("fracciones no dejan tres fechas para development")
    calibration_days = set(days[:calibration_count])
    holdout_days = set(days[calibration_count:calibration_count + holdout_count])
    development_days = set(days) - calibration_days - holdout_days

    def collect(selected: set[date]) -> tuple[CorpusEntry, ...]:
        return tuple(
            sorted(
                (entry for day in selected for entry in groups[day]),
                key=lambda entry: (entry.recorded_on, entry.session_id, entry.clip_id),
            )
        )

    development = collect(development_days)
    calibration = collect(calibration_days)
    holdout = collect(holdout_days)
    return DatasetSplit.create(
        manifest,
        development,
        calibration,
        holdout,
        seed,
    )
