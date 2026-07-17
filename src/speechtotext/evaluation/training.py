from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import Literal

import numpy as np

from speechtotext.asr.base import VerifiedLocalAsrBackend
from speechtotext.asr.types import TranscriptionRequest
from speechtotext.audio.fingerprint import PipelineProvenance
from speechtotext.confidence.calibration import (
    CalibratorArtifact,
    select_operating_threshold,
)
from speechtotext.evaluation.manifest import CorpusManifest
from speechtotext.evaluation.splits import DatasetSplit


@dataclass(frozen=True)
class LabeledFeatureExample:
    clip_id: str
    session_id: str
    recorded_on: date
    features: Mapping[str, float]
    label: int

    def __post_init__(self) -> None:
        if not self.clip_id.strip() or not self.session_id.strip():
            raise ValueError("clip_id y session_id son obligatorios")
        if self.label not in {0, 1} or isinstance(self.label, bool):
            raise ValueError("label debe ser 0 o 1")
        frozen = {str(key): float(value) for key, value in self.features.items()}
        if not frozen or any(not key or not math.isfinite(value) for key, value in frozen.items()):
            raise ValueError("features deben tener nombres y valores finitos")
        object.__setattr__(self, "features", MappingProxyType(frozen))


@dataclass(frozen=True, init=False)
class LabeledFeaturePartition:
    name: Literal["development", "calibration"]
    examples: tuple[LabeledFeatureExample, ...]
    manifest_fingerprint: str
    split_fingerprint: str
    partition_ids: tuple[str, ...]

    @classmethod
    def from_split(
        cls,
        name: Literal["development", "calibration"],
        manifest: CorpusManifest,
        split: DatasetSplit,
        examples: tuple[LabeledFeatureExample, ...],
    ) -> "LabeledFeaturePartition":
        if name not in {"development", "calibration"}:
            raise ValueError("partition debe ser development o calibration")
        selected = split.partition(name, manifest)
        expected = {
            entry.clip_id: entry for entry in selected if entry.kind == "speech"
        }
        rows = tuple(examples)
        ids = tuple(row.clip_id for row in rows)
        if not rows or len(set(ids)) != len(ids) or set(ids) != set(expected):
            raise ValueError("examples deben cubrir speech clips una sola vez")
        for row in rows:
            entry = expected[row.clip_id]
            if (row.session_id, row.recorded_on) != (
                entry.session_id,
                entry.recorded_on,
            ):
                raise ValueError("metadata de example no coincide con manifest/split")
        instance = object.__new__(cls)
        object.__setattr__(instance, "name", name)
        object.__setattr__(instance, "examples", rows)
        object.__setattr__(instance, "manifest_fingerprint", manifest.version)
        object.__setattr__(instance, "split_fingerprint", split.fingerprint)
        object.__setattr__(instance, "partition_ids", tuple(sorted(ids)))
        return instance

    @property
    def fingerprint(self) -> str:
        payload = {
            "schema_version": "speechtotext.feature-partition/v1",
            "name": self.name,
            "manifest_fingerprint": self.manifest_fingerprint,
            "split_fingerprint": self.split_fingerprint,
            "partition_ids": list(self.partition_ids),
            "examples": [
                {
                    "clip_id": item.clip_id,
                    "session_id": item.session_id,
                    "recorded_on": item.recorded_on.isoformat(),
                    "features": dict(item.features),
                    "label": item.label,
                }
                for item in self.examples
            ],
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def fit_segment_usable_calibrator(
    development: LabeledFeaturePartition,
    calibration: LabeledFeaturePartition,
    *,
    backend: VerifiedLocalAsrBackend,
    pipeline: PipelineProvenance,
    request: TranscriptionRequest,
    usable_max_wer: float,
    min_precision_lower_95: float,
    artifact_version: str,
) -> CalibratorArtifact:
    if not isinstance(backend, VerifiedLocalAsrBackend):
        raise TypeError("fit exige VerifiedLocalAsrBackend")
    backend.model_artifact.require_active()
    if (
        backend.backend_artifact_kind != "local_model_manifest"
        or backend.backend_artifact_fingerprint
        != backend.model_artifact.fingerprint
    ):
        raise ValueError("binding local de modelo inconsistente")
    if not isinstance(development, LabeledFeaturePartition) or not isinstance(
        calibration, LabeledFeaturePartition
    ):
        raise TypeError("fit exige LabeledFeaturePartition selladas")
    if development.name != "development" or calibration.name != "calibration":
        raise ValueError("fit/seleccion requieren development y calibration")
    if (
        development.manifest_fingerprint != calibration.manifest_fingerprint
        or development.split_fingerprint != calibration.split_fingerprint
    ):
        raise ValueError("fit/calibration deben pertenecer al mismo manifest/split")
    fit_sessions = {item.session_id for item in development.examples}
    cal_sessions = {item.session_id for item in calibration.examples}
    fit_days = {item.recorded_on for item in development.examples}
    cal_days = {item.recorded_on for item in calibration.examples}
    if fit_sessions & cal_sessions or fit_days & cal_days:
        raise ValueError("fit y calibration deben ser disjuntos por sesion y fecha")
    fit_features = [item.features for item in development.examples]
    cal_features = [item.features for item in calibration.examples]
    fit_labels = [item.label for item in development.examples]
    cal_labels = [item.label for item in calibration.examples]
    feature_names = tuple(fit_features[0])
    all_features = (*fit_features, *cal_features)
    if not feature_names or any(tuple(example) != feature_names for example in all_features):
        raise ValueError("todos los ejemplos deben tener las mismas features y orden")
    fit_truth = np.asarray(fit_labels, dtype=np.int8)
    cal_truth = np.asarray(cal_labels, dtype=np.int8)
    if set(fit_truth.tolist()) != {0, 1} or set(cal_truth.tolist()) != {0, 1}:
        raise ValueError("development y calibration deben contener ambas clases")
    fit_matrix = np.asarray(
        [[float(example[name]) for name in feature_names] for example in fit_features],
        dtype=np.float64,
    )
    cal_matrix = np.asarray(
        [[float(example[name]) for name in feature_names] for example in cal_features],
        dtype=np.float64,
    )
    if not np.isfinite(fit_matrix).all() or not np.isfinite(cal_matrix).all():
        raise ValueError("features deben ser finitas")
    means = np.mean(fit_matrix, axis=0)
    scales = np.std(fit_matrix, axis=0)
    scales = np.where(scales > 0.0, scales, 1.0)
    fit_standardized = (fit_matrix - means) / scales
    cal_standardized = (cal_matrix - means) / scales
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise RuntimeError(
            'falta el extra evaluation: pip install -e ".[evaluation]"'
        ) from exc
    estimator = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=0,
    )
    estimator.fit(fit_standardized, fit_truth)
    probabilities = estimator.predict_proba(cal_standardized)[:, 1].tolist()
    selection = select_operating_threshold(
        probabilities,
        cal_truth.tolist(),
        min_precision_lower_95=min_precision_lower_95,
    )
    if selection.accepted == 0:
        raise ValueError("ningun umbral cumple el limite inferior de precision")
    return CalibratorArtifact(
        schema_version="speechtotext.calibrator/v1",
        artifact_version=artifact_version,
        target="segment_usable",
        usable_max_wer=usable_max_wer,
        expected_language=request.language,
        feature_names=feature_names,
        feature_means=tuple(float(value) for value in means),
        feature_scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in estimator.coef_[0]),
        intercept=float(estimator.intercept_[0]),
        operating_threshold=selection.threshold,
        selection_correct=selection.correct,
        selection_accepted=selection.accepted,
        selection_total=selection.total,
        precision_lower_95=selection.precision_lower_95,
        backend=backend.backend_id,
        model=backend.model_id,
        model_version=backend.model_version,
        backend_artifact_kind=backend.backend_artifact_kind,
        backend_artifact_fingerprint=backend.backend_artifact_fingerprint,
        backend_config_fingerprint=backend.config_fingerprint,
        fit_split_fingerprint=development.fingerprint,
        calibration_split_fingerprint=calibration.fingerprint,
        pipeline_fingerprint=pipeline.fingerprint,
        request_fingerprint=request.fingerprint,
    )
