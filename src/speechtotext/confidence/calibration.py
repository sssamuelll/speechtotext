from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Mapping, Sequence

from speechtotext.asr.base import CalibratedAsrBackend, VerifiedLocalAsrBackend
from speechtotext.asr.types import TranscriptionRequest, TranscriptionResult
from speechtotext.audio.fingerprint import PipelineProvenance
from speechtotext.audio.types import AudioView
from speechtotext.models import VerifiedModelArtifact
from speechtotext.statistics import one_sided_success_lower

SCHEMA_VERSION = "speechtotext.calibrator/v1"


@dataclass(frozen=True)
class CalibratorArtifact:
    schema_version: str
    artifact_version: str
    target: str
    usable_max_wer: float
    expected_language: str
    feature_names: tuple[str, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    operating_threshold: float
    selection_correct: int
    selection_accepted: int
    selection_total: int
    precision_lower_95: float
    backend: str
    model: str
    model_version: str
    backend_artifact_kind: str
    backend_artifact_fingerprint: str
    backend_config_fingerprint: str
    fit_split_fingerprint: str
    calibration_split_fingerprint: str
    pipeline_fingerprint: str
    request_fingerprint: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.target != "segment_usable":
            raise ValueError("schema o target de calibrador incompatible")
        for name in (
            "artifact_version",
            "expected_language",
            "backend",
            "model",
            "model_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} debe ser string no vacio")
        if self.backend_artifact_kind not in {
            "local_model_manifest",
            "provider_model_descriptor",
        }:
            raise ValueError("backend_artifact_kind incompatible")
        if (
            isinstance(self.usable_max_wer, bool)
            or not isinstance(self.usable_max_wer, (int, float))
            or not math.isfinite(self.usable_max_wer)
            or not 0.0 <= self.usable_max_wer <= 1.0
        ):
            raise ValueError("usable_max_wer/lenguaje esperado invalidos")
        if (
            not isinstance(self.feature_names, tuple)
            or not all(
                isinstance(values, tuple)
                for values in (
                    self.feature_means,
                    self.feature_scales,
                    self.coefficients,
                )
            )
        ):
            raise ValueError("vectores del calibrador deben ser tuplas")
        size = len(self.feature_names)
        if (
            size == 0
            or any(not isinstance(name, str) or not name for name in self.feature_names)
            or len(set(self.feature_names)) != size
            or not all(
                len(values) == size
                for values in (
                    self.feature_means,
                    self.feature_scales,
                    self.coefficients,
                )
            )
        ):
            raise ValueError("dimensiones del calibrador incompatibles")
        finite_parameters = (
            *self.feature_means,
            *self.feature_scales,
            *self.coefficients,
            self.intercept,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in finite_parameters
        ):
            raise ValueError("parametros del calibrador deben ser finitos")
        if any(scale <= 0.0 for scale in self.feature_scales):
            raise ValueError("feature_scales debe ser positivo")
        if (
            isinstance(self.operating_threshold, bool)
            or not isinstance(self.operating_threshold, (int, float))
            or not math.isfinite(self.operating_threshold)
            or not 0.0 <= self.operating_threshold <= 1.0
        ):
            raise ValueError("operating_threshold fuera de rango")
        selection_counts = (
            self.selection_correct,
            self.selection_accepted,
            self.selection_total,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in selection_counts):
            raise ValueError("los conteos de seleccion deben ser enteros")
        if not (
            0 <= self.selection_correct
            <= self.selection_accepted
            <= self.selection_total
            and self.selection_accepted > 0
        ):
            raise ValueError("conteos de seleccion invalidos")
        expected_lower = one_sided_success_lower(
            self.selection_correct,
            self.selection_accepted,
        )
        if (
            isinstance(self.precision_lower_95, bool)
            or not isinstance(self.precision_lower_95, (int, float))
            or not math.isfinite(self.precision_lower_95)
            or not math.isclose(
            self.precision_lower_95,
            expected_lower,
            rel_tol=1e-12,
            abs_tol=1e-12,
            )
        ):
            raise ValueError("precision_lower_95 no coincide con los conteos")
        for name in (
            "fit_split_fingerprint",
            "calibration_split_fingerprint",
            "backend_artifact_fingerprint",
            "backend_config_fingerprint",
            "pipeline_fingerprint",
            "request_fingerprint",
        ):
            raw = getattr(self, name)
            if not isinstance(raw, str):
                raise ValueError(f"{name} debe ser SHA-256 hexadecimal")
            value = raw.lower()
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} debe ser SHA-256 hexadecimal")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "feature_means", tuple(self.feature_means))
        object.__setattr__(self, "feature_scales", tuple(self.feature_scales))
        object.__setattr__(self, "coefficients", tuple(self.coefficients))

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        for key in ("feature_names", "feature_means", "feature_scales", "coefficients"):
            data[key] = list(data[key])
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CalibratorArtifact":
        if not isinstance(data, Mapping):
            raise ValueError("calibrador debe ser un mapping")
        expected = set(cls.__dataclass_fields__)
        if set(data) != expected:
            raise ValueError(
                f"campos de calibrador invalidos: {sorted(set(data) ^ expected)}"
            )
        converted = dict(data)
        for key in ("feature_names", "feature_means", "feature_scales", "coefficients"):
            value = converted[key]
            if not isinstance(value, list):
                raise ValueError(f"{key} debe ser una lista JSON")
            converted[key] = tuple(value)
        return cls(**converted)

    @property
    def version(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class LogisticCalibrator:
    def __init__(self, artifact: CalibratorArtifact) -> None:
        if not isinstance(artifact, CalibratorArtifact):
            raise TypeError("LogisticCalibrator exige CalibratorArtifact")
        self._artifact = artifact

    @property
    def artifact(self) -> CalibratorArtifact:
        return self._artifact

    def _validate_identity(
        self,
        *,
        backend: str,
        model: str,
        model_version: str,
        backend_artifact_kind: str,
        backend_artifact_fingerprint: str,
        backend_config_fingerprint: str,
        pipeline_fingerprint: str,
        request_fingerprint: str,
        expected_language: str,
        usable_max_wer: float,
    ) -> None:
        artifact = self.artifact
        if (backend, model, model_version) != (
            artifact.backend,
            artifact.model,
            artifact.model_version,
        ):
            raise ValueError("calibrador incompatible con backend/modelo/version")
        if (
            backend_artifact_kind != artifact.backend_artifact_kind
            or backend_artifact_fingerprint
            != artifact.backend_artifact_fingerprint
            or backend_config_fingerprint != artifact.backend_config_fingerprint
        ):
            raise ValueError("calibrador incompatible con artefacto/config ASR")
        if (
            pipeline_fingerprint != artifact.pipeline_fingerprint
            or request_fingerprint != artifact.request_fingerprint
        ):
            raise ValueError("calibrador incompatible con pipeline/request")
        if (
            expected_language != artifact.expected_language
            or not math.isclose(
                usable_max_wer,
                artifact.usable_max_wer,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("calibrador incompatible con lenguaje/usable_max_wer")

    def validate_binding_for_identity(
        self,
        *,
        backend: str,
        model: str,
        model_version: str,
        backend_artifact_kind: str,
        backend_artifact_fingerprint: str,
        backend_config_fingerprint: str,
        pipeline: PipelineProvenance,
        request: TranscriptionRequest,
        expected_language: str,
        usable_max_wer: float,
    ) -> None:
        if not isinstance(pipeline, PipelineProvenance) or not isinstance(
            request, TranscriptionRequest
        ):
            raise TypeError("pipeline/request invalidos")
        self._validate_identity(
            backend=backend,
            model=model,
            model_version=model_version,
            backend_artifact_kind=backend_artifact_kind,
            backend_artifact_fingerprint=backend_artifact_fingerprint,
            backend_config_fingerprint=backend_config_fingerprint,
            pipeline_fingerprint=pipeline.fingerprint,
            request_fingerprint=request.fingerprint,
            expected_language=expected_language,
            usable_max_wer=usable_max_wer,
        )

    def validate_for(
        self,
        *,
        backend: CalibratedAsrBackend,
        pipeline: PipelineProvenance,
        request: TranscriptionRequest,
        expected_language: str,
        usable_max_wer: float,
    ) -> None:
        if not isinstance(backend, CalibratedAsrBackend):
            raise TypeError("calibrador exige CalibratedAsrBackend")
        is_local = isinstance(backend, VerifiedLocalAsrBackend)
        if backend.backend_artifact_kind == "local_model_manifest" and not is_local:
            raise TypeError("binding local exige VerifiedLocalAsrBackend")
        if is_local:
            if not isinstance(backend.model_artifact, VerifiedModelArtifact):
                raise TypeError("backend local exige VerifiedModelArtifact")
            backend.model_artifact.require_active()
            if (
                backend.backend_artifact_kind != "local_model_manifest"
                or backend.backend_artifact_fingerprint
                != backend.model_artifact.fingerprint
            ):
                raise ValueError("binding local de modelo inconsistente")
        self.validate_binding_for_identity(
            backend=backend.backend_id,
            model=backend.model_id,
            model_version=backend.model_version,
            backend_artifact_kind=backend.backend_artifact_kind,
            backend_artifact_fingerprint=backend.backend_artifact_fingerprint,
            backend_config_fingerprint=backend.config_fingerprint,
            pipeline=pipeline,
            request=request,
            expected_language=expected_language,
            usable_max_wer=usable_max_wer,
        )

    def predict_proba(self, features: Mapping[str, float]) -> float:
        if not isinstance(features, Mapping) or tuple(features) != self.artifact.feature_names:
            raise ValueError(
                f"features incompatibles: esperado {self.artifact.feature_names}, "
                f"recibido {tuple(features)}"
            )
        values = tuple(features[name] for name in self.artifact.feature_names)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("features del calibrador deben ser numericas y finitas")
        z = self.artifact.intercept
        for name, mean, scale, coefficient in zip(
            self.artifact.feature_names,
            self.artifact.feature_means,
            self.artifact.feature_scales,
            self.artifact.coefficients,
        ):
            z += coefficient * ((features[name] - mean) / scale)
        if z >= 0.0:
            return 1.0 / (1.0 + math.exp(-z))
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)

    def apply(
        self,
        result: TranscriptionResult,
        features: Mapping[str, float],
        *,
        backend: CalibratedAsrBackend,
        view: AudioView,
        request: TranscriptionRequest,
        expected_language: str,
        usable_max_wer: float,
    ) -> TranscriptionResult:
        self.validate_for(
            backend=backend,
            pipeline=view.provenance,
            request=request,
            expected_language=expected_language,
            usable_max_wer=usable_max_wer,
        )
        self._validate_identity(
            backend=result.backend,
            model=result.model,
            model_version=result.model_version,
            backend_artifact_kind=backend.backend_artifact_kind,
            backend_artifact_fingerprint=backend.backend_artifact_fingerprint,
            backend_config_fingerprint=backend.config_fingerprint,
            pipeline_fingerprint=view.provenance.fingerprint,
            request_fingerprint=request.fingerprint,
            expected_language=expected_language,
            usable_max_wer=usable_max_wer,
        )
        return replace(
            result,
            calibrated_confidence=self.predict_proba(features),
            calibrator_version=self.artifact.version,
        )


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    precision: float
    precision_lower_95: float
    coverage: float
    correct: int
    accepted: int
    total: int


def select_operating_threshold(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    min_precision_lower_95: float,
) -> ThresholdSelection:
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("probabilities y labels deben ser no vacios e igual longitud")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
        for value in probabilities
    ) or any(type(label) is not int or label not in {0, 1} for label in labels):
        raise ValueError("probabilities/labels invalidos")
    if (
        isinstance(min_precision_lower_95, bool)
        or not isinstance(min_precision_lower_95, (int, float))
        or not math.isfinite(min_precision_lower_95)
        or not 0.0 < min_precision_lower_95 <= 1.0
    ):
        raise ValueError("min_precision_lower_95 debe estar en (0, 1]")
    candidates = sorted({float(value) for value in probabilities}, reverse=True)
    valid: list[ThresholdSelection] = []
    total = len(labels)
    for threshold in candidates:
        indices = [i for i, value in enumerate(probabilities) if value >= threshold]
        correct = sum(int(labels[i] == 1) for i in indices)
        precision = correct / len(indices)
        precision_lower_95 = one_sided_success_lower(correct, len(indices))
        if precision_lower_95 >= min_precision_lower_95:
            valid.append(
                ThresholdSelection(
                    threshold,
                    precision,
                    precision_lower_95,
                    len(indices) / total,
                    correct,
                    len(indices),
                    total,
                )
            )
    if not valid:
        return ThresholdSelection(1.0, 0.0, 0.0, 0.0, 0, 0, total)
    return max(valid, key=lambda item: (item.coverage, item.threshold))


def serialize_calibrator_artifact(artifact: CalibratorArtifact) -> bytes:
    return (
        json.dumps(
            artifact.to_dict(),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def parse_calibrator_artifact_bytes(
    payload: bytes,
    *,
    expected_fingerprint: str,
) -> CalibratorArtifact:
    def reject_constant(value: str):
        raise ValueError(f"constante JSON no finita: {value}")

    def strict_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"clave JSON duplicada: {key}")
            result[key] = value
        return result

    if not isinstance(payload, bytes) or len(payload) > 1_000_000:
        raise ValueError("payload de calibrador invalido")
    if (
        not isinstance(expected_fingerprint, str)
        or len(expected_fingerprint) != 64
        or any(char not in "0123456789abcdef" for char in expected_fingerprint)
    ):
        raise ValueError("expected_fingerprint de calibrador invalido")
    try:
        data = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("calibrador JSON invalido") from error
    if not isinstance(data, dict):
        raise ValueError("el calibrador debe ser un objeto JSON")
    artifact = CalibratorArtifact.from_dict(data)
    if artifact.version != expected_fingerprint:
        raise ValueError("calibrador no coincide con el trust anchor")
    return artifact
