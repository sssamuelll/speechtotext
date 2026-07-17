from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Sequence

from speechtotext.models.manifest import VerifiedModelArtifact

_PROVENANCE_FACTORY_TOKEN = object()


def _validate_json(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("el pipeline solo admite JSON finito")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("las claves del pipeline deben ser strings")
            _validate_json(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_json(item)
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError(f"valor no serializable en pipeline: {type(value).__name__}")


def _freeze_json(value: object) -> object:
    _validate_json(value)
    if isinstance(value, Mapping):
        return MappingProxyType({
            key: _freeze_json(item) for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class PipelineStep:
    name: str
    version: str
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or not isinstance(self.version, str)
            or not self.version.strip()
            or not isinstance(self.parameters, Mapping)
        ):
            raise ValueError("name y version de un step son obligatorios")
        object.__setattr__(self, "parameters", _freeze_json(self.parameters))

    def to_dict(self) -> dict[str, object]:
        data = {
            "name": self.name,
            "version": self.version,
            "parameters": _thaw_json(self.parameters),
        }
        _validate_json(data)
        return data


@dataclass(frozen=True, init=False)
class PipelineProvenance:
    sample_rate: int
    parent_fingerprint: str | None
    steps: tuple[PipelineStep, ...]
    model_fingerprints: tuple[str, ...]
    thresholds: Mapping[str, object]

    @classmethod
    def _create(
        cls,
        sample_rate: int,
        parent_fingerprint: str | None,
        steps: Sequence[PipelineStep],
        models: Sequence[VerifiedModelArtifact],
        thresholds: Mapping[str, object],
        *,
        _factory_token=None,
    ) -> "PipelineProvenance":
        if _factory_token is not _PROVENANCE_FACTORY_TOKEN:
            raise TypeError(
                "PipelineProvenance solo puede crearse mediante su factory publica"
            )
        model_values = tuple(models)
        step_values = tuple(steps)
        if type(sample_rate) is not int or sample_rate <= 0:
            raise ValueError("sample_rate debe ser un entero positivo")
        if not isinstance(thresholds, Mapping):
            raise ValueError("thresholds debe ser un mapping JSON")
        if any(not isinstance(model, VerifiedModelArtifact) for model in model_values):
            raise TypeError("models exige VerifiedModelArtifact")
        if any(not isinstance(step, PipelineStep) for step in step_values):
            raise TypeError("steps exige PipelineStep")
        instance = object.__new__(cls)
        object.__setattr__(instance, "sample_rate", sample_rate)
        object.__setattr__(instance, "parent_fingerprint", parent_fingerprint)
        object.__setattr__(instance, "steps", step_values)
        object.__setattr__(
            instance,
            "model_fingerprints",
            tuple(model.fingerprint for model in model_values),
        )
        object.__setattr__(instance, "thresholds", _freeze_json(thresholds))
        instance._validate()
        return instance

    @classmethod
    def capture(
        cls,
        *,
        sample_rate: int,
        step: PipelineStep,
        models: Sequence[VerifiedModelArtifact] = (),
        thresholds: Mapping[str, object] | None = None,
    ) -> "PipelineProvenance":
        return cls._create(
            sample_rate,
            None,
            (step,),
            models,
            {} if thresholds is None else thresholds,
            _factory_token=_PROVENANCE_FACTORY_TOKEN,
        )

    @classmethod
    def derive(
        cls,
        parent: "PipelineProvenance",
        *,
        sample_rate: int,
        steps: Sequence[PipelineStep],
        models: Sequence[VerifiedModelArtifact] = (),
        thresholds: Mapping[str, object] | None = None,
    ) -> "PipelineProvenance":
        if not isinstance(parent, PipelineProvenance):
            raise TypeError("parent exige PipelineProvenance")
        if not steps:
            raise ValueError("derive exige al menos una transformacion de audio")
        return cls._create(
            sample_rate,
            parent.fingerprint,
            tuple(steps),
            models,
            {} if thresholds is None else thresholds,
            _factory_token=_PROVENANCE_FACTORY_TOKEN,
        )

    def _validate(self) -> None:
        if type(self.sample_rate) is not int or self.sample_rate <= 0:
            raise ValueError("sample_rate debe ser un entero positivo")
        if not self.steps:
            raise ValueError("pipeline exige steps")
        fingerprints = (*self.model_fingerprints,)
        if self.parent_fingerprint is not None:
            fingerprints = (self.parent_fingerprint, *fingerprints)
        if any(
            len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in fingerprints
        ):
            raise ValueError("fingerprint de parent/modelo invalido")
        _validate_json(self.thresholds)
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "model_fingerprints", tuple(self.model_fingerprints))
        object.__setattr__(self, "thresholds", _freeze_json(self.thresholds))

    def _payload(self) -> dict[str, object]:
        payload = {
            "schema_version": "speechtotext.pipeline/v1",
            "sample_rate": self.sample_rate,
            "parent_fingerprint": self.parent_fingerprint,
            "steps": [step.to_dict() for step in self.steps],
            "model_fingerprints": list(self.model_fingerprints),
            "thresholds": _thaw_json(self.thresholds),
        }
        _validate_json(payload)
        return payload

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self._payload(), ensure_ascii=True, allow_nan=False,
            separators=(",", ":"), sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, object],
        *,
        parent: "PipelineProvenance | None",
        models: Sequence[VerifiedModelArtifact],
    ) -> "PipelineProvenance":
        if parent is not None and not isinstance(parent, PipelineProvenance):
            raise TypeError("parent exige PipelineProvenance")
        expected = {
            "schema_version", "sample_rate", "parent_fingerprint", "steps",
            "model_fingerprints", "thresholds", "fingerprint",
        }
        if (
            not isinstance(data, Mapping)
            or set(data) != expected
            or data["schema_version"] != "speechtotext.pipeline/v1"
            or not isinstance(data["fingerprint"], str)
            or not isinstance(data["steps"], list)
            or not isinstance(data["model_fingerprints"], list)
            or not isinstance(data["thresholds"], Mapping)
            or any(
                not isinstance(item, Mapping)
                or set(item) != {"name", "version", "parameters"}
                or not isinstance(item["name"], str)
                or not isinstance(item["version"], str)
                or not isinstance(item["parameters"], Mapping)
                for item in data["steps"]
            )
            or any(not isinstance(item, str) for item in data["model_fingerprints"])
        ):
            raise ValueError("schema de pipeline invalido")
        if type(data["sample_rate"]) is not int or data["sample_rate"] <= 0:
            raise ValueError("sample_rate debe ser un entero positivo")
        provenance = cls._create(
            data["sample_rate"],
            None if parent is None else parent.fingerprint,
            tuple(PipelineStep(item["name"], item["version"], item["parameters"])
                  for item in data["steps"]),
            models,
            data["thresholds"],
            _factory_token=_PROVENANCE_FACTORY_TOKEN,
        )
        if (
            data["parent_fingerprint"] != provenance.parent_fingerprint
            or tuple(data["model_fingerprints"]) != provenance.model_fingerprints
            or data["fingerprint"] != provenance.fingerprint
        ):
            raise ValueError("fingerprint declarado no coincide con provenance")
        return provenance
