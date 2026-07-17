from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import BinaryIO, Literal

from speechtotext.models.filesystem import (
    ModelFileLease,
    ModelFilesystem,
    ModelFilesystemError,
    _safe_relative,
    default_model_filesystem,
)


SCHEMA_VERSION = "speechtotext.model/v1"
_MAX_MANIFEST_BYTES = 1_048_576
_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)
_MANIFEST_TRUST_TOKEN = object()
_ARTIFACT_FACTORY_TOKEN = object()
_FIELDS = {
    "schema_version",
    "model_id",
    "source",
    "revision_kind",
    "revision",
    "license",
    "format",
    "sample_rate",
    "preprocessing",
    "files",
}


class ModelIntegrityError(ModelFilesystemError):
    """A local artifact does not match its approved manifest."""


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    stream.seek(0)
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        if not isinstance(block, bytes):
            raise ModelIntegrityError("lectura del modelo no devolvio bytes")
        digest.update(block)
    stream.seek(0)
    return digest.hexdigest()


def _read_bounded(stream: BinaryIO, limit: int = _MAX_MANIFEST_BYTES) -> bytes:
    stream.seek(0)
    payload = stream.read(limit + 1)
    stream.seek(0)
    if not isinstance(payload, bytes):
        raise ModelIntegrityError("lectura de manifest no devolvio bytes")
    if len(payload) > limit:
        raise ValueError("manifest JSON excede 1 MiB")
    return payload


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contiene clave duplicada: {key}")
        result[key] = value
    return result


def _reject_constant(value: str):
    raise ValueError(f"JSON contiene constante no finita: {value}")


def _canonical(data: object) -> bytes:
    return json.dumps(
        data,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("preprocessing exige claves string")
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("preprocessing solo admite JSON finito")
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise ValueError(f"preprocessing no admite {type(value).__name__}")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class ModelFile:
    path: str
    sha256: str

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ModelFile:
        if (
            not isinstance(data, Mapping)
            or set(data) != {"path", "sha256"}
            or type(data["path"]) is not str
            or type(data["sha256"]) is not str
        ):
            raise ValueError("cada archivo exige exactamente path y sha256")
        path = _safe_relative(data["path"])
        digest = data["sha256"]
        if _SHA256.fullmatch(digest) is None:
            raise ValueError("sha256 debe tener 64 caracteres hexadecimales lowercase")
        return cls(path, digest)

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True, init=False)
class ModelManifest:
    schema_version: str
    model_id: str
    source: str
    revision_kind: Literal["git_commit", "content_digest"]
    revision: str
    license: str
    format: str
    sample_rate: int
    preprocessing: Mapping[str, object]
    files: tuple[ModelFile, ...]
    _trusted_fingerprint: str | None
    _manifest_relative_path: str | None
    _trust_token: object | None

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ModelManifest:
        if not isinstance(data, Mapping):
            raise ValueError("manifest debe ser un objeto JSON")
        unknown = set(data) - _FIELDS
        missing = _FIELDS - set(data)
        if unknown:
            raise ValueError(f"campos desconocidos: {sorted(unknown)}")
        if missing:
            raise ValueError(f"campos requeridos ausentes: {sorted(missing)}")
        text_fields = (
            "schema_version",
            "model_id",
            "source",
            "revision_kind",
            "revision",
            "license",
            "format",
        )
        if any(type(data[name]) is not str for name in text_fields):
            raise ValueError("metadata de modelo exige strings")
        if data["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"schema_version incompatible: {data['schema_version']}")
        if any(
            not data[name].strip()
            for name in ("model_id", "source", "license", "format")
        ):
            raise ValueError("metadata obligatoria de modelo vacia")
        if type(data["sample_rate"]) is not int or data["sample_rate"] <= 0:
            raise ValueError("sample_rate debe ser un entero positivo")
        if type(data["preprocessing"]) is not dict:
            raise ValueError("preprocessing debe ser un objeto JSON")
        files_raw = data["files"]
        if type(files_raw) is not list or not files_raw:
            raise ValueError("files debe ser una lista no vacia")
        files = tuple(ModelFile.from_dict(item) for item in files_raw)
        folded = [item.path.casefold() for item in files]
        if len(folded) != len(set(folded)):
            raise ValueError("paths de modelo duplicados por case-fold")
        revision = data["revision"].lower()
        git_commit = (
            data["revision_kind"] == "git_commit"
            and len(revision) in {40, 64}
            and all(char in "0123456789abcdef" for char in revision)
        )
        content_digest = (
            data["revision_kind"] == "content_digest"
            and revision.startswith("sha256:")
            and len(revision) == 71
            and all(char in "0123456789abcdef" for char in revision[7:])
        )
        if not (git_commit or content_digest):
            raise ValueError("revision inmutable invalida")
        manifest = object.__new__(cls)
        values = {
            "schema_version": SCHEMA_VERSION,
            "model_id": data["model_id"],
            "source": data["source"],
            "revision_kind": data["revision_kind"],
            "revision": revision,
            "license": data["license"],
            "format": data["format"],
            "sample_rate": data["sample_rate"],
            "preprocessing": _freeze_json(data["preprocessing"]),
            "files": files,
            "_trusted_fingerprint": None,
            "_manifest_relative_path": None,
            "_trust_token": None,
        }
        for name, value in values.items():
            object.__setattr__(manifest, name, value)
        _canonical(manifest.to_dict())
        return manifest

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "source": self.source,
            "revision_kind": self.revision_kind,
            "revision": self.revision,
            "license": self.license,
            "format": self.format,
            "sample_rate": self.sample_rate,
            "preprocessing": _thaw_json(self.preprocessing),
            "files": [item.to_dict() for item in self.files],
        }

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(self.to_dict())).hexdigest()


def parse_model_manifest_bytes(payload: bytes) -> ModelManifest:
    if type(payload) is not bytes or len(payload) > _MAX_MANIFEST_BYTES:
        raise ValueError("payload JSON de manifest invalido o excede 1 MiB")
    try:
        data = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("manifest JSON invalido") from error
    if type(data) is not dict:
        raise ValueError("el manifest debe ser un objeto JSON")
    return ModelManifest.from_dict(data)


def _translate_filesystem_error(error: ModelFilesystemError) -> ModelIntegrityError:
    if isinstance(error, ModelIntegrityError):
        return error
    return ModelIntegrityError(str(error))


def load_model_manifest(
    path: Path,
    *,
    model_root: Path,
    expected_fingerprint: str,
    filesystem: ModelFilesystem | None = None,
) -> ModelManifest:
    if type(expected_fingerprint) is not str or _SHA256.fullmatch(expected_fingerprint) is None:
        raise ValueError("expected_fingerprint debe ser SHA-256 lowercase")
    fs = default_model_filesystem() if filesystem is None else filesystem
    try:
        with fs.lease_read_only_root(Path(model_root)) as root_lease:
            manifest_relative = fs.relative_path(Path(path), root_lease)
            with fs.lease_file(manifest_relative, root_lease) as lease:
                manifest = parse_model_manifest_bytes(_read_bounded(lease.stream))
    except ModelFilesystemError as error:
        raise _translate_filesystem_error(error) from error
    if not hmac.compare_digest(manifest.fingerprint, expected_fingerprint):
        raise ModelIntegrityError("manifest no coincide con el trust anchor")
    if manifest_relative.casefold() in {
        item.path.casefold() for item in manifest.files
    }:
        raise ValueError("el manifest no puede declararse como peso de si mismo")
    object.__setattr__(manifest, "_trusted_fingerprint", expected_fingerprint)
    object.__setattr__(manifest, "_manifest_relative_path", manifest_relative)
    object.__setattr__(manifest, "_trust_token", _MANIFEST_TRUST_TOKEN)
    return manifest


@dataclass(frozen=True, init=False)
class VerifiedModelArtifact:
    _manifest: ModelManifest
    _requested_root: Path
    _filesystem: ModelFilesystem
    _stack: ExitStack | None
    _root: Path | None
    _active: bool
    _consumed: bool
    _factory_token: object
    _file_identities: Mapping[str, tuple[int, bytes]]

    def __enter__(self) -> VerifiedModelArtifact:
        if self._factory_token is not _ARTIFACT_FACTORY_TOKEN:
            raise ModelIntegrityError("artefacto no proviene de verify_model_files")
        if self._consumed:
            raise ModelIntegrityError("artefacto de modelo ya fue consumido")
        object.__setattr__(self, "_consumed", True)
        stack = ExitStack()
        try:
            try:
                root_lease = stack.enter_context(
                    self._filesystem.lease_read_only_root(self._requested_root)
                )
                manifest_relative = self._manifest._manifest_relative_path
                if manifest_relative is None:
                    raise ModelIntegrityError("manifest no tiene trust anchor")
                manifest_lease = stack.enter_context(
                    self._filesystem.lease_file(manifest_relative, root_lease)
                )
                current = parse_model_manifest_bytes(
                    _read_bounded(manifest_lease.stream)
                )
                if (
                    not hmac.compare_digest(
                        current.fingerprint,
                        self._manifest._trusted_fingerprint or "",
                    )
                    or current.to_dict() != self._manifest.to_dict()
                ):
                    raise ModelIntegrityError(
                        "manifest cambio despues del trust check"
                    )
                expected_inventory = tuple(
                    sorted(
                        (
                            manifest_relative,
                            *(item.path for item in self._manifest.files),
                        ),
                        key=str.casefold,
                    )
                )
                if self._filesystem.inventory(root_lease) != expected_inventory:
                    raise ModelIntegrityError(
                        "inventario de modelo incompleto o con extras"
                    )
                identities: dict[str, tuple[int, bytes]] = {
                    manifest_relative: (
                        manifest_lease.volume_serial,
                        manifest_lease.file_id,
                    )
                }
                for item in self._manifest.files:
                    lease: ModelFileLease = stack.enter_context(
                        self._filesystem.lease_file(item.path, root_lease)
                    )
                    if lease.size < 0:
                        raise ModelIntegrityError("identidad de modelo invalida")
                    actual = _sha256_stream(lease.stream)
                    if not hmac.compare_digest(actual, item.sha256):
                        raise ModelIntegrityError(
                            f"sha256 invalido para {item.path}: "
                            f"esperado {item.sha256}, obtenido {actual}"
                        )
                    identities[item.path] = (lease.volume_serial, lease.file_id)
                if self._filesystem.inventory(root_lease) != expected_inventory:
                    raise ModelIntegrityError(
                        "inventario de modelo cambio durante el audit"
                    )
            except ModelFilesystemError as error:
                raise _translate_filesystem_error(error) from error
            object.__setattr__(self, "_root", root_lease.root)
            object.__setattr__(self, "_stack", stack)
            object.__setattr__(self, "_file_identities", MappingProxyType(identities))
            object.__setattr__(self, "_active", True)
            return self
        except BaseException:
            stack.close()
            raise

    def __exit__(self, exc_type, exc, traceback) -> None:
        stack = self._stack
        object.__setattr__(self, "_active", False)
        object.__setattr__(self, "_stack", None)
        object.__setattr__(self, "_root", None)
        object.__setattr__(self, "_file_identities", MappingProxyType({}))
        if stack is not None:
            stack.close()

    def require_active(self) -> None:
        if not self._active or self._stack is None or self._root is None:
            raise ModelIntegrityError("artefacto de modelo no esta activo")

    @property
    def manifest(self) -> ModelManifest:
        self.require_active()
        return self._manifest

    @property
    def root(self) -> Path:
        self.require_active()
        assert self._root is not None
        return self._root

    @property
    def fingerprint(self) -> str:
        self.require_active()
        return self._manifest.fingerprint

    @property
    def file_identities(self) -> Mapping[str, tuple[int, bytes]]:
        self.require_active()
        return self._file_identities


def verify_model_files(
    manifest: ModelManifest,
    root: Path,
    *,
    filesystem: ModelFilesystem | None = None,
) -> VerifiedModelArtifact:
    if not isinstance(manifest, ModelManifest):
        raise TypeError("verify_model_files exige ModelManifest sellado")
    if (
        manifest._trust_token is not _MANIFEST_TRUST_TOKEN
        or manifest._trusted_fingerprint is None
        or not hmac.compare_digest(
            manifest._trusted_fingerprint, manifest.fingerprint
        )
        or manifest._manifest_relative_path is None
    ):
        raise ModelIntegrityError("ModelManifest no proviene de un trust anchor")
    artifact = object.__new__(VerifiedModelArtifact)
    values = {
        "_manifest": manifest,
        "_requested_root": Path(root),
        "_filesystem": default_model_filesystem() if filesystem is None else filesystem,
        "_stack": None,
        "_root": None,
        "_active": False,
        "_consumed": False,
        "_factory_token": _ARTIFACT_FACTORY_TOKEN,
        "_file_identities": MappingProxyType({}),
    }
    for name, value in values.items():
        object.__setattr__(artifact, name, value)
    return artifact
