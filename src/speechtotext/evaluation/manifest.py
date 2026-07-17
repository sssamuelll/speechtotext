from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Literal, Mapping

from speechtotext.audio.types import SpeechRegion
from speechtotext.evaluation.filesystem import (
    CorpusFilesystem,
    default_corpus_filesystem,
)

CorpusKind = Literal["speech", "silence", "noise", "other_voice", "replay", "tts"]
SpoofLabel = Literal["bona_fide", "replay", "tts", "other_voice", "unknown"]
AssetRole = Literal["primary_audio", "derived", "backup"]
SCHEMA_VERSION = "speechtotext.corpus/v1"
MAX_CORPUS_MANIFEST_BYTES = 8 * 1024 * 1024
_WIN_FORBIDDEN = frozenset('<>:"\\|?*')
_WIN_RESERVED = frozenset({
    "clock$", "con", "conin$", "conout$", "prn", "aux", "nul",
})
_WIN_DEVICE_SUFFIXES = frozenset((*"123456789", "¹", "²", "³"))
_ENTRY_FIELDS = {
    "clip_id",
    "assets",
    "session_id",
    "recorded_on",
    "kind",
    "speaker",
    "condition",
    "source_id",
    "duration_ms",
    "transcript",
    "speech_regions",
    "intent",
    "slots",
    "spoof_label",
    "provenance",
    "consent_or_license",
    "retention_until",
}
_KINDS = {"speech", "silence", "noise", "other_voice", "replay", "tts"}
_SPOOF = {"bona_fide", "replay", "tts", "other_voice", "unknown"}
CORPUS_CONDITIONS = frozenset({
    "clean", "noise", "silence",
    "clean_near", "clean_fast", "clean_low", "clean_distance",
    "noise_keyboard", "noise_household", "other_voice_media",
    "silence_non_vocal", "replay_tts", "wake_continuous", "wake_paused",
    "short_technical", "tts_noise",
})


def _safe_relative(value: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError("asset path exige una ruta relativa segura")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("asset path exige una ruta relativa segura")
    for part in path.parts:
        stem = part.split(".", 1)[0].casefold()
        device = stem in _WIN_RESERVED or (
            len(stem) == 4
            and stem[:3] in {"com", "lpt"}
            and stem[3] in _WIN_DEVICE_SUFFIXES
        )
        if (
            part.endswith((".", " "))
            or any(ord(char) < 32 or char in _WIN_FORBIDDEN for char in part)
            or device
            or len(part.encode("utf-16-le")) // 2 > 255
        ):
            raise ValueError("asset path exige una ruta relativa segura")
    if len(value.encode("utf-16-le")) // 2 > 32767:
        raise ValueError("asset path exige una ruta relativa segura")
    return value


def _is_within(path: Path, root: Path) -> bool:
    path = Path(os.path.abspath(path))
    root = Path(os.path.abspath(root))
    return path == root or root in path.parents


@dataclass(frozen=True)
class CorpusAsset:
    role: AssetRole
    path: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "path": self.path, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CorpusAsset":
        if not isinstance(data, dict) or set(data) != {"role", "path", "sha256"}:
            raise ValueError("campos de asset invalidos")
        role = data["role"]
        path = data["path"]
        digest = data["sha256"]
        if type(role) is not str or type(path) is not str or type(digest) is not str:
            raise ValueError("tipo de asset invalido")
        if role not in {"primary_audio", "derived", "backup"}:
            raise ValueError("role de asset invalido")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("sha256 de asset invalido")
        return cls(role, _safe_relative(path), digest)


@dataclass(frozen=True)
class CorpusEntry:
    clip_id: str
    assets: tuple[CorpusAsset, ...]
    session_id: str
    recorded_on: date
    kind: CorpusKind
    speaker: str
    condition: str
    source_id: str
    duration_ms: int
    transcript: str
    speech_regions: tuple[SpeechRegion, ...]
    intent: str | None
    slots: Mapping[str, str]
    spoof_label: SpoofLabel
    provenance: str
    consent_or_license: str
    retention_until: date

    @property
    def primary_audio(self) -> CorpusAsset:
        return next(asset for asset in self.assets if asset.role == "primary_audio")

    @property
    def audio_path(self) -> str:
        return self.primary_audio.path

    @property
    def audio_sha256(self) -> str:
        return self.primary_audio.sha256

    def to_dict(self) -> dict[str, object]:
        return {
            "clip_id": self.clip_id,
            "assets": [asset.to_dict() for asset in self.assets],
            "session_id": self.session_id,
            "recorded_on": self.recorded_on.isoformat(),
            "kind": self.kind,
            "speaker": self.speaker,
            "condition": self.condition,
            "source_id": self.source_id,
            "duration_ms": self.duration_ms,
            "transcript": self.transcript,
            "speech_regions": [
                {"start_s": region.start_s, "end_s": region.end_s}
                for region in self.speech_regions
            ],
            "intent": self.intent,
            "slots": dict(self.slots),
            "spoof_label": self.spoof_label,
            "provenance": self.provenance,
            "consent_or_license": self.consent_or_license,
            "retention_until": self.retention_until.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CorpusEntry":
        if not isinstance(data, dict):
            raise ValueError("cada entry de corpus debe ser un objeto")
        if set(data) != _ENTRY_FIELDS:
            raise ValueError(
                f"campos de corpus invalidos: {sorted(set(data) ^ _ENTRY_FIELDS)}"
            )
        string_fields = (
            "clip_id", "session_id", "recorded_on", "kind", "speaker",
            "condition", "source_id", "transcript", "spoof_label",
            "provenance", "consent_or_license", "retention_until",
        )
        if any(type(data[name]) is not str for name in string_fields):
            raise ValueError("tipo de campo de corpus invalido")
        if type(data["duration_ms"]) is not int:
            raise ValueError("tipo de duration_ms invalido: exige un entero")
        if data["intent"] is not None and type(data["intent"]) is not str:
            raise ValueError("intent exige string o null")
        kind = data["kind"]
        spoof = data["spoof_label"]
        if kind not in _KINDS or spoof not in _SPOOF:
            raise ValueError("kind o spoof_label invalido")
        assets_raw = data["assets"]
        regions_raw = data["speech_regions"]
        slots = data["slots"]
        if (
            not isinstance(assets_raw, list)
            or not assets_raw
            or not isinstance(regions_raw, list)
            or not isinstance(slots, dict)
            or any(type(key) is not str or type(value) is not str for key, value in slots.items())
        ):
            raise ValueError("assets/speech_regions/slots con tipo invalido")
        assets = tuple(CorpusAsset.from_dict(item) for item in assets_raw)
        primary_count = sum(asset.role == "primary_audio" for asset in assets)
        asset_paths = [asset.path.casefold() for asset in assets]
        if primary_count != 1 or len(set(asset_paths)) != len(asset_paths):
            raise ValueError("se exige un primary_audio y paths de asset unicos")
        regions: list[SpeechRegion] = []
        for item in regions_raw:
            if not isinstance(item, dict) or set(item) != {"start_s", "end_s"}:
                raise ValueError("speech_region invalida")
            start, end = item["start_s"], item["end_s"]
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in (start, end)
            ):
                raise ValueError("speech_region con tipo invalido: exige numeros finitos")
            regions.append(SpeechRegion(float(start), float(end)))
        recorded_on = date.fromisoformat(data["recorded_on"])
        retention_until = date.fromisoformat(data["retention_until"])
        if (
            recorded_on.isoformat() != data["recorded_on"]
            or retention_until.isoformat() != data["retention_until"]
        ):
            raise ValueError("fechas de corpus deben usar YYYY-MM-DD canonico")
        if retention_until < recorded_on:
            raise ValueError("retention_until no puede preceder recorded_on")
        entry = cls(
            clip_id=data["clip_id"],
            assets=assets,
            session_id=data["session_id"],
            recorded_on=recorded_on,
            kind=kind,
            speaker=data["speaker"],
            condition=data["condition"],
            source_id=data["source_id"],
            duration_ms=data["duration_ms"],
            transcript=data["transcript"],
            speech_regions=tuple(regions),
            intent=data["intent"],
            slots=dict(slots),
            spoof_label=spoof,
            provenance=data["provenance"],
            consent_or_license=data["consent_or_license"],
            retention_until=retention_until,
        )
        required_text = (
            entry.clip_id,
            entry.session_id,
            entry.condition,
            entry.source_id,
            entry.provenance,
            entry.consent_or_license,
        )
        if any(not value.strip() for value in required_text):
            raise ValueError("campos de identidad/procedencia no pueden estar vacios")
        if entry.condition not in CORPUS_CONDITIONS:
            raise ValueError("condition fuera de la taxonomia versionada")
        if entry.duration_ms <= 0:
            raise ValueError("duration_ms debe ser positivo")
        return entry


@dataclass(frozen=True)
class CorpusManifest:
    schema_version: str
    dataset_id: str
    created_on: date
    entries: tuple[CorpusEntry, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "created_on": self.created_on.isoformat(),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @property
    def version(self) -> str:
        encoded = json.dumps(
            self.to_dict(), ensure_ascii=True, allow_nan=False,
            separators=(",", ":"), sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _validate_asset_location(asset: CorpusAsset, dataset_root: Path) -> None:
    root = dataset_root.resolve()
    path = (root / Path(asset.path)).resolve()
    if not _is_within(path, root):
        raise ValueError("asset path escapa de dataset_root")


def parse_corpus_manifest_bytes(
    payload: bytes,
    *,
    dataset_root: Path,
) -> CorpusManifest:
    if (
        type(payload) is not bytes
        or not payload
        or len(payload) > MAX_CORPUS_MANIFEST_BYTES
    ):
        raise ValueError("payload de corpus vacio, invalido o sobre el limite")
    root = Path(os.path.abspath(dataset_root))

    def reject_constant(value: str):
        raise ValueError(f"constante JSON no finita: {value}")

    def strict_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"clave JSON duplicada: {key}")
            result[key] = value
        return result

    try:
        data = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("manifest de corpus JSON invalido") from error
    expected = {"schema_version", "dataset_id", "created_on", "entries"}
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError("campos top-level de corpus invalidos")
    if data["schema_version"] != SCHEMA_VERSION:
        raise ValueError("schema_version de corpus incompatible")
    if type(data["dataset_id"]) is not str or not data["dataset_id"].strip():
        raise ValueError("dataset_id de corpus invalido")
    if type(data["created_on"]) is not str:
        raise ValueError("created_on de corpus invalido")
    entries_raw = data["entries"]
    if not isinstance(entries_raw, list) or not entries_raw:
        raise ValueError("entries debe ser una lista no vacia")
    entries = tuple(CorpusEntry.from_dict(item) for item in entries_raw)
    ids = [entry.clip_id for entry in entries]
    if len(set(ids)) != len(ids):
        raise ValueError("clip_id duplicado")
    all_asset_paths = [asset.path.casefold() for entry in entries for asset in entry.assets]
    if len(set(all_asset_paths)) != len(all_asset_paths):
        raise ValueError("asset path declarado por mas de una entrada")
    for entry in entries:
        for asset in entry.assets:
            _validate_asset_location(asset, root)
    created_on = date.fromisoformat(data["created_on"])
    if created_on.isoformat() != data["created_on"]:
        raise ValueError("created_on debe usar YYYY-MM-DD canonico")
    return CorpusManifest(
        schema_version=SCHEMA_VERSION,
        dataset_id=data["dataset_id"],
        created_on=created_on,
        entries=entries,
    )


def _read_manifest_bounded(stream) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = stream.read(min(65536, MAX_CORPUS_MANIFEST_BYTES + 1 - total))
        if not isinstance(chunk, bytes):
            raise ValueError("stream de manifest debe ser binario")
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        total += len(chunk)
        if total > MAX_CORPUS_MANIFEST_BYTES:
            raise ValueError("payload de corpus sobre el limite")


def load_corpus_manifest(
    path: Path,
    *,
    dataset_root: Path,
    repo_root: Path,
    filesystem: CorpusFilesystem | None = None,
) -> CorpusManifest:
    repo = Path(os.path.abspath(repo_root))
    root = Path(os.path.abspath(dataset_root))
    manifest_path = Path(os.path.abspath(path))
    if _is_within(root, repo) or _is_within(manifest_path, repo):
        raise ValueError("dataset y manifest deben vivir fuera de Git")
    if not _is_within(manifest_path, root):
        raise ValueError("manifest debe vivir dentro de dataset_root privado")
    adapter = filesystem or default_corpus_filesystem()
    with adapter.lease_manifest(manifest_path, root) as lease:
        payload = _read_manifest_bounded(lease.stream)
    return parse_corpus_manifest_bytes(payload, dataset_root=root)
