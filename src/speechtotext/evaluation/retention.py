from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

from speechtotext.evaluation.filesystem import (
    CorpusAssetLease,
    CorpusFileIdentity,
    CorpusFilesystem,
    CorpusFilesystemError,
    PrivateFileLease,
    PrivateJournalLease,
    default_corpus_filesystem,
    lease_corpus_asset,
)
from speechtotext.evaluation.manifest import (
    MAX_CORPUS_MANIFEST_BYTES,
    CorpusAsset,
    CorpusManifest,
    parse_corpus_manifest_bytes,
)
from speechtotext.evaluation.privacy import protected_ref

RECEIPT_SCHEMA = "speechtotext.purge-receipt/v1"


class DatasetSecurityError(RuntimeError):
    """El OS no pudo demostrar seguridad del dataset, o la evidencia cambio."""


class RetentionError(RuntimeError):
    """Una operacion de retencion no pudo completarse de forma segura."""


def _abspath(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _within(path: Path, root: Path) -> bool:
    path = _abspath(path)
    root = _abspath(root)
    return path == root or root in path.parents


def _canonical(obj: object) -> bytes:
    return (
        json.dumps(
            obj, ensure_ascii=True, allow_nan=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RetentionItem:
    clip_ref: str
    retention_until: date
    expired: bool
    condition: str


@dataclass(frozen=True)
class PurgeReceipt:
    schema: str
    operation_ref: str
    dataset_ref: str
    manifest_ref: str
    planned_at: str
    finished_at: str
    dry_run: bool
    status: str
    items: tuple[Mapping[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "operation_ref": self.operation_ref,
            "dataset_ref": self.dataset_ref,
            "manifest_ref": self.manifest_ref,
            "planned_at": self.planned_at,
            "finished_at": self.finished_at,
            "dry_run": self.dry_run,
            "status": self.status,
            "items": [dict(item) for item in self.items],
        }


class DatasetSecurityEvidence(AbstractContextManager):
    """Evidencia sellada y ligada al root/manifest y a un scope de assets.

    Es un context manager obligatorio: mantiene abierto el lease read-only del
    manifest desde el audit hasta cerrar todo flujo. No expone un constructor
    publico con booleanos; `audit_dataset_security` es la unica via.
    """

    __slots__ = (
        "sufficient",
        "encryption_provider",
        "checked_at",
        "_dataset_root",
        "_repo_root",
        "_manifest_path",
        "_manifest",
        "_reports_root",
        "_secrets_root",
        "_authorized_asset_digests",
        "_observed_asset_paths",
        "_filesystem",
        "_active",
        "_root_identity",
        "_manifest_sha256",
        "_manifest_version",
        "_manifest_lease",
        "_manifest_lease_ctx",
        "_lock_ctx",
        "_update_ctx",
        "_update_lease",
        "_mutation_started",
        "_manifest_replaced",
    )

    def __enter__(self) -> "DatasetSecurityEvidence":
        fs = self._filesystem
        ctx = fs.lease_manifest(self._manifest_path, self._dataset_root)
        lease = ctx.__enter__()
        self._manifest_lease_ctx = ctx
        self._manifest_lease = lease
        payload = self._read_lease_bytes(lease)
        parsed = parse_corpus_manifest_bytes(payload, dataset_root=self._dataset_root)
        if parsed.to_dict() != self._manifest.to_dict():
            ctx.__exit__(None, None, None)
            raise DatasetSecurityError("manifest reledo difiere del auditado")
        self._manifest_sha256 = hashlib.sha256(payload).hexdigest()
        self._manifest_version = parsed.version
        self._root_identity = fs.identity(self._dataset_root)
        acl_ok = bool(fs.current_user_only_acl(self._dataset_root))
        enc_ok, provider = fs.encryption_at_rest(self._dataset_root)
        self.encryption_provider = provider
        self.checked_at = _now()
        self.sufficient = acl_ok and bool(enc_ok)
        self._active = True
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._active = False
        for attr in ("_update_ctx", "_lock_ctx", "_manifest_lease_ctx"):
            ctx = getattr(self, attr, None)
            if ctx is not None:
                setattr(self, attr, None)
                try:
                    ctx.__exit__(None, None, None)
                except Exception:
                    pass

    # -- validation -------------------------------------------------------
    @staticmethod
    def _read_lease_bytes(lease: PrivateFileLease) -> bytes:
        lease.stream.seek(0)
        return lease.stream.read()

    def _require_active(self) -> None:
        if not self._active:
            raise DatasetSecurityError("la evidencia ya no esta activa")

    def require_for(self, dataset_root: Path, manifest: CorpusManifest) -> None:
        self._require_active()
        if not self.sufficient:
            raise DatasetSecurityError("evidencia insuficiente de seguridad")
        if _abspath(dataset_root) != _abspath(self._dataset_root):
            raise DatasetSecurityError("root distinto al auditado")
        if self._filesystem.identity(self._dataset_root) != self._root_identity:
            raise DatasetSecurityError("identidad del root cambio")
        payload = self._read_lease_bytes(self._manifest_lease)
        if hashlib.sha256(payload).hexdigest() != self._manifest_sha256:
            raise DatasetSecurityError("el manifest cambio bajo el lease")
        parsed = parse_corpus_manifest_bytes(payload, dataset_root=self._dataset_root)
        if parsed.to_dict() != manifest.to_dict():
            raise DatasetSecurityError("manifest no coincide con la evidencia")

    def require_asset(self, asset: CorpusAsset, lease: CorpusAssetLease) -> None:
        self._require_active()
        expected = self._authorized_asset_digests.get(asset.path)
        if expected is None or expected != asset.sha256:
            raise DatasetSecurityError("asset fuera del scope autorizado")
        if lease.asset.path != asset.path or lease.verified_sha256 != asset.sha256:
            raise DatasetSecurityError("el lease no corresponde al asset autorizado")
        if lease.identity.link_count != 1:
            raise DatasetSecurityError("hardlink no permitido en asset autorizado")
        key = (lease.identity.volume_serial, lease.identity.file_id)
        seen = self._observed_asset_paths.get(key)
        if seen is not None and seen != asset.path:
            raise DatasetSecurityError("dos paths autorizados comparten file id")
        self._observed_asset_paths[key] = asset.path

    # -- manifest mutation transition ------------------------------------
    def prepare_manifest_mutation(self) -> None:
        self._require_active()
        if self._mutation_started:
            raise DatasetSecurityError("la mutacion ya fue iniciada")
        fs = self._filesystem
        lock_ctx = fs.lease_manifest_mutation_lock(
            self._manifest_path, self._dataset_root
        )
        lock_ctx.__enter__()
        self._lock_ctx = lock_ctx
        # Cierra el lease read-only y adquiere el de update sosteniendo el lock.
        self._manifest_lease_ctx.__exit__(None, None, None)
        self._manifest_lease_ctx = None
        update_ctx = fs.lease_manifest_for_update(
            self._manifest_path, self._dataset_root
        )
        update_lease = update_ctx.__enter__()
        self._update_ctx = update_ctx
        self._update_lease = update_lease
        self._manifest_lease = update_lease
        payload = self._read_lease_bytes(update_lease)
        if hashlib.sha256(payload).hexdigest() != self._manifest_sha256:
            raise DatasetSecurityError("carrera durante la transicion de mutacion")
        self._mutation_started = True

    def replace_manifest(self, payload: bytes) -> CorpusFileIdentity:
        self._require_active()
        if not self._mutation_started:
            raise DatasetSecurityError("no hay transaccion de mutacion activa")
        if self._manifest_replaced:
            raise DatasetSecurityError("un segundo replace no esta permitido")
        identity = self._filesystem.atomic_replace_manifest(self._update_lease, payload)
        self._manifest_replaced = True
        return identity

    # -- journal ----------------------------------------------------------
    def reserve_or_resume_purge_journal(
        self, path: Path, initial_record: bytes
    ) -> AbstractContextManager[PrivateJournalLease]:
        self._require_active()
        if not _within(path, self._reports_root):
            raise RetentionError("el journal de purga debe vivir en reports/")
        return self._filesystem.reserve_or_resume_private_journal(
            path, self._reports_root, initial_record, max_bytes=MAX_CORPUS_MANIFEST_BYTES
        )

    def append_purge_journal(
        self, lease: PrivateJournalLease, record: bytes
    ) -> None:
        self._require_active()
        self._filesystem.append_private_journal(lease, record)

    def read_report_ref_key(self, path: Path) -> bytes:
        self._require_active()
        if not _within(path, self._secrets_root):
            raise RetentionError("la clave debe vivir en secrets/")
        with self._filesystem.lease_private_file(path, self._secrets_root) as lease:
            lease.stream.seek(0)
            data = lease.stream.read(65)
        if not (32 <= len(data) <= 64):
            raise RetentionError("clave de referencia con longitud invalida")
        return data

    def write_json_report(self, path: Path, payload: Mapping[str, object]) -> None:
        self._require_active()
        if not _within(path, self._reports_root):
            raise RetentionError("el reporte debe vivir en reports/")
        self._filesystem.atomic_write_private(
            path, self._reports_root, _canonical(dict(payload))
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "sufficient": self.sufficient,
            "encryption_provider": self.encryption_provider,
            "checked_at": self.checked_at,
        }


def audit_dataset_security(
    dataset_root: Path,
    repo_root: Path,
    manifest_path: Path,
    manifest: CorpusManifest,
    *,
    assets: tuple[CorpusAsset, ...] | None = None,
    filesystem: CorpusFilesystem | None = None,
) -> DatasetSecurityEvidence:
    root = _abspath(dataset_root)
    repo = _abspath(repo_root)
    mpath = _abspath(manifest_path)
    if _within(root, repo) or _within(mpath, repo):
        raise DatasetSecurityError("dataset y manifest deben vivir fuera de Git")
    if not _within(mpath, root):
        raise DatasetSecurityError("manifest debe vivir dentro del dataset_root")
    reports_root = root / "reports"
    secrets_root = root / "secrets"
    declared = {
        asset.path: asset.sha256
        for entry in manifest.entries
        for asset in entry.assets
    }
    for asset_path in declared:
        target = root.joinpath(*Path(asset_path).parts)
        if _within(target, reports_root) or _within(target, secrets_root):
            raise DatasetSecurityError("un asset no puede vivir en reports/ o secrets/")
    if _within(mpath, reports_root) or _within(mpath, secrets_root):
        raise DatasetSecurityError("el manifest no puede vivir en reports/ o secrets/")
    if assets is None:
        scope = dict(declared)
    else:
        scope = {}
        for asset in assets:
            if declared.get(asset.path) != asset.sha256:
                raise DatasetSecurityError("scope contiene un asset no declarado")
            if asset.path in scope:
                raise DatasetSecurityError("scope contiene un asset duplicado")
            scope[asset.path] = asset.sha256
    evidence = DatasetSecurityEvidence.__new__(DatasetSecurityEvidence)
    evidence._dataset_root = root
    evidence._repo_root = repo
    evidence._manifest_path = mpath
    evidence._manifest = manifest
    evidence._reports_root = reports_root
    evidence._secrets_root = secrets_root
    evidence._authorized_asset_digests = MappingProxyType(scope)
    evidence._observed_asset_paths = {}
    evidence._filesystem = filesystem or default_corpus_filesystem()
    evidence._active = False
    evidence._manifest_lease = None
    evidence._manifest_lease_ctx = None
    evidence._lock_ctx = None
    evidence._update_ctx = None
    evidence._update_lease = None
    evidence._mutation_started = False
    evidence._manifest_replaced = False
    evidence.sufficient = False
    evidence.encryption_provider = "unproven"
    evidence.checked_at = ""
    return evidence


def _require_evidence(security_evidence: DatasetSecurityEvidence) -> None:
    if not isinstance(security_evidence, DatasetSecurityEvidence):
        raise RetentionError("se requiere DatasetSecurityEvidence ligada")


def list_retention(
    manifest: CorpusManifest,
    *,
    security_evidence: DatasetSecurityEvidence,
    ref_key: bytes,
    today: Callable[[], date] = date.today,
) -> tuple[RetentionItem, ...]:
    _require_evidence(security_evidence)
    security_evidence.require_for(security_evidence._dataset_root, manifest)
    effective = today()
    return tuple(
        RetentionItem(
            clip_ref=protected_ref(ref_key, "clip", entry.clip_id),
            retention_until=entry.retention_until,
            expired=entry.retention_until < effective,
            condition=entry.condition,
        )
        for entry in manifest.entries
    )


def renew_retention(
    manifest_path: Path,
    dataset_root: Path,
    *,
    clip_ids: tuple[str, ...],
    until: date,
    confirm: bool,
    security_evidence: DatasetSecurityEvidence,
    filesystem: CorpusFilesystem | None = None,
) -> CorpusManifest:
    _require_evidence(security_evidence)
    if not confirm:
        raise ValueError("renew_retention exige confirm=True")
    manifest = security_evidence._manifest
    security_evidence.require_for(dataset_root, manifest)
    targets = set(clip_ids)
    known = {entry.clip_id for entry in manifest.entries}
    if not targets or not targets.issubset(known):
        raise ValueError("clip_ids desconocidos para renovacion")
    for entry in manifest.entries:
        if entry.clip_id in targets and until < entry.retention_until:
            raise ValueError("la retencion solo puede extenderse, no acortarse")
    new_entries = tuple(
        dataclasses.replace(entry, retention_until=until)
        if entry.clip_id in targets
        else entry
        for entry in manifest.entries
    )
    new_manifest = dataclasses.replace(manifest, entries=new_entries)
    security_evidence.prepare_manifest_mutation()
    security_evidence.replace_manifest(_manifest_bytes(new_manifest))
    return new_manifest


def _manifest_bytes(manifest: CorpusManifest) -> bytes:
    return json.dumps(manifest.to_dict(), ensure_ascii=True, allow_nan=False).encode(
        "utf-8"
    )


def purge_expired(
    manifest: CorpusManifest,
    dataset_root: Path,
    *,
    receipt_path: Path,
    confirm: bool = False,
    manifest_path: Path | None = None,
    security_evidence: DatasetSecurityEvidence,
    filesystem: CorpusFilesystem,
    ref_key: bytes,
    today: Callable[[], date] = date.today,
) -> PurgeReceipt:
    _require_evidence(security_evidence)
    if not _within(receipt_path, security_evidence._reports_root):
        raise RetentionError("receipt_path debe vivir en reports/")
    if confirm and manifest_path is None:
        raise ValueError("una purga confirmada exige manifest_path")
    security_evidence.require_for(dataset_root, manifest)

    effective = today()
    expired = tuple(
        entry for entry in manifest.entries if entry.retention_until < effective
    )
    operation_id = hashlib.sha256(
        f"{manifest.version}\0{effective.isoformat()}".encode("utf-8")
    ).hexdigest()[:16]
    operation_ref = protected_ref(ref_key, "operation", operation_id)
    dataset_ref = protected_ref(ref_key, "dataset", manifest.dataset_id)
    manifest_ref = protected_ref(ref_key, "manifest", manifest.version)
    planned_at = _now()

    def clip_ref(clip_id: str) -> str:
        return protected_ref(ref_key, "clip", clip_id)

    binding = _canonical({
        "schema": RECEIPT_SCHEMA,
        "operation_ref": operation_ref,
        "dataset_ref": dataset_ref,
        "manifest_ref": manifest_ref,
        "effective": effective.isoformat(),
    })
    plan_items = tuple(
        {"clip_ref": clip_ref(entry.clip_id), "role": asset.role, "state": "planned"}
        for entry in expired
        for asset in entry.assets
    )

    with security_evidence.reserve_or_resume_purge_journal(
        receipt_path, binding
    ) as journal:
        security_evidence.append_purge_journal(
            journal,
            _canonical({
                "kind": "planned",
                "operation_ref": operation_ref,
                "clips": [clip_ref(entry.clip_id) for entry in expired],
            }),
        )
        if not confirm:
            return PurgeReceipt(
                schema=RECEIPT_SCHEMA,
                operation_ref=operation_ref,
                dataset_ref=dataset_ref,
                manifest_ref=manifest_ref,
                planned_at=planned_at,
                finished_at=_now(),
                dry_run=True,
                status="planned",
                items=plan_items,
            )

        security_evidence.append_purge_journal(
            journal, _canonical({"kind": "started", "operation_ref": operation_ref})
        )
        security_evidence.prepare_manifest_mutation()

        item_states: dict[tuple[str, str], str] = {
            (clip_ref(entry.clip_id), asset.role): "failed"
            for entry in expired
            for asset in entry.assets
        }
        removed: list[str] = []
        for entry in expired:
            entry_ok = True
            for asset in entry.assets:
                key = (clip_ref(entry.clip_id), asset.role)
                security_evidence.append_purge_journal(
                    journal,
                    _canonical({
                        "kind": "intent",
                        "operation_ref": operation_ref,
                        "clip_ref": key[0],
                        "role": asset.role,
                    }),
                )
                try:
                    lease_cm = lease_corpus_asset(
                        asset, dataset_root, filesystem=filesystem
                    )
                    with lease_cm as lease:
                        security_evidence.require_for(dataset_root, manifest)
                        security_evidence.require_asset(asset, lease)
                        try:
                            filesystem.delete_leased(lease)
                        except (CorpusFilesystemError, OSError) as delete_error:
                            del delete_error
                            item_states[key] = "failed"
                            entry_ok = False
                            _append_outcome(
                                security_evidence, journal, operation_ref, key,
                                asset.role, "failed",
                            )
                            continue
                        item_states[key] = "deleted"
                        _append_outcome(
                            security_evidence, journal, operation_ref, key,
                            asset.role, "deleted",
                        )
                except CorpusFilesystemError as lease_error:
                    raise RetentionError(str(lease_error)) from lease_error
            if entry_ok:
                removed.append(entry.clip_id)

        if removed:
            security_evidence.require_for(dataset_root, manifest)
            remaining = tuple(
                entry for entry in manifest.entries if entry.clip_id not in removed
            )
            new_manifest = dataclasses.replace(manifest, entries=remaining)
            security_evidence.replace_manifest(_manifest_bytes(new_manifest))

        all_deleted = all(state == "deleted" for state in item_states.values())
        status = "complete" if all_deleted else "partial"
        return PurgeReceipt(
            schema=RECEIPT_SCHEMA,
            operation_ref=operation_ref,
            dataset_ref=dataset_ref,
            manifest_ref=manifest_ref,
            planned_at=planned_at,
            finished_at=_now(),
            dry_run=False,
            status=status,
            items=tuple(
                {"clip_ref": key[0], "role": key[1], "state": state}
                for key, state in item_states.items()
            ),
        )


def _append_outcome(
    evidence: DatasetSecurityEvidence,
    journal: PrivateJournalLease,
    operation_ref: str,
    key: tuple[str, str],
    role: str,
    result: str,
) -> None:
    evidence.append_purge_journal(
        journal,
        _canonical({
            "kind": "outcome",
            "operation_ref": operation_ref,
            "clip_ref": key[0],
            "role": role,
            "result": result,
        }),
    )


def initialize_report_ref_key(
    dataset_root: Path,
    repo_root: Path,
    output: Path,
    *,
    filesystem: CorpusFilesystem | None = None,
) -> None:
    root = _abspath(dataset_root)
    repo = _abspath(repo_root)
    if _within(root, repo):
        raise DatasetSecurityError("dataset debe vivir fuera de Git")
    secrets_root = root / "secrets"
    if not _within(output, secrets_root):
        raise DatasetSecurityError("la clave debe crearse dentro de secrets/")
    adapter = filesystem or default_corpus_filesystem()
    adapter.create_private_secret(output, secrets_root, size=32)
