import dataclasses
import io
import json
from datetime import date

import pytest

from speechtotext.evaluation.filesystem import (
    CorpusAssetLease,
    CorpusFileIdentity,
    CorpusFilesystemError,
)
from speechtotext.evaluation.manifest import (
    MAX_CORPUS_MANIFEST_BYTES,
    CorpusAsset,
    load_corpus_manifest,
)
from speechtotext.evaluation.retention import (
    DatasetSecurityError,
    RetentionError,
    audit_dataset_security,
    initialize_report_ref_key,
    purge_expired,
    renew_retention,
)


RETENTION_REF_KEY = b"retention-reference-key-tests!!!"


def reload_manifest(corpus, fs_adapter):
    return load_corpus_manifest(
        corpus.manifest_path,
        dataset_root=corpus.root,
        repo_root=corpus.repo,
        filesystem=fs_adapter,
    )


def security_evidence(corpus, fs_adapter):
    fs_adapter.configure_security(acl_ok=True, encryption_ok=True)
    manifest = reload_manifest(corpus, fs_adapter)
    return audit_dataset_security(
        corpus.root,
        corpus.repo,
        corpus.manifest_path,
        manifest,
        filesystem=fs_adapter,
    )


def test_purge_es_dry_run_por_default_y_solo_enumera_expirados(
    corpus, fs_adapter
):
    with security_evidence(corpus, fs_adapter) as evidence:
        receipt = purge_expired(
            corpus.manifest,
            corpus.root,
            receipt_path=corpus.root / "reports" / "purge-plan.journal.jsonl",
            today=lambda: date(2027, 1, 13),
            filesystem=fs_adapter,
            security_evidence=evidence,
            ref_key=RETENTION_REF_KEY,
        )
    assert receipt.dry_run is True
    assert receipt.status == "planned"
    assert corpus.all_declared_assets_exist()
    assert fs_adapter.journal_events[:2] == ["reserved", "planned_flushed"]


def test_purge_confirmado_nunca_toca_un_archivo_no_declarado(corpus, fs_adapter):
    undeclared = corpus.root / "clips" / "keep.wav"
    undeclared.write_bytes(b"keep")
    with security_evidence(corpus, fs_adapter) as evidence:
        receipt = purge_expired(
            corpus.manifest,
            corpus.root,
            today=lambda: date(2027, 1, 13),
            confirm=True,
            manifest_path=corpus.manifest_path,
            receipt_path=corpus.root / "reports" / "purge-complete.json",
            filesystem=fs_adapter,
            security_evidence=evidence,
            ref_key=RETENTION_REF_KEY,
        )
    assert receipt.status == "complete"
    assert undeclared.read_bytes() == b"keep"


def test_renew_extiende_atomico_y_no_acepta_acortar(corpus, fs_adapter):
    with security_evidence(corpus, fs_adapter) as evidence:
        renewed = renew_retention(
            corpus.manifest_path,
            corpus.root,
            clip_ids=("day1-001",),
            until=date(2027, 7, 12),
            confirm=True,
            filesystem=fs_adapter,
            security_evidence=evidence,
        )
    assert renewed.entries[0].retention_until == date(2027, 7, 12)
    with security_evidence(corpus, fs_adapter) as evidence:
        with pytest.raises(ValueError, match="solo puede extender"):
            renew_retention(
                corpus.manifest_path,
                corpus.root,
                clip_ids=("day1-001",),
                until=date(2027, 1, 1),
                confirm=True,
                filesystem=fs_adapter,
                security_evidence=evidence,
            )


def test_purge_rechaza_symlink_o_reparse_antes_de_borrar(corpus, fs_adapter):
    fs_adapter.replace_parent_with_reparse_point("clips", corpus.outside)
    with pytest.raises((DatasetSecurityError, RetentionError), match="reparse"):
        with security_evidence(corpus, fs_adapter) as evidence:
            purge_expired(
                corpus.manifest,
                corpus.root,
                today=lambda: date(2027, 1, 13),
                confirm=True,
                manifest_path=corpus.manifest_path,
                receipt_path=corpus.root / "reports" / "purge-reparse.json",
                filesystem=fs_adapter,
                security_evidence=evidence,
                ref_key=RETENTION_REF_KEY,
            )
    assert corpus.outside_asset.exists()


def test_fallo_parcial_deja_entrada_y_emite_recibo_solo_metadata(corpus, fs_adapter):
    fs_adapter.fail_delete_for("derived/day1-001.features.json")
    with security_evidence(corpus, fs_adapter) as evidence:
        receipt = purge_expired(
            corpus.manifest,
            corpus.root,
            today=lambda: date(2027, 1, 13),
            confirm=True,
            manifest_path=corpus.manifest_path,
            receipt_path=corpus.root / "reports" / "purge-partial.json",
            filesystem=fs_adapter,
            security_evidence=evidence,
            ref_key=RETENTION_REF_KEY,
        )
    payload = receipt.to_dict()
    assert receipt.status == "partial"
    assert "day1-001" in {
        entry.clip_id for entry in reload_manifest(corpus, fs_adapter).entries
    }
    serialized = json.dumps(payload)
    assert "hey Jarvis" not in serialized
    assert corpus.primary_sha256 not in serialized
    assert "day1-001" not in serialized


@pytest.mark.parametrize("acl_ok,encryption_ok", [(False, True), (True, False)])
def test_evidencia_de_seguridad_faltante_bloquea_corpus(
    corpus, fs_adapter, acl_ok, encryption_ok
):
    fs_adapter.configure_security(acl_ok=acl_ok, encryption_ok=encryption_ok)
    with audit_dataset_security(
        corpus.root,
        corpus.repo,
        corpus.manifest_path,
        corpus.manifest,
        filesystem=fs_adapter,
    ) as evidence:
        assert evidence.sufficient is False
        with pytest.raises(DatasetSecurityError, match="evidencia insuficiente"):
            evidence.require_for(corpus.root, corpus.manifest)


# -- require_for anti-TOCTOU (finding 2) ---------------------------------


def test_require_for_detecta_root_distinto(corpus, fs_adapter):
    with security_evidence(corpus, fs_adapter) as evidence:
        with pytest.raises(DatasetSecurityError, match="root distinto"):
            evidence.require_for(corpus.outside, corpus.manifest)


def test_require_for_detecta_identidad_de_root_cambiada(corpus, fs_adapter):
    with security_evidence(corpus, fs_adapter) as evidence:
        fs_adapter.force_identity_change(corpus.root)
        with pytest.raises(DatasetSecurityError, match="identidad del root cambio"):
            evidence.require_for(corpus.root, corpus.manifest)


def test_require_for_detecta_manifest_cambiado_bajo_lease(corpus, fs_adapter):
    with security_evidence(corpus, fs_adapter) as evidence:
        # El handle read-only conserva los bytes; alterarlos simula un swap que
        # la revalidacion SHA debe detectar antes de operar.
        evidence._manifest_lease.stream.seek(0)
        evidence._manifest_lease.stream.write(b"XXXX")
        with pytest.raises(DatasetSecurityError, match="cambio bajo el lease"):
            evidence.require_for(corpus.root, corpus.manifest)


def test_require_for_rechaza_manifest_construido_a_mano(corpus, fs_adapter):
    with security_evidence(corpus, fs_adapter) as evidence:
        forged = dataclasses.replace(corpus.manifest, dataset_id="otro-dataset")
        with pytest.raises(DatasetSecurityError, match="no coincide con la evidencia"):
            evidence.require_for(corpus.root, forged)


# -- require_asset JIT authorization (finding 1) -------------------------


def _asset_lease(asset: CorpusAsset, identity: CorpusFileIdentity) -> CorpusAssetLease:
    return CorpusAssetLease(
        asset=asset,
        stream=io.BytesIO(b""),
        identity=identity,
        verified_sha256=asset.sha256,
    )


def test_require_asset_rechaza_asset_fuera_de_scope(corpus, fs_adapter):
    with security_evidence(corpus, fs_adapter) as evidence:
        ghost = CorpusAsset("primary_audio", "clips/ghost.wav", "0" * 64)
        identity = CorpusFileIdentity(1, b"\x01" * 16, 5, 10, link_count=1)
        with pytest.raises(DatasetSecurityError, match="fuera del scope autorizado"):
            evidence.require_asset(ghost, _asset_lease(ghost, identity))


def test_require_asset_rechaza_hardlink_en_el_primer_lease_jit(corpus, fs_adapter):
    with security_evidence(corpus, fs_adapter) as evidence:
        primary = corpus.manifest.entries[0].primary_audio
        linked = CorpusFileIdentity(1, b"\x02" * 16, 5, 10, link_count=2)
        with pytest.raises(DatasetSecurityError, match="hardlink no permitido"):
            evidence.require_asset(primary, _asset_lease(primary, linked))


def test_require_asset_rechaza_dos_paths_con_el_mismo_file_id(corpus, fs_adapter):
    with security_evidence(corpus, fs_adapter) as evidence:
        entry = corpus.manifest.entries[0]
        primary = entry.primary_audio
        derived = next(a for a in entry.assets if a.role == "derived")
        shared = CorpusFileIdentity(7, b"\x09" * 16, 5, 10, link_count=1)
        evidence.require_asset(primary, _asset_lease(primary, shared))
        with pytest.raises(DatasetSecurityError, match="comparten file id"):
            evidence.require_asset(derived, _asset_lease(derived, shared))


# -- manifest mutation transition (finding 3) ----------------------------


def test_manifest_mutation_replay_readonly_bloquea_transicion(corpus, fs_adapter):
    with security_evidence(corpus, fs_adapter) as replay:
        replay.require_for(corpus.root, corpus.manifest)
        with security_evidence(corpus, fs_adapter) as mutator:
            with pytest.raises(CorpusFilesystemError, match="bloquea la transicion"):
                mutator.prepare_manifest_mutation()


def test_manifest_mutation_transiciona_a_lease_replaceable(corpus, fs_adapter):
    with security_evidence(corpus, fs_adapter) as evidence:
        evidence.require_for(corpus.root, corpus.manifest)
        evidence.prepare_manifest_mutation()
        # Un segundo lock exclusivo falla mientras la mutacion esta activa.
        with pytest.raises(CorpusFilesystemError, match="lock de mutacion"):
            with fs_adapter.lease_manifest_mutation_lock(
                corpus.manifest_path, corpus.root
            ):
                pass
        identity = evidence.replace_manifest(b'{"schema_version": "x"}')
        assert identity is not None
        # Ni un segundo replace ni require_for son validos tras el commit.
        with pytest.raises(DatasetSecurityError, match="segundo replace"):
            evidence.replace_manifest(b'{"schema_version": "x"}')
        with pytest.raises(DatasetSecurityError, match="consumida"):
            evidence.require_for(corpus.root, corpus.manifest)


# -- durable journal barrier (finding 4) ---------------------------------


def test_journal_es_barrera_durable_y_recuperable(corpus, fs_adapter):
    receipt_path = corpus.root / "reports" / "purge.journal.jsonl"
    with security_evidence(corpus, fs_adapter) as evidence:
        receipt = purge_expired(
            corpus.manifest,
            corpus.root,
            today=lambda: date(2027, 1, 13),
            confirm=True,
            manifest_path=corpus.manifest_path,
            receipt_path=receipt_path,
            filesystem=fs_adapter,
            security_evidence=evidence,
            ref_key=RETENTION_REF_KEY,
        )
    assert receipt.status == "complete"
    events = fs_adapter.journal_events
    assert events[:3] == ["reserved", "planned_flushed", "started_flushed"]
    # Barrera: cada intent durable precede su delete, y el delete su outcome.
    assert events[3:] == ["intent_flushed", "delete", "outcome_flushed"] * 3
    # Durabilidad: los records quedaron en disco antes de cada side effect.
    on_disk = receipt_path.read_bytes()
    assert on_disk.count(b'"kind":"intent"') == 3
    assert on_disk.count(b'"kind":"outcome"') == 3
    # Recuperable: una reanudacion relee los intents persistidos.
    with fs_adapter.reserve_or_resume_private_journal(
        receipt_path,
        corpus.root / "reports",
        b"resume",
        max_bytes=MAX_CORPUS_MANIFEST_BYTES,
    ) as resumed:
        assert resumed.existing_payload.count(b'"kind":"intent"') == 3


# -- read_report_bytes (finding 5) ---------------------------------------


def test_read_report_bytes_usa_un_solo_lease_y_limite(corpus, fs_adapter):
    report = corpus.root / "reports" / "r.json"
    report.write_bytes(b'{"ok": true}')
    with security_evidence(corpus, fs_adapter) as evidence:
        assert evidence.read_report_bytes(report, max_bytes=64) == b'{"ok": true}'
        with pytest.raises(RetentionError, match="excede max_bytes"):
            evidence.read_report_bytes(report, max_bytes=4)
        outside = corpus.root / "secrets" / "leak.json"
        outside.write_bytes(b"{}")
        with pytest.raises(RetentionError, match="reports/"):
            evidence.read_report_bytes(outside, max_bytes=64)


# -- initialize_report_ref_key (finding 7) -------------------------------


def test_initialize_report_ref_key_create_new(corpus, fs_adapter):
    fuera = corpus.root / "reports" / "k.key"
    with pytest.raises(DatasetSecurityError, match="secrets/"):
        initialize_report_ref_key(
            corpus.root, corpus.repo, fuera, filesystem=fs_adapter
        )
    key = corpus.root / "secrets" / "ref.key"
    initialize_report_ref_key(corpus.root, corpus.repo, key, filesystem=fs_adapter)
    assert key.exists()
    with pytest.raises(CorpusFilesystemError, match="secreto existente"):
        initialize_report_ref_key(
            corpus.root, corpus.repo, key, filesystem=fs_adapter
        )
