import json
from datetime import date

import pytest

from speechtotext.evaluation.manifest import load_corpus_manifest
from speechtotext.evaluation.retention import (
    DatasetSecurityError,
    RetentionError,
    audit_dataset_security,
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
