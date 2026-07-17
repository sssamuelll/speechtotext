from datetime import date
import hashlib
import json

import pytest

from speechtotext.evaluation.manifest import (
    MAX_CORPUS_MANIFEST_BYTES,
    load_corpus_manifest,
    parse_corpus_manifest_bytes,
)


def _entry(digest: str):
    return {
        "clip_id": "day1-001",
        "assets": [
            {
                "role": "primary_audio",
                "path": "clips/day1-001.wav",
                "sha256": digest,
            },
            {
                "role": "derived",
                "path": "derived/day1-001.features.json",
                "sha256": "1" * 64,
            },
            {
                "role": "backup",
                "path": "backups/day1-001.wav.enc",
                "sha256": "2" * 64,
            },
        ],
        "session_id": "session-day1",
        "recorded_on": "2026-07-16",
        "kind": "speech",
        "speaker": "Samuel",
        "condition": "clean_near",
        "source_id": "desktop-mic",
        "duration_ms": 1500,
        "transcript": "hey Jarvis abre el proyecto",
        "speech_regions": [{"start_s": 0.10, "end_s": 1.40}],
        "intent": "open_project",
        "slots": {"project": "aurelius"},
        "spoof_label": "bona_fide",
        "provenance": "recorded_by_owner",
        "consent_or_license": "owner-consent-2026-07-16",
        "retention_until": "2027-01-12",
    }


def _manifest(entry):
    return {
        "schema_version": "speechtotext.corpus/v1",
        "dataset_id": "samuel-desktop-2026",
        "created_on": "2026-07-16",
        "entries": [entry],
    }


def _write_manifest(path, entry):
    path.write_text(
        json.dumps(_manifest(entry)),
        encoding="utf-8",
    )


def test_manifest_privado_carga_inventario_desde_lease(tmp_path, fs_adapter):
    repo = tmp_path / "repo"
    dataset = tmp_path / "private"
    repo.mkdir()
    (dataset / "clips").mkdir(parents=True)
    audio = dataset / "clips" / "day1-001.wav"
    audio.write_bytes(b"audio")
    digest = hashlib.sha256(b"audio").hexdigest()
    manifest_path = dataset / "manifest.json"
    _write_manifest(manifest_path, _entry(digest))
    manifest = load_corpus_manifest(
        manifest_path,
        dataset_root=dataset,
        repo_root=repo,
        filesystem=fs_adapter,
    )
    assert manifest.entries[0].recorded_on == date(2026, 7, 16)
    assert manifest.entries[0].retention_until == date(2027, 1, 12)
    assert manifest.entries[0].primary_audio.path == "clips/day1-001.wav"


def test_manifest_rechaza_dataset_dentro_del_repo(tmp_path, fs_adapter):
    repo = tmp_path / "repo"
    dataset = repo / "corpus"
    dataset.mkdir(parents=True)
    path = dataset / "manifest.json"
    _write_manifest(path, _entry("0" * 64))
    with pytest.raises(ValueError, match="fuera de Git"):
        load_corpus_manifest(
            path, dataset_root=dataset, repo_root=repo, filesystem=fs_adapter
        )


def test_manifest_rechaza_path_traversal(tmp_path, fs_adapter):
    repo = tmp_path / "repo"
    dataset = tmp_path / "private"
    repo.mkdir()
    dataset.mkdir()
    entry = _entry("0" * 64)
    entry["assets"][0]["path"] = "../outside.wav"
    path = dataset / "manifest.json"
    _write_manifest(path, entry)
    with pytest.raises(ValueError, match="ruta relativa segura"):
        load_corpus_manifest(
            path, dataset_root=dataset, repo_root=repo, filesystem=fs_adapter
        )


def test_manifest_rechaza_payload_sobredimensionado_y_claves_duplicadas(tmp_path):
    with pytest.raises(ValueError, match="limite"):
        parse_corpus_manifest_bytes(
            b" " * (MAX_CORPUS_MANIFEST_BYTES + 1), dataset_root=tmp_path
        )
    payload = json.dumps(_manifest(_entry("0" * 64)))
    payload = payload.replace(
        '"dataset_id": "samuel-desktop-2026"',
        '"dataset_id": "first", "dataset_id": "second"',
    )
    with pytest.raises(ValueError, match="duplicada"):
        parse_corpus_manifest_bytes(payload.encode("utf-8"), dataset_root=tmp_path)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("duration_ms", True),
        ("clip_id", 7),
        ("recorded_on", 20260716),
        ("asset_path", 7),
        ("region_start", True),
        ("slot_value", 7),
        ("dataset_id", 7),
    ],
)
def test_manifest_rechaza_coerciones_de_tipos(tmp_path, field, bad):
    entry = _entry("0" * 64)
    payload = _manifest(entry)
    if field == "asset_path":
        entry["assets"][0]["path"] = bad
    elif field == "region_start":
        entry["speech_regions"][0]["start_s"] = bad
    elif field == "slot_value":
        entry["slots"]["project"] = bad
    elif field == "dataset_id":
        payload["dataset_id"] = bad
    else:
        entry[field] = bad
    with pytest.raises(ValueError, match="tipo|invalido"):
        parse_corpus_manifest_bytes(
            json.dumps(payload).encode("utf-8"), dataset_root=tmp_path
        )


@pytest.mark.parametrize(
    "unsafe",
    [
        r"clips\day1.wav",
        "clips/day1.wav:zone.identifier",
        "clips/CON.wav",
        "clips/com1.txt",
        "clips/CONIN$.wav",
        "clips/day1.wav.",
        "clips/day1.wav ",
        "clips//day1.wav",
        "clips/./day1.wav",
        "clips/day\x001.wav",
        r"\\?\C:\outside.wav",
    ],
)
def test_manifest_rechaza_paths_ambiguos_en_windows(tmp_path, unsafe):
    entry = _entry("0" * 64)
    entry["assets"][0]["path"] = unsafe
    with pytest.raises(ValueError, match="ruta relativa segura"):
        parse_corpus_manifest_bytes(
            json.dumps(_manifest(entry)).encode("utf-8"), dataset_root=tmp_path
        )


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_manifest_rechaza_json_no_finito(tmp_path, token):
    payload = json.dumps(_manifest(_entry("0" * 64)))
    payload = payload.replace('"duration_ms": 1500', f'"duration_ms": {token}')
    with pytest.raises(ValueError, match="no finita"):
        parse_corpus_manifest_bytes(payload.encode("utf-8"), dataset_root=tmp_path)


def test_manifest_exige_un_solo_primario_y_assets_unicos(tmp_path, fs_adapter):
    repo = tmp_path / "repo"
    dataset = tmp_path / "private"
    repo.mkdir()
    dataset.mkdir()
    entry = _entry("0" * 64)
    entry["assets"].append(dict(entry["assets"][0]))
    path = dataset / "manifest.json"
    _write_manifest(path, entry)
    with pytest.raises(ValueError, match="primary_audio"):
        load_corpus_manifest(
            path, dataset_root=dataset, repo_root=repo, filesystem=fs_adapter
        )
