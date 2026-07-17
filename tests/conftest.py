import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pytest

from speechtotext.evaluation.filesystem import FakeCorpusFilesystem
from speechtotext.evaluation.manifest import CorpusManifest, load_corpus_manifest


@dataclass
class _Corpus:
    repo: Path
    root: Path
    outside: Path
    manifest_path: Path
    manifest: CorpusManifest
    primary_sha256: str
    outside_asset: Path
    _asset_relpaths: tuple[str, ...]

    def all_declared_assets_exist(self) -> bool:
        return all(
            (self.root / Path(rel)).is_file() for rel in self._asset_relpaths
        )


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def fs_adapter() -> FakeCorpusFilesystem:
    return FakeCorpusFilesystem()


@pytest.fixture
def corpus(tmp_path, fs_adapter) -> _Corpus:
    repo = tmp_path / "repo"
    root = tmp_path / "private"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    (root / "clips").mkdir(parents=True)
    (root / "derived").mkdir(parents=True)
    (root / "backups").mkdir(parents=True)
    (root / "reports").mkdir(parents=True)
    (root / "secrets").mkdir(parents=True)

    primary_bytes = b"primary-audio-bytes"
    derived_bytes = b"{\"features\": [1, 2, 3]}"
    backup_bytes = b"encrypted-backup-bytes"
    (root / "clips" / "day1-001.wav").write_bytes(primary_bytes)
    (root / "derived" / "day1-001.features.json").write_bytes(derived_bytes)
    (root / "backups" / "day1-001.wav.enc").write_bytes(backup_bytes)

    outside_asset = outside / "leaked.wav"
    outside_asset.write_bytes(b"outside-audio")

    asset_relpaths = (
        "clips/day1-001.wav",
        "derived/day1-001.features.json",
        "backups/day1-001.wav.enc",
    )
    entry = {
        "clip_id": "day1-001",
        "assets": [
            {"role": "primary_audio", "path": asset_relpaths[0],
             "sha256": _digest(primary_bytes)},
            {"role": "derived", "path": asset_relpaths[1],
             "sha256": _digest(derived_bytes)},
            {"role": "backup", "path": asset_relpaths[2],
             "sha256": _digest(backup_bytes)},
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
    manifest_doc = {
        "schema_version": "speechtotext.corpus/v1",
        "dataset_id": "samuel-desktop-2026",
        "created_on": "2026-07-16",
        "entries": [entry],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_doc), encoding="utf-8")

    manifest = load_corpus_manifest(
        manifest_path, dataset_root=root, repo_root=repo, filesystem=fs_adapter
    )
    return _Corpus(
        repo=repo,
        root=root,
        outside=outside,
        manifest_path=manifest_path,
        manifest=manifest,
        primary_sha256=_digest(primary_bytes),
        outside_asset=outside_asset,
        _asset_relpaths=asset_relpaths,
    )
