from dataclasses import replace
from datetime import date, timedelta

import pytest

from speechtotext.evaluation.manifest import (
    CorpusAsset,
    CorpusEntry,
    CorpusManifest,
)
from speechtotext.evaluation.splits import split_by_recording_day


def _entry(index: int, day: int, suffix: str) -> CorpusEntry:
    recorded = date(2026, 7, 1) + timedelta(days=day)
    return CorpusEntry(
        clip_id=f"clip-{index}-{suffix}",
        assets=(
            CorpusAsset(
                "primary_audio",
                f"clips/{index}-{suffix}.wav",
                f"{index:064x}",
            ),
        ),
        session_id=f"session-{day}-{suffix}",
        recorded_on=recorded,
        kind="speech",
        speaker="Samuel",
        condition="clean",
        source_id="mic",
        duration_ms=1000,
        transcript="hola",
        speech_regions=(),
        intent=None,
        slots={},
        spoof_label="bona_fide",
        provenance="owner",
        consent_or_license="owner-consent",
        retention_until=recorded + timedelta(days=180),
    )


def _manifest(entries) -> CorpusManifest:
    return CorpusManifest(
        "speechtotext.corpus/v1",
        "dataset",
        date(2026, 7, 1),
        tuple(entries),
    )


def test_split_no_mezcla_dias_y_es_determinista():
    entries = tuple(
        _entry(day, day, suffix)
        for day in range(9)
        for suffix in ("a", "b")
    )
    manifest = _manifest(entries)
    first = split_by_recording_day(manifest)
    second = split_by_recording_day(manifest)
    assert first == second
    partitions = tuple(
        first.partition(name, manifest)
        for name in ("development", "calibration", "holdout")
    )
    day_sets = [{entry.recorded_on for entry in part} for part in partitions]
    assert day_sets[0].isdisjoint(day_sets[1])
    assert day_sets[0].isdisjoint(day_sets[2])
    assert day_sets[1].isdisjoint(day_sets[2])
    assert all(partitions)
    assert all(len(days) >= 3 for days in day_sets)
    assert len(first.fingerprint) == 64


def test_split_fingerprint_cambia_si_cambia_dataset():
    entries = tuple(_entry(day, day, "a") for day in range(9))
    first_manifest = _manifest(entries)
    first = split_by_recording_day(first_manifest)
    changed = replace(entries[0], transcript="contenido distinto con el mismo id")
    second_manifest = _manifest((changed, *entries[1:]))
    second = split_by_recording_day(second_manifest)
    assert first.fingerprint != second.fingerprint
    with pytest.raises(ValueError, match="manifest no coincide"):
        first.partition("development", second_manifest)


def test_split_exige_nueve_dias():
    entries = tuple(_entry(day, day, "a") for day in range(8))
    with pytest.raises(ValueError, match="nueve fechas"):
        split_by_recording_day(_manifest(entries))


def test_split_rechaza_una_sesion_que_cruza_fechas():
    entries = tuple(
        replace(_entry(day, day, "a"), session_id="shared" if day < 2 else f"s-{day}")
        for day in range(9)
    )
    with pytest.raises(ValueError, match="sesion abarca mas de una fecha"):
        split_by_recording_day(_manifest(entries))
