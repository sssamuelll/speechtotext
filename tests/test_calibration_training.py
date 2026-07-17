from dataclasses import replace
from datetime import date, timedelta
import hashlib
import json

import pytest

from speechtotext.asr import TranscriptionRequest
from speechtotext.audio import PipelineProvenance, PipelineStep
from speechtotext.evaluation.training import (
    LabeledFeatureExample,
    LabeledFeaturePartition,
    fit_segment_usable_calibrator,
)
from speechtotext.evaluation.manifest import CorpusAsset, CorpusEntry, CorpusManifest
from speechtotext.evaluation.splits import DatasetSplit
from speechtotext.models import load_model_manifest, verify_model_files
from speechtotext.models.filesystem import FakeModelFilesystem


ASR_PIPELINE = PipelineProvenance.capture(
    sample_rate=16000,
    step=PipelineStep("test-source", "1", {}),
)
REQUEST = TranscriptionRequest(language="es")


class FakeBackend:
    backend_id = "fake"

    def __init__(self, model_artifact):
        self.model_artifact = model_artifact
        self.warm_calls = 0
        self.inference_calls = 0

    @property
    def model_id(self):
        return self.model_artifact.manifest.model_id

    @property
    def model_version(self):
        return self.model_artifact.manifest.revision

    @property
    def backend_artifact_kind(self):
        return "local_model_manifest"

    @property
    def backend_artifact_fingerprint(self):
        return self.model_artifact.fingerprint

    @property
    def config_fingerprint(self):
        return "1" * 64

    def warm(self):
        self.warm_calls += 1

    def transcribe(self, clip, request):
        self.inference_calls += 1
        raise AssertionError("training no debe ejecutar inferencia")


@pytest.fixture
def backend(tmp_path):
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(b"weights")
    data = {
        "schema_version": "speechtotext.model/v1",
        "model_id": "fake-model",
        "source": "https://example.invalid/fake-model",
        "revision_kind": "git_commit",
        "revision": "a" * 40,
        "license": "MIT",
        "format": "ctranslate2",
        "sample_rate": 16000,
        "preprocessing": {"mono": True},
        "files": [{
            "path": "model.bin",
            "sha256": hashlib.sha256(b"weights").hexdigest(),
        }],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    expected = hashlib.sha256(json.dumps(
        data, ensure_ascii=True, allow_nan=False,
        separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")).hexdigest()
    filesystem = FakeModelFilesystem(root_read_only=True)
    manifest = load_model_manifest(
        manifest_path, model_root=tmp_path,
        expected_fingerprint=expected, filesystem=filesystem,
    )
    with verify_model_files(
        manifest, tmp_path, filesystem=filesystem
    ) as artifact:
        yield FakeBackend(artifact)


def _entry(clip_id, *, day, session):
    recorded_on = date(2026, 7, 1) + timedelta(days=day)
    return CorpusEntry(
        clip_id=clip_id,
        assets=(CorpusAsset("primary_audio", f"clips/{clip_id}.wav", "0" * 64),),
        session_id=session,
        recorded_on=recorded_on,
        kind="speech",
        speaker="test-speaker",
        condition="clean",
        source_id="test-mic",
        duration_ms=1000,
        transcript="hola",
        speech_regions=(),
        intent=None,
        slots={},
        spoof_label="bona_fide",
        provenance="synthetic-test",
        consent_or_license="test-only",
        retention_until=date(2027, 1, 1),
    )


def _partitions(values=(-2.0, -1.0, 1.0, 2.0)):
    labels = (0, 0, 1, 1)
    development_entries = tuple(
        _entry(f"development-{index}", day=0, session="fit-session")
        for index in range(4)
    )
    calibration_entries = tuple(
        _entry(f"calibration-{index}", day=10, session="cal-session")
        for index in range(4)
    )
    holdout_entries = (_entry("holdout-0", day=20, session="holdout-session"),)
    manifest = CorpusManifest(
        "speechtotext.corpus/v1",
        "training-test",
        date(2026, 7, 1),
        development_entries + calibration_entries + holdout_entries,
    )
    split = DatasetSplit.create(
        manifest,
        development_entries,
        calibration_entries,
        holdout_entries,
        20260716,
    )

    def build(name, entries):
        examples = tuple(
            LabeledFeatureExample(
                clip_id=entry.clip_id,
                session_id=entry.session_id,
                recorded_on=entry.recorded_on,
                features={"x": value},
                label=label,
            )
            for entry, value, label in zip(entries, values, labels, strict=True)
        )
        return LabeledFeaturePartition.from_split(
            name, manifest, split, examples
        )

    return (
        build("development", development_entries),
        build("calibration", calibration_entries),
        manifest,
        split,
    )


def test_fit_exporta_coeficientes_puros_y_ordena_probabilidades(backend):
    development, calibration, _, _ = _partitions()
    artifact = fit_segment_usable_calibrator(
        development,
        calibration,
        backend=backend,
        pipeline=ASR_PIPELINE,
        request=REQUEST,
        usable_max_wer=0.10,
        min_precision_lower_95=0.20,
        artifact_version="fw-small-segment-usable-1",
    )
    assert artifact.feature_names == ("x",)
    assert artifact.feature_means == pytest.approx((0.0,))
    assert artifact.feature_scales[0] > 0.0
    assert len(artifact.coefficients) == 1
    assert 0.0 <= artifact.operating_threshold <= 1.0
    assert artifact.selection_correct == artifact.selection_accepted
    assert artifact.selection_accepted > 0
    assert artifact.selection_total == 4
    assert artifact.precision_lower_95 >= 0.20
    assert artifact.fit_split_fingerprint == development.fingerprint
    assert artifact.calibration_split_fingerprint == calibration.fingerprint
    assert artifact.pipeline_fingerprint == ASR_PIPELINE.fingerprint
    assert artifact.request_fingerprint == REQUEST.fingerprint
    assert artifact.backend_artifact_kind == "local_model_manifest"
    assert (
        artifact.backend_artifact_fingerprint
        == backend.model_artifact.fingerprint
    )
    assert artifact.backend_config_fingerprint == backend.config_fingerprint
    assert backend.warm_calls == backend.inference_calls == 0


def test_fit_es_determinista(backend):
    kwargs = dict(
        backend=backend,
        pipeline=ASR_PIPELINE,
        request=REQUEST,
        usable_max_wer=0.10,
        min_precision_lower_95=0.20,
        artifact_version="fw-small-segment-usable-1",
    )
    development, calibration, _, _ = _partitions()
    first = fit_segment_usable_calibrator(development, calibration, **kwargs)
    second = fit_segment_usable_calibrator(development, calibration, **kwargs)
    assert first == second
    assert first.version == second.version
    assert backend.warm_calls == backend.inference_calls == 0


def test_particion_rechaza_duplicados_metadata_adulterada_y_holdout(backend):
    development, _, manifest, split = _partitions()
    duplicated = tuple(development.examples[0] for _ in development.examples)
    with pytest.raises(ValueError, match="una sola vez"):
        LabeledFeaturePartition.from_split(
            "development", manifest, split, duplicated
        )
    altered = (replace(development.examples[0], session_id="other"),) + tuple(
        development.examples[1:]
    )
    with pytest.raises(ValueError, match="metadata"):
        LabeledFeaturePartition.from_split(
            "development", manifest, split, altered
        )
    with pytest.raises(ValueError, match="development|calibration"):
        LabeledFeaturePartition.from_split(
            "holdout", manifest, split, development.examples
        )
    with pytest.raises(TypeError):
        LabeledFeaturePartition("development", development.examples)
    assert backend.warm_calls == backend.inference_calls == 0
