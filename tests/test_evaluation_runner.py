# tests/test_evaluation_runner.py
from datetime import date, timedelta
import hashlib
import json

import numpy as np
import pytest

from speechtotext.asr import NativeSignals, TranscriptionRequest, TranscriptionResult
from speechtotext.audio import (
    AudioClip,
    AudioQualityReport,
    AudioView,
    AudioViews,
    PipelineProvenance,
    PipelineStep,
    QualityThresholds,
)
from speechtotext.evaluation.manifest import CorpusAsset, CorpusEntry, CorpusManifest
from speechtotext.evaluation.runner import (
    collect_labeled_feature_partition,
    EvaluationConfig,
    run_evaluation,
)
from speechtotext.evaluation.retention import audit_dataset_security
from speechtotext.evaluation.splits import DatasetSplit
from speechtotext.models import load_model_manifest, verify_model_files
from speechtotext.models.filesystem import FakeModelFilesystem
from speechtotext.models.manifest import ModelIntegrityError

REPORT_REF_KEY = b"report-reference-key-for-tests!!"
TEST_ENVIRONMENT = {
    "schema_version": "speechtotext.environment/v1",
    "git_ref": "git-revision:" + "a" * 32,
    "python": "3.11.9",
    "implementation": "CPython",
    "platform": "Windows-11-test",
    "machine": "AMD64",
    "processor": "test-cpu",
    "executable_name": "python.exe",
    "memory": {"rss": 100, "peak_rss": 200},
    "packages": {
        "speechtotext": "0.1.0",
        "av": "18.0.0",
        "ctranslate2": "4.5.0",
        "faster-whisper": "1.2.1",
        "huggingface-hub": "0.30.0",
        "numpy": "2.4.6",
        "onnxruntime": "1.20.0",
        "scikit-learn": "1.9.0",
        "scipy": "1.17.1",
        "tokenizers": "0.21.0",
    },
}
TEST_PIPELINE = PipelineProvenance.capture(
    sample_rate=16000,
    step=PipelineStep("fake-load", "1", {}),
)


@pytest.fixture
def security_factory(request):
    def create(tmp_path, manifest, fs_adapter, *, assets=None):
        fs_adapter.configure_security(acl_ok=True, encryption_ok=True)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
        repo = tmp_path / "repo"
        repo.mkdir(exist_ok=True)
        (tmp_path / "reports").mkdir(exist_ok=True)
        context = audit_dataset_security(
            tmp_path,
            repo,
            manifest_path,
            manifest,
            assets=assets,
            filesystem=fs_adapter,
        )
        evidence = context.__enter__()
        request.addfinalizer(lambda: context.__exit__(None, None, None))
        return evidence

    return create


def _entry(tmp_path, clip_id, kind, transcript, day, *, condition=None):
    rel = f"clips/{clip_id}.wav"
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(clip_id.encode())
    recorded = date(2026, 7, day)
    return CorpusEntry(
        clip_id=clip_id,
        assets=(
            CorpusAsset(
                "primary_audio",
                rel,
                hashlib.sha256(clip_id.encode()).hexdigest(),
            ),
        ),
        session_id=f"session-{day}",
        recorded_on=recorded,
        kind=kind,
        speaker="Samuel" if kind == "speech" else "none",
        condition=condition or ("clean" if kind == "speech" else "silence"),
        source_id="desktop-mic",
        duration_ms=1000,
        transcript=transcript,
        speech_regions=(),
        intent=None,
        slots={},
        spoof_label="bona_fide" if kind == "speech" else "unknown",
        provenance="owner",
        consent_or_license="owner-consent",
        retention_until=recorded + timedelta(days=180),
    )


def _loader(
    entry,
    lease,
    expected_pipeline,
    sample_rate,
    gain_db,
):
    view = AudioView.capture(
        np.zeros(sample_rate, dtype=np.float32),
        sample_rate,
        step=PipelineStep("fake-load", "1", {}),
    )
    assert expected_pipeline == view.provenance
    voice_ms = 500 if entry.kind == "speech" else 0
    quality = AudioQualityReport(
        1000,
        voice_ms,
        -25.0 if voice_ms else -120.0,
        -25.0 if voice_ms else -120.0,
        -6.0 if voice_ms else -120.0,
        0.0,
        -45.0,
        20.0 if voice_ms else None,
        0.0,
        0.0,
        0,
        0,
        (),
    )
    return AudioClip(
        0.0,
        1.0,
        entry.source_id,
        (),
        quality,
        AudioViews(view, view, view),
    )


def _verified_model(
    tmp_path, request, *, model_id="fake-model", revision="a" * 40
):
    root = tmp_path / f"model-{model_id}"
    root.mkdir(exist_ok=True)
    model_path = root / f"{model_id}.bin"
    payload = model_id.encode("utf-8")
    model_path.write_bytes(payload)
    data = {
        "schema_version": "speechtotext.model/v1",
        "model_id": model_id,
        "source": "https://example.invalid/model",
        "revision_kind": "git_commit",
        "revision": revision,
        "license": "MIT",
        "format": "ctranslate2",
        "sample_rate": 16000,
        "preprocessing": {"mono": True},
        "files": [{
            "path": model_path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    expected = hashlib.sha256(json.dumps(
        data, ensure_ascii=True, allow_nan=False,
        separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")).hexdigest()
    filesystem = FakeModelFilesystem(root_read_only=True)
    manifest = load_model_manifest(
        manifest_path, model_root=root,
        expected_fingerprint=expected, filesystem=filesystem,
    )
    context = verify_model_files(manifest, root, filesystem=filesystem)
    artifact = context.__enter__()
    request.addfinalizer(lambda: context.__exit__(None, None, None))
    return artifact


class FakeBackend:
    backend_id = "fake"

    def __init__(self, model_artifact):
        self.model_artifact = model_artifact
        self.warmed = 0
        self.calls = []

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
        self.warmed += 1

    def transcribe(self, clip, request):
        self.calls.append((clip.source_id, request.language))
        return TranscriptionResult(
            "hola mundo",
            "es",
            (),
            (),
            "fake",
            self.model_id,
            self.model_version,
            120,
            NativeSignals(0.01, -0.1, 1.0, 0.99),
            "segment_usable",
            None,
            None,
            (),
        )


def test_runner_bloquea_silencio_y_escribe_reporte_atomico(
    tmp_path, request, fs_adapter, security_factory
):
    speech = _entry(tmp_path, "speech", "speech", "hola mundo", 1)
    silence = _entry(tmp_path, "silence", "silence", "", 1)
    manifest = CorpusManifest(
        "speechtotext.corpus/v1",
        "dataset",
        date(2026, 7, 1),
        (speech, silence),
    )
    split = DatasetSplit.create(manifest, (speech, silence), (), (), 20260716)
    backend = FakeBackend(_verified_model(tmp_path, request))
    output = tmp_path / "reports" / "dev.json"
    report = run_evaluation(
        manifest=manifest,
        split=split,
        partition="development",
        dataset_root=tmp_path,
        backend=backend,
        request=TranscriptionRequest(language="es"),
        config=EvaluationConfig(as_of=date(2026, 7, 16)),
        thresholds=QualityThresholds(160, -45.0, 6.0, 0.01),
        pipeline=TEST_PIPELINE,
        output_path=output,
        calibrator=None,
        clip_loader=_loader,
        asset_lease_factory=fs_adapter.lease_asset,
        security_evidence=security_factory(tmp_path, manifest, fs_adapter),
        environment=TEST_ENVIRONMENT,
        memory_probe=lambda: {"rss": 100, "peak_rss": 200},
        report_ref_key=REPORT_REF_KEY,
        today=lambda: date(2026, 7, 16),
    )
    assert backend.warmed == 1
    assert len(backend.calls) == 1
    assert report["counts"] == {
        "clips": 2,
        "eligible": 1,
        "blocked_before_asr": 1,
        "transcribed": 1,
    }
    assert report["asr"]["wer_overall"]["sufficient_evidence"] is False
    assert report["asr"]["word_errors"] == 0
    assert report["asr"]["reference_words"] == 2
    assert report["asr"]["wer"] == 0.0
    assert report["safety"]["silence_transcripts"]["errors"] == 0
    assert report["safety"]["silence_transcripts"]["trials"] == 1
    assert report["safety"]["silence_transcripts"]["upper_95"] is None
    assert report["safety"]["silence_transcripts"]["sufficient_evidence"] is False
    assert report["engine_latency_ms"]["p95"] == 120.0
    assert report["engine_latency_ms"]["acceptance_gate"] is False
    assert report["corpus"]["sufficient_evidence"] is False
    assert set(report["corpus"]["insufficient_reason_codes"]) == {
        "duration_below_30m",
        "recording_days_below_3",
        "recording_sessions_below_3",
        "required_conditions_missing",
    }
    assert report["acceptance_gate"]["status"] == "insufficient_evidence"
    assert report["acceptance_gate"]["passed"] is False
    assert report["memory_bytes"]["max_peak_rss_observed"] == 200
    serialized = output.read_text(encoding="utf-8")
    assert json.loads(serialized) == report
    assert "hola mundo" not in serialized
    assert str(tmp_path.resolve()) not in serialized
    assert speech.audio_sha256 not in serialized
    assert "model.bin" not in serialized
    assert report["backend"]["model_ref"].startswith("model:")
    assert not output.with_suffix(".json.tmp").exists()


def test_tres_bloques_perfectos_pero_diminutos_no_aprueban(
    tmp_path, request, fs_adapter, security_factory
):
    entries = (
        _entry(tmp_path, "clean-1", "speech", "hola mundo", 1),
        _entry(tmp_path, "noise-2", "speech", "hola mundo", 2, condition="noise"),
        _entry(tmp_path, "clean-3", "speech", "hola mundo", 3),
        _entry(tmp_path, "silence-3", "silence", "", 3),
    )
    manifest = CorpusManifest(
        "speechtotext.corpus/v1", "dataset", date(2026, 7, 1), entries
    )
    report = run_evaluation(
        manifest=manifest,
        split=DatasetSplit.create(manifest, entries, (), (), 20260716),
        partition="development",
        dataset_root=tmp_path,
        backend=FakeBackend(_verified_model(tmp_path, request)),
        request=TranscriptionRequest(language="es"),
        config=EvaluationConfig(as_of=date(2026, 7, 16)),
        thresholds=QualityThresholds(160, -45.0, 6.0, 0.01),
        pipeline=TEST_PIPELINE,
        output_path=tmp_path / "reports" / "tiny.json",
        calibrator=None,
        clip_loader=_loader,
        asset_lease_factory=fs_adapter.lease_asset,
        security_evidence=security_factory(tmp_path, manifest, fs_adapter),
        environment=TEST_ENVIRONMENT,
        memory_probe=lambda: {"rss": 100, "peak_rss": 200},
        report_ref_key=REPORT_REF_KEY,
        today=lambda: date(2026, 7, 16),
    )
    assert report["engine_latency_ms"]["blocks"] == 3
    assert report["engine_latency_ms"]["p95"] == 120.0
    assert report["engine_latency_ms"]["acceptance_gate"] is False
    assert report["corpus"]["insufficient_reason_codes"] == ["duration_below_30m"]
    assert report["acceptance_gate"]["status"] == "insufficient_evidence"


def test_collector_liga_split_y_nunca_abre_holdout(
    tmp_path, request, fs_adapter, security_factory
):
    development = _entry(tmp_path, "development", "speech", "hola mundo", 1)
    calibration = _entry(tmp_path, "calibration", "speech", "hola mundo", 2)
    holdout = _entry(tmp_path, "holdout", "speech", "secreto", 3)
    manifest = CorpusManifest(
        "speechtotext.corpus/v1",
        "dataset",
        date(2026, 7, 1),
        (development, calibration, holdout),
    )
    split = DatasetSplit.create(
        manifest, (development,), (calibration,), (holdout,), 20260716
    )
    backend = FakeBackend(_verified_model(tmp_path, request))
    evidence = security_factory(
        tmp_path,
        manifest,
        fs_adapter,
        assets=development.assets,
    )
    partition = collect_labeled_feature_partition(
        manifest=manifest,
        split=split,
        partition="development",
        dataset_root=tmp_path,
        backend=backend,
        request=TranscriptionRequest(language="es"),
        config=EvaluationConfig(as_of=date(2026, 7, 16)),
        thresholds=QualityThresholds(160, -45.0, 6.0, 0.01),
        pipeline=TEST_PIPELINE,
        security_evidence=evidence,
        clip_loader=_loader,
        asset_lease_factory=fs_adapter.lease_asset,
        today=lambda: date(2026, 7, 16),
    )
    assert partition.partition_ids == ("development",)
    assert partition.manifest_fingerprint == manifest.version
    assert partition.split_fingerprint == split.fingerprint
    assert [call[0] for call in backend.calls] == [development.source_id]

    untouched = FakeBackend(
        _verified_model(tmp_path, request, model_id="other")
    )
    with pytest.raises(ValueError, match="holdout"):
        collect_labeled_feature_partition(
            manifest=manifest,
            split=split,
            partition="holdout",
            dataset_root=tmp_path,
            backend=untouched,
            request=TranscriptionRequest(language="es"),
            config=EvaluationConfig(as_of=date(2026, 7, 16)),
            thresholds=QualityThresholds(160, -45.0, 6.0, 0.01),
            pipeline=TEST_PIPELINE,
            security_evidence=evidence,
            today=lambda: date(2026, 7, 16),
        )
    assert untouched.warmed == 0 and untouched.calls == []


@pytest.mark.parametrize(
    "mutate",
    [
        "backend",
        "model",
        "version",
        "pipeline",
        "request",
    ],
)
def test_calibrador_incompatible_falla_antes_de_warm_y_lease(
    tmp_path, request, fs_adapter, security_factory, mutate
):
    speech = _entry(tmp_path, "speech", "speech", "hola mundo", 1)
    manifest = CorpusManifest(
        "speechtotext.corpus/v1", "dataset", date(2026, 7, 1), (speech,)
    )
    split = DatasetSplit.create(manifest, (speech,), (), (), 20260716)
    backend = FakeBackend(_verified_model(tmp_path, request))
    leases = []

    def counting_lease(rel, root):
        leases.append(rel)
        return fs_adapter.lease_asset(rel, root)

    calibrator = _IncompatibleCalibrator(mutate)
    with pytest.raises(ValueError):
        run_evaluation(
            manifest=manifest,
            split=split,
            partition="development",
            dataset_root=tmp_path,
            backend=backend,
            request=TranscriptionRequest(language="es"),
            config=EvaluationConfig(as_of=date(2026, 7, 16)),
            thresholds=QualityThresholds(160, -45.0, 6.0, 0.01),
            pipeline=TEST_PIPELINE,
            output_path=tmp_path / "reports" / "x.json",
            calibrator=calibrator,
            clip_loader=_loader,
            asset_lease_factory=counting_lease,
            security_evidence=security_factory(tmp_path, manifest, fs_adapter),
            environment=TEST_ENVIRONMENT,
            memory_probe=lambda: {"rss": 100, "peak_rss": 200},
            report_ref_key=REPORT_REF_KEY,
            today=lambda: date(2026, 7, 16),
        )
    assert backend.warmed == 0
    assert backend.calls == []
    assert leases == []


class _IncompatibleCalibrator:
    def __init__(self, mutate):
        self.mutate = mutate

    def validate_for(
        self, *, backend, pipeline, request, expected_language, usable_max_wer
    ):
        raise ValueError(f"calibrator incompatible: {self.mutate}")


def test_duration_manifest_no_es_evidencia(tmp_path, monkeypatch):
    # Real PyAV decode is banned in tests (F1 constraint); monkeypatch the
    # module-level decode so the REAL load_entry_clip tolerance logic runs on a
    # synthetic one-second capture.
    import io
    from types import SimpleNamespace

    from speechtotext.audio.level import apply_fixed_gain
    from speechtotext.evaluation import runner as runner_mod

    samples = np.full(16000, 0.01, dtype=np.float32)
    capture = AudioView.capture(
        samples,
        16000,
        step=PipelineStep(
            "pyav-decode",
            "1",
            {"layout": "mono", "dtype": "float32", "sample_rate": 16000},
        ),
    )
    monkeypatch.setattr(
        runner_mod, "decode_audio", lambda stream, *, sample_rate: capture
    )
    gain = apply_fixed_gain(capture.samples, 0.0)
    analysis = AudioView.derive(
        capture,
        gain.samples,
        steps=(
            PipelineStep(
                "fixed-gain",
                "1",
                {"gain_db": 0.0, "max_gain_db": 18.0, "peak_limit_dbfs": -1.0},
            ),
        ),
    )
    expected_pipeline = analysis.provenance
    lease = SimpleNamespace(stream=io.BytesIO(b"ignored"))

    # Declara 30 minutos sobre un asset decodificado de un segundo -> tolerancia.
    over = _entry(tmp_path, "over", "speech", "hola", 1)
    object.__setattr__(over, "duration_ms", 1_800_000)
    with pytest.raises(ValueError, match="tolerancia"):
        runner_mod.load_entry_clip(over, lease, expected_pipeline, 16000, 0.0)

    # Dentro de max(50 ms, 1 %): usa exactamente el decoded_duration_ms.
    close = _entry(tmp_path, "close", "speech", "hola", 1)
    object.__setattr__(close, "duration_ms", 1005)
    clip = runner_mod.load_entry_clip(close, lease, expected_pipeline, 16000, 0.0)
    assert clip.quality.duration_ms == 1000


def test_split_ligado_al_manifest_falla_antes_de_warm_y_lease(
    tmp_path, request, fs_adapter, security_factory
):
    speech = _entry(tmp_path, "speech", "speech", "hola mundo", 1)
    manifest = CorpusManifest(
        "speechtotext.corpus/v1", "dataset", date(2026, 7, 1), (speech,)
    )
    split = DatasetSplit.create(manifest, (speech,), (), (), 20260716)
    backend = FakeBackend(_verified_model(tmp_path, request))
    leases = []

    def counting_lease(rel, root):
        leases.append(rel)
        return fs_adapter.lease_asset(rel, root)

    mutated = _entry(tmp_path, "speech", "speech", "otra cosa", 1)
    other_manifest = CorpusManifest(
        "speechtotext.corpus/v1", "dataset", date(2026, 7, 1), (mutated,)
    )
    with pytest.raises(ValueError):
        run_evaluation(
            manifest=other_manifest,
            split=split,
            partition="development",
            dataset_root=tmp_path,
            backend=backend,
            request=TranscriptionRequest(language="es"),
            config=EvaluationConfig(as_of=date(2026, 7, 16)),
            thresholds=QualityThresholds(160, -45.0, 6.0, 0.01),
            pipeline=TEST_PIPELINE,
            output_path=tmp_path / "reports" / "x.json",
            calibrator=None,
            clip_loader=_loader,
            asset_lease_factory=counting_lease,
            security_evidence=security_factory(
                tmp_path, other_manifest, fs_adapter
            ),
            environment=TEST_ENVIRONMENT,
            memory_probe=lambda: {"rss": 100, "peak_rss": 200},
            report_ref_key=REPORT_REF_KEY,
            today=lambda: date(2026, 7, 16),
        )
    assert backend.warmed == 0
    assert leases == []


def test_modelo_asr_cambia_model_ref_sin_cambiar_pipeline_ref(
    tmp_path, request, fs_adapter, security_factory
):
    speech = _entry(tmp_path, "speech", "speech", "hola mundo", 1)
    manifest = CorpusManifest(
        "speechtotext.corpus/v1", "dataset", date(2026, 7, 1), (speech,)
    )
    split = DatasetSplit.create(manifest, (speech,), (), (), 20260716)

    def _run(model_id, out):
        return run_evaluation(
            manifest=manifest,
            split=split,
            partition="development",
            dataset_root=tmp_path,
            backend=FakeBackend(
                _verified_model(tmp_path, request, model_id=model_id)
            ),
            request=TranscriptionRequest(language="es"),
            config=EvaluationConfig(as_of=date(2026, 7, 16)),
            thresholds=QualityThresholds(160, -45.0, 6.0, 0.01),
            pipeline=TEST_PIPELINE,
            output_path=tmp_path / "reports" / out,
            calibrator=None,
            clip_loader=_loader,
            asset_lease_factory=fs_adapter.lease_asset,
            security_evidence=security_factory(tmp_path, manifest, fs_adapter),
            environment=TEST_ENVIRONMENT,
            memory_probe=lambda: {"rss": 100, "peak_rss": 200},
            report_ref_key=REPORT_REF_KEY,
            today=lambda: date(2026, 7, 16),
        )

    first = _run("model-a", "a.json")
    second = _run("model-b", "b.json")
    assert first["pipeline_ref"] == second["pipeline_ref"]
    assert first["backend"]["model_ref"] != second["backend"]["model_ref"]


def test_retencion_expirada_falla_antes_de_warm_y_lease(
    tmp_path, request, fs_adapter, security_factory
):
    speech = _entry(tmp_path, "speech", "speech", "hola mundo", 1)
    object.__setattr__(speech, "retention_until", date(2026, 7, 10))
    manifest = CorpusManifest(
        "speechtotext.corpus/v1", "dataset", date(2026, 7, 1), (speech,)
    )
    split = DatasetSplit.create(manifest, (speech,), (), (), 20260716)
    backend = FakeBackend(_verified_model(tmp_path, request))
    leases = []

    def counting_lease(rel, root):
        leases.append(rel)
        return fs_adapter.lease_asset(rel, root)

    output = tmp_path / "reports" / "x.json"
    with pytest.raises(ValueError, match="corpus_retention_expired"):
        run_evaluation(
            manifest=manifest,
            split=split,
            partition="development",
            dataset_root=tmp_path,
            backend=backend,
            request=TranscriptionRequest(language="es"),
            config=EvaluationConfig(as_of=date(2026, 7, 5)),
            thresholds=QualityThresholds(160, -45.0, 6.0, 0.01),
            pipeline=TEST_PIPELINE,
            output_path=output,
            calibrator=None,
            clip_loader=_loader,
            asset_lease_factory=counting_lease,
            security_evidence=security_factory(tmp_path, manifest, fs_adapter),
            environment=TEST_ENVIRONMENT,
            memory_probe=lambda: {"rss": 100, "peak_rss": 200},
            report_ref_key=REPORT_REF_KEY,
            today=lambda: date(2026, 7, 15),
        )
    assert backend.warmed == 0
    assert backend.calls == []
    assert leases == []
    assert not output.exists()


def test_job_ref_es_estable_y_sensible_a_inputs(
    tmp_path, request, fs_adapter, security_factory
):
    speech = _entry(tmp_path, "speech", "speech", "hola mundo", 1)
    manifest = CorpusManifest(
        "speechtotext.corpus/v1", "dataset", date(2026, 7, 1), (speech,)
    )
    split = DatasetSplit.create(manifest, (speech,), (), (), 20260716)

    def _run(*, model_id="fake-model", thresholds=None, as_of=date(2026, 7, 16), out):
        return run_evaluation(
            manifest=manifest,
            split=split,
            partition="development",
            dataset_root=tmp_path,
            backend=FakeBackend(
                _verified_model(tmp_path, request, model_id=model_id)
            ),
            request=TranscriptionRequest(language="es"),
            config=EvaluationConfig(as_of=as_of),
            thresholds=thresholds or QualityThresholds(160, -45.0, 6.0, 0.01),
            pipeline=TEST_PIPELINE,
            output_path=tmp_path / "reports" / out,
            calibrator=None,
            clip_loader=_loader,
            asset_lease_factory=fs_adapter.lease_asset,
            security_evidence=security_factory(tmp_path, manifest, fs_adapter),
            environment=TEST_ENVIRONMENT,
            memory_probe=lambda: {"rss": 100, "peak_rss": 200},
            report_ref_key=REPORT_REF_KEY,
            today=lambda: date(2026, 7, 16),
        )

    base = _run(out="base.json")
    same = _run(out="same.json")
    assert base["job_ref"] == same["job_ref"]

    other_model = _run(model_id="other", out="model.json")
    assert other_model["job_ref"] != base["job_ref"]

    other_threshold = _run(
        thresholds=QualityThresholds(200, -45.0, 6.0, 0.01), out="thr.json"
    )
    assert other_threshold["job_ref"] != base["job_ref"]

    other_as_of = _run(as_of=date(2026, 7, 17), out="asof.json")
    assert other_as_of["job_ref"] != base["job_ref"]


def test_config_rechaza_gain_no_finito_o_sobre_18_antes_de_warm():
    with pytest.raises(ValueError):
        EvaluationConfig(as_of=date(2026, 7, 16), gain_db=float("nan"))
    with pytest.raises(ValueError):
        EvaluationConfig(as_of=date(2026, 7, 16), gain_db=18.5)


def test_resultado_con_identidad_distinta_al_backend_falla_aunque_no_haya_calibrador(
    tmp_path, request, fs_adapter, security_factory
):
    for mutate in ("backend", "model", "version"):
        speech = _entry(tmp_path, "speech", "speech", "hola mundo", 1)
        manifest = CorpusManifest(
            "speechtotext.corpus/v1", "dataset", date(2026, 7, 1), (speech,)
        )
        split = DatasetSplit.create(manifest, (speech,), (), (), 20260716)
        backend = _MislabeledBackend(
            _verified_model(tmp_path, request, model_id=f"m-{mutate}"), mutate
        )
        with pytest.raises(ValueError, match="incompatible"):
            run_evaluation(
                manifest=manifest,
                split=split,
                partition="development",
                dataset_root=tmp_path,
                backend=backend,
                request=TranscriptionRequest(language="es"),
                config=EvaluationConfig(as_of=date(2026, 7, 16)),
                thresholds=QualityThresholds(160, -45.0, 6.0, 0.01),
                pipeline=TEST_PIPELINE,
                output_path=tmp_path / "reports" / f"{mutate}.json",
                calibrator=None,
                clip_loader=_loader,
                asset_lease_factory=fs_adapter.lease_asset,
                security_evidence=security_factory(tmp_path, manifest, fs_adapter),
                environment=TEST_ENVIRONMENT,
                memory_probe=lambda: {"rss": 100, "peak_rss": 200},
                report_ref_key=REPORT_REF_KEY,
                today=lambda: date(2026, 7, 16),
            )


class _MislabeledBackend(FakeBackend):
    def __init__(self, model_artifact, mutate):
        super().__init__(model_artifact)
        self._mutate = mutate

    def transcribe(self, clip, request):
        self.calls.append((clip.source_id, request.language))
        backend = "wrong" if self._mutate == "backend" else self.backend_id
        model = "wrong" if self._mutate == "model" else self.model_id
        version = "wrong" if self._mutate == "version" else self.model_version
        return TranscriptionResult(
            "hola mundo",
            "es",
            (),
            (),
            backend,
            model,
            version,
            120,
            NativeSignals(0.01, -0.1, 1.0, 0.99),
            "segment_usable",
            None,
            None,
            (),
        )


@pytest.mark.parametrize(
    "field, value",
    [
        ("python", "3.11/9"),
        ("platform", "C:\\evil"),
        ("processor", "a" * 64),
        ("implementation", "has:colon"),
    ],
)
def test_environment_malicioso_falla_antes_de_warm(
    tmp_path, request, fs_adapter, security_factory, field, value
):
    speech = _entry(tmp_path, "speech", "speech", "hola mundo", 1)
    manifest = CorpusManifest(
        "speechtotext.corpus/v1", "dataset", date(2026, 7, 1), (speech,)
    )
    split = DatasetSplit.create(manifest, (speech,), (), (), 20260716)
    backend = FakeBackend(_verified_model(tmp_path, request))
    hostile = dict(TEST_ENVIRONMENT)
    hostile[field] = value
    output = tmp_path / "reports" / "x.json"
    with pytest.raises(ValueError):
        run_evaluation(
            manifest=manifest,
            split=split,
            partition="development",
            dataset_root=tmp_path,
            backend=backend,
            request=TranscriptionRequest(language="es"),
            config=EvaluationConfig(as_of=date(2026, 7, 16)),
            thresholds=QualityThresholds(160, -45.0, 6.0, 0.01),
            pipeline=TEST_PIPELINE,
            output_path=output,
            calibrator=None,
            clip_loader=_loader,
            asset_lease_factory=fs_adapter.lease_asset,
            security_evidence=security_factory(tmp_path, manifest, fs_adapter),
            environment=hostile,
            memory_probe=lambda: {"rss": 100, "peak_rss": 200},
            report_ref_key=REPORT_REF_KEY,
            today=lambda: date(2026, 7, 16),
        )
    assert backend.warmed == 0
    assert not output.exists()


def test_model_artifact_cerrado_falla_antes_de_warm_y_audio_lease(
    tmp_path, request, fs_adapter, security_factory
):
    speech = _entry(tmp_path, "speech", "speech", "hola mundo", 1)
    manifest = CorpusManifest(
        "speechtotext.corpus/v1", "dataset", date(2026, 7, 1), (speech,)
    )
    split = DatasetSplit.create(manifest, (speech,), (), (), 20260716)

    root = tmp_path / "model-closed"
    root.mkdir(exist_ok=True)
    model_path = root / "closed.bin"
    payload = b"closed"
    model_path.write_bytes(payload)
    data = {
        "schema_version": "speechtotext.model/v1",
        "model_id": "closed",
        "source": "https://example.invalid/model",
        "revision_kind": "git_commit",
        "revision": "a" * 40,
        "license": "MIT",
        "format": "ctranslate2",
        "sample_rate": 16000,
        "preprocessing": {"mono": True},
        "files": [{
            "path": model_path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    expected = hashlib.sha256(json.dumps(
        data, ensure_ascii=True, allow_nan=False,
        separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")).hexdigest()
    filesystem = FakeModelFilesystem(root_read_only=True)
    model_manifest = load_model_manifest(
        manifest_path, model_root=root,
        expected_fingerprint=expected, filesystem=filesystem,
    )
    context = verify_model_files(model_manifest, root, filesystem=filesystem)
    artifact = context.__enter__()
    context.__exit__(None, None, None)
    backend = FakeBackend(artifact)
    leases = []

    def counting_lease(rel, root):
        leases.append(rel)
        return fs_adapter.lease_asset(rel, root)

    output = tmp_path / "reports" / "closed.json"
    with pytest.raises(ModelIntegrityError):
        run_evaluation(
            manifest=manifest,
            split=split,
            partition="development",
            dataset_root=tmp_path,
            backend=backend,
            request=TranscriptionRequest(language="es"),
            config=EvaluationConfig(as_of=date(2026, 7, 16)),
            thresholds=QualityThresholds(160, -45.0, 6.0, 0.01),
            pipeline=TEST_PIPELINE,
            output_path=output,
            calibrator=None,
            clip_loader=_loader,
            asset_lease_factory=counting_lease,
            security_evidence=security_factory(tmp_path, manifest, fs_adapter),
            environment=TEST_ENVIRONMENT,
            memory_probe=lambda: {"rss": 100, "peak_rss": 200},
            report_ref_key=REPORT_REF_KEY,
            today=lambda: date(2026, 7, 16),
        )
    assert backend.warmed == 0
    assert leases == []
    assert not output.exists()

    with pytest.raises(ModelIntegrityError):
        collect_labeled_feature_partition(
            manifest=manifest,
            split=split,
            partition="development",
            dataset_root=tmp_path,
            backend=backend,
            request=TranscriptionRequest(language="es"),
            config=EvaluationConfig(as_of=date(2026, 7, 16)),
            thresholds=QualityThresholds(160, -45.0, 6.0, 0.01),
            pipeline=TEST_PIPELINE,
            security_evidence=security_factory(tmp_path, manifest, fs_adapter),
            clip_loader=_loader,
            asset_lease_factory=counting_lease,
            today=lambda: date(2026, 7, 16),
        )
    assert leases == []
