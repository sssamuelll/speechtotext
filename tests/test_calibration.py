from dataclasses import replace
import hashlib
import json

import pytest

from speechtotext.asr import NativeSignals, TranscriptionResult
from speechtotext.confidence.calibration import (
    CalibratorArtifact,
    LogisticCalibrator,
    parse_calibrator_artifact_bytes,
    select_operating_threshold,
    serialize_calibrator_artifact,
)
from speechtotext.confidence import ASR_FEATURE_NAMES
from speechtotext.confidence.runtime import CalibratingLocalAsrBackend
from speechtotext.asr.types import TranscriptionRequest
from speechtotext.audio import AudioClip, AudioQualityReport, AudioView, AudioViews
from speechtotext.audio.fingerprint import PipelineProvenance, PipelineStep
from speechtotext.models import load_model_manifest, verify_model_files
from speechtotext.models.filesystem import FakeModelFilesystem
from speechtotext.statistics import one_sided_success_lower


FIT_SPLIT_FINGERPRINT = "f" * 64
CALIBRATION_SPLIT_FINGERPRINT = "c" * 64
BACKEND_ARTIFACT_FINGERPRINT = "d" * 64
BACKEND_CONFIG_FINGERPRINT = "e" * 64

PIPELINE = PipelineProvenance.capture(
    sample_rate=16000, step=PipelineStep("test-source", "1", {})
)
REQUEST = TranscriptionRequest(language="es")


def _provider_descriptor_fingerprint(response_mapper_version="openai-audio/v1"):
    descriptor = {
        "schema_version": "speechtotext.provider-model-descriptor/v1",
        "provider": "openai",
        "model": "gpt-4o-transcribe",
        "identity_kind": "time_bounded_alias",
        "promotion_policy_version": "cloud-promotion/v1",
        "response_mapper_version": response_mapper_version,
        "alias": "gpt-4o-transcribe",
        "observed_at": "2026-07-16T00:00:00Z",
        "not_after": "2026-07-23T00:00:00Z",
    }
    payload = json.dumps(
        descriptor,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _artifact():
    return CalibratorArtifact(
        schema_version="speechtotext.calibrator/v1",
        artifact_version="segment-usable-fw-small-1",
        target="segment_usable",
        usable_max_wer=0.10,
        expected_language="es",
        feature_names=("x",),
        feature_means=(0.0,),
        feature_scales=(1.0,),
        coefficients=(1.0,),
        intercept=0.0,
        operating_threshold=0.8,
        selection_correct=1,
        selection_accepted=1,
        selection_total=1,
        precision_lower_95=one_sided_success_lower(1, 1),
        backend="faster-whisper",
        model="small",
        model_version="a" * 40,
        backend_artifact_kind="local_model_manifest",
        backend_artifact_fingerprint=BACKEND_ARTIFACT_FINGERPRINT,
        backend_config_fingerprint=BACKEND_CONFIG_FINGERPRINT,
        fit_split_fingerprint=FIT_SPLIT_FINGERPRINT,
        calibration_split_fingerprint=CALIBRATION_SPLIT_FINGERPRINT,
        pipeline_fingerprint=PIPELINE.fingerprint,
        request_fingerprint=REQUEST.fingerprint,
    )


def _result():
    return TranscriptionResult(
        "hola", "es", (), (), "faster-whisper", "small", "a" * 40, 10,
        NativeSignals(None, None, None, None),
        "segment_usable", None, None, (),
    )


def _view():
    return AudioView.capture(
        [0.0], 16000, step=PipelineStep("test-source", "1", {})
    )


def _clip():
    view = AudioView.capture(
        [0.0] * 16_000,
        16000,
        step=PipelineStep("test-source", "1", {}),
    )
    quality = AudioQualityReport(
        1000, 800, -25.0, -25.0, -1.0, 0.0, -45.0, 20.0,
        0.0, 0.0, 0, 0, (),
    )
    return AudioClip(0.0, 1.0, "test", (), quality, AudioViews(view, view, view))


class _Backend:
    backend_id = "faster-whisper"

    def __init__(self, model_artifact, config_fingerprint=BACKEND_CONFIG_FINGERPRINT):
        self.model_artifact = model_artifact
        self._config_fingerprint = config_fingerprint

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
        return self._config_fingerprint

    def warm(self):
        return None

    def transcribe(self, clip, request):
        raise AssertionError("este backend de identidad no ejecuta inferencia")


class _RemoteBackend:
    backend_id = "openai"
    model_id = "gpt-4o-transcribe"
    model_version = "gpt-4o-transcribe"
    backend_artifact_kind = "provider_model_descriptor"
    config_fingerprint = "8" * 64

    def __init__(self, response_mapper_version="openai-audio/v1"):
        self.backend_artifact_fingerprint = _provider_descriptor_fingerprint(
            response_mapper_version
        )

    def warm(self):
        return None

    def transcribe(self, clip, request):
        raise AssertionError("este backend de identidad no ejecuta transporte")


class _ReturningBackend(_Backend):
    def __init__(self, model_artifact):
        super().__init__(model_artifact)
        self.warm_calls = 0
        self.transcribe_calls = 0

    def warm(self):
        self.warm_calls += 1

    def transcribe(self, clip, request):
        self.transcribe_calls += 1
        return _result()


@pytest.fixture
def backend(tmp_path, request):
    payload = b"calibration-test-model"
    (tmp_path / "model.bin").write_bytes(payload)
    data = {
        "schema_version": "speechtotext.model/v1",
        "model_id": "small",
        "source": "https://example.invalid/small",
        "revision_kind": "git_commit",
        "revision": "a" * 40,
        "license": "MIT",
        "format": "ctranslate2",
        "sample_rate": 16000,
        "preprocessing": {"mono": True},
        "files": [{
            "path": "model.bin",
            "sha256": hashlib.sha256(payload).hexdigest(),
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
    context = verify_model_files(manifest, tmp_path, filesystem=filesystem)
    artifact = context.__enter__()
    request.addfinalizer(lambda: context.__exit__(None, None, None))
    return _Backend(artifact)


def _artifact_for(backend):
    return replace(
        _artifact(),
        backend_artifact_fingerprint=backend.backend_artifact_fingerprint,
        backend_config_fingerprint=backend.config_fingerprint,
    )


def _full_feature_artifact_for(backend):
    size = len(ASR_FEATURE_NAMES)
    return replace(
        _artifact_for(backend),
        feature_names=ASR_FEATURE_NAMES,
        feature_means=(0.0,) * size,
        feature_scales=(1.0,) * size,
        coefficients=(0.0,) * size,
        intercept=0.0,
    )


def test_logistic_predict_y_apply_guardan_version(backend):
    artifact = _artifact_for(backend)
    calibrator = LogisticCalibrator(artifact)
    assert calibrator.predict_proba({"x": 0.0}) == pytest.approx(0.5)
    calibrated = calibrator.apply(
        _result(),
        {"x": 2.0},
        backend=backend,
        view=_view(),
        request=REQUEST,
        expected_language="es",
        usable_max_wer=0.10,
    )
    assert calibrated.calibrated_confidence == pytest.approx(0.880797, rel=1e-5)
    assert calibrated.calibrator_version == artifact.version


def test_calibrator_rechaza_features_incompatibles():
    with pytest.raises(ValueError, match="features incompatibles"):
        LogisticCalibrator(_artifact()).predict_proba({"y": 1.0})


def test_calibrator_prevalida_identidad_planeada_sin_backend_activo():
    artifact = _artifact()
    calibrator = LogisticCalibrator(artifact)
    identity = {
        "backend": artifact.backend,
        "model": artifact.model,
        "model_version": artifact.model_version,
        "backend_artifact_kind": artifact.backend_artifact_kind,
        "backend_artifact_fingerprint": artifact.backend_artifact_fingerprint,
        "backend_config_fingerprint": artifact.backend_config_fingerprint,
    }
    calibrator.validate_binding_for_identity(
        **identity,
        pipeline=PIPELINE,
        request=REQUEST,
        expected_language="es",
        usable_max_wer=0.10,
    )
    identity["backend_artifact_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="artefacto/config"):
        calibrator.validate_binding_for_identity(
            **identity,
            pipeline=PIPELINE,
            request=REQUEST,
            expected_language="es",
            usable_max_wer=0.10,
        )


def test_calibrator_rechaza_pipeline_o_request_incompatible(backend):
    calibrator = LogisticCalibrator(_artifact_for(backend))
    with pytest.raises(ValueError, match="pipeline/request"):
        calibrator.apply(
            _result(),
            {"x": 1.0},
            backend=backend,
            view=AudioView.capture(
                [0.0], 16000, step=PipelineStep("different", "1", {})
            ),
            request=REQUEST,
            expected_language="es",
            usable_max_wer=0.10,
        )


@pytest.mark.parametrize(
    "expected_language,usable_max_wer",
    [("en", 0.10), ("es", 0.20)],
)
def test_calibrator_rechaza_semantica_de_label_incompatible(
    backend, expected_language, usable_max_wer
):
    calibrator = LogisticCalibrator(_artifact_for(backend))
    with pytest.raises(ValueError, match="lenguaje/usable_max_wer"):
        calibrator.validate_for(
            backend=backend,
            pipeline=PIPELINE,
            request=REQUEST,
            expected_language=expected_language,
            usable_max_wer=usable_max_wer,
        )


def test_calibrator_rechaza_config_backend_distinta(backend):
    calibrator = LogisticCalibrator(_artifact_for(backend))
    changed = _Backend(backend.model_artifact, "b" * 64)
    with pytest.raises(ValueError, match="artefacto/config"):
        calibrator.validate_for(
            backend=changed,
            pipeline=PIPELINE,
            request=REQUEST,
            expected_language="es",
            usable_max_wer=0.10,
        )


def test_calibrator_liga_provider_descriptor_sin_asumir_alias_inmutable():
    backend = _RemoteBackend()
    artifact = replace(
        _artifact(),
        backend=backend.backend_id,
        model=backend.model_id,
        model_version=backend.model_version,
        backend_artifact_kind=backend.backend_artifact_kind,
        backend_artifact_fingerprint=backend.backend_artifact_fingerprint,
        backend_config_fingerprint=backend.config_fingerprint,
    )
    calibrated = LogisticCalibrator(artifact).apply(
        replace(
            _result(),
            backend=backend.backend_id,
            model=backend.model_id,
            model_version=backend.model_version,
        ),
        {"x": 1.0},
        backend=backend,
        view=_view(),
        request=REQUEST,
        expected_language="es",
        usable_max_wer=0.10,
    )
    assert calibrated.calibrator_version == artifact.version
    assert artifact.backend_artifact_kind == "provider_model_descriptor"
    changed_descriptor = _RemoteBackend(response_mapper_version="openai-audio/v2")
    assert (
        changed_descriptor.backend_artifact_fingerprint
        != backend.backend_artifact_fingerprint
    )
    with pytest.raises(ValueError, match="artefacto/config"):
        LogisticCalibrator(artifact).validate_for(
            backend=changed_descriptor,
            pipeline=PIPELINE,
            request=REQUEST,
            expected_language="es",
            usable_max_wer=0.10,
        )


def test_decorator_local_nunca_publica_resultado_sin_calibrar(backend):
    raw = _ReturningBackend(backend.model_artifact)
    wrapped = CalibratingLocalAsrBackend(
        raw,
        LogisticCalibrator(_full_feature_artifact_for(raw)),
        pipeline=PIPELINE,
        request=REQUEST,
        expected_language="es",
        usable_max_wer=0.10,
    )
    wrapped.warm()
    result = wrapped.transcribe(_clip(), REQUEST)
    assert raw.warm_calls == raw.transcribe_calls == 1
    assert result.calibrated_confidence is not None
    assert result.calibrator_version == wrapped.calibrator_version
    assert wrapped.model_artifact is raw.model_artifact


def test_decorator_rechaza_binding_antes_de_warm_o_inferencia(backend):
    raw = _ReturningBackend(backend.model_artifact)
    with pytest.raises(ValueError, match="pipeline/request"):
        CalibratingLocalAsrBackend(
            raw,
            LogisticCalibrator(_artifact_for(raw)),
            pipeline=PipelineProvenance.capture(
                sample_rate=16000,
                step=PipelineStep("different", "1", {}),
            ),
            request=REQUEST,
            expected_language="es",
            usable_max_wer=0.10,
        )
    assert raw.warm_calls == raw.transcribe_calls == 0


def test_artifact_roundtrip_canonico_en_bytes():
    payload = serialize_calibrator_artifact(_artifact())
    assert payload.endswith(b"\n")
    assert parse_calibrator_artifact_bytes(
        payload, expected_fingerprint=_artifact().version,
    ) == _artifact()


def test_artifact_recalcula_y_rechaza_limite_persistido_adulterado():
    with pytest.raises(ValueError, match="precision_lower_95 no coincide"):
        replace(_artifact(), precision_lower_95=0.9999)


def test_calibrator_no_permite_rebinding_de_artifact():
    calibrator = LogisticCalibrator(_artifact())
    with pytest.raises(AttributeError):
        calibrator.artifact = replace(_artifact(), artifact_version="other")


def test_threshold_maximiza_cobertura_bajo_precision():
    selection = select_operating_threshold(
        probabilities=[0.90] * 300 + [0.80],
        labels=[1] * 300 + [0],
        min_precision_lower_95=0.99,
    )
    assert selection.threshold == pytest.approx(0.90)
    assert selection.precision == 1.0
    assert selection.precision_lower_95 >= 0.99
    assert selection.coverage == pytest.approx(300 / 301)
    assert (selection.correct, selection.accepted) == (300, 300)


def test_precision_puntual_alta_no_sustituye_el_limite_inferior():
    selection = select_operating_threshold(
        probabilities=[0.90] * 100,
        labels=[1] * 100,
        min_precision_lower_95=0.99,
    )
    assert selection.accepted == 0
    assert selection.precision_lower_95 == 0.0


@pytest.mark.parametrize(
    "field", ["feature_means", "feature_scales", "coefficients", "intercept"]
)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_artifact_rechaza_parametros_no_finitos(field, bad):
    artifact = _artifact()
    value = bad
    if field != "intercept":
        values = list(getattr(artifact, field))
        values[0] = bad
        value = tuple(values)
    with pytest.raises(ValueError, match="finitos"):
        replace(artifact, **{field: value})


def test_parse_rechaza_constante_json_nan():
    payload = _artifact().to_dict()
    payload["intercept"] = float("nan")
    with pytest.raises(ValueError, match="JSON no finita"):
        parse_calibrator_artifact_bytes(
            json.dumps(payload).encode("utf-8"),
            expected_fingerprint=_artifact().version,
        )


def test_parse_rechaza_reescritura_sin_trust_anchor_promovido():
    payload = serialize_calibrator_artifact(
        replace(_artifact(), operating_threshold=0.81)
    )
    with pytest.raises(ValueError, match="trust anchor"):
        parse_calibrator_artifact_bytes(
            payload, expected_fingerprint=_artifact().version,
        )


def test_parse_rechaza_claves_duplicadas_y_vectores_coercibles():
    payload = serialize_calibrator_artifact(_artifact()).replace(
        b'{"artifact_version":',
        b'{"artifact_version":"duplicado","artifact_version":',
        1,
    )
    with pytest.raises(ValueError, match="duplicada"):
        parse_calibrator_artifact_bytes(
            payload, expected_fingerprint=_artifact().version,
        )
    data = _artifact().to_dict()
    data["feature_names"] = "x"
    with pytest.raises(ValueError, match="lista JSON"):
        parse_calibrator_artifact_bytes(
            json.dumps(data).encode("utf-8"),
            expected_fingerprint=_artifact().version,
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "0.5"])
def test_runtime_rechaza_features_y_probabilidades_no_numericas_o_no_finitas(bad):
    with pytest.raises(ValueError, match="numericas y finitas"):
        LogisticCalibrator(_artifact()).predict_proba({"x": bad})
    with pytest.raises(ValueError, match="probabilities/labels"):
        select_operating_threshold([bad], [1], min_precision_lower_95=0.9)
