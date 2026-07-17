from contextlib import contextmanager
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from speechtotext.asr import TranscriptionRequest
from speechtotext.asr.faster_whisper import (
    FasterWhisperBackend,
    FasterWhisperConfig,
)
from speechtotext.audio import (
    AudioClip, AudioQualityReport, AudioView, AudioViews, PipelineStep,
)
from speechtotext.models import load_model_manifest, verify_model_files
from speechtotext.models.filesystem import FakeModelFilesystem


@contextmanager
def _backend(tmp_path, segments, info, calls):
    model_file = tmp_path / "model.bin"
    model_file.write_bytes(b"weights")
    data = {
            "schema_version": "speechtotext.model/v1",
            "model_id": "faster-whisper-small",
            "source": "https://example.invalid/model",
            "revision_kind": "git_commit",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "license": "MIT",
            "format": "ctranslate2",
            "sample_rate": 16000,
            "preprocessing": {"mono": True, "dtype": "float32"},
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
        manifest_path,
        model_root=tmp_path,
        expected_fingerprint=expected,
        filesystem=filesystem,
    )

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            calls["audio"] = audio
            calls["kwargs"] = kwargs
            return iter(segments), info

    def factory(path, **kwargs):
        calls["factory_path"] = path
        calls["factory_kwargs"] = kwargs
        return FakeModel()

    ticks = iter([10.0, 10.125])
    with verify_model_files(
        manifest, tmp_path, filesystem=filesystem
    ) as verified:
        yield FasterWhisperBackend(
            FasterWhisperConfig(),
            verified,
            model_factory=factory,
            clock=lambda: next(ticks),
        ), filesystem


def _clip():
    view = AudioView.capture(
        np.zeros(16000, dtype=np.float32),
        16000,
        step=PipelineStep("test-source", "1", {}),
    )
    quality = AudioQualityReport(
        1000, 800, -25.0, -25.0, -6.0, 0.0, -45.0, 20.0,
        0.0, 0.0, 0, 0, (),
    )
    return AudioClip(1.0, 2.0, "mic", (), quality, AudioViews(view, view, view))


def test_backend_extrae_palabras_senales_y_opciones(tmp_path):
    words = [SimpleNamespace(start=0.1, end=0.4, word=" hola", probability=0.91)]
    segments = [
        SimpleNamespace(
            start=0.0,
            end=1.0,
            text=" hola",
            words=words,
            no_speech_prob=0.10,
            avg_logprob=-0.20,
            compression_ratio=1.1,
        ),
        SimpleNamespace(
            start=1.0,
            end=3.0,
            text=" mundo",
            words=None,
            no_speech_prob=0.30,
            avg_logprob=-0.40,
            compression_ratio=1.4,
        ),
    ]
    info = SimpleNamespace(language="es", language_probability=0.98)
    calls = {}
    with _backend(tmp_path, segments, info, calls) as (backend, _filesystem):
        assert backend.model_artifact.root == tmp_path.resolve()
        assert backend.model_id == backend.model_artifact.manifest.model_id
        assert backend.model_version == backend.model_artifact.manifest.revision
        result = backend.transcribe(
            _clip(),
            TranscriptionRequest(
                language="es", hotwords=("Aurelius",), context="Tachira"
            ),
        )
        assert result.text == "hola mundo"
        assert result.words[0].confidence == pytest.approx(0.91)
        assert result.native_signals.no_speech == pytest.approx(0.30)
        assert result.native_signals.avg_logprob == pytest.approx((-0.2 + -0.8) / 3)
        assert result.native_signals.compression_ratio == pytest.approx(1.4)
        assert result.native_signals.language_probability == pytest.approx(0.98)
        assert result.latency_ms == 125
        assert calls["kwargs"]["condition_on_previous_text"] is False
        assert calls["kwargs"]["hotwords"] == "Aurelius"
        assert calls["kwargs"]["initial_prompt"] == "Tachira"
        assert calls["factory_kwargs"]["local_files_only"] is True


def test_backend_vacio_no_inventa_senales(tmp_path):
    calls = {}
    context = _backend(
        tmp_path,
        [],
        SimpleNamespace(language="es", language_probability=0.8),
        calls,
    )
    with context as (backend, _filesystem):
        result = backend.transcribe(_clip(), TranscriptionRequest())
        assert result.text == ""
        assert result.native_signals.no_speech is None
        assert result.native_signals.avg_logprob is None
        assert result.warnings == ("empty_transcript",)


def test_lease_activo_bloquea_cambio_y_cierre_invalida_backend(tmp_path):
    calls = {}
    context = _backend(
        tmp_path,
        [],
        SimpleNamespace(language="es", language_probability=1.0),
        calls,
    )
    with context as (backend, filesystem):
        with pytest.raises(PermissionError, match="sharing violation"):
            filesystem.replace(tmp_path / "model.bin", b"tampered")
        backend.warm()
    with pytest.raises(Exception, match="activo"):
        backend.transcribe(_clip(), TranscriptionRequest())


def test_backend_rechaza_duck_artifact_y_no_permite_rebinding(tmp_path):
    fake = SimpleNamespace(
        manifest=SimpleNamespace(format="ctranslate2"),
        root=tmp_path.resolve(),
        fingerprint="0" * 64,
    )
    with pytest.raises(TypeError, match="VerifiedModelArtifact"):
        FasterWhisperBackend(FasterWhisperConfig(), fake)
    context = _backend(
        tmp_path,
        [],
        SimpleNamespace(language="es", language_probability=1.0),
        {},
    )
    with context as (backend, _filesystem):
        with pytest.raises(AttributeError):
            backend.model_artifact = fake


def test_config_fingerprint_liga_todos_los_parametros_efectivos():
    base = FasterWhisperConfig()
    assert base.fingerprint == FasterWhisperConfig().fingerprint
    assert base.fingerprint != FasterWhisperConfig(compute_type="float32").fingerprint
    assert base.fingerprint != FasterWhisperConfig(cpu_threads=2).fingerprint
    with pytest.raises(ValueError, match="cpu_threads"):
        FasterWhisperConfig(cpu_threads=True)
