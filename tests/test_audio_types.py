from dataclasses import replace
import math

import numpy as np
import pytest

from speechtotext.audio import (
    AudioClip,
    AudioQualityReport,
    AudioView,
    AudioViews,
    PipelineProvenance,
    PipelineStep,
    SpeechRegion,
)

def _view(samples=None, rate=16000):
    if samples is None:
        samples = np.zeros(rate, dtype=np.float32)
    return AudioView.capture(
        np.asarray(samples),
        rate,
        step=PipelineStep("decode", "1", {"layout": "mono"}),
    )


def _quality(duration_ms=1000):
    return AudioQualityReport(
        duration_ms=duration_ms,
        effective_voice_ms=0,
        input_rms_dbfs=-20.0,
        processed_rms_dbfs=-20.0,
        peak_dbfs=-12.0,
        clipping_ratio=0.0,
        noise_floor_dbfs=None,
        snr_db=None,
        requested_gain_db=0.0,
        applied_gain_db=0.0,
        dropped_frames=0,
        discontinuities=0,
        warnings=(),
    )


def test_audio_view_normaliza_float32_mono_contiguo_e_inmutable():
    original = np.array([0.0, 0.25, -0.25], dtype=np.float64)
    view = AudioView.capture(
        original, 16000, step=PipelineStep("decode", "1", {"layout": "mono"})
    )
    assert view.samples.dtype == np.float32
    assert view.samples.ndim == 1
    assert view.samples.flags.c_contiguous
    assert view.samples.flags.writeable is False
    original[1] = 99.0
    assert view.samples[1] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        view.samples.setflags(write=True)
    with pytest.raises(ValueError):
        view.samples[0] = 1.0


@pytest.mark.parametrize(
    "samples",
    [np.zeros((2, 2), dtype=np.float32), np.array([0.0, np.nan], dtype=np.float32)],
)
def test_audio_view_rechaza_multicanal_o_no_finito(samples):
    with pytest.raises(ValueError):
        AudioView.capture(
            samples, 16000, step=PipelineStep("decode", "1", {"layout": "mono"})
        )


def test_audio_view_rechaza_parent_y_provenance_simulados():
    fake_provenance = type("FakeProvenance", (), {"sample_rate": 16000})()
    with pytest.raises(TypeError, match="PipelineProvenance"):
        AudioView._create([0.0], 16000, fake_provenance)
    with pytest.raises(TypeError, match="AudioView"):
        AudioView.derive(
            object(),
            [0.0],
            steps=(PipelineStep("gain", "1", {}),),
        )


def test_audio_view_factory_privado_no_es_una_ruta_publica():
    provenance = PipelineProvenance.capture(
        sample_rate=16000,
        step=PipelineStep("capture", "1", {}),
    )
    with pytest.raises(TypeError, match="factory"):
        AudioView._create([0.0], 16000, provenance)


def test_audio_view_no_tiene_constructor_publico():
    provenance = PipelineProvenance.capture(
        sample_rate=16000,
        step=PipelineStep("capture", "1", {}),
    )
    with pytest.raises(TypeError):
        AudioView(np.zeros(1, dtype=np.float32), 16000, provenance)


def test_audio_clip_conserva_pausas_y_resuelve_vistas():
    view = _view(np.zeros(32_000, dtype=np.float32))
    clip = AudioClip(
        started_at=10.0,
        ended_at=12.0,
        source_id="desktop-mic",
        speech_regions=(SpeechRegion(0.2, 0.8), SpeechRegion(1.1, 1.8)),
        quality=_quality(2000),
        views=AudioViews(capture=view, analysis=view, asr=view),
    )
    assert clip.speech_regions == (
        SpeechRegion(0.2, 0.8),
        SpeechRegion(1.1, 1.8),
    )
    assert clip.view("asr") is view
    with pytest.raises(KeyError, match="identity"):
        clip.view("identity")
    with pytest.raises(KeyError, match="desconocida"):
        clip.view("raw")


def test_audio_clip_rechaza_region_fuera_de_su_duracion():
    view = _view()
    with pytest.raises(ValueError, match="speech region"):
        AudioClip(
            10.0,
            11.0,
            "mic",
            (SpeechRegion(0.5, 1.5),),
            _quality(),
            AudioViews(view, view, view),
        )


def test_audio_view_derive_representa_resample_48k_a_16k():
    capture = _view(np.zeros(480, dtype=np.float32), rate=48000)
    asr = AudioView.derive(
        capture,
        np.zeros(160, dtype=np.float32),
        sample_rate=16000,
        steps=(PipelineStep("resample", "1", {"from": 48000, "to": 16000}),),
    )
    assert asr.sample_rate == 16000
    assert asr.provenance.sample_rate == 16000
    assert asr.pipeline_fingerprint != capture.pipeline_fingerprint


@pytest.mark.parametrize("sample_rate", [True, 0, -1, 16000.5])
def test_audio_view_rechaza_sample_rate_no_entero_positivo(sample_rate):
    with pytest.raises(ValueError, match="sample_rate"):
        AudioView.capture(
            [0.0], sample_rate,
            step=PipelineStep("decode", "1", {"layout": "mono"}),
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_contratos_rechazan_tiempos_y_metricas_no_finitas(value):
    with pytest.raises(ValueError):
        SpeechRegion(value, 1.0)
    with pytest.raises(ValueError):
        replace(_quality(), processed_rms_dbfs=value)
    with pytest.raises(ValueError):
        AudioClip(
            value, 1.0, "mic", (), _quality(),
            AudioViews(_view(), _view(), _view()),
        )


def test_quality_rechaza_conteos_rangos_e_inconsistencia():
    with pytest.raises(ValueError):
        replace(_quality(), effective_voice_ms=2000)
    with pytest.raises(ValueError):
        replace(_quality(), clipping_ratio=1.01)
    with pytest.raises(ValueError):
        replace(_quality(), dropped_frames=-1)
    with pytest.raises(ValueError, match="enteros"):
        replace(_quality(), duration_ms=True)
    with pytest.raises(ValueError, match="enteros"):
        replace(_quality(), dropped_frames=0.5)


def test_audio_views_rechaza_duck_types_y_clip_rechaza_duraciones_divergentes():
    one_second = _view()
    half_second = _view(np.zeros(8_000, dtype=np.float32))
    with pytest.raises(TypeError, match="AudioView"):
        AudioViews(one_second, object(), one_second)
    with pytest.raises(ValueError, match="duracion de vistas"):
        AudioClip(
            0.0, 1.0, "mic", (), _quality(),
            AudioViews(one_second, half_second, one_second),
        )
    with pytest.raises(ValueError, match="quality.duration_ms"):
        AudioClip(
            0.0, 1.0, "mic", (), _quality(900),
            AudioViews(one_second, one_second, one_second),
        )


@pytest.mark.parametrize("bad", [True, object(), "warning"])
def test_quality_rechaza_metricas_o_warnings_con_tipo_incorrecto(bad):
    if bad == "warning":
        with pytest.raises(ValueError, match="warnings"):
            replace(_quality(), warnings=bad)
    else:
        with pytest.raises(ValueError, match="metricas"):
            replace(_quality(), processed_rms_dbfs=bad)
