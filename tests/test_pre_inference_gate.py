from dataclasses import replace
import math

import pytest
from speechtotext.audio import AudioQualityReport
from speechtotext.audio.gate import QualityThresholds, evaluate_pre_inference


def _report():
    return AudioQualityReport(
        duration_ms=1000,
        effective_voice_ms=600,
        input_rms_dbfs=-30.0,
        processed_rms_dbfs=-24.0,
        peak_dbfs=-3.0,
        clipping_ratio=0.0,
        noise_floor_dbfs=-45.0,
        snr_db=21.0,
        requested_gain_db=6.0,
        applied_gain_db=6.0,
        dropped_frames=0,
        discontinuities=0,
        warnings=(),
    )


def _thresholds():
    return QualityThresholds(
        min_effective_voice_ms=160,
        min_processed_rms_dbfs=-45.0,
        min_snr_db=6.0,
        max_clipping_ratio=0.01,
    )


def test_gate_acepta_audio_util():
    decision = evaluate_pre_inference(_report(), _thresholds())
    assert decision.eligible is True
    assert decision.reason_codes == ()


def test_gate_bloquea_silencio_sin_llamarlo_voz_corta():
    decision = evaluate_pre_inference(
        replace(_report(), effective_voice_ms=0, snr_db=None),
        _thresholds(),
    )
    assert decision.eligible is False
    assert decision.reason_codes[0] == "silence"
    assert "too_short" not in decision.reason_codes


def test_gate_acumula_razones_en_orden_estable():
    bad = replace(
        _report(),
        effective_voice_ms=100,
        processed_rms_dbfs=-60.0,
        snr_db=2.0,
        clipping_ratio=0.2,
        dropped_frames=3,
        discontinuities=1,
    )
    decision = evaluate_pre_inference(bad, _thresholds())
    assert decision.reason_codes == (
        "too_short",
        "level_too_low",
        "snr_too_low",
        "clipping",
        "dropped_audio",
        "discontinuous_audio",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_processed_rms_dbfs", math.nan),
        ("min_snr_db", math.inf),
        ("max_clipping_ratio", math.nan),
    ],
)
def test_thresholds_rechazan_valores_no_finitos(field, value):
    values = {
        "min_effective_voice_ms": 160,
        "min_processed_rms_dbfs": -45.0,
        "min_snr_db": 6.0,
        "max_clipping_ratio": 0.01,
    }
    values[field] = value
    with pytest.raises(ValueError):
        QualityThresholds(**values)
