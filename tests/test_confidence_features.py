import math

from speechtotext.asr import NativeSignals, TranscriptionResult
from speechtotext.audio import AudioQualityReport
from speechtotext.confidence.features import ASR_FEATURE_NAMES, extract_asr_features


def _quality(snr=15.0):
    return AudioQualityReport(
        2000, 1500, -30.0, -24.0, -3.0, 0.001, -40.0, snr,
        6.0, 6.0, 0, 0, (),
    )


def _result(signals=None, text="hola hola mundo", language="es"):
    return TranscriptionResult(
        text=text,
        language=language,
        words=(),
        segments=(),
        backend="faster-whisper",
        model="small",
        model_version="rev1",
        latency_ms=100,
        native_signals=signals or NativeSignals(0.1, -0.2, 1.1, 0.98),
        confidence_target="segment_usable",
        calibrated_confidence=None,
        calibrator_version=None,
        warnings=(),
    )


def test_features_tienen_orden_y_repeticion_definidos():
    features = extract_asr_features(_result(), _quality(), expected_language="es")
    assert tuple(features) == ASR_FEATURE_NAMES
    assert features["token_repetition_ratio"] == 1.0 - 2.0 / 3.0
    assert features["language_matches"] == 1.0
    assert features["missing_snr"] == 0.0
    assert features["missing_native"] == 0.0


def test_features_imputan_missing_sin_nan():
    result = _result(NativeSignals(None, None, None, None), text="")
    features = extract_asr_features(result, _quality(snr=None), expected_language="es")
    assert features["missing_snr"] == 1.0
    assert features["missing_native"] == 1.0
    assert all(math.isfinite(value) for value in features.values())
