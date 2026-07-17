import math
import subprocess
import sys

import pytest

from speechtotext.asr import (
    AsrBackend,
    AsrError,
    NativeSignals,
    SegmentNativeSignals,
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
)


def test_asr_publico_no_importa_faster_whisper():
    assert AsrBackend.__name__ == "AsrBackend"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import speechtotext.asr; "
            "assert 'faster_whisper' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0


def test_resultado_conserva_senales_palabras_y_target():
    word = TranscriptionWord("hola", 0.1, 0.4, 0.92)
    segment = TranscriptionSegment(
        0.1,
        0.5,
        "hola",
        (word,),
        SegmentNativeSignals(0.02, -0.15, 1.1),
    )
    result = TranscriptionResult(
        text="hola",
        language="es",
        words=(word,),
        segments=(segment,),
        backend="faster-whisper",
        model="small",
        model_version="rev1",
        latency_ms=120,
        native_signals=NativeSignals(0.02, -0.15, 1.1, 0.99),
        confidence_target="segment_usable",
        calibrated_confidence=None,
        calibrator_version=None,
        warnings=(),
    )
    assert result.segments[0].native_signals.no_speech == pytest.approx(0.02)
    assert result.confidence_target == "segment_usable"


def test_resultado_rechaza_confianza_sin_version():
    with pytest.raises(ValueError, match="calibrator_version"):
        TranscriptionResult(
            text="hola",
            language="es",
            words=(),
            segments=(),
            backend="fake",
            model="fake",
            model_version="1",
            latency_ms=1,
            native_signals=NativeSignals(None, None, None, None),
            confidence_target="segment_usable",
            calibrated_confidence=0.9,
            calibrator_version=None,
            warnings=(),
        )


def test_asr_error_expone_codigo_y_recuperabilidad():
    error = AsrError("model_unavailable", False, "modelo ausente")
    assert str(error) == "modelo ausente"
    assert error.code == "model_unavailable"
    assert error.recoverable is False


def test_request_fingerprint_es_determinista_y_sensible_al_contexto():
    base = TranscriptionRequest(language="es", context="catalog-v1")
    assert base.fingerprint == TranscriptionRequest(
        language="es", context="catalog-v1"
    ).fingerprint
    assert base.fingerprint != TranscriptionRequest(
        language="es", context="catalog-v2"
    ).fingerprint


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_asr_rechaza_timestamps_y_senales_no_finitas(value):
    with pytest.raises(ValueError):
        TranscriptionWord("hola", value, 1.0, 0.9)
    with pytest.raises(ValueError):
        TranscriptionSegment(
            value,
            1.0,
            "hola",
            (),
            SegmentNativeSignals(0.1, -0.2, 1.0),
        )
    with pytest.raises(ValueError):
        SegmentNativeSignals(value, -0.2, 1.0)
    with pytest.raises(ValueError):
        NativeSignals(0.1, value, 1.0, 0.9)


def test_resultado_exige_identidad_lenguaje_y_latencia_entera():
    base = dict(
        text="",
        language="es",
        words=(),
        segments=(),
        backend="fake",
        model="model",
        model_version="1",
        latency_ms=1,
        native_signals=NativeSignals(None, None, None, None),
        confidence_target="segment_usable",
        calibrated_confidence=None,
        calibrator_version=None,
        warnings=(),
    )
    for field in ("language", "backend", "model", "model_version"):
        with pytest.raises(ValueError):
            TranscriptionResult(**{**base, field: ""})
    with pytest.raises(ValueError):
        TranscriptionResult(**{**base, "latency_ms": 1.5})


def test_tipos_asr_rechazan_coerciones_y_contenedores_mutables():
    base = dict(
        text="",
        language="es",
        words=(),
        segments=(),
        backend="fake",
        model="model",
        model_version="1",
        latency_ms=1,
        native_signals=NativeSignals(None, None, None, None),
        confidence_target="segment_usable",
        calibrated_confidence=None,
        calibrator_version=None,
        warnings=(),
    )
    with pytest.raises(TypeError, match="hotwords"):
        TranscriptionRequest(hotwords="Aurelius")
    with pytest.raises(TypeError, match="beam_size"):
        TranscriptionRequest(beam_size=True)
    with pytest.raises(TypeError, match="word_timestamps"):
        TranscriptionRequest(word_timestamps=1)
    with pytest.raises(TypeError, match="timestamps"):
        TranscriptionWord("hola", True, 1.0, 0.9)
    with pytest.raises(TypeError, match="senales"):
        NativeSignals("0.1", None, None, None)
    with pytest.raises(TypeError, match="words"):
        TranscriptionSegment(
            0.0, 1.0, "hola", [], SegmentNativeSignals(None, None, None)
        )
    with pytest.raises(TypeError, match="warnings"):
        TranscriptionResult(**{**base, "warnings": ["mutable"]})
