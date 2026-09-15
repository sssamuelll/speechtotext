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


def test_the_public_asr_api_does_not_import_faster_whisper():
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


def test_the_result_preserves_native_signals_and_words():
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
        warnings=(),
    )
    assert result.segments[0].native_signals.no_speech == pytest.approx(0.02)


def test_asr_error_exposes_its_code_and_recoverability():
    error = AsrError("model_unavailable", False, "model missing")
    assert str(error) == "model missing"
    assert error.code == "model_unavailable"
    assert error.recoverable is False


def test_the_request_fingerprint_is_deterministic_and_context_sensitive():
    base = TranscriptionRequest(language="es", context="catalog-v1")
    assert base.fingerprint == TranscriptionRequest(
        language="es", context="catalog-v1"
    ).fingerprint
    assert base.fingerprint != TranscriptionRequest(
        language="es", context="catalog-v2"
    ).fingerprint


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_asr_rejects_nonfinite_timestamps_and_signals(value):
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


def test_the_result_requires_identity_language_and_integer_latency():
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
        warnings=(),
    )
    for field in ("language", "backend", "model", "model_version"):
        with pytest.raises(ValueError):
            TranscriptionResult(**{**base, field: ""})
    with pytest.raises(ValueError):
        TranscriptionResult(**{**base, "latency_ms": 1.5})


def test_asr_types_reject_coercions_and_mutable_containers():
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
        warnings=(),
    )
    with pytest.raises(TypeError, match="hotwords"):
        TranscriptionRequest(hotwords="Bézier")
    with pytest.raises(TypeError, match="beam_size"):
        TranscriptionRequest(beam_size=True)
    with pytest.raises(TypeError, match="word_timestamps"):
        TranscriptionRequest(word_timestamps=1)
    with pytest.raises(TypeError, match="timestamps"):
        TranscriptionWord("hola", True, 1.0, 0.9)
    with pytest.raises(TypeError, match="signals"):
        NativeSignals("0.1", None, None, None)
    with pytest.raises(TypeError, match="words"):
        TranscriptionSegment(
            0.0, 1.0, "hola", [], SegmentNativeSignals(None, None, None)
        )
    with pytest.raises(TypeError, match="warnings"):
        TranscriptionResult(**{**base, "warnings": ["mutable"]})


def test_the_result_has_no_calibration_fields():
    from dataclasses import fields

    from speechtotext.asr.types import TranscriptionResult

    names = {field.name for field in fields(TranscriptionResult)}
    assert names == {
        "text", "language", "words", "segments", "backend", "model",
        "model_version", "latency_ms", "native_signals", "warnings",
    }


def test_the_request_validates_vad_as_a_bool_and_includes_it_in_the_fingerprint():
    from speechtotext.asr.types import TranscriptionRequest

    assert TranscriptionRequest().vad is False
    assert TranscriptionRequest(vad=True).to_dict()["vad"] is True
    assert TranscriptionRequest(vad=True).fingerprint != TranscriptionRequest().fingerprint
    with pytest.raises(TypeError, match="vad"):
        TranscriptionRequest(vad=1)


def test_the_request_accepts_auto_as_the_language():
    from speechtotext.asr.types import TranscriptionRequest

    assert TranscriptionRequest(language="auto").language == "auto"
