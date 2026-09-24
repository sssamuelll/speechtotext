import threading
from types import SimpleNamespace

import numpy as np
import pytest

from speechtotext.asr import AsrBackend, AsrError, Caps, TranscriptionRequest
from speechtotext.asr.faster_whisper import FasterWhisperBackend, FasterWhisperConfig

_INFO = SimpleNamespace(language="es", language_probability=0.98)


def _backend(model, segments, info, calls, **kwargs):
    class FakeModel:
        def transcribe(self, audio, **opts):
            calls["audio"] = audio
            calls["kwargs"] = opts
            return iter(segments), info

    def factory(path, **factory_kwargs):
        calls["factory_path"] = path
        calls["factory_kwargs"] = factory_kwargs
        return FakeModel()

    ticks = iter([10.0, 10.125])
    return FasterWhisperBackend(
        model, FasterWhisperConfig(), model_factory=factory,
        clock=lambda: next(ticks), **kwargs,
    )


def _samples(seconds: float = 1.0) -> np.ndarray:
    return np.zeros(int(seconds * 16000), dtype=np.float32)


def test_the_backend_extracts_words_native_signals_and_options():
    words = [SimpleNamespace(start=0.1, end=0.4, word=" hola", probability=0.91)]
    segments = [
        SimpleNamespace(start=0.0, end=1.0, text=" hola", words=words,
                        no_speech_prob=0.10, avg_logprob=-0.20, compression_ratio=1.1),
        SimpleNamespace(start=1.0, end=3.0, text=" mundo", words=None,
                        no_speech_prob=0.30, avg_logprob=-0.40, compression_ratio=1.4),
    ]
    calls = {}
    backend = _backend("large-v3", segments, _INFO, calls)
    assert isinstance(backend, AsrBackend)
    assert backend.model_id == "large-v3"
    assert backend.model_version == "unpinned"
    samples = _samples()
    result = backend.transcribe(
        samples,
        TranscriptionRequest(language="es", hotwords=("Bézier",), context="meeting"),
    )
    assert result.text == "hola mundo"
    assert result.segments[0].text == " hola"          # raw: the leading space comes from the engine
    assert result.words[0].text == " hola"
    assert calls["audio"] is samples
    assert calls["kwargs"]["vad_filter"] is False
    assert result.model == "large-v3"
    assert result.model_version == "unpinned"
    assert result.words[0].confidence == pytest.approx(0.91)
    assert result.native_signals.no_speech == pytest.approx(0.30)
    assert result.native_signals.avg_logprob == pytest.approx((-0.2 + -0.8) / 3)
    assert result.native_signals.compression_ratio == pytest.approx(1.4)
    assert result.native_signals.language_probability == pytest.approx(0.98)
    assert result.latency_ms == 125
    assert calls["kwargs"]["condition_on_previous_text"] is False
    assert calls["kwargs"]["hotwords"] == "Bézier"
    assert calls["kwargs"]["initial_prompt"] == "meeting"


def test_a_model_name_lets_faster_whisper_resolve_it():
    calls = {}
    _backend("large-v3", [], _INFO, calls).warm()
    assert calls["factory_path"] == "large-v3"
    assert calls["factory_kwargs"]["local_files_only"] is False
    assert calls["factory_kwargs"]["device"] == "cpu"
    assert calls["factory_kwargs"]["compute_type"] == "int8"


def test_a_directory_path_loads_local_files_only(tmp_path):
    calls = {}
    backend = _backend(tmp_path, [], _INFO, calls)
    assert backend.model_id == tmp_path.name
    backend.warm()
    assert calls["factory_path"] == str(tmp_path)
    assert calls["factory_kwargs"]["local_files_only"] is True


def test_the_model_version_is_supplied_by_the_caller():
    revision = "0123456789abcdef0123456789abcdef01234567"
    backend = _backend("large-v3", [], _INFO, {}, model_version=revision)
    assert backend.model_version == revision
    assert backend.transcribe(_samples(), TranscriptionRequest()).model_version == revision


def test_warm_loads_the_model_only_once():
    calls = {"n": 0}

    def factory(path, **kwargs):
        calls["n"] += 1
        return SimpleNamespace(transcribe=lambda audio, **opts: (iter([]), _INFO))

    backend = FasterWhisperBackend("small", model_factory=factory)
    backend.warm()
    backend.warm()
    backend.transcribe(_samples(), TranscriptionRequest())
    assert calls["n"] == 1
    assert backend.config == FasterWhisperConfig()


def test_an_empty_backend_result_does_not_invent_native_signals():
    info = SimpleNamespace(language="es", language_probability=0.8)
    result = _backend("small", [], info, {}).transcribe(_samples(), TranscriptionRequest())
    assert result.text == ""
    assert result.native_signals.no_speech is None
    assert result.native_signals.avg_logprob is None
    assert result.warnings == ("empty_transcript",)


def test_an_invalid_model_is_rejected_during_construction():
    with pytest.raises(TypeError, match="model"):
        FasterWhisperBackend(123)
    with pytest.raises(ValueError, match="model"):
        FasterWhisperBackend("   ")
    with pytest.raises(ValueError, match="model_version"):
        FasterWhisperBackend("small", model_version="")


def test_the_config_fingerprint_binds_all_effective_parameters():
    base = FasterWhisperConfig()
    assert base.fingerprint == FasterWhisperConfig().fingerprint
    assert base.fingerprint != FasterWhisperConfig(compute_type="float32").fingerprint
    assert base.fingerprint != FasterWhisperConfig(cpu_threads=2).fingerprint
    with pytest.raises(ValueError, match="cpu_threads"):
        FasterWhisperConfig(cpu_threads=True)


def test_the_backend_declares_caps_and_the_engine_version():
    backend = _backend("large-v3", [], _INFO, {})
    assert isinstance(backend, AsrBackend)
    assert backend.caps == Caps("honored", "honored", "honored")
    assert backend.engine_version.startswith("faster-whisper")
    assert (backend.quant, backend.device) == ("int8", "cpu")


def test_vad_and_auto_are_sent_to_the_engine():
    calls = {}
    _backend("large-v3", [], _INFO, calls).transcribe(
        _samples(), TranscriptionRequest(language="auto", vad=True),
    )
    assert calls["kwargs"]["vad_filter"] is True
    assert calls["kwargs"]["language"] is None


def _raw(start, end, text):
    return SimpleNamespace(start=start, end=end, text=text, words=None,
                           no_speech_prob=None, avg_logprob=None, compression_ratio=None)


def _lazy(segments, pulled):
    """Like faster-whisper's generator: it records each segment as the engine hands it over."""
    for segment in segments:
        pulled.append(segment.text)
        yield segment


def test_each_segment_reaches_the_callback_before_the_next_is_decoded():
    pulled, seen = [], []
    backend = _backend("large-v3", _lazy([_raw(0.0, 1.0, " one"), _raw(1.0, 2.5, " two")], pulled),
                       _INFO, {})
    result = backend.transcribe(
        _samples(3.0), TranscriptionRequest(),
        on_segment=lambda s: seen.append((s.start, s.end, s.text, list(pulled))),
    )
    assert seen == [(0.0, 1.0, " one", [" one"]), (1.0, 2.5, " two", [" one", " two"])]
    assert [s.text for s in result.segments] == [" one", " two"]


def test_cancel_stops_between_segments_and_decodes_no_further():
    pulled, stop = [], threading.Event()
    segments = [_raw(0.0, 1.0, " one"), _raw(1.0, 2.0, " two"), _raw(2.0, 3.0, " three")]
    backend = _backend("large-v3", _lazy(segments, pulled), _INFO, {})
    with pytest.raises(AsrError) as ei:
        backend.transcribe(_samples(3.0), TranscriptionRequest(),
                           on_segment=lambda s: stop.set(), cancel=stop)
    assert (ei.value.code, ei.value.recoverable) == ("cancelled", True)
    assert str(ei.value) == "transcription cancelled"
    assert pulled == [" one"]


def test_cancel_before_starting_never_reaches_the_model():
    stop = threading.Event()
    stop.set()

    class Untouchable:
        def transcribe(self, audio, **opts):
            pytest.fail("decoded after cancel")

    backend = FasterWhisperBackend("large-v3", model_factory=lambda path, **kw: Untouchable())
    with pytest.raises(AsrError) as ei:
        backend.transcribe(_samples(), TranscriptionRequest(), cancel=stop)
    assert ei.value.code == "cancelled"


def test_a_callback_error_is_not_reported_as_an_engine_failure():
    backend = _backend("large-v3", iter([_raw(0.0, 1.0, " one")]), _INFO, {})

    def broken(segment):
        raise ValueError("bug in the caller")

    with pytest.raises(ValueError, match="bug in the caller"):
        backend.transcribe(_samples(), TranscriptionRequest(), on_segment=broken)


def test_an_error_while_decoding_is_still_an_engine_failure():
    # Guard: passes today and must keep passing once the loop reads one segment at a time.
    def failing():
        yield _raw(0.0, 1.0, " one")
        raise RuntimeError("CUDA error: out of range")

    backend = _backend("large-v3", failing(), _INFO, {})
    with pytest.raises(AsrError) as ei:
        backend.transcribe(_samples(), TranscriptionRequest())
    assert ei.value.code == "backend_failed" and "CUDA error" in str(ei.value)
