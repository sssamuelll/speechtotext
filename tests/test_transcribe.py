"""core.transcribe: route, backend factory, decoding, and orchestration."""
import wave
from pathlib import Path

import numpy as np
import pytest

from speechtotext.asr.faster_whisper import FasterWhisperBackend
from speechtotext.asr.whispercpp import WhisperCppBackend
from speechtotext.audio.io import AudioDecodeError
from speechtotext.core import transcribe as core
from speechtotext.core.transcribe import EngineInfo, Route, load_audio, make_backend, resolve_route


# --- route ---------------------------------------------------------------------------

def test_resolve_route_probes_and_delegates(monkeypatch):
    observed = {}
    monkeypatch.setattr(core.probe, "machine", lambda: "MAQUINA")

    def choose(m, model, **kw):
        observed.update(m=m, model=model, **kw)
        return Route("faster-whisper", "cpu", "int8", "")

    monkeypatch.setattr(core.probe, "choose_route", choose)
    assert resolve_route(engine="auto", device="cpu", compute_type="int8", model="small").device == "cpu"
    assert observed == {"m": "MAQUINA", "model": "small", "engine": "auto", "device": "cpu",
                        "compute_type": "int8"}


def test_the_default_route_on_the_test_machine_is_cpu_int8():
    # conftest sets a machine without a GPU: 'auto' resolves to faster-whisper on CPU int8.
    r = resolve_route()
    assert (r.engine, r.device, r.compute_type, r.reason) == (
        "faster-whisper", "cpu", "int8", "no usable GPU: CPU")
    assert r.eta_factor == round(1 / 1.27, 3) and r.estimated is True


# --- factory -------------------------------------------------------------------------

def test_make_backend_builds_the_type_and_config():
    fw = make_backend("faster-whisper", "large-v3", "cpu", "int8")
    assert isinstance(fw, FasterWhisperBackend)
    assert fw.model_id == "large-v3"
    assert (fw.config.device, fw.config.compute_type) == ("cpu", "int8")
    wc = make_backend("whispercpp", "small", "cuda", "q5_0")
    assert isinstance(wc, WhisperCppBackend)


def test_make_backend_distributes_threads_when_chunking():
    import os

    single = make_backend("faster-whisper", "large-v3", "cpu", "int8")
    assert (single.config.cpu_threads, single.config.num_workers) == (0, 1)
    four_jobs = make_backend("faster-whisper", "large-v3", "cpu", "int8", jobs=4)
    assert four_jobs.config.num_workers == 4
    assert four_jobs.config.cpu_threads == max(1, (os.cpu_count() or 1) // 4)


# --- decoding ------------------------------------------------------------------------

def _wav(path: Path, seconds: float, rate: int = 8000) -> Path:
    n = int(seconds * rate)
    t = np.arange(n) / rate
    pcm = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path


def test_load_audio_decodes_to_16k_mono_float32(tmp_path):
    samples = load_audio(_wav(tmp_path / "tono.wav", 0.5, rate=8000))
    assert samples.dtype == np.float32 and samples.ndim == 1
    assert abs(len(samples) - 8000) <= 160  # 0.5 s at 16 kHz, allowing for resampler tolerance
    assert 0.3 < float(np.abs(samples).max()) <= 1.0


def test_load_audio_rejects_non_audio(tmp_path):
    garbage = tmp_path / "x.wav"
    garbage.write_bytes(b"RIFF this is not a wav")
    with pytest.raises(AudioDecodeError):
        load_audio(garbage)


# --- types ---------------------------------------------------------------------------

def test_engine_info_omits_diarization_when_not_applicable():
    info = EngineInfo("faster-whisper", "faster-whisper 1.2", "large-v3", "int8", "cpu")
    assert info.to_dict() == {
        "name": "faster-whisper", "version": "faster-whisper 1.2", "model": "large-v3",
        "quant": "int8", "device": "cpu", "selection": "explicit",
    }
    assert EngineInfo("whispercpp", "v", "small", "q5_0", "cuda", diarization="segment").to_dict()["diarization"] == "segment"
    with_diarizer = EngineInfo("faster-whisper", "v", "large-v3", "int8", "cpu", diarization="word",
                               diarizer="nvidia/Nemotron-3-Diarization").to_dict()
    assert with_diarizer["diarizer"] == "nvidia/Nemotron-3-Diarization"


# --- orchestration -------------------------------------------------------------------

import json
import threading

from speechtotext.asr import AsrError, Caps
from speechtotext.asr.types import (
    NativeSignals, SegmentNativeSignals, TranscriptionResult, TranscriptionSegment,
    TranscriptionWord,
)
from speechtotext.core import chunked


class FakeBackend:
    """Fake engine: returns the given segments (times LOCAL to the chunk)."""

    def __init__(self, segments=((1.0, 2.0, " hola"),), *, language="es", probability=0.9,
                 boom=None, wrap=False, backend_id="faster-whisper",
                 caps=Caps("honored", "honored", "honored")):
        self.segments, self.language, self.probability, self.boom = segments, language, probability, boom
        self.wrap = wrap
        self.backend_id, self.caps = backend_id, caps
        self.quant = "q5_0" if backend_id == "whispercpp" else "int8"
        self.device = "cuda" if backend_id == "whispercpp" else "cpu"
        self.model_id, self.model_version, self.engine_version = "large-v3", "unpinned", "fake 1.0"
        self.calls, self.warmed = [], 0

    def warm(self):
        self.warmed += 1

    def transcribe(self, samples, request):
        self.calls.append((len(samples), request))
        if self.boom is not None:
            if self.wrap:
                raise AsrError("backend_failed", True, str(self.boom))
            raise self.boom
        segs = tuple(
            TranscriptionSegment(s, e, t, (TranscriptionWord(t, s, e, None),) if request.word_timestamps else (),
                                 SegmentNativeSignals(None, None, None))
            for s, e, t in self.segments
        )
        return TranscriptionResult(
            text="".join(t for _, _, t in self.segments).strip(), language=self.language,
            words=(), segments=segs, backend=self.backend_id, model=self.model_id,
            model_version=self.model_version, latency_ms=1,
            native_signals=NativeSignals(None, None, None, self.probability), warnings=(),
        )


def _zeros(seconds):
    return np.zeros(int(seconds * 16000), dtype=np.float32)


def test_one_chunk_produces_the_complete_transcript():
    backend = FakeBackend([(1.0, 2.0, " it's 8.30"), (2.0, 20.0, " and that")])
    t = core.transcribe(_zeros(30.0), backend=backend, language="es", chunk=False)
    assert backend.warmed == 1 and len(backend.calls) == 1
    assert t.duration == 30.0
    assert [s.text for s in t.segments] == [" it's 8:30", " and that"]  # normalize_hours
    assert (t.language, t.language_probability) == ("es", 1.0)  # forced -> 1.0
    assert t.speech_s == 19.0
    assert t.gaps == [[20.0, 30.0]]
    assert t.engine.to_dict() == {
        "name": "faster-whisper", "version": "fake 1.0", "model": "large-v3",
        "quant": "int8", "device": "cpu", "selection": "explicit",
    }
    assert t.request.vad is False and t.warnings == () and t.diarization is None


def test_auto_language_uses_the_detected_language_and_its_probability():
    t = core.transcribe(_zeros(5.0), backend=FakeBackend(language="en", probability=0.7), chunk=False)
    assert (t.language, t.language_probability) == ("en", 0.7)


def test_progress_uses_callback_and_decodes_once(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(core, "load_audio", lambda p: (calls.append(p), _zeros(10.0))[1])
    events = []
    t = core.transcribe(tmp_path / "a.wav", backend=FakeBackend(), chunk=False,
                        on_progress=events.append)
    assert calls == [tmp_path / "a.wav"]
    # Two decode events: before (indeterminate) and after, with done = total = duration,
    # which is what the CLI needs for the ETA
    assert [e.stage for e in events] == ["decode", "decode", "load", "transcribe"]
    assert (events[0].done, events[0].total) == (0, None)
    assert (events[1].done, events[1].total, events[1].detail) == (10.0, 10.0, "a.wav")
    assert events[-1].done == 1 and events[-1].total == 1 and "(nuevo)" in events[-1].detail
    assert t.duration == 10.0


def test_accepts_the_path_as_a_string(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "load_audio",
                        lambda p: _zeros(3.0) if isinstance(p, Path) else pytest.fail("raw str"))
    t = core.transcribe(str(tmp_path / "a.wav"), backend=FakeBackend(), chunk=False)
    assert t.duration == 3.0


def test_impossible_flags_stop_before_decoding(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "load_audio", lambda p: pytest.fail("decoded before validating"))
    with pytest.raises(ValueError, match="does not exist"):
        core.transcribe(tmp_path / "a.wav", engine="chatgpt")


def test_rejected_hotwords_stop_before_loading_the_model():
    backend = FakeBackend(backend_id="whispercpp", caps=Caps("rejected", "degraded", "degraded"))
    with pytest.raises(AsrError) as ei:
        core.transcribe(_zeros(5.0), backend=backend, hotwords=("Bézier",), chunk=False)
    assert ei.value.code == "unsupported_option" and ei.value.recoverable is False
    assert "--hotwords has no effect" in str(ei.value) and "faster-whisper" in str(ei.value)
    assert backend.warmed == 0 and backend.calls == []


def test_degraded_caps_warn_and_disable_the_knob():
    backend = FakeBackend(backend_id="whispercpp", caps=Caps("rejected", "degraded", "degraded"))
    t = core.transcribe(_zeros(5.0), backend=backend, vad=True, word_timestamps=True, chunk=False)
    assert any("has no VAD" in w for w in t.warnings)
    assert any("per-segment attribution" in w for w in t.warnings)
    assert t.request.vad is False and t.request.word_timestamps is False
    assert backend.calls[0][1].vad is False


def test_oom_is_translated_with_engine_specific_advice():
    with pytest.raises(AsrError) as ei:
        core.transcribe(_zeros(5.0), chunk=False,
                        backend=FakeBackend(boom=RuntimeError("mkl_malloc: failed to allocate memory")))
    assert ei.value.code == "out_of_memory" and "-m medium" in str(ei.value) and "VRAM" not in str(ei.value)
    with pytest.raises(AsrError) as ei:
        core.transcribe(_zeros(5.0), chunk=False,
                        backend=FakeBackend(backend_id="whispercpp",
                                            boom=RuntimeError("ggml_cuda: failed to allocate 1.28 GB")))
    assert "VRAM" in str(ei.value) and "--engine faster-whisper" in str(ei.value) and "medium" not in str(ei.value)


def test_unrelated_runtimeerror_propagates_raw_for_one_chunk():
    with pytest.raises(RuntimeError, match="^cable suelto$"):
        core.transcribe(_zeros(5.0), chunk=False, backend=FakeBackend(boom=RuntimeError("cable suelto")))


def test_cancellation_before_starting_skips_the_backend():
    backend = FakeBackend()
    stop_event = threading.Event()
    stop_event.set()
    with pytest.raises(AsrError) as ei:
        core.transcribe(_zeros(5.0), backend=backend, cancel=stop_event, chunk=False)
    assert ei.value.code == "cancelled" and backend.calls == []


def test_a_provided_backend_prevents_building_another_one(monkeypatch):
    monkeypatch.setattr(core, "make_backend", lambda *a, **k: pytest.fail("must not construct"))
    core.transcribe(_zeros(5.0), backend=FakeBackend(), chunk=False)


def test_the_core_clamps_jobs_for_whispercpp_and_distributes_them_for_faster(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    monkeypatch.setattr(core, "load_audio", lambda p: _zeros(1300.0))
    observed = []

    def backend_factory(engine, model, device, compute_type, jobs=1):
        observed.append((engine, jobs))
        return FakeBackend(backend_id=engine, caps=Caps("rejected", "degraded", "degraded")
                           if engine == "whispercpp" else Caps("honored", "honored", "honored"))

    monkeypatch.setattr(core, "make_backend", backend_factory)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    core.transcribe(audio, engine="whispercpp", model="small", chunk=True, jobs=4)   # 3 fixed chunks
    core.transcribe(audio, engine="faster-whisper", chunk=True, jobs=4)
    assert observed == [("whispercpp", 1), ("faster-whisper", 3)]


def test_multiple_chunks_reassemble_in_order_with_global_times(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    audio = tmp_path / "largo.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(core, "load_audio", lambda p: _zeros(1200.0))
    monkeypatch.setattr(core, "plan_chunks", lambda path, dur: [(0.0, 600.0), (600.0, 1200.0)])
    backend = FakeBackend([(1.0, 2.0, " t")])
    events = []
    t = core.transcribe(audio, backend=backend, chunk=True, jobs=2, on_progress=events.append)
    assert [(s.start, s.end) for s in t.segments] == [(1.0, 2.0), (601.0, 602.0)]
    assert sorted(n for n, _ in backend.calls) == [600 * 16000, 600 * 16000]
    details = [e.detail for e in events if e.stage == "transcribe"]
    assert len(details) == 2 and all("(nuevo)" in d for d in details)
    assert t.language_probability is None  # Multiple chunks: none measured a single probability


def test_checkpoint_avoids_the_engine_and_is_clipped_when_read(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(core, "load_audio", lambda p: _zeros(600.0))
    monkeypatch.setattr(core, "plan_chunks", lambda path, dur: [(0.0, 600.0)])
    backend = FakeBackend()
    request, _ = core._apply_caps(backend, core._request(language="auto", vad=False, hotwords=(),
                                                         beam_size=5, word_timestamps=False))
    p = chunked.chunk_path(core._identity(audio, backend, request), 0.0, 600.0)
    p.write_text(json.dumps({"language": "es", "segments": [
        {"start": 1.0, "end": 2.0, "text": " cache"},
        {"start": 599.9, "end": 629.9, "text": " Gracias por ver el video."},  # Phantom over the padding
    ]}), encoding="utf-8")
    events = []
    t = core.transcribe(audio, backend=backend, chunk=True, on_progress=events.append)
    assert backend.calls == [] and backend.warmed == 1
    assert [s.text for s in t.segments] == [" cache"]
    assert any("(cache)" in e.detail for e in events)
    assert t.language == "es"


def test_the_new_chunk_leaves_a_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(core, "load_audio", lambda p: _zeros(600.0))
    monkeypatch.setattr(core, "plan_chunks", lambda path, dur: [(0.0, 600.0)])
    core.transcribe(audio, backend=FakeBackend(), chunk=True)
    written_files = list((tmp_path / "chunks").glob("*.json"))
    assert len(written_files) == 1
    assert json.loads(written_files[0].read_text(encoding="utf-8"))["segments"][0]["text"] == " hola"


def test_samples_without_a_file_are_chunked_without_a_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    backend = FakeBackend()
    t = core.transcribe(_zeros(1300.0), backend=backend, chunk=True)
    assert len(backend.calls) == 3  # pick_cuts without silences: 600 + 600 + 100
    assert not (tmp_path / "chunks").exists()
    assert len(t.segments) == 3


def test_the_first_failure_cancels_pending_chunks_and_names_the_chunk(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    monkeypatch.setattr(core, "load_audio", lambda p: _zeros(1200.0))
    monkeypatch.setattr(core, "plan_chunks", lambda path, dur: [(0.0, 600.0), (600.0, 1200.0)])
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    with pytest.raises(AsrError) as ei:
        core.transcribe(audio, backend=FakeBackend(boom=RuntimeError("cable")), chunk=True, jobs=1)
    assert ei.value.code == "backend_failed" and "failed on chunk 1/2" in str(ei.value)


def test_oom_wrapped_by_the_real_backend_is_also_translated():
    boom = RuntimeError("mkl_malloc: failed to allocate memory")
    with pytest.raises(AsrError) as ei:
        core.transcribe(_zeros(5.0), chunk=False, backend=FakeBackend(boom=boom, wrap=True))
    assert ei.value.code == "out_of_memory" and "-m medium" in str(ei.value)


def test_a_wrapped_failure_across_multiple_chunks_names_the_chunk(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    monkeypatch.setattr(core, "load_audio", lambda p: _zeros(1200.0))
    monkeypatch.setattr(core, "plan_chunks", lambda path, dur: [(0.0, 600.0), (600.0, 1200.0)])
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    with pytest.raises(AsrError) as ei:
        core.transcribe(audio, backend=FakeBackend(boom=RuntimeError("cable"), wrap=True), chunk=True, jobs=1)
    assert ei.value.code == "backend_failed" and "failed on chunk 1/2" in str(ei.value)


def test_wrapped_cancellation_is_not_reclassified():
    stop_event = threading.Event()
    stop_event.set()
    with pytest.raises(AsrError) as ei:
        core.transcribe(_zeros(5.0), backend=FakeBackend(), cancel=stop_event, chunk=False)
    assert ei.value.code == "cancelled"


def test_diarization_uses_the_same_samples_and_measures_beforehand(monkeypatch):
    from speechtotext.speakers import diarization, registry

    observed = {}

    def fake_diarize(samples, sample_rate, num_speakers=None):
        observed.update(n=len(samples), sr=sample_rate, k=num_speakers)
        return [(0.0, 30.0, "SPEAKER_00")], {"SPEAKER_00": np.array([1.0, 0.0])}

    monkeypatch.setattr(diarization, "diarize", fake_diarize)
    monkeypatch.setattr(registry, "get_embeddings", lambda model: {"Alice": np.array([1.0, 0.0])})
    backend = FakeBackend([(0.0, 30.0, " Gracias.")])
    t = core.transcribe(_zeros(30.0), backend=backend, diarize=True, speakers=1, chunk=False)
    assert observed == {"n": 30 * 16000, "sr": 16000, "k": 1}
    assert backend.calls[0][1].word_timestamps is True  # Diarization requests words
    assert t.segments[0].speaker == "Alice"
    assert t.segments[0].src_dur == 30.0            # Span emitted by ASR, for is_suspect
    assert t.speech_s == 30.0 and t.gaps == []      # Measured BEFORE diarization
    assert t.engine.diarization == "word"
    assert t.engine.diarizer == diarization.EMBEDDING_MODEL   # pyannote stays the default
    assert t.diarization.speakers == 1 and t.diarization.identified == 1 and t.diarization.auto is False


def test_diarization_without_the_extra_is_an_error_with_a_code(monkeypatch):
    from speechtotext.speakers import diarization

    def without_pyannote(samples, sample_rate, num_speakers=None):
        raise ImportError("No module named 'pyannote'")

    monkeypatch.setattr(diarization, "diarize", without_pyannote)
    with pytest.raises(AsrError) as ei:
        core.transcribe(_zeros(5.0), backend=FakeBackend(), diarize=True, chunk=False)
    assert ei.value.code == "diarize_unavailable" and "[diarize]" in str(ei.value)


# --- the second diarizer: Nemotron ------------------------------------------------------

def _nemotron(monkeypatch, turns, *, enrolled=None, missing=None):
    """Stub the model boundary (nemotron.diarize, its dependency check, the voice registry);
    assignment, naming and the report run for real."""
    from speechtotext.speakers import nemotron, registry

    observed = {}

    def fake_diarize(samples, sample_rate):
        observed.update(n=len(samples), sr=sample_rate)
        return turns

    monkeypatch.setattr(nemotron, "diarize", fake_diarize)
    monkeypatch.setattr(nemotron, "missing", lambda: missing)
    monkeypatch.setattr(registry, "get_embeddings", lambda model: dict(enrolled or {}))
    return observed


def test_nemotron_diarizes_the_same_samples_and_the_json_says_which_diarizer_ran(monkeypatch):
    observed = _nemotron(monkeypatch, [(0.0, 30.0, "speaker_0")])
    t = core.transcribe(_zeros(30.0), backend=FakeBackend([(0.0, 30.0, " Gracias.")]),
                        diarize=True, diarizer="nemotron", chunk=False)
    assert observed == {"n": 30 * 16000, "sr": 16000}
    assert t.segments[0].speaker == "Speaker 1"
    assert t.engine.to_dict()["diarizer"] == "nvidia/Nemotron-3-Diarization"
    assert t.diarization.speakers == 1 and t.diarization.auto is True
    assert t.warnings == ()


def test_nemotron_refuses_a_speaker_count_before_any_work(monkeypatch):
    # The model takes no count: honoring it is impossible and dropping it quietly would let
    # the user believe it applied. The run does not start.
    _nemotron(monkeypatch, [(0.0, 30.0, "speaker_0")])
    backend = FakeBackend([(0.0, 30.0, " x")])
    with pytest.raises(AsrError) as ei:
        core.transcribe(_zeros(30.0), backend=backend, diarize=True, diarizer="nemotron",
                        speakers=2, chunk=False)
    assert ei.value.code == "unsupported_option" and "--speakers" in str(ei.value)
    assert backend.calls == [] and backend.warmed == 0


def test_nemotron_leaves_enrolled_voices_uncompared_and_says_so(monkeypatch):
    voices = {"Alice": np.array([1.0, 0.0]), "Bob": np.array([0.0, 1.0])}
    _nemotron(monkeypatch, [(0.0, 30.0, "speaker_0")], enrolled=voices)
    backend = FakeBackend([(0.0, 30.0, " x")])
    t = core.transcribe(_zeros(30.0), backend=backend, diarize=True, diarizer="nemotron", chunk=False)
    assert t.segments[0].speaker == "Speaker 1"                # no name was put on anyone
    assert (t.diarization.enrolled, t.diarization.identified) == (0, 0)
    assert len([w for w in t.warnings if "2 enrolled voices" in w]) == 1

    # --no-identify asked for no names: there is nothing to warn about.
    t = core.transcribe(_zeros(30.0), backend=backend, diarize=True, diarizer="nemotron",
                        identify=False, chunk=False)
    assert t.warnings == ()


def test_a_missing_nemotron_stops_before_the_transcription_starts(monkeypatch):
    # The check exists so that a missing dependency costs milliseconds, not an hour of ASR.
    _nemotron(monkeypatch, [], missing="transformers is not installed; pip install ...")
    backend = FakeBackend()
    with pytest.raises(AsrError) as ei:
        core.transcribe(_zeros(5.0), backend=backend, diarize=True, diarizer="nemotron", chunk=False)
    assert ei.value.code == "diarize_unavailable" and "transformers is not installed" in str(ei.value)
    assert backend.calls == [] and backend.warmed == 0


def test_the_nemotron_check_only_runs_when_diarizing(monkeypatch):
    _nemotron(monkeypatch, [], missing="transformers is not installed; pip install ...")
    t = core.transcribe(_zeros(5.0), backend=FakeBackend(), diarizer="nemotron", chunk=False)
    assert t.diarization is None and "diarizer" not in t.engine.to_dict()


def test_an_unknown_diarizer_is_rejected_before_decoding():
    backend = FakeBackend()
    with pytest.raises(ValueError, match="pyannote"):
        core.transcribe(_zeros(5.0), backend=backend, diarize=True, diarizer="whisperx", chunk=False)
    assert backend.calls == []


# --- engine warnings and mid-pool cancellation ------------------------------------------

def test_engine_warnings_reach_the_transcript_without_being_duplicated(tmp_path, monkeypatch):
    from dataclasses import replace as dc_replace

    class WarningBackend(FakeBackend):
        def transcribe(self, samples, request):
            return dc_replace(super().transcribe(samples, request), warnings=("empty_transcript",))

    t = core.transcribe(_zeros(5.0), backend=WarningBackend(), chunk=False)
    assert t.warnings == ("faster-whisper: empty_transcript",)

    monkeypatch.setattr(core, "load_audio", lambda p: _zeros(1200.0))
    monkeypatch.setattr(core, "plan_chunks", lambda path, dur: [(0.0, 600.0), (600.0, 1200.0)])
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    t = core.transcribe(audio, backend=WarningBackend(), chunk=True, jobs=1)
    assert t.warnings == ("faster-whisper: empty_transcript",)   # Two chunks, one warning


def test_caps_warnings_come_before_engine_warnings():
    from dataclasses import replace as dc_replace

    class WarningBackend(FakeBackend):
        def transcribe(self, samples, request):
            return dc_replace(super().transcribe(samples, request), warnings=("empty_transcript",))

    backend = WarningBackend(backend_id="whispercpp", caps=Caps("rejected", "degraded", "degraded"))
    t = core.transcribe(_zeros(5.0), backend=backend, vad=True, chunk=False)
    assert "has no VAD" in t.warnings[0] and t.warnings[-1] == "whispercpp: empty_transcript"


def test_provided_route_does_not_probe_again_and_marks_selection(monkeypatch):
    monkeypatch.setattr(core.probe, "machine", lambda: pytest.fail("probed twice"))
    observed = []
    monkeypatch.setattr(core, "make_backend", lambda *a, **k: (observed.append(a), FakeBackend())[1])
    route = Route("faster-whisper", "cpu", "int8", "")
    t = core.transcribe(_zeros(5.0), route=route, chunk=False)                     # engine default: auto
    assert observed[0][:4] == ("faster-whisper", "large-v3", "cpu", "int8")
    assert t.engine.selection == "auto"
    t = core.transcribe(_zeros(5.0), route=route, engine="faster-whisper", chunk=False)
    assert t.engine.selection == "explicit"


def test_selection_is_auto_only_when_the_probe_chose_the_route(monkeypatch):
    monkeypatch.setattr(core, "make_backend", lambda *a, **k: FakeBackend())
    assert core.transcribe(_zeros(5.0), chunk=False).engine.selection == "auto"
    assert core.transcribe(_zeros(5.0), engine="faster-whisper", chunk=False).engine.selection == "explicit"
    assert core.transcribe(_zeros(5.0), backend=FakeBackend(), chunk=False).engine.selection == "explicit"


def test_mid_pool_cancellation_does_not_launch_more_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "load_audio", lambda p: _zeros(1800.0))
    monkeypatch.setattr(core, "plan_chunks",
                        lambda path, dur: [(0.0, 600.0), (600.0, 1200.0), (1200.0, 1800.0)])
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    stop_event = threading.Event()

    class CancellingBackend(FakeBackend):
        def transcribe(self, samples, request):
            stop_event.set()     # The first chunk requests a stop from within
            return super().transcribe(samples, request)

    backend = CancellingBackend()
    with pytest.raises(AsrError) as ei:
        core.transcribe(audio, backend=backend, cancel=stop_event, chunk=True, jobs=1)
    assert ei.value.code == "cancelled"
    assert len(backend.calls) == 1     # Chunks 2 and 3 never reached the engine
