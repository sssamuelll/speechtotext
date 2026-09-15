import logging
import re
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from speechtotext.cli.app import app
from speechtotext.speakers import registry

runner = CliRunner()


def _seg(start, end, text="hi there"):
    return SimpleNamespace(start=start, end=end, text=text)


def _info(duration, language="es", language_probability=1.0):
    return SimpleNamespace(
        duration=duration, language=language, language_probability=language_probability
    )


def _result(segments, info):
    """Build a TranscriptionResult from the tests' fake SimpleNamespace segments:
    raw faster-whisper dialect (no_speech_prob) with optional words."""
    from speechtotext.asr.types import (
        NativeSignals, SegmentNativeSignals, TranscriptionResult, TranscriptionSegment,
        TranscriptionWord,
    )

    segs = []
    for s in segments:
        words = tuple(TranscriptionWord(w.word, w.start, w.end, None)
                      for w in (getattr(s, "words", None) or ()))
        segs.append(TranscriptionSegment(s.start, s.end, s.text, words, SegmentNativeSignals(
            getattr(s, "no_speech_prob", None), getattr(s, "avg_logprob", None),
            getattr(s, "compression_ratio", None))))
    return TranscriptionResult(
        text="".join(s.text for s in segments).strip(), language=info.language, words=(),
        segments=tuple(segs), backend="fake", model="fake", model_version="1", latency_ms=1,
        native_signals=NativeSignals(None, None, None, info.language_probability), warnings=(),
    )


def _fake_transcribe(monkeypatch, tmp_path, segments, info, boom=None, calls=None):
    """Cut the path off just before the engine: the audio is never opened and the
    backend is fake. calls is an optional list recording each (samples, request)."""
    from speechtotext.asr import Caps
    from speechtotext.core import transcribe as core_transcribe

    audio = tmp_path / "charla.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(core_transcribe, "load_audio",
                        lambda p: np.zeros(int(info.duration * 16000), dtype=np.float32))
    monkeypatch.setattr(core_transcribe, "should_chunk", lambda d, c: False)

    class FakeBackend:
        def __init__(self, engine, model, device, compute_type, jobs=1):
            self.backend_id, self.model_id, self.device, self.quant = engine, model, device, compute_type
            self.model_version = "1"
            self.engine_version = "whisper.cpp v1.9.1" if engine == "whispercpp" else "faster-whisper 1.2.0"
            self.caps = (Caps("rejected", "degraded", "degraded") if engine == "whispercpp"
                         else Caps("honored", "honored", "honored"))

        def warm(self):
            pass

        def transcribe(self, samples, request):
            if calls is not None:
                calls.append((samples, request))
            if boom is not None:
                raise boom
            return _result(segments, info)

    monkeypatch.setattr(core_transcribe, "make_backend", FakeBackend)
    return audio


def _invoke(audio, tmp_path, *extra, catch=True):
    return runner.invoke(
        app,
        ["transcribe", str(audio), "-f", "txt", "-o", str(tmp_path / "out")] + list(extra),
        catch_exceptions=catch,
    )


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _flattened(output: str) -> str:
    """Normalize Rich's rendered output so assertions can be made against it.

    Two things are outside our control: it wraps at 80 columns under CliRunner, and
    with color it colors the first hyphen of a flag separately from the rest —
    `'\\x1b[1m-\\x1b[0m\\x1b[1m-threshold\\x1b[0m'` — so `--threshold` no longer exists
    as a substring. Local runs without color and CI with color, so asserting against
    the raw text passes here and fails there. The rendered form is not anyone's
    contract, so clean it before inspecting it.
    """
    return " ".join(_ANSI.sub("", output).split())


def test_voices_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    result = runner.invoke(app, ["voices"])
    assert result.exit_code == 0
    assert "No enrolled voices" in result.stdout


def test_forget_missing_returns_error(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    result = runner.invoke(app, ["forget", "Nadie"])
    assert result.exit_code == 1


def test_voices_lists_enrolled(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    registry.enroll("Alice", np.array([1.0, 2.0], dtype=np.float32), seconds=10.0, model="m")
    result = runner.invoke(app, ["voices"])
    assert "Alice" in result.stdout


def test_transcribe_still_registered():
    # The transcribe command still exists as a named subcommand.
    result = runner.invoke(app, ["transcribe", "--help"])
    assert result.exit_code == 0
    assert "diarize" in result.stdout


# --- 1.1 · coverage in the summary line ---------------------------------------------


def test_the_summary_lists_the_gaps(tmp_path, monkeypatch):
    # The run that uncovered the issue: 2206 s of audio, text at 0-300 and 600-850.
    # The "try --no-vad" advice only applies with VAD enabled; it is no longer the default.
    segs = [_seg(0.0, 300.0), _seg(600.0, 850.0)]
    audio = _fake_transcribe(monkeypatch, tmp_path, segs, _info(2206.0))
    result = _invoke(audio, tmp_path, "--vad")
    assert result.exit_code == 0
    assert "25%" in result.stdout
    assert "2 gaps without text: 05:00-10:00 (300 s), 14:10-36:46 (1356 s)" in result.stdout
    assert "try --no-vad" in result.stdout


def test_high_coverage_with_a_real_gap_lists_it(tmp_path, monkeypatch):
    # 90%: the number looks healthy, yet there are 100 s without text. This is THE band
    # identified by the plan (§1.3: "70-90%, exactly where costly content is lost") and
    # the case that the removed threshold never triggered. Under the old code this run
    # printed "90%" and nothing else; if this line disappears, Phase 1 loses its purpose.
    segs = [_seg(0.0, 900.0)]
    audio = _fake_transcribe(monkeypatch, tmp_path, segs, _info(1000.0))
    result = _invoke(audio, tmp_path)
    assert result.exit_code == 0
    assert "90%" in result.stdout
    assert "1 gap without text: 15:00-16:40 (100 s)" in result.stdout


def test_the_summary_explicitly_says_when_there_are_no_gaps(tmp_path, monkeypatch):
    # One contiguous segment: the absence of gaps is stated, not left unmentioned.
    segs = [_seg(0.0, 900.0)]
    audio = _fake_transcribe(monkeypatch, tmp_path, segs, _info(900.0))
    result = _invoke(audio, tmp_path)
    assert result.exit_code == 0
    assert "100%" in result.stdout
    assert "no gaps of 5 s or more" in result.stdout
    assert "--no-vad" not in result.stdout


def test_no_vad_is_not_suggested_when_it_is_already_disabled(tmp_path, monkeypatch):
    # Runs 2, 3, and 5 of the real case already used --no-vad: repeating the advice
    # there is guaranteed noise. The gap line remains; the suggestion does not.
    segs = [_seg(0.0, 300.0)]
    audio = _fake_transcribe(monkeypatch, tmp_path, segs, _info(2206.0))
    result = _invoke(audio, tmp_path, "--no-vad")
    assert result.exit_code == 0
    assert "1 gap without text: 05:00-36:46 (1906 s)" in result.stdout
    assert "--no-vad" not in result.stdout


# --- 5.1.3 · one quantity, calculated before diarization -----------------------------


def _diarize_recompressing(monkeypatch, output):
    """Mirror real diarization: recompress each span to the extent of its words.
    Measuring afterward publishes a different number under the same name — C-13."""
    from speechtotext.core import transcribe as core_transcribe
    from speechtotext.core.segments import LabeledSegment
    from speechtotext.core.transcribe import DiarizationReport

    labeled = [LabeledSegment(s.start, s.end, s.text) for s in output]
    monkeypatch.setattr(core_transcribe, "_diarize",
                        lambda samples, segs, *a: (labeled, DiarizationReport(1, 0, 0, 0, None, True)))


def test_json_measures_what_asr_emitted_not_the_diarized_output(tmp_path, monkeypatch):
    # The wiring that fixes C-13. Without this test the regression stays green: moving
    # `cov = sum(...)` below _diarize is enough, and the other 604 tests still pass.
    import json

    _diarize_recompressing(monkeypatch, [_seg(0.0, 1.0), _seg(600.0, 601.0)])
    segs = [_seg(0.0, 300.0), _seg(600.0, 850.0)]
    audio = _fake_transcribe(monkeypatch, tmp_path, segs, _info(2206.0))
    result = _invoke(audio, tmp_path, "-f", "json", "--diarize")
    assert result.exit_code == 0
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert payload["speech_s"] == 550.0  # 300 + 250; post-diarization would yield 2.0
    assert payload["gaps"] == [[300.0, 600.0], [850.0, 2206.0]]
    # And this is exactly what the console said: one quantity, two channels.
    assert "2 gaps without text: 05:00-10:00 (300 s), 14:10-36:46 (1356 s)" in result.stdout


# --- 5.2.1 · Typer validates range contracts, not the --help prose -------------------


def test_an_out_of_range_threshold_exits_with_code_2(tmp_path, monkeypatch):
    # With 2.0, voice identification was silently disabled (assign_names breaks the
    # loop on the first candidate and returns {}); with -1, it named everything.
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "--threshold", "2.0")
    assert result.exit_code == 2
    assert "--threshold" in _flattened(result.stderr)


def test_zero_speakers_exits_with_code_2(tmp_path, monkeypatch):
    # 0 is falsy, and speakers/diarization.py reinterpreted it as "auto" without warning:
    # exactly the silent substitution prohibited by the capabilities contract.
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "--speakers", "0")
    assert result.exit_code == 2
    assert "--speakers" in _flattened(result.stderr)


def test_zero_beam_size_exits_with_code_2(tmp_path, monkeypatch):
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "--beam-size", "0")
    assert result.exit_code == 2
    assert "--beam-size" in _flattened(result.stderr)


def test_find_with_an_out_of_range_threshold_exits_with_code_2(tmp_path, monkeypatch):
    # find re-exposes the same flags: fixing only transcribe left an opening.
    from speechtotext.core import finder

    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    audio = tmp_path / "programa.wav"
    audio.write_bytes(b"x" * 100)
    # If validation is missing, the callback runs: make it finish quickly with exit 0.
    monkeypatch.setattr(finder, "load_or_build_index", lambda *a, **k: ([], True))
    result = runner.invoke(app, ["find", str(audio), "sismica", "--threshold", "2.0"])
    assert result.exit_code == 2
    assert "--threshold" in _flattened(result.stderr)


def test_find_with_zero_speakers_exits_with_code_2(tmp_path, monkeypatch):
    from speechtotext.core import finder

    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    audio = tmp_path / "programa.wav"
    audio.write_bytes(b"x" * 100)
    monkeypatch.setattr(finder, "load_or_build_index", lambda *a, **k: ([], True))
    result = runner.invoke(app, ["find", str(audio), "sismica", "--speakers", "0"])
    assert result.exit_code == 2
    assert "--speakers" in _flattened(result.stderr)


# --- 5.2.3 · the [?] marker survives --diarize ---------------------------------------


def test_diarize_marks_suspicious_text_the_same_as_without_diarization(tmp_path, monkeypatch):
    # The plan's canonical case: a 30 s ASR segment with a single 1 s word. Without
    # --diarize the gate measures the span's 30 s; with --diarize the span is recompressed
    # to 1 s and only src_dur preserves the marker. Real end-to-end route: diarize and the
    # voice registry are stubbed; assign_segments/apply_names/post-processing are real.
    from speechtotext.speakers import diarization

    word = SimpleNamespace(start=0.4, end=1.4, word=" Gracias.")
    seg = SimpleNamespace(start=0.0, end=30.0, text=" Gracias.", words=[word])
    audio = _fake_transcribe(monkeypatch, tmp_path, [seg], _info(30.0))
    monkeypatch.setattr(
        diarization, "diarize",
        lambda samples, sample_rate, num_speakers=None: (
            [(0.0, 30.0, "SPEAKER_00")], {"SPEAKER_00": np.array([1.0])}
        ),
    )
    monkeypatch.setattr(registry, "get_embeddings", lambda model: {})

    without_diarization = _invoke(audio, tmp_path)
    assert without_diarization.exit_code == 0
    assert "[?] Gracias." in (tmp_path / "out.txt").read_text(encoding="utf-8")

    with_diarization = _invoke(audio, tmp_path, "--diarize")
    assert with_diarization.exit_code == 0
    text = (tmp_path / "out.txt").read_text(encoding="utf-8")
    assert "[?] Gracias." in text  # With the span recompressed to 1 s, only src_dur marks it.
    assert "Speaker 1" in text


# --- 5.2.4 · diarization quality report ---------------------------------------------


def _fake_diarization(monkeypatch, tmp_path, turns, clusters, enrolled):
    """Stub the model boundary: diarize and the voice registry. The rest of the route
    (assign_segments, assign_names, apply_names, and the report) runs for real."""
    from speechtotext.speakers import diarization

    monkeypatch.setattr(
        diarization, "diarize", lambda samples, sample_rate, num_speakers=None: (turns, clusters)
    )
    monkeypatch.setattr(
        registry, "get_embeddings",
        # Vector space matters: if the CLI requests another model's embedding space,
        # there are no voices.
        lambda model: enrolled if model == diarization.EMBEDDING_MODEL else {},
    )


def test_the_diarization_report_with_no_enrolled_voices(tmp_path, monkeypatch):
    # Two clusters and no enrolled voices: the count appears and the score line does not.
    clusters = {"SPEAKER_00": np.array([1.0, 0.0]), "SPEAKER_01": np.array([0.0, 1.0])}
    turns = [(0.0, 5.0, "SPEAKER_00"), (5.0, 9.0, "SPEAKER_01")]
    _fake_diarization(monkeypatch, tmp_path, turns, clusters, {})
    segs = [_seg(0.0, 5.0), _seg(5.0, 9.0)]
    audio = _fake_transcribe(monkeypatch, tmp_path, segs, _info(9.0))
    result = _invoke(audio, tmp_path, "--diarize")
    assert result.exit_code == 0
    output = _flattened(result.stdout)
    assert "2 speakers · 0% unattributed" in output
    assert "voices identified" not in output  # Without a registry, the clause does not apply.
    assert "best score" not in output


def test_the_diarization_report_shows_the_best_score_below_threshold(tmp_path, monkeypatch):
    # An enrolled voice that does not reach the threshold: show the best score and the
    # threshold, or identification fails in the dark (the plan's real case: 0.38 < 0.50).
    clusters = {"SPEAKER_00": np.array([1.0, 3.0]), "SPEAKER_01": np.array([0.0, 1.0])}
    turns = [(0.0, 5.0, "SPEAKER_00"), (5.0, 9.0, "SPEAKER_01")]
    enrolled = {"Alice": np.array([1.0, 0.0])}  # Cosine with SPEAKER_00: 1/sqrt(10) = 0.32.
    _fake_diarization(monkeypatch, tmp_path, turns, clusters, enrolled)
    segs = [_seg(0.0, 5.0), _seg(5.0, 9.0)]
    audio = _fake_transcribe(monkeypatch, tmp_path, segs, _info(9.0))
    result = _invoke(audio, tmp_path, "--diarize")
    assert result.exit_code == 0
    output = _flattened(result.stdout)
    assert "0 of 1 voices identified" in output
    assert "best score 0.32 < 0.50" in output


def test_the_diarization_report_suggests_speakers_when_the_automatic_count_is_too_high(
    tmp_path, monkeypatch
):
    clusters = {f"SPEAKER_{i:02d}": np.array([1.0, 0.0]) for i in range(6)}
    turns = [(float(i), float(i + 1), f"SPEAKER_{i:02d}") for i in range(6)]
    _fake_diarization(monkeypatch, tmp_path, turns, clusters, {})
    segs = [_seg(float(i), float(i + 1)) for i in range(6)]
    audio = _fake_transcribe(monkeypatch, tmp_path, segs, _info(6.0))

    result = _invoke(audio, tmp_path, "--diarize")
    assert result.exit_code == 0
    assert "6 speakers" in _flattened(result.stdout)
    assert "--speakers N" in result.stdout

    # The suggestion does not apply when the user specifies the number.
    result = _invoke(audio, tmp_path, "--diarize", "--speakers", "6")
    assert result.exit_code == 0
    assert "--speakers N" not in result.stdout


# --- 1.6 · measured vs forced language ---------------------------------------------


def test_a_forced_language_does_not_report_probability(tmp_path, monkeypatch):
    # Explicit -l: nothing was detected; the user's choice was honored (default is auto).
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "-l", "es")
    assert "(forced)" in result.stdout
    assert "prob=" not in result.stdout
    assert "Language detected" not in result.stdout


def test_auto_language_reports_probability(tmp_path, monkeypatch):
    info = _info(10.0, language="en", language_probability=0.87)
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], info)
    result = _invoke(audio, tmp_path, "-l", "auto")
    assert "Language detected" in result.stdout
    assert "prob=0.87" in result.stdout


def test_auto_language_omits_an_unknown_probability(tmp_path, monkeypatch):
    # The chunked route under --language auto dropped the real probability: omit, do not invent.
    info = _info(10.0, language="en", language_probability=None)
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], info)
    result = _invoke(audio, tmp_path, "-l", "auto")
    assert "Language detected" in result.stdout
    assert "prob=" not in result.stdout


# --- 1.8 · faster-whisper logger ----------------------------------------------------


def test_the_faster_whisper_logger_is_at_info():
    # Without this, "VAD filter removed X of audio" never appears and Q1 cannot be closed.
    assert logging.getLogger("faster_whisper").level == logging.INFO


# --- 1.9 · memory guard -------------------------------------------------------------


def test_oom_exits_with_an_actionable_message(tmp_path, monkeypatch):
    boom = RuntimeError("mkl_malloc: failed to allocate memory")
    audio = _fake_transcribe(monkeypatch, tmp_path, [], _info(2206.0), boom=boom)
    result = _invoke(audio, tmp_path)
    assert result.exit_code == 1
    assert "memory" in result.stdout
    assert "medium" in result.stdout


def test_an_unrelated_runtimeerror_propagates(tmp_path, monkeypatch):
    # Swallowing every RuntimeError would turn a bug into an incorrect RAM message.
    boom = RuntimeError("the model does not exist")
    audio = _fake_transcribe(monkeypatch, tmp_path, [], _info(2206.0), boom=boom)
    with pytest.raises(RuntimeError, match="the model does not exist"):
        _invoke(audio, tmp_path, catch=False)


# --- multi-engine · CAPS, clamping, and the engine declared in output ---------------


def test_hotwords_with_whispercpp_are_rejected_before_construction(tmp_path, monkeypatch):
    # --prompt is inert under -mc 0 (measured 2026-07-27): reject, do not degrade.
    calls = []
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0), calls=calls)
    result = _invoke(audio, tmp_path, "--engine", "whispercpp", "--hotwords", "Bézier")
    assert result.exit_code == 2
    assert "--hotwords has no effect" in result.stdout
    assert "faster-whisper" in result.stdout
    assert calls == []  # It never reached the backend: neither model nor cache.


def test_an_invalid_engine_fails(tmp_path, monkeypatch):
    calls = []
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0), calls=calls)
    result = _invoke(audio, tmp_path, "--engine", "chatgpt")
    assert result.exit_code == 2
    assert "does not exist" in result.stderr  # Our BadParameter, not "no such option."
    assert calls == []


def test_an_unmappable_compute_type_with_whispercpp_is_rejected(tmp_path, monkeypatch):
    # fp16 on the 980 pages under WDDM (0.53x real time): reject with the measurement.
    calls = []
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0), calls=calls)
    result = _invoke(audio, tmp_path, "--engine", "whispercpp", "--compute-type", "float16")
    assert result.exit_code == 2
    assert "WDDM paging" in result.stderr  # The rejection cites the measurement, not a generic.
    assert calls == []


def test_vad_warning_with_whispercpp(tmp_path, monkeypatch):
    # Explicit --vad: the default is already False, so there is nothing to degrade otherwise.
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "--engine", "whispercpp", "--vad")
    assert result.exit_code == 0
    assert "has no VAD" in result.stdout


def test_whispercpp_does_not_suggest_no_vad(tmp_path, monkeypatch):
    # With gaps in the timeline, the "try --no-vad" advice is absurd under whispercpp:
    # the engine has no VAD to disable. The gap line remains; the suggestion does not.
    segs = [_seg(0.0, 300.0)]
    audio = _fake_transcribe(monkeypatch, tmp_path, segs, _info(2206.0))
    result = _invoke(audio, tmp_path, "--engine", "whispercpp")
    assert result.exit_code == 0
    assert "1 gap without text: 05:00-36:46 (1906 s)" in result.stdout
    assert "try --no-vad" not in result.stdout


def test_diarize_warning_with_whispercpp(tmp_path, monkeypatch):
    from speechtotext.core import transcribe as core_transcribe
    from speechtotext.core.transcribe import DiarizationReport

    # Real diarization needs pyannote; only the preceding warning matters here.
    monkeypatch.setattr(core_transcribe, "_diarize",
                        lambda samples, segs, *a: ([], DiarizationReport(0, 0, 0, 0, None, True)))
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "--engine", "whispercpp", "--diarize")
    assert result.exit_code == 0
    assert "per-segment attribution" in result.stdout


def test_jobs_are_clamped_with_whispercpp_cuda(tmp_path, monkeypatch):
    # 4 subprocesses × 1.28 GB against 4096 MiB: WDDM silently pages 25x (measured).
    calls = []
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0), calls=calls)
    result = _invoke(audio, tmp_path, "--engine", "whispercpp", "-d", "cuda", "-j", "4")
    assert result.exit_code == 0
    assert "jobs=1" in result.stdout
    assert len(calls) == 1


def test_the_summary_includes_the_default_engine(tmp_path, monkeypatch):
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path)
    assert result.exit_code == 0
    assert "faster-whisper" in result.stdout  # The summary always declares the engine.
    assert "Engine whisper.cpp" not in result.stdout  # Extra header only for a nondefault engine.


def test_the_summary_and_header_with_whispercpp(tmp_path, monkeypatch):
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "--engine", "whispercpp")
    assert result.exit_code == 0
    assert "whisper.cpp v1.9.1" in result.stdout  # Header with the pinned version.
    assert "q5_0" in result.stdout  # The EFFECTIVE quantization, not 'auto'.
    assert "whispercpp" in result.stdout  # Resolved engine in the summary.


def test_whispercpp_oom_advises_about_vram(tmp_path, monkeypatch):
    # CUDA OOM stderr also contains 'alloc': RAM advice would be a false friend.
    boom = RuntimeError("ggml_cuda: failed to allocate 1.28 GB")
    audio = _fake_transcribe(monkeypatch, tmp_path, [], _info(2206.0), boom=boom)
    result = _invoke(audio, tmp_path, "--engine", "whispercpp", "-d", "cuda")
    assert result.exit_code == 1
    assert "VRAM" in result.stdout
    assert "--engine faster-whisper" in result.stdout
    assert "medium" not in result.stdout  # The CPU RAM advice does not appear.


def test_faster_whisper_oom_advises_about_ram(tmp_path, monkeypatch):
    # The counterpart: under the default, the advice remains about CPU RAM.
    boom = RuntimeError("mkl_malloc: failed to allocate memory")
    audio = _fake_transcribe(monkeypatch, tmp_path, [], _info(2206.0), boom=boom)
    result = _invoke(audio, tmp_path)
    assert result.exit_code == 1
    assert "medium" in result.stdout
    assert "VRAM" not in result.stdout


# --- engine identity in JSON (G2) and effective device -------------------------------


def test_json_declares_the_faster_whisper_engine(tmp_path, monkeypatch):
    # The engine block must reach the real FILE, not merely exist as a kwarg in formats:
    # this test wires CLI -> write_json end to end.
    import json

    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "-f", "json")
    assert result.exit_code == 0
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    eng = payload["engine"]
    assert eng["name"] == "faster-whisper"
    assert eng["version"].startswith("faster-whisper ")
    assert eng["quant"] == "int8"  # The effective value for auto+cpu.
    assert eng["device"] == "cpu"
    assert eng["selection"] == "auto"  # --engine auto (default): the probe chose it.
    assert "diarization" not in eng  # Without --diarize, nothing is claimed.


def test_json_declares_the_whispercpp_engine_and_cuda_device(tmp_path, monkeypatch):
    # The pinned binary is a CUDA build and ALWAYS runs on the GPU (measured in smoke):
    # labeling it CPU would be false. The effective device is CUDA even without -d.
    import json

    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "-f", "json", "--engine", "whispercpp")
    assert result.exit_code == 0
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    eng = payload["engine"]
    assert eng == {
        "name": "whispercpp", "version": "whisper.cpp v1.9.1", "model": "large-v3",
        "quant": "q5_0", "device": "cuda", "selection": "explicit",
    }
    assert "cuda" in result.stdout  # The header does not say CPU either.


def test_whispercpp_rejects_an_unpinned_model(tmp_path, monkeypatch):
    # Without pre-validation, ensure_model blows up with a raw RuntimeError mid-run;
    # the error must arrive before anything is constructed and list what is available.
    calls = []
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0), calls=calls)
    result = _invoke(audio, tmp_path, "--engine", "whispercpp", "-m", "medium")
    assert result.exit_code == 2
    assert "is not pinned" in result.stderr
    assert "large-v3" in result.stderr and "small" in result.stderr
    assert calls == []  # The backend was never called.


# --- bench · table of measured configs (core ALWAYS stubbed, never engines) ----------


def _bench_row(**over):
    row = {
        "engine": "faster-whisper", "model": "small", "quant": "int8", "device": "cpu",
        "load_s": 1.2, "transcribe_s": 7.0, "x_realtime": 8.5,
        "peak_ram_mb": 900.0, "peak_vram_mb": None, "segments": 12, "chars": 800,
        "capabilities": {"hotwords": True, "word_timestamps": True,
                         "native_signals": True, "vad": True},
        "wer_ref": 0.419, "error": None,
    }
    row.update(over)
    return row


def _bench_table(results, skipped=()):
    return {
        "schema_version": "speechtotext.bench/v1",
        "measured_at": "2026-07-27T00:00:00+00:00",
        "machine": {"cpu": "x", "logical_cores": 8, "ram_gb": 16.0, "gpu": None},
        "audio": {"source": "a.wav", "duration_s": 60.0, "sha1": "0" * 40},
        "results": results,
        "skipped": list(skipped),
    }


def test_bench_show_without_a_table(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    result = runner.invoke(app, ["bench", "--show"])
    assert result.exit_code == 1
    assert "speechtotext bench" in result.stdout  # The message says HOW to measure it.


def test_bench_show_with_a_table(tmp_path, monkeypatch):
    from speechtotext.core import benchmark

    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    benchmark.write_table(_bench_table([_bench_row()]))
    result = runner.invoke(app, ["bench", "--show"])
    assert result.exit_code == 0
    assert "faster-whisper" in result.stdout
    assert "8.5" in result.stdout  # x_realtime is visible.


def test_bench_audio_writes_the_table_and_reports_skipped_configs(tmp_path, monkeypatch):
    from speechtotext.cli import app as cli_app
    from speechtotext.core import audio as core_audio
    from speechtotext.core import benchmark

    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    src = tmp_path / "charla.mp3"
    src.write_bytes(b"ID3")
    wav = tmp_path / "t.wav"
    wav.write_bytes(b"RIFF")
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")
    monkeypatch.setattr(core_audio, "transcode_to_wav", lambda b, **k: wav)
    monkeypatch.setattr(cli_app, "_trim_wav", lambda w, s: clip)
    monkeypatch.setattr(cli_app, "_wav_seconds", lambda p: 42.0)

    viable = {"engine": "faster-whisper", "model": "tiny", "quant": "int8", "device": "cpu",
              "capabilities": _bench_row()["capabilities"], "wer_ref": None}
    monkeypatch.setattr(benchmark, "available_configs", lambda: ([viable], []))

    def fake_run_benchmark(wav_path, duration_s, configs, *, progress=None):
        results = []
        for cfg in configs:
            res = _bench_row(engine=cfg["engine"], model=cfg["model"])
            results.append(res)
            if progress:
                progress(cfg, res)
        return _bench_table(
            results,
            skipped=[{"engine": "whispercpp", "model": "small", "reason": "exe missing"}],
        )

    monkeypatch.setattr(benchmark, "run_benchmark", fake_run_benchmark)
    result = runner.invoke(app, ["bench", str(src)])
    assert result.exit_code == 0
    assert (tmp_path / "bench.json").exists()  # Real write_table, home directory seeded.
    assert "bench.json" in result.stdout  # The path is printed.
    assert "exe missing" in result.stdout  # Skipped configs are explained.
    assert "tiny" in result.stdout  # One progress row per config.


def test_bench_quick_skips_the_slow_models(tmp_path, monkeypatch):
    from speechtotext.cli import app as cli_app
    from speechtotext.core import audio as core_audio
    from speechtotext.core import benchmark

    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    src = tmp_path / "charla.mp3"
    src.write_bytes(b"ID3")
    wav = tmp_path / "t.wav"
    wav.write_bytes(b"RIFF")
    monkeypatch.setattr(core_audio, "transcode_to_wav", lambda b, **k: wav)
    monkeypatch.setattr(cli_app, "_trim_wav", lambda w, s: wav)
    monkeypatch.setattr(cli_app, "_wav_seconds", lambda p: 42.0)
    caps = _bench_row()["capabilities"]
    viable_configs = [
        {"engine": "faster-whisper", "model": m, "quant": "int8", "device": "cpu",
         "capabilities": caps, "wer_ref": None}
        for m in ("tiny", "medium", "large-v3")
    ]
    monkeypatch.setattr(benchmark, "available_configs", lambda: (viable_configs, []))
    seen = {}

    def fake_run_benchmark(wav_path, duration_s, configs, *, progress=None):
        seen["models"] = [c["model"] for c in configs]
        return _bench_table([])

    monkeypatch.setattr(benchmark, "run_benchmark", fake_run_benchmark)
    result = runner.invoke(app, ["bench", str(src), "--quick"])
    assert result.exit_code == 0
    assert seen["models"] == ["tiny"]  # medium and large-v3 from fw are excluded.


def test_a_bench_config_with_an_error_is_marked(tmp_path, monkeypatch):
    from speechtotext.core import benchmark

    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    broken = _bench_row(model="medium", load_s=None, transcribe_s=None, x_realtime=None,
                        peak_ram_mb=None, segments=None, chars=None, error="child died rc=1")
    benchmark.write_table(_bench_table([_bench_row(), broken]))
    result = runner.invoke(app, ["bench", "--show"])
    assert result.exit_code == 0
    assert "child died" in result.stdout  # The broken row is visible, not hidden.
    assert "1 with error" in result.stdout


def test_whispercpp_warns_about_device_remapping(tmp_path, monkeypatch):
    # Silently overriding an explicit -d cpu would be the forbidden silent substitution:
    # CUDA remapping is always announced unless CUDA was requested.
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "--engine", "whispercpp", "-d", "cpu")
    assert result.exit_code == 0
    assert "runs on the GPU; device=cuda" in result.stdout

    result = _invoke(audio, tmp_path, "--engine", "whispercpp", "-d", "cuda")
    assert result.exit_code == 0
    assert "runs on the GPU" not in result.stdout  # A CUDA request receives no noise.


def test_bench_quick_documents_skipped_configs_under_skipped(tmp_path, monkeypatch):
    # A table with unexplained missing rows would make its consumer choose without knowing
    # candidates are missing: quick-skipped configs go under skipped with their reason.
    import json

    from speechtotext.core import benchmark

    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path / "home"))
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    from speechtotext.cli import app as app_mod

    monkeypatch.setattr(app_mod, "_trim_wav", lambda wav, s: wav)
    monkeypatch.setattr(
        "speechtotext.core.audio.transcode_to_wav", lambda b: tmp_path / "t.wav"
    )
    (tmp_path / "t.wav").write_bytes(b"RIFF")
    monkeypatch.setattr("speechtotext.cli.app._wav_seconds", lambda p: 60.0)
    monkeypatch.setattr(
        benchmark, "available_configs",
        lambda: ([
            {"engine": "faster-whisper", "model": "small"},
            {"engine": "faster-whisper", "model": "medium"},
            {"engine": "faster-whisper", "model": "large-v3"},
        ], []),
    )
    monkeypatch.setattr(
        benchmark, "run_benchmark",
        lambda clip, d, configs, progress=None: {
            "results": [], "skipped": [], "machine": {}, "audio": {},
        },
    )
    result = runner.invoke(app, ["bench", str(audio), "--quick"], catch_exceptions=False)
    assert result.exit_code == 0
    table = json.loads((tmp_path / "home" / "bench.json").read_text(encoding="utf-8"))
    reasons = {(s["engine"], s["model"]): s["reason"] for s in table["skipped"]}
    assert reasons[("faster-whisper", "medium")] == "skipped by --quick"
    assert reasons[("faster-whisper", "large-v3")] == "skipped by --quick"


def test_bench_with_broken_ffmpeg_exits_with_a_message(tmp_path, monkeypatch):
    # The audio-clipping error path: red message and exit 1, not a raw traceback.
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(
        "speechtotext.core.audio.transcode_to_wav", lambda b: tmp_path / "t.wav"
    )
    (tmp_path / "t.wav").write_bytes(b"RIFF")
    from speechtotext.cli import app as app_mod

    def boom(wav, s):
        raise RuntimeError("ffmpeg could not clip the audio: corrupt track")

    monkeypatch.setattr(app_mod, "_trim_wav", boom)
    result = runner.invoke(app, ["bench", str(audio)])
    assert result.exit_code == 1
    assert "Could not clip" in result.stdout


def test_chunking_is_announced_and_each_chunk_is_listed_outside_a_tty(tmp_path, monkeypatch):
    from speechtotext.core import transcribe as core_transcribe

    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(1.0, 2.0)], _info(1200.0))
    monkeypatch.setattr(core_transcribe, "should_chunk", lambda d, c: True)
    monkeypatch.setattr(core_transcribe, "plan_chunks", lambda path, dur: [(0.0, 600.0), (600.0, 1200.0)])
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    result = _invoke(audio, tmp_path, "-j", "2")
    assert result.exit_code == 0, result.stdout
    output = _flattened(result.stdout)
    assert "Chunked (jobs=2)" in output
    assert "[1/2]" in output and "[2/2]" in output
    assert "(nuevo)" in output


# --- spec §5.2 defaults, ETA, and uncertain language --------------------------------


def test_cli_defaults_match_the_spec(tmp_path, monkeypatch):
    import json

    calls = []
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0), calls=calls)
    result = _invoke(audio, tmp_path, "-f", "json")
    assert result.exit_code == 0, result.stdout
    (_, request), = calls
    assert (request.language, request.vad, request.beam_size, request.hotwords) == ("auto", False, 5, ())
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert payload["engine"]["model"] == "large-v3"
    assert payload["engine"]["device"] == "cpu"      # conftest: machine without a GPU.
    assert payload["engine"]["selection"] == "auto"
    assert "Language detected" in result.stdout
    assert "engine faster-whisper" in _flattened(result.stdout)


def test_eta_is_printed_after_decoding(tmp_path, monkeypatch):
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(600.0))
    result = _invoke(audio, tmp_path)
    assert result.exit_code == 0, result.stdout
    assert "Duration 10.0 min · ETA ~8 min (estimated)" in _flattened(result.stdout)


def test_an_eta_measured_with_bench_does_not_say_estimated(tmp_path, monkeypatch):
    from speechtotext.core import benchmark

    benchmark.write_table({"schema_version": "speechtotext.bench/v1", "results": [
        {"engine": "faster-whisper", "model": "large-v3", "quant": "int8", "device": "cpu",
         "x_realtime": 2.0, "error": None, "capabilities": {}, "wer_ref": None}],
        "skipped": [], "recommendations": []})
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(600.0))
    result = _invoke(audio, tmp_path)
    assert "ETA ~5 min (measured with bench)" in _flattened(result.stdout)


def test_a_route_without_an_eta_says_so(tmp_path, monkeypatch):
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(60.0))
    result = _invoke(audio, tmp_path, "-m", "medium")
    assert "ETA not measured for this route" in _flattened(result.stdout)


def test_an_uncertain_language_suggests_setting_it(tmp_path, monkeypatch):
    info = _info(10.0, language="pt", language_probability=0.41)
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], info)
    result = _invoke(audio, tmp_path)
    assert "prob=0.41" in result.stdout and "set it with -l" in result.stdout
    info = _info(10.0, language="pt", language_probability=0.9)
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], info)
    assert "set it with -l" not in _invoke(audio, tmp_path).stdout


def test_a_model_that_does_not_fit_in_ram_stops_without_changing_it(tmp_path, monkeypatch):
    from speechtotext.core import probe

    calls = []
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0), calls=calls)
    monkeypatch.setattr(probe, "machine", lambda: probe.Machine("win32", 4, 4.0, False, None, None, None))
    result = _invoke(audio, tmp_path)
    assert result.exit_code == 1
    assert "large-v3 needs ~6 GB" in result.stdout and "-m small" in result.stdout
    assert calls == []


def test_the_auto_route_warns_and_announces_the_whispercpp_download(tmp_path, monkeypatch):
    import json

    from speechtotext.core import probe

    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    monkeypatch.setattr(probe, "machine",
                        lambda: probe.Machine("win32", 12, 32.0, True, "GTX 980", 3.5, None))
    result = _invoke(audio, tmp_path, "-f", "json")
    assert result.exit_code == 0, result.stdout
    output = _flattened(result.stdout)
    assert "GPU with 3.5 GB free: quantized whisper.cpp" in output
    assert "whisper.cpp v1.9.1 is not installed: downloading now (~646 MB, once)" in output
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert (payload["engine"]["name"], payload["engine"]["device"]) == ("whispercpp", "cuda")


def test_an_installed_whispercpp_does_not_announce_a_download(tmp_path, monkeypatch):
    from pathlib import Path

    from speechtotext.core import probe

    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    monkeypatch.setattr(probe, "machine", lambda: probe.Machine(
        "win32", 12, 32.0, True, "GTX 980", 3.5, Path("C:/x/whisper-cli.exe")))
    result = _invoke(audio, tmp_path)
    assert result.exit_code == 0 and "downloading now" not in result.stdout


# --- probe and models ---------------------------------------------------------------


def test_probe_prints_the_machine_and_routes(monkeypatch):
    from speechtotext.core import probe

    monkeypatch.setattr(probe, "machine",
                        lambda: probe.Machine("win32", 12, 31.9, True, "GTX 980", 3.46, None))
    result = runner.invoke(app, ["probe"])
    assert result.exit_code == 0, result.stdout
    output = _flattened(result.stdout)
    # _flattened collapses every run of spaces to one (" ".join(text.split())): the real
    # command uses several spaces to align columns, but only one survives here.
    assert "platform win32" in output and "ram_gb 31.9" in output
    assert "gpu GTX 980" in output and "vram_free 3.46 GB" in output
    assert "whispercpp not installed" in output
    assert ("large-v3 whispercpp · cuda · q5_0 · ~8.0x real time (estimated) · "
            "GPU with 3.5 GB free: quantized whisper.cpp") in output
    assert "small whispercpp · cuda · q5_0 · ~15.6x real time (estimated)" in output


def test_probe_says_when_the_model_does_not_fit(monkeypatch):
    from speechtotext.core import probe

    monkeypatch.setattr(probe, "machine", lambda: probe.Machine("linux", 4, 4.0, False, None, None, None))
    result = runner.invoke(app, ["probe"])
    assert result.exit_code == 0, result.stdout
    output = _flattened(result.stdout)
    assert "large-v3 large-v3 needs ~6 GB" in output
    assert "small faster-whisper · cpu · int8 · ~6.4x real time (estimated) · no usable GPU: CPU" in output
    assert "ram_gb 4.0" in output


def test_probe_without_measurements_does_not_invent_them(monkeypatch):
    from speechtotext.core import probe

    monkeypatch.setattr(probe, "machine", lambda: probe.Machine("darwin", 8, None, False, None, None, None))
    output = _flattened(runner.invoke(app, ["probe"]).stdout)
    assert "ram_gb not measured" in output and "vram_free -" in output and "gpu -" in output


def _models_double(monkeypatch, installed_models=(), size=None, boom=None):
    from pathlib import Path

    from speechtotext.core import models

    observed = {}
    monkeypatch.setattr(models, "installed",
                        lambda engine=None: [m for m in installed_models if engine in (None, m.engine)])
    monkeypatch.setattr(models, "remote_size", lambda engine, name: size)

    def ensure(engine, name, on_progress=None):
        if boom:
            raise boom
        observed["ensure"] = (engine, name)
        return Path("C:/hf/snap")

    def remove(engine, name):
        if boom:
            raise boom
        observed["remove"] = (engine, name)

    monkeypatch.setattr(models, "ensure", ensure)
    monkeypatch.setattr(models, "remove", remove)
    return observed


def test_an_empty_models_list_suggests_pull(monkeypatch):
    _models_double(monkeypatch)
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0, result.stdout
    assert "models pull large-v3" in _flattened(result.stdout)


def test_models_lists_a_table(monkeypatch, tmp_path):
    from speechtotext.core.models import ModelInfo

    _models_double(monkeypatch, [
        ModelInfo("faster-whisper", "large-v3", tmp_path / "x", 3_090_839_273, False),
        ModelInfo("whispercpp", "small", tmp_path / "y.bin", 487_601_967, True),
    ])
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0, result.stdout
    output = _flattened(result.stdout)
    assert "large-v3" in output and "2.9 GB" in output and "465 MB" in output and "yes" in output
    assert "Data in" in output


def test_models_pull_announces_the_size_and_downloads(monkeypatch):
    observed = _models_double(monkeypatch, size=3_090_839_273)
    result = runner.invoke(app, ["models", "pull", "large-v3"])
    assert result.exit_code == 0, result.stdout
    assert "Downloading large-v3 (faster-whisper, ~2.9 GB)" in _flattened(result.stdout)
    assert observed["ensure"] == ("faster-whisper", "large-v3")
    result = runner.invoke(app, ["models", "pull", "small", "--engine", "whispercpp"])
    assert result.exit_code == 0 and observed["ensure"] == ("whispercpp", "small")


def test_models_pull_without_a_size_does_not_invent_one(monkeypatch):
    _models_double(monkeypatch, size=None)
    result = runner.invoke(app, ["models", "pull", "small"])
    assert "Downloading small (faster-whisper)..." in _flattened(result.stdout)


def test_models_pull_exits_with_code_1_on_download_failure(monkeypatch):
    _models_double(monkeypatch, boom=RuntimeError("sha256 of ggml-small.bin does not match the pin"))
    result = runner.invoke(app, ["models", "pull", "small", "--engine", "whispercpp"])
    assert result.exit_code == 1 and "sha256" in result.stdout


def test_models_rm(monkeypatch):
    observed = _models_double(monkeypatch)
    result = runner.invoke(app, ["models", "rm", "small"])
    assert result.exit_code == 0 and observed["remove"] == ("faster-whisper", "small")
    _models_double(monkeypatch, boom=FileNotFoundError("small (faster-whisper) is not installed"))
    result = runner.invoke(app, ["models", "rm", "small"])
    assert result.exit_code == 1 and "is not installed" in result.stdout


# --- I3 · the jobs=1 warning only under explicit --engine ---------------------------


def test_an_auto_route_to_whispercpp_does_not_scold_about_jobs(tmp_path, monkeypatch):
    from pathlib import Path

    from speechtotext.core import probe

    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    monkeypatch.setattr(probe, "machine", lambda: probe.Machine(
        "win32", 12, 32.0, True, "GTX 980", 3.5, Path("C:/x/whisper-cli.exe")))
    result = _invoke(audio, tmp_path)
    assert result.exit_code == 0, result.stdout
    assert "does not parallelize" not in result.stdout


def test_an_explicit_whispercpp_engine_with_multiple_jobs_warns(tmp_path, monkeypatch):
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "--engine", "whispercpp", "-j", "4")
    assert result.exit_code == 0, result.stdout
    assert "the GPU does not parallelize; jobs=1" in result.stdout


# --- I4/I5 · one probe, selection follows how the engine was chosen -----------------


def test_the_cli_probes_only_once(tmp_path, monkeypatch):
    from speechtotext.core import probe

    fixed_machine = probe.machine()
    calls = []
    monkeypatch.setattr(probe, "machine", lambda: (calls.append(1), fixed_machine)[1])
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    assert _invoke(audio, tmp_path).exit_code == 0
    assert len(calls) == 1


def test_mcp_calls_serve(monkeypatch):
    from speechtotext.cli import mcp_server

    called = []
    monkeypatch.setattr(mcp_server, "serve", lambda: called.append(True))
    result = runner.invoke(app, ["mcp"])
    assert result.exit_code == 0, result.output
    assert called == [True]
