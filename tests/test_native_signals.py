"""Native signals per segment (no_speech, avg_logprob, compression_ratio).

Phase 2 of the transcription-quality plan / PR-2 of the multi-engine plan: faster-whisper
already emits them for free on every Segment, but the CLI route discarded them.
`is_suspect` (core/formats.py) read `no_speech` from an attribute nobody set: dead branch.

Law G5 (core/engines.py): a signal the engine does not emit is OMITTED, never filled in.
whisper.cpp does not emit them -> None end to end, and JSON omits the key.
"""
import json
from types import SimpleNamespace

from speechtotext.core.chunked import (
    TimedSegment,
    TimedWord,
    seg_from_dict,
    seg_to_dict,
    shift_segments,
)
from speechtotext.core.formats import is_suspect, write_json
from speechtotext.core.segments import LabeledSegment, native_signals
from speechtotext.speakers.diarization import apply_names, assign_segments


def _raw(start, end, text="hi there", **kw):
    """Raw faster-whisper Segment: its field name there is `no_speech_prob`."""
    return SimpleNamespace(start=start, end=end, text=text, words=None, **kw)


def _info(duration=60.0):
    return SimpleNamespace(duration=duration, language="es", language_probability=1.0)


# --- native_signals: one reader for both dialects ---


def test_native_signals_reads_the_faster_whisper_dialect():
    seg = _raw(0.0, 2.0, no_speech_prob=0.8, avg_logprob=-0.4, compression_ratio=1.7)
    assert native_signals(seg) == (0.8, -0.4, 1.7)


def test_native_signals_reads_its_own_dialect():
    seg = TimedSegment(0.0, 2.0, "hola", None, no_speech=0.8,
                       avg_logprob=-0.4, compression_ratio=1.7)
    assert native_signals(seg) == (0.8, -0.4, 1.7)


def test_native_signals_without_signals_is_none_not_zero():
    """whisper.cpp (_parse_ojf) emits none: zero would be a fabricated measurement."""
    assert native_signals(_raw(0.0, 2.0)) == (None, None, None)


# --- shift_segments: the extraction point of the chunked route ---


def test_shift_segments_copies_the_signals():
    out = shift_segments(
        [_raw(1.0, 3.0, no_speech_prob=0.9, avg_logprob=-0.2, compression_ratio=2.1)],
        10.0,
    )
    assert (out[0].no_speech, out[0].avg_logprob, out[0].compression_ratio) == (0.9, -0.2, 2.1)
    assert (out[0].start, out[0].end) == (11.0, 13.0)  # the offset still applies


def test_shift_segments_without_signals_leaves_none():
    out = shift_segments([_raw(0.0, 2.0)], 0.0)
    assert (out[0].no_speech, out[0].avg_logprob, out[0].compression_ratio) == (None, None, None)


# --- checkpoint: round trip and compatibility with an old cache ---


def test_checkpoint_round_trip_preserves_the_signals():
    seg = TimedSegment(0.0, 2.0, "hola", [TimedWord(0.0, 1.0, "hola")],
                       no_speech=0.7, avg_logprob=-0.3, compression_ratio=1.9)
    back = seg_from_dict(json.loads(json.dumps(seg_to_dict(seg))))
    assert (back.no_speech, back.avg_logprob, back.compression_ratio) == (0.7, -0.3, 1.9)
    assert back.words[0].word == "hola"


def test_old_checkpoint_without_signals_does_not_blow_up():
    """The .json files in ~/.speechtotext/chunks written before this lack the keys:
    they degrade to None (= today's behavior), not to KeyError."""
    back = seg_from_dict({"start": 0.0, "end": 2.0, "text": "hola"})
    assert (back.no_speech, back.avg_logprob, back.compression_ratio) == (None, None, None)


def test_checkpoint_omits_the_missing_signals():
    d = seg_to_dict(TimedSegment(0.0, 2.0, "hola"))
    assert "no_speech" not in d and "avg_logprob" not in d and "compression_ratio" not in d


# --- is_suspect: the branch that was dead ---


def test_is_suspect_triggers_on_no_speech():
    """Short, dense segment: today it does NOT trigger on the density heuristic.
    Only no_speech can mark it, and until now nobody set it."""
    seg = LabeledSegment(0.0, 2.0, "hi there how are you", no_speech=0.75)
    assert is_suspect(seg) is True


def test_is_suspect_does_not_trigger_with_low_no_speech():
    seg = LabeledSegment(0.0, 2.0, "hi there how are you", no_speech=0.5)
    assert is_suspect(seg) is False


def test_is_suspect_without_a_signal_falls_back_to_density_as_before():
    dense = LabeledSegment(0.0, 2.0, "hi there how are you")
    sparse = LabeledSegment(0.0, 12.0, "eh")
    assert is_suspect(dense) is False
    assert is_suspect(sparse) is True


# --- diarization: signals survive assignment by speaker ---


def test_coarse_assign_segments_propagates_the_signals():
    raw = TimedSegment(0.0, 4.0, "hi there", None,
                       no_speech=0.8, avg_logprob=-0.5, compression_ratio=2.0)
    out = assign_segments([raw], [(0.0, 4.0, "SPEAKER_00")])
    assert len(out) == 1
    assert (out[0].no_speech, out[0].avg_logprob, out[0].compression_ratio) == (0.8, -0.5, 2.0)


def test_word_level_assign_segments_propagates_to_each_run():
    """The N runs of a segment inherit its signals, just like src_dur: they come
    from the same decoding window."""
    words = [TimedWord(0.0, 1.0, "hola"), TimedWord(2.0, 3.0, "adios")]
    raw = TimedSegment(0.0, 3.0, "hola adios", words,
                       no_speech=0.8, avg_logprob=-0.5, compression_ratio=2.0)
    out = assign_segments([raw], [(0.0, 1.5, "SPEAKER_00"), (1.5, 3.0, "SPEAKER_01")])
    assert len(out) == 2
    for seg in out:
        assert (seg.no_speech, seg.avg_logprob, seg.compression_ratio) == (0.8, -0.5, 2.0)


def test_apply_names_preserves_the_signals():
    labeled = [LabeledSegment(0.0, 2.0, "hola", "SPEAKER_00",
                              no_speech=0.8, avg_logprob=-0.5, compression_ratio=2.0)]
    out = apply_names(labeled, {"SPEAKER_00": "Alice"})
    assert out[0].speaker == "Alice"
    assert (out[0].no_speech, out[0].avg_logprob, out[0].compression_ratio) == (0.8, -0.5, 2.0)


# --- JSON: the machine-readable artifact exposes them ---


def test_write_json_emits_the_signals(tmp_path):
    path = tmp_path / "out.json"
    segs = [LabeledSegment(0.0, 2.0, "hi there", no_speech=0.75,
                           avg_logprob=-0.3, compression_ratio=1.8)]
    write_json(segs, _info(), path)
    seg = json.loads(path.read_text(encoding="utf-8"))["segments"][0]
    assert seg["no_speech"] == 0.75
    assert seg["avg_logprob"] == -0.3
    assert seg["compression_ratio"] == 1.8
    assert seg["suspect"] is True  # the marker comes from the signal, not density


def test_write_json_omits_the_missing_signals(tmp_path):
    path = tmp_path / "out.json"
    write_json([LabeledSegment(0.0, 2.0, "hi there")], _info(), path)
    seg = json.loads(path.read_text(encoding="utf-8"))["segments"][0]
    assert "no_speech" not in seg
    assert "avg_logprob" not in seg
    assert "compression_ratio" not in seg


# --- the end-to-end CLI: the direct route also propagates them ---


def test_cli_transcribe_carries_the_signals_into_json(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from speechtotext.cli.app import app

    audio = tmp_path / "charla.wav"
    audio.write_bytes(b"RIFF")
    info = _info(duration=60.0)
    segs = [_raw(0.0, 2.0, no_speech_prob=0.75, avg_logprob=-0.3, compression_ratio=1.8)]
    from speechtotext.asr import Caps
    from speechtotext.asr.types import (
        NativeSignals, SegmentNativeSignals, TranscriptionResult, TranscriptionSegment,
    )
    from speechtotext.core import transcribe as core_transcribe

    import numpy as np

    monkeypatch.setattr(core_transcribe, "load_audio", lambda p: np.zeros(60 * 16000, dtype=np.float32))
    monkeypatch.setattr(core_transcribe, "should_chunk", lambda d, c: False)
    result = TranscriptionResult(
        text="hi there", language="es", words=(),
        segments=tuple(TranscriptionSegment(s.start, s.end, s.text, (), SegmentNativeSignals(
            s.no_speech_prob, s.avg_logprob, s.compression_ratio)) for s in segs),
        backend="faster-whisper", model="small", model_version="1", latency_ms=1,
        native_signals=NativeSignals(None, None, None, 1.0), warnings=(),
    )
    fake = SimpleNamespace(backend_id="faster-whisper", model_id="small", device="cpu", quant="int8",
                           model_version="1", engine_version="faster-whisper 1.2.0",
                           caps=Caps("honored", "honored", "honored"), warm=lambda: None,
                           transcribe=lambda samples, request, **kw: result)
    monkeypatch.setattr(core_transcribe, "make_backend", lambda *a, **k: fake)

    # -o with a nonexistent path and no trailing separator is a BASE path, not a folder
    # (_resolve_output_base, cli/app.py:70): JSON goes to out.json, not out/charla.json.
    result = CliRunner().invoke(
        app, ["transcribe", str(audio), "-f", "json", "-o", str(tmp_path / "out")]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    seg = payload["segments"][0]
    assert seg["no_speech"] == 0.75
    assert seg["suspect"] is True


# --- pathological values: the engine may emit garbage, the artifact may not ---


def test_native_signals_discards_nonfinite_values():
    """NaN/inf are not measurements: they enter as None (G5), not as numbers.
    Without this, json.dumps writes the NaN/Infinity tokens — invalid JSON under
    RFC 8259, which Python reads back but jq and JSON.parse reject — and on top of
    that `NaN > 0.6` is False, so the most broken signal would be the only unmarked one."""
    seg = _raw(0.0, 2.0, no_speech_prob=float("nan"),
               avg_logprob=float("-inf"), compression_ratio=float("inf"))
    assert native_signals(seg) == (None, None, None)


def test_write_json_does_not_emit_invalid_tokens(tmp_path):
    path = tmp_path / "out.json"
    segs = shift_segments([_raw(0.0, 2.0, no_speech_prob=float("nan"))], 0.0)
    write_json([LabeledSegment(s.start, s.end, s.text, no_speech=s.no_speech)
                for s in segs], _info(), path)
    raw_json = path.read_text(encoding="utf-8")
    assert "NaN" not in raw_json and "Infinity" not in raw_json
    json.loads(raw_json)  # read back: if tokens were invalid, the repo itself would tolerate them


def test_write_json_rounds_the_signals(tmp_path):
    """faster-whisper's float32 arrives as -0.30000001192092896; the rest of the
    payload (start/end/language_probability) is already rounded, and these were
    not the exception."""
    path = tmp_path / "out.json"
    segs = [LabeledSegment(0.0, 2.0, "hi there", no_speech=0.1234567,
                           avg_logprob=-0.30000001192092896, compression_ratio=1.23456789)]
    write_json(segs, _info(), path)
    seg = json.loads(path.read_text(encoding="utf-8"))["segments"][0]
    assert seg["no_speech"] == 0.1235
    assert seg["avg_logprob"] == -0.3
    assert seg["compression_ratio"] == 1.2346
