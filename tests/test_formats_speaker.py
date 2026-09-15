import json
from pathlib import Path
from types import SimpleNamespace

from speechtotext.core.segments import LabeledSegment
from speechtotext.core.formats import (
    is_suspect,
    write_txt,
    write_srt,
    write_vtt,
    write_json,
)


# Real evidence from the brief: "Gracias." is 8 characters in 30 s = 0.27 char/s,
# far below the 12-15 char/s of spoken Spanish and the 3-5 of scattered whispering.
HALLUCINATED = LabeledSegment(0, 30, "Gracias.")
REAL = LabeledSegment(30, 33, "Okay, so that's settled.")


def test_txt_groups_consecutive_speaker(tmp_path):
    segs = [
        LabeledSegment(0, 1, "hello", "Alice"),
        LabeledSegment(1, 2, "there", "Alice"),
        LabeledSegment(2, 3, "fine", "Bob"),
    ]
    p = tmp_path / "o.txt"
    write_txt(segs, p)
    assert p.read_text(encoding="utf-8") == "Alice: hello there\nBob: fine\n"


def test_txt_without_speaker_unchanged(tmp_path):
    segs = [LabeledSegment(0, 1, "hola"), LabeledSegment(1, 2, "chao")]
    p = tmp_path / "o.txt"
    write_txt(segs, p)
    assert p.read_text(encoding="utf-8") == "hola\nchao\n"


def test_srt_prefixes_speaker(tmp_path):
    segs = [LabeledSegment(0, 1, "hola", "Alice")]
    p = tmp_path / "o.srt"
    write_srt(segs, p)
    assert "Alice: hola" in p.read_text(encoding="utf-8")


def test_json_has_speaker_and_speakers(tmp_path):
    segs = [LabeledSegment(0, 1, "hola", "Bob")]
    info = SimpleNamespace(language="es", language_probability=1.0, duration=1.0)
    p = tmp_path / "o.json"
    write_json(segs, info, p)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["speakers"] == ["Bob"]
    assert data["segments"][0]["speaker"] == "Bob"


def test_json_without_speaker_omits_fields(tmp_path):
    segs = [LabeledSegment(0, 1, "hola")]
    info = SimpleNamespace(language="es", language_probability=1.0, duration=1.0)
    p = tmp_path / "o.json"
    write_json(segs, info, p)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "speakers" not in data
    assert "speaker" not in data["segments"][0]


def test_vtt_prefixes_speaker(tmp_path):
    segs = [LabeledSegment(0, 1, "hola", "Alice")]
    p = tmp_path / "o.vtt"
    write_vtt(segs, p)
    assert "Alice: hola" in p.read_text(encoding="utf-8")


# --- 1.2: seconds with speech and gaps ---


def test_json_has_speech_s_and_gaps(tmp_path):
    segs = [LabeledSegment(0, 1, "hola"), LabeledSegment(40, 41, "chao")]
    info = SimpleNamespace(language="es", language_probability=1.0, duration=60.0)
    p = tmp_path / "o.json"
    write_json(segs, info, p)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["speech_s"] == 2.0
    assert data["gaps"] == [[1, 40], [41, 60]]
    # "duration" remains the file's duration, not the speech duration
    assert data["duration"] == 60.0


def test_json_gaps_ignores_short_gaps(tmp_path):
    # 2 s of silence between segments: breathing, not lost audio
    segs = [LabeledSegment(0, 1, "hola"), LabeledSegment(3, 4, "chao")]
    info = SimpleNamespace(language="es", language_probability=1.0, duration=4.0)
    p = tmp_path / "o.json"
    write_json(segs, info, p)
    assert json.loads(p.read_text(encoding="utf-8"))["gaps"] == []


def test_passed_speech_s_and_gaps_are_emitted_unchanged(tmp_path):
    # The CLI calculates the metric only once, before diarization (5.1.3): if the
    # kwargs arrive, they are emitted unchanged even if the segments imply another value.
    segs = [LabeledSegment(0, 1, "hola"), LabeledSegment(40, 41, "chao")]
    info = SimpleNamespace(language="es", language_probability=1.0, duration=60.0)
    p = tmp_path / "o.json"
    write_json(segs, info, p, speech_s=99.0, gaps=[[1, 2]])
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["speech_s"] == 99.0
    assert data["gaps"] == [[1, 2]]


# --- 1.5: [?] marker ---


def test_is_suspect_triggers_by_density():
    assert is_suspect(HALLUCINATED)
    assert not is_suspect(REAL)


def test_is_suspect_no_speech_overrides_density():
    # the field does not exist until Phase 2: the branch is inert today and activates itself
    long_but_spoken = LabeledSegment(0, 30, "x" * 100)
    assert not is_suspect(long_but_spoken)
    long_but_spoken.no_speech = 0.9
    assert is_suspect(long_but_spoken)


def test_txt_marks_suspect_without_diarization(tmp_path):
    p = tmp_path / "o.txt"
    write_txt([HALLUCINATED, REAL], p)
    assert p.read_text(encoding="utf-8") == (
        "[?] Gracias.\nOkay, so that's settled.\n"
    )


def test_txt_marks_suspect_with_diarization(tmp_path):
    # The shape the real route DOES produce under --diarize: the span is compressed
    # to the word extent (speakers/diarization.py), while the ASR segment duration
    # travels in src_dur. The hand-built LabeledSegment(0, 30, "Gracias.", "Alice")
    # previously here certified an impossible case: with --diarize set, the route
    # never emits a 30 s span with one word — it emits a ~1 s span with src_dur=30.0.
    # The fallback branch (without src_dur) remains covered by the non-diarized tests
    # in this same file.
    segs = [
        LabeledSegment(0.4, 1.4, "Gracias.", "Alice", src_dur=30.0),
        LabeledSegment(30.2, 32.8, "Okay, so that's settled.", "Bob", src_dur=3.0),
    ]
    p = tmp_path / "o.txt"
    write_txt(segs, p)
    assert p.read_text(encoding="utf-8") == (
        "Alice: [?] Gracias.\nBob: Okay, so that's settled.\n"
    )


def test_srt_marks_suspect(tmp_path):
    p = tmp_path / "o.srt"
    write_srt([HALLUCINATED, REAL], p)
    txt = p.read_text(encoding="utf-8")
    assert "[?] Gracias." in txt
    assert "[?] Bueno" not in txt


def test_vtt_marks_suspect(tmp_path):
    p = tmp_path / "o.vtt"
    write_vtt([HALLUCINATED, REAL], p)
    txt = p.read_text(encoding="utf-8")
    assert "[?] Gracias." in txt
    assert "[?] Bueno" not in txt


def test_json_marks_suspect_and_leaves_the_text_clean(tmp_path):
    info = SimpleNamespace(language="es", language_probability=1.0, duration=33.0)
    p = tmp_path / "o.json"
    write_json([HALLUCINATED, REAL], info, p)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["segments"][0]["suspect"] is True
    assert data["segments"][0]["text"] == "Gracias."
    assert "suspect" not in data["segments"][1]


# --- 1.6: measured vs forced language ---


def test_json_omits_language_probability_when_it_is_none(tmp_path):
    segs = [LabeledSegment(0, 1, "hola")]
    info = SimpleNamespace(language="es", language_probability=None, duration=1.0)
    p = tmp_path / "o.json"
    write_json(segs, info, p)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "language_probability" not in data
    assert data["language"] == "es"


def test_json_preserves_forced_language_probability(tmp_path):
    segs = [LabeledSegment(0, 1, "hola")]
    info = SimpleNamespace(language="es", language_probability=1.0, duration=1.0)
    p = tmp_path / "o.json"
    write_json(segs, info, p)
    assert json.loads(p.read_text(encoding="utf-8"))["language_probability"] == 1.0


# --- multi-engine (plan 2.6): engine block in JSON ---

# The CLI builds the dict; formats only emits it. This is the contract shape.
ENGINE_INFO = {
    "name": "whispercpp",
    "version": "v1.9.1",
    "model": "large-v3",
    "quant": "q5_0",
    "device": "cuda",
    "selection": "explicit",
}


def _info(duration=1.0, prob=1.0):
    return SimpleNamespace(language="es", language_probability=prob, duration=duration)


def test_json_without_engine_info_omits_the_key(tmp_path):
    # regression: old call sites (without the kwarg) produce the same payload as before
    p = tmp_path / "o.json"
    write_json([LabeledSegment(0, 1, "hola")], _info(), p)
    assert "engine" not in json.loads(p.read_text(encoding="utf-8"))


def test_json_engine_info_is_complete(tmp_path):
    p = tmp_path / "o.json"
    write_json([LabeledSegment(0, 1, "hola")], _info(), p, engine_info=dict(ENGINE_INFO))
    assert json.loads(p.read_text(encoding="utf-8"))["engine"] == ENGINE_INFO


def test_json_diarization_goes_inside_the_engine_block(tmp_path):
    p = tmp_path / "o.json"
    write_json(
        [LabeledSegment(0, 1, "hola", "Alice")],
        _info(),
        p,
        engine_info={**ENGINE_INFO, "diarization": "segment"},
    )
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["engine"]["diarization"] == "segment"
    assert "diarization" not in data  # a single home for engine metadata (G4)


def test_pipeline_takes_the_whispercpp_fixture_to_write_json(tmp_path):
    """End-to-end dual contract: the synthetic -ojf fixture goes through the core
    parser, becomes LabeledSegments (the bridge the CLI provides today), and comes
    out through write_json. test_whispercpp_backend.py does not cover this final leg;
    it lives here."""
    from speechtotext.asr.whispercpp import parse_ojf

    fixture = Path(__file__).parent / "fixtures" / "whispercpp_ojf.json"
    raw_segments, language = parse_ojf(json.loads(fixture.read_text(encoding="utf-8")))
    segs = [LabeledSegment(s.start, s.end, s.text) for s in raw_segments]
    # the parser does not fabricate duration (law G5); the orchestrator does — here, the test
    info = SimpleNamespace(
        language=language,
        language_probability=None,
        duration=segs[-1].end,
    )
    p = tmp_path / "o.json"
    write_json(segs, info, p, engine_info=dict(ENGINE_INFO))
    data = json.loads(p.read_text(encoding="utf-8"))
    # (a) engine language_probability=None -> absent key, never 0.0
    assert "language_probability" not in data
    assert data["language"] == "es"
    # (b) text survives intact (write_json only trims the decorative leading space)
    assert data["segments"][0]["text"] == "Segment one of the test track."
    assert data["segments"][0]["start"] == 0.0
    assert data["segments"][0]["end"] == 19.92
    assert len(data["segments"]) == len(segs)
    # (c) the engine block comes out complete with all 6 keys
    assert data["engine"] == ENGINE_INFO
