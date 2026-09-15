"""WhisperCppBackend: the subprocess never runs in tests; `run` is stubbed."""
import itertools
import json
import os
import shutil
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from speechtotext.asr import AsrBackend, Caps, TranscriptionRequest
from speechtotext.asr import whispercpp
from speechtotext.asr.whispercpp import WhisperCppBackend, parse_ojf
from speechtotext.core import enginepin

FIXTURE = Path(__file__).parent / "fixtures" / "whispercpp_ojf.json"


def test_the_fixture_is_synthetic_and_contains_no_machine_paths():
    """Release guard: this fixture ships in the public repo. If someone regenerates it
    by pasting real output from their machine, they find out here, not in filter-repo."""
    raw = FIXTURE.read_text(encoding="utf-8")
    assert "\\" not in raw, "Windows path in the fixture"
    assert "Users" not in raw, "machine path in the fixture"
    payload = json.loads(raw)
    texts = [s["text"].strip() for s in payload["transcription"]]
    assert len(texts) == 6
    assert all(t.startswith("Segment ") for t in texts), texts


def _run_stub(write="fixture", rc=0, stderr=b""):
    """`write`: 'fixture' copies the synthetic fixture; a dict writes that JSON; None
    writes nothing; a raw str writes garbage. Returns (run, seen)."""
    seen = {}

    def fake_run(cmd, capture_output=None, timeout=None):
        seen["cmd"] = list(cmd)
        seen["timeout"] = timeout
        base = cmd[cmd.index("-of") + 1]
        seen["base"] = base
        seen["wav"] = cmd[cmd.index("-f") + 1]
        seen["wav_existia"] = os.path.exists(seen["wav"])
        if write == "fixture":
            shutil.copyfile(FIXTURE, base + ".json")
        elif isinstance(write, dict):
            Path(base + ".json").write_text(json.dumps(write), encoding="utf-8")
        elif isinstance(write, str):
            Path(base + ".json").write_text(write, encoding="utf-8")
        return SimpleNamespace(returncode=rc, stdout=b"", stderr=stderr)

    return fake_run, seen


def _backend(run, **kw):
    # cycle, not iter: test_timeout_scales_proportionally_with_a_floor calls transcribe()
    # twice on the same backend (4 wall-clock reads); a two-item iter([...]) would run out.
    ticks = itertools.cycle([10.0, 10.5])
    return WhisperCppBackend(
        "large-v3", exe=Path("C:/wcpp/Release/whisper-cli.exe"),
        model_path=Path("C:/wcpp/models/ggml.bin"), run=run, clock=lambda: next(ticks), **kw,
    )


def _samples(seconds=100.0):
    return np.zeros(int(seconds * 16000), dtype=np.float32)


def test_caps_and_identity_match_the_contract(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    backend = _backend(lambda *a, **k: None)
    assert isinstance(backend, AsrBackend)
    assert backend.backend_id == "whispercpp"
    assert backend.caps == Caps("rejected", "degraded", "degraded")
    assert backend.quant == "q5_0"
    assert backend.device == "cuda"
    assert backend.model_id == "large-v3"
    assert backend.model_version == enginepin.MODELS_PIN["large-v3-q5_0"]["sha256"]
    assert backend.engine_version == f"whisper.cpp {enginepin.ENGINE_PIN['version']}"


def test_outside_win32_uses_native_for_device_and_an_unpinned_version(monkeypatch):
    # The PATH binary determines the device based on its build but does not report it;
    # labeling it cuda on macOS would lie in the JSON. Same label as core.probe.choose_route.
    monkeypatch.setattr(sys, "platform", "darwin")
    backend = _backend(lambda *a, **k: None)
    assert backend.device == "native"
    assert backend.engine_version == "whisper.cpp (PATH, unpinned)"
    assert backend.model_version == enginepin.MODELS_PIN["large-v3-q5_0"]["sha256"]  # The ggml is pinned.


def test_an_unpinned_model_is_rejected_during_construction():
    with pytest.raises(ValueError, match="is not pinned"):
        WhisperCppBackend("medium")


def test_warm_resolves_the_exe_and_model_from_the_pins(monkeypatch, tmp_path):
    monkeypatch.setattr(enginepin, "ensure_engine", lambda: tmp_path / "whisper-cli.exe")
    monkeypatch.setattr(enginepin, "ensure_model", lambda name: tmp_path / f"{name}.bin")
    backend = WhisperCppBackend("small")
    backend.warm()
    assert backend._exe == tmp_path / "whisper-cli.exe"
    assert backend._model_path == tmp_path / "small.bin"


def test_transcribe_parses_the_fixture():
    run, seen = _run_stub()
    result = _backend(run).transcribe(_samples(), TranscriptionRequest(language="es"))
    assert len(result.segments) == 6
    assert (result.segments[0].start, result.segments[0].end) == (0.0, 19.92)
    assert result.segments[0].text == " Segment one of the test track."
    assert result.segments[0].words == ()
    assert result.segments[0].native_signals.no_speech is None
    assert result.language == "es"
    assert result.native_signals.language_probability is None
    assert result.backend == "whispercpp" and result.model == "large-v3"
    assert result.latency_ms == 500
    cmd = seen["cmd"]
    assert cmd[0] == str(Path("C:/wcpp/Release/whisper-cli.exe"))
    assert cmd[cmd.index("-m") + 1] == str(Path("C:/wcpp/models/ggml.bin"))
    assert cmd[cmd.index("-l") + 1] == "es"
    assert cmd[cmd.index("-bs") + 1] == "5"
    assert cmd[cmd.index("-mc") + 1] == "0"  # Without it: loops and 3x wall time (measured 2026-07-27).
    assert "-np" in cmd and "-ojf" in cmd


def test_the_temp_wav_is_pcm16_mono_16k_and_gets_deleted():
    run, seen = _run_stub()
    samples = np.full(16000, 0.5, dtype=np.float32)
    recorded = {}

    def spy(cmd, **kw):
        with wave.open(cmd[cmd.index("-f") + 1]) as w:
            recorded.update(rate=w.getframerate(), ch=w.getnchannels(), width=w.getsampwidth(),
                           n=w.getnframes())
        return run(cmd, **kw)

    _backend(spy).transcribe(samples, TranscriptionRequest())
    assert recorded == {"rate": 16000, "ch": 1, "width": 2, "n": 16000}
    assert seen["wav_existia"] is True
    assert not os.path.exists(seen["wav"])
    assert not os.path.exists(seen["base"]) and not os.path.exists(seen["base"] + ".json")


def test_auto_is_passed_as_auto():
    run, seen = _run_stub()
    _backend(run).transcribe(_samples(), TranscriptionRequest(language="auto"))
    assert seen["cmd"][seen["cmd"].index("-l") + 1] == "auto"


def test_timeout_scales_proportionally_with_a_floor():
    run, seen = _run_stub()
    backend = _backend(run)
    backend.transcribe(_samples(100.0), TranscriptionRequest())
    assert seen["timeout"] == 400  # 4x duration.
    backend.transcribe(_samples(10.0), TranscriptionRequest())
    assert seen["timeout"] == 120  # Floor for a cold JIT.


def test_a_nonzero_rc_fails_with_the_tail_of_stderr():
    stderr = "\n".join(f"linea {i}" for i in range(20)).encode()
    run, seen = _run_stub(write=None, rc=3, stderr=stderr)
    with pytest.raises(RuntimeError) as ei:
        _backend(run).transcribe(_samples(), TranscriptionRequest())
    msg = str(ei.value)
    assert "whisper-cli rc=3" in msg and "linea 19" in msg and "linea 5" not in msg
    assert not os.path.exists(seen["base"] + ".json")


def test_missing_or_malformed_json_fails_with_a_cause():
    run, _ = _run_stub(write=None)
    with pytest.raises(RuntimeError, match="JSON"):
        _backend(run).transcribe(_samples(), TranscriptionRequest())
    run, seen = _run_stub(write="{esto no es json")
    with pytest.raises(RuntimeError, match="JSON"):
        _backend(run).transcribe(_samples(), TranscriptionRequest())
    assert not os.path.exists(seen["base"] + ".json")


def test_the_parser_filters_empty_text_segments():
    payload = {
        "result": {"language": "es"},
        "transcription": [
            {"offsets": {"from": 0, "to": 1000}, "text": "   "},
            {"offsets": {"from": 1000, "to": 2500}, "text": " hola"},
        ],
    }
    segs, lang = parse_ojf(payload)
    assert [(s.start, s.end, s.text) for s in segs] == [(1.0, 2.5, " hola")]
    assert lang == "es"


def test_hotwords_are_rejected_before_running():
    from speechtotext.asr import AsrError

    backend = _backend(lambda *a, **k: pytest.fail("must not run"))
    with pytest.raises(AsrError) as ei:
        backend.transcribe(_samples(1.0), TranscriptionRequest(hotwords=("Bézier",)))
    assert ei.value.code == "unsupported_option"


def test_a_non_ascii_temp_base_fails_before_running(monkeypatch, tmp_path):
    base = tmp_path / "output-ø"

    def fake_mkstemp(**kw):
        return os.open(str(base), os.O_CREAT | os.O_RDWR), str(base)

    monkeypatch.setattr(whispercpp.tempfile, "mkstemp", fake_mkstemp)
    with pytest.raises(RuntimeError, match="non-ASCII"):
        _backend(lambda *a, **k: pytest.fail("must not run")).transcribe(
            _samples(), TranscriptionRequest(),
        )
    assert not base.exists()
