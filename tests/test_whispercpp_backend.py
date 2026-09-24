"""WhisperCppBackend: the subprocess never runs in tests; `popen` is stubbed."""
import itertools
import json
import os
import shutil
import sys
import threading
import wave
from pathlib import Path

import numpy as np
import pytest

from speechtotext.asr import AsrBackend, AsrError, Caps, TranscriptionRequest
from speechtotext.asr import whispercpp
from speechtotext.asr.whispercpp import WhisperCppBackend, parse_ojf, parse_live_line
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


def _popen_stub(lines=(), rc=0, stderr_bytes=b"", write="fixture", hang=False):
    """A fake whisper-cli. It writes the -ojf JSON as `write` says ('fixture' copies the
    synthetic fixture, a dict writes that JSON, None writes nothing, a str writes garbage),
    prints `lines` on stdout and exits with `rc`; with `hang` it waits instead until it is
    terminated or killed. Returns (popen, seen)."""
    seen = {}

    class FakeProc:
        def __init__(self, cmd, stdout=None, stderr=None):
            seen.update(cmd=list(cmd), proc=self, base=cmd[cmd.index("-of") + 1],
                        wav=cmd[cmd.index("-f") + 1])
            seen["wav_existed"] = os.path.exists(seen["wav"])
            base = seen["base"]
            if write == "fixture":
                shutil.copyfile(FIXTURE, base + ".json")
            elif isinstance(write, dict):
                Path(base + ".json").write_text(json.dumps(write), encoding="utf-8")
            elif isinstance(write, str):
                Path(base + ".json").write_text(write, encoding="utf-8")
            stderr.write(stderr_bytes)
            self.returncode = None
            self.terminated = self.killed = False
            self._ended = threading.Event()
            self.stdout = self._print()

        def _print(self):
            yield from lines
            if hang:
                self._ended.wait(5)
            else:
                self._end(rc)

        def _end(self, code):
            if self.returncode is None:
                self.returncode = code
            self._ended.set()

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self._ended.wait(5)
            return self.returncode

        def terminate(self):
            self.terminated = True
            self._end(-15)

        def kill(self):
            self.killed = True
            self._end(-9)

    return FakeProc, seen


def _backend(popen, **kw):
    # cycle, not iter: some tests call transcribe() twice on the same backend.
    ticks = itertools.cycle([10.0, 10.5])
    return WhisperCppBackend(
        "large-v3", exe=Path("C:/wcpp/Release/whisper-cli.exe"),
        model_path=Path("C:/wcpp/models/ggml.bin"), popen=popen, clock=lambda: next(ticks),
        poll_s=0.01, **kw,
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
    popen, seen = _popen_stub()
    result = _backend(popen).transcribe(_samples(), TranscriptionRequest(language="es"))
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
    popen, seen = _popen_stub()
    samples = np.full(16000, 0.5, dtype=np.float32)
    recorded = {}

    def spy(cmd, **kw):
        with wave.open(cmd[cmd.index("-f") + 1]) as w:
            recorded.update(rate=w.getframerate(), ch=w.getnchannels(), width=w.getsampwidth(),
                           n=w.getnframes())
        return popen(cmd, **kw)

    _backend(spy).transcribe(samples, TranscriptionRequest())
    assert recorded == {"rate": 16000, "ch": 1, "width": 2, "n": 16000}
    assert seen["wav_existed"] is True
    assert not os.path.exists(seen["wav"])
    assert not os.path.exists(seen["base"]) and not os.path.exists(seen["base"] + ".json")


def test_auto_is_passed_as_auto():
    popen, seen = _popen_stub()
    _backend(popen).transcribe(_samples(), TranscriptionRequest(language="auto"))
    assert seen["cmd"][seen["cmd"].index("-l") + 1] == "auto"


def test_timeout_scales_proportionally_with_a_floor():
    assert whispercpp._timeout_for(100 * 16000) == 400   # 4x the duration
    assert whispercpp._timeout_for(10 * 16000) == 120    # floor for a cold JIT


def test_a_nonzero_rc_fails_with_the_tail_of_stderr():
    stderr = "\n".join(f"linea {i}" for i in range(20)).encode()
    popen, seen = _popen_stub(write=None, rc=3, stderr_bytes=stderr)
    with pytest.raises(RuntimeError) as ei:
        _backend(popen).transcribe(_samples(), TranscriptionRequest())
    msg = str(ei.value)
    assert "whisper-cli rc=3" in msg and "linea 19" in msg and "linea 5" not in msg
    assert not os.path.exists(seen["base"] + ".json")


def test_missing_or_malformed_json_fails_with_a_cause():
    popen, _ = _popen_stub(write=None)
    with pytest.raises(RuntimeError, match="JSON"):
        _backend(popen).transcribe(_samples(), TranscriptionRequest())
    popen, seen = _popen_stub(write="{esto no es json")
    with pytest.raises(RuntimeError, match="JSON"):
        _backend(popen).transcribe(_samples(), TranscriptionRequest())
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


def test_live_lines_become_segments_with_local_times():
    segment = parse_live_line(b"[00:01:02.340 --> 01:00:00.000]   hello\r\n")
    assert (segment.start, segment.end, segment.text) == (62.34, 3600.0, " hello")
    assert segment.words == () and segment.native_signals.no_speech is None


def test_live_lines_tolerate_crlf_blank_lines_and_bad_bytes():
    assert parse_live_line(b"\r\n") is None                                  # the blank first line
    assert parse_live_line(b"whisper_init_from_file: loading model\n") is None
    assert parse_live_line(b"[00:00:00.000 --> 00:00:01.000]    \r\n") is None  # no text
    segment = parse_live_line(b"[00:00:01.000 --> 00:00:02.000]   caf\xe9\r\n")  # not UTF-8
    assert segment is not None and segment.text.startswith(" caf")
    segment = parse_live_line(b"[00:00:02.000 --> 00:00:03.000]   next\n")  # LF-only (macOS/Linux)
    assert segment.text == " next"


def test_each_live_line_reaches_the_callback_and_the_json_is_still_the_result():
    popen, _ = _popen_stub(lines=[b"\r\n",
                                  b"[00:00:00.000 --> 00:00:01.500]   Segment one\r\n",
                                  b"[00:00:01.500 --> 00:00:03.000]   Segment two\r\n"])
    got = []
    result = _backend(popen).transcribe(_samples(), TranscriptionRequest(), on_segment=got.append)
    assert [(s.start, s.end, s.text) for s in got] == [(0.0, 1.5, " Segment one"),
                                                       (1.5, 3.0, " Segment two")]
    assert len(result.segments) == 6        # the fixture's six: the JSON decides the result


def test_cancel_terminates_whisper_cli_and_reports_cancelled():
    stop = threading.Event()
    popen, seen = _popen_stub(lines=[b"[00:00:00.000 --> 00:00:01.000]   one\r\n"], hang=True)
    with pytest.raises(AsrError) as ei:
        _backend(popen).transcribe(_samples(), TranscriptionRequest(),
                                   on_segment=lambda s: stop.set(), cancel=stop)
    assert (ei.value.code, str(ei.value)) == ("cancelled", "transcription cancelled")
    assert seen["proc"].terminated
    assert not os.path.exists(seen["wav"]) and not os.path.exists(seen["base"] + ".json")


def test_cancel_before_starting_never_launches_whisper_cli():
    stop = threading.Event()
    stop.set()
    backend = _backend(lambda *a, **k: pytest.fail("launched after cancel"))
    with pytest.raises(AsrError) as ei:
        backend.transcribe(_samples(), TranscriptionRequest(), cancel=stop)
    assert ei.value.code == "cancelled"


def test_a_callback_error_kills_whisper_cli_and_propagates():
    popen, seen = _popen_stub(lines=[b"[00:00:00.000 --> 00:00:01.000]   one\r\n"], hang=True)

    def broken(segment):
        raise ValueError("bug in the caller")

    with pytest.raises(ValueError, match="bug in the caller"):
        _backend(popen).transcribe(_samples(), TranscriptionRequest(), on_segment=broken)
    assert seen["proc"].killed
    assert not os.path.exists(seen["wav"])


def test_a_run_past_its_timeout_is_killed(monkeypatch):
    monkeypatch.setattr(whispercpp, "_timeout_for", lambda n_samples: 0.05)
    popen, seen = _popen_stub(hang=True)
    with pytest.raises(RuntimeError, match="did not finish"):
        _backend(popen).transcribe(_samples(), TranscriptionRequest())
    assert seen["proc"].killed
