"""Checks for the hotword lexicon and HF environment defaults on Windows."""
import os
import re
import sys
from types import SimpleNamespace

from typer.testing import CliRunner

from speechtotext.cli.app import _load_hotwords_file, _resolve_hotwords, app
import numpy as np

runner = CliRunner()


def test_load_hotwords_file_handles_lines_and_commas(tmp_path):
    p = tmp_path / "lex.txt"
    p.write_text("Kestrel\nGlen Hollow, Meridian\n\n  Larkspur  \n", encoding="utf-8")
    assert _load_hotwords_file(p) == "Kestrel, Glen Hollow, Meridian, Larkspur"


def test_load_hotwords_file_tolerates_a_bom(tmp_path):
    p = tmp_path / "lex.txt"
    p.write_text("Larkspur, Windmere", encoding="utf-8-sig")  # Windows editor with a BOM
    assert _load_hotwords_file(p) == "Larkspur, Windmere"


def test_resolve_with_nothing_returns_none(tmp_path):
    # Without flags there are no hotwords: no global defaults that poison another audio file.
    assert _resolve_hotwords(None, None) is None
    assert _resolve_hotwords("   ", None) is None


def test_resolve_combines_file_and_inline_hotwords(tmp_path):
    p = tmp_path / "lex.txt"
    p.write_text("Larkspur", encoding="utf-8")
    assert _resolve_hotwords("Meridian", p) == "Larkspur, Meridian"


def test_hf_environment_defaults_are_set_on_windows():
    # Importing the CLI above should already have set these environment variables via setdefault.
    if sys.platform == "win32":
        assert os.environ.get("HF_HUB_DISABLE_SYMLINKS") == "1"
        assert os.environ.get("HF_HUB_DISABLE_XET") == "1"


# --- 5.2.2 · The prose states the actual mechanism and the long list warns ---------------


def _fake_transcribe(monkeypatch, tmp_path):
    """Stop execution just before the engine: the audio is never opened."""
    from speechtotext.asr import Caps
    from speechtotext.asr.types import (
        NativeSignals, SegmentNativeSignals, TranscriptionResult, TranscriptionSegment,
    )
    from speechtotext.core import transcribe as core_transcribe

    audio = tmp_path / "charla.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(core_transcribe, "load_audio", lambda p: np.zeros(160000, dtype=np.float32))
    monkeypatch.setattr(core_transcribe, "should_chunk", lambda d, c: False)
    result = TranscriptionResult(
        text="hola que tal", language="es", words=(),
        segments=(TranscriptionSegment(0.0, 9.0, "hola que tal", (), SegmentNativeSignals(None, None, None)),),
        backend="faster-whisper", model="small", model_version="1", latency_ms=1,
        native_signals=NativeSignals(None, None, None, 1.0), warnings=(),
    )
    fake = SimpleNamespace(backend_id="faster-whisper", model_id="small", device="cpu", quant="int8",
                           model_version="1", engine_version="faster-whisper 1.2.0",
                           caps=Caps("honored", "honored", "honored"), warm=lambda: None,
                           transcribe=lambda samples, request, **kw: result)
    monkeypatch.setattr(core_transcribe, "make_backend", lambda *a, **k: fake)
    return audio


def _invoke(audio, tmp_path, *extra):
    return runner.invoke(
        app,
        ["transcribe", str(audio), "-f", "txt", "-o", str(tmp_path / "out")] + list(extra),
    )


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(output: str) -> str:
    # Rich wraps at 80 columns under CliRunner and, when color is enabled, inserts escape
    # sequences inside tokens. The rendered output is not the contract, so clean it first.
    return " ".join(_ANSI.sub("", output).split())


def test_resolve_hotwords_docstring_does_not_contain_the_old_spanish_term_for_bias():
    # This cycle's defect WAS a docstring that claimed a false mechanism
    # ("probabilistic bias," plan §4): prose is tested like code.
    assert "sesgo" not in _resolve_hotwords.__doc__


def test_transcribe_help_does_not_contain_the_old_spanish_term_for_bias():
    result = runner.invoke(app, ["transcribe", "--help"])
    assert result.exit_code == 0
    assert "sesgar" not in result.stdout


def test_long_hotword_list_prints_the_count_and_warning(tmp_path, monkeypatch):
    # The real-case list: 25 terms. It must print the count and the warning with
    # the measurement (9 coverage points lost, 2026-08-03).
    audio = _fake_transcribe(monkeypatch, tmp_path)
    hotword_list = ", ".join(f"Proper Noun {i:02d}" for i in range(25))
    result = _invoke(audio, tmp_path, "--hotwords", hotword_list)
    assert result.exit_code == 0
    plain_output = _plain(result.stdout)
    assert "25 terms" in plain_output
    assert "characters" in plain_output
    assert "degraded coverage by 9 points" in plain_output
    assert "prior text" in plain_output


def test_short_hotword_list_does_not_warn(tmp_path, monkeypatch):
    # With 3 terms, the blackout measured no loss: the count is printed, but no warning.
    audio = _fake_transcribe(monkeypatch, tmp_path)
    result = _invoke(audio, tmp_path, "--hotwords", "Meridian, Larkspur, Kestrel")
    assert result.exit_code == 0
    plain_output = _plain(result.stdout)
    assert "3 terms" in plain_output
    assert "degraded" not in plain_output
