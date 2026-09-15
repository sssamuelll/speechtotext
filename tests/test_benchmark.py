"""Multi-engine benchmark tests (speechtotext.bench/v1 schema). No real engines."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from speechtotext.core import benchmark, probe
from speechtotext.core.benchmark_child import run_child

ALL_CAPS = ("hotwords", "word_timestamps", "native_signals", "vad")


def _cfg(engine: str, model: str) -> dict:
    return next(
        c for c in benchmark.candidate_configs(platform="win32")
        if c["engine"] == engine and c["model"] == model
    )


# --- candidate_configs -----------------------------------------------------------


def test_candidate_configs_returns_seven_entries_with_capabilities():
    configs = benchmark.candidate_configs(platform="win32")
    assert len(configs) == 7
    fw = [c for c in configs if c["engine"] == "faster-whisper"]
    assert [c["model"] for c in fw] == ["tiny", "base", "small", "medium", "large-v3"]
    assert all(c["device"] == "cpu" and c["quant"] == "int8" for c in fw)
    assert all(c["capabilities"] == dict.fromkeys(ALL_CAPS, True) for c in fw)
    wc = [c for c in configs if c["engine"] == "whispercpp"]
    assert [c["model"] for c in wc] == ["small", "large-v3"]
    assert all(c["device"] == "cuda" and c["quant"] == "q5_0" for c in wc)
    assert all(c["capabilities"] == dict.fromkeys(ALL_CAPS, False) for c in wc)


def test_candidate_configs_wer_ref():
    wer = {(c["engine"], c["model"]): c["wer_ref"] for c in benchmark.candidate_configs(platform="win32")}
    assert wer[("faster-whisper", "small")] == 0.419
    assert wer[("faster-whisper", "large-v3")] == 0.355
    assert wer[("whispercpp", "large-v3")] == 0.355
    assert wer[("faster-whisper", "tiny")] is None
    assert wer[("whispercpp", "small")] is None


def test_candidate_configs_outside_win32_uses_the_native_device_label():
    wc = [c for c in benchmark.candidate_configs(platform="darwin") if c["engine"] == "whispercpp"]
    assert wc and all(c["device"] == "native" for c in wc)
    assert all(c["device"] == "cpu" for c in benchmark.candidate_configs(platform="darwin")
               if c["engine"] == "faster-whisper")


# --- available_configs -----------------------------------------------------------


def _machine(**over):
    base = dict(platform="win32", cpu_count=8, ram_gb=32.0, cuda=False, gpu_name=None,
                vram_free_gb=None, whispercpp=None)
    base.update(over)
    return probe.Machine(**base)


def test_available_configs_skips_whispercpp_without_a_binary(monkeypatch):
    monkeypatch.setattr(probe, "machine", lambda: _machine(cuda=True, vram_free_gb=3.5))
    viable, skipped = benchmark.available_configs()
    assert all(c["engine"] == "faster-whisper" for c in viable) and len(viable) == 5
    assert [s["model"] for s in skipped] == ["small", "large-v3"]
    assert all(s["engine"] == "whispercpp" and "missing" in s["reason"] for s in skipped)


def test_available_configs_skips_whispercpp_without_a_gpu(monkeypatch):
    monkeypatch.setattr(probe, "machine", lambda: _machine(whispercpp=Path("C:/x/whisper-cli.exe")))
    viable, skipped = benchmark.available_configs()
    assert len(viable) == 5 and all("nvidia-smi" in s["reason"] for s in skipped)


def test_available_configs_with_a_gpu_and_binary_measures_all_seven(monkeypatch):
    monkeypatch.setattr(probe, "machine",
                        lambda: _machine(cuda=True, vram_free_gb=3.5, whispercpp=Path("C:/x/whisper-cli.exe")))
    viable, skipped = benchmark.available_configs()
    assert len(viable) == 7 and skipped == []


def test_available_configs_outside_win32_does_not_require_nvidia(monkeypatch):
    monkeypatch.setattr(probe, "machine", lambda: _machine(
        platform="darwin", whispercpp=Path("/opt/homebrew/bin/whisper-cli")))
    viable, skipped = benchmark.available_configs()
    assert skipped == []
    assert [c["device"] for c in viable if c["engine"] == "whispercpp"] == ["native", "native"]


def test_machine_info_matches_the_v1_schema_shape(monkeypatch):
    monkeypatch.setattr(probe, "machine",
                        lambda: _machine(gpu_name="GTX 980", cuda=True, vram_free_gb=3.5, ram_gb=31.9))
    info = benchmark.machine_info()
    assert set(info) == {"cpu", "logical_cores", "ram_gb", "gpu"}
    assert (info["logical_cores"], info["ram_gb"], info["gpu"]) == (8, 31.9, "GTX 980")


def test_table_caps_come_from_the_backends():
    from speechtotext.asr.whispercpp import WhisperCppBackend

    assert benchmark._CAPS["faster-whisper"] == {
        "hotwords": True, "word_timestamps": True, "native_signals": True, "vad": True}
    assert benchmark._CAPS["whispercpp"] == {
        "hotwords": False, "word_timestamps": False, "native_signals": False, "vad": False}
    assert benchmark._CAPS["whispercpp"]["vad"] == (WhisperCppBackend.caps.vad == "honored")


# --- run_config ------------------------------------------------------------------


def _stub_run(stdout: str, returncode: int = 0):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")

    return run, calls


def test_run_config_ok():
    payload = {"load_s": 1.5, "transcribe_s": 10.0, "segments": 3, "chars": 120,
               "peak_ram_mb": 512.5, "error": None}
    run, calls = _stub_run("engine noise\n" + json.dumps(payload) + "\n")
    res = benchmark.run_config(_cfg("faster-whisper", "small"), "x.wav", 60.0, run=run)
    assert calls[0][:3] == [sys.executable, "-m", "speechtotext.core.benchmark_child"]
    assert calls[0][3:] == ["faster-whisper", "small", "cpu", "int8", "x.wav"]
    assert res["error"] is None
    assert res["load_s"] == 1.5 and res["transcribe_s"] == 10.0
    assert res["x_realtime"] == 6.0
    assert res["peak_ram_mb"] == 512.5 and res["peak_vram_mb"] is None
    assert res["segments"] == 3 and res["chars"] == 120
    assert res["wer_ref"] == 0.419
    assert res["capabilities"]["hotwords"] is True


def test_run_config_returns_the_child_error():
    run, _ = _stub_run(json.dumps({"error": "CUDA crashed"}) + "\n")
    res = benchmark.run_config(_cfg("faster-whisper", "tiny"), "x.wav", 60.0, run=run)
    assert res["error"] == "CUDA crashed"
    assert res["load_s"] is None and res["x_realtime"] is None


def test_run_config_reports_an_error_for_garbage_output():
    run, _ = _stub_run("this is not JSON\n")
    res = benchmark.run_config(_cfg("faster-whisper", "tiny"), "x.wav", 60.0, run=run)
    assert res["error"] is not None and "JSON" in res["error"]


def test_run_config_includes_the_return_code_when_the_child_dies():
    run, _ = _stub_run("", returncode=1)
    res = benchmark.run_config(_cfg("faster-whisper", "tiny"), "x.wav", 60.0, run=run)
    assert "rc=1" in res["error"]


# --- run_benchmark ---------------------------------------------------------------


def test_run_benchmark_builds_the_complete_table(monkeypatch, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFFfake")
    row = {"engine": "faster-whisper", "model": "tiny", "error": None}
    monkeypatch.setattr(benchmark, "run_config", lambda cfg, w, d, **k: dict(row, model=cfg["model"]))
    monkeypatch.setattr(
        benchmark, "available_configs",
        lambda: ([], [{"engine": "whispercpp", "model": "small", "reason": "exe missing"}]),
    )
    monkeypatch.setattr(
        benchmark, "machine_info",
        lambda: {"cpu": "fake", "logical_cores": 8, "ram_gb": 16.0, "gpu": None},
    )
    seen_models = []
    configs = [_cfg("faster-whisper", "tiny"), _cfg("faster-whisper", "base")]
    table = benchmark.run_benchmark(wav, 60.0, configs, progress=lambda c, r: seen_models.append(c["model"]))
    assert table["schema_version"] == "speechtotext.bench/v1"
    assert [r["model"] for r in table["results"]] == ["tiny", "base"]
    assert seen_models == ["tiny", "base"]
    assert table["skipped"][0]["reason"] == "exe missing"
    assert table["audio"]["sha1"] == hashlib.sha1(b"RIFFfake").hexdigest()
    assert table["audio"]["duration_s"] == 60.0
    assert table["machine"]["cpu"] == "fake"


# --- write/read ------------------------------------------------------------------


def test_write_read_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    assert benchmark.bench_path() == tmp_path / "bench.json"
    assert benchmark.read_table() is None
    # recommendations is present: without the key, read_table would derive it (backward compatibility)
    # and the round trip would no longer preserve identity.
    table = {"schema_version": "speechtotext.bench/v1", "results": [], "skipped": [],
             "recommendations": []}
    benchmark.write_table(table)
    assert benchmark.read_table() == table


# --- benchmark_child.run_child ---------------------------------------------------


class _FakeBackend:
    def __init__(self, engine="faster-whisper"):
        from speechtotext.asr import Caps

        self.backend_id = engine
        self.caps = (Caps("rejected", "degraded", "degraded") if engine == "whispercpp"
                     else Caps("honored", "honored", "honored"))
        self.model_id, self.model_version, self.engine_version = "tiny", "1", "fake"
        self.quant, self.device = "int8", "cpu"
        self.request = None
        self.warmed = 0

    def warm(self):
        self.warmed += 1

    def transcribe(self, samples, request):
        from speechtotext.asr.types import (
            NativeSignals, SegmentNativeSignals, TranscriptionResult, TranscriptionSegment,
        )

        self.request = request
        no_signals = SegmentNativeSignals(None, None, None)
        return TranscriptionResult(
            text="hola mundo", language="es", words=(),
            segments=(TranscriptionSegment(0.0, 1.0, " hola ", (), no_signals),
                      TranscriptionSegment(1.0, 2.0, "mundo", (), no_signals)),
            backend=self.backend_id, model="tiny", model_version="1", latency_ms=1,
            native_signals=NativeSignals(None, None, None, None), warnings=(),
        )


@pytest.fixture
def _without_decoding(monkeypatch):
    import numpy as np

    from speechtotext.core import benchmark_child

    monkeypatch.setattr(benchmark_child, "load_audio", lambda p: np.zeros(16000, dtype=np.float32))


def test_run_child_with_a_fake_engine(_without_decoding):
    eng = _FakeBackend()
    res = run_child(lambda *a: eng, "faster-whisper", "tiny", "cpu", "int8", "x.wav")
    assert res["error"] is None
    assert res["segments"] == 2 and res["chars"] == 9
    assert res["load_s"] >= 0 and res["transcribe_s"] >= 0
    assert res["peak_ram_mb"] > 0  # The child measures ITS own peak, on any OS
    assert eng.warmed == 1
    # Canonical contract request: identical for every engine, so the timings are comparable
    assert eng.request.language == "es" and eng.request.beam_size == 5
    assert eng.request.vad is True and eng.request.word_timestamps is False


def test_run_child_whispercpp_applies_caps(_without_decoding):
    eng = _FakeBackend("whispercpp")
    res = run_child(lambda *a: eng, "whispercpp", "small", "cuda", "q5_0", "x.wav")
    assert res["error"] is None
    assert eng.request.vad is False and eng.request.word_timestamps is False


def test_run_child_reports_a_factory_failure(_without_decoding):
    def factory(*_a):
        raise RuntimeError("no model")

    res = run_child(factory, "faster-whisper", "tiny", "cpu", "int8", "x.wav")
    assert "no model" in res["error"]
    assert set(res) == {"error"}


# --- recommendations by use case -------------------------------------------------------


def _row(engine, model, x_rt, wer=None, error=None, caps=None):
    if caps is None:
        caps = benchmark._CAPS[engine]
    return {
        "engine": engine, "model": model, "quant": "int8" if engine == "faster-whisper" else "q5_0",
        "device": "cpu" if engine == "faster-whisper" else "cuda",
        "x_realtime": x_rt, "wer_ref": wer, "error": error, "capabilities": dict(caps),
    }


def _realistic_table():
    # Shape of the table measured on 2026-07-27 on the real machine.
    return [
        _row("faster-whisper", "base", 53.96),
        _row("faster-whisper", "small", 21.15, wer=0.419),
        _row("faster-whisper", "large-v3", 4.89, wer=0.355),
        _row("whispercpp", "large-v3", 10.66, wer=0.355),
    ]


def test_recommend_covers_all_declared_cases():
    recs = benchmark.recommend(_realistic_table())
    assert [r["case"] for r in recs] == [c["case"] for c in benchmark.USE_CASES]
    assert all(r["description"] and r["reason"] for r in recs)


def test_recommend_live_conversation_requires_a_resident_engine():
    # whispercpp is a subprocess (it loads the model per phrase): even if it is fast,
    # live conversation can only choose faster-whisper.
    recs = {r["case"]: r for r in benchmark.recommend(_realistic_table())}
    conv = recs["live_conversation"]["choice"]
    assert conv["engine"] == "faster-whisper"
    assert conv["model"] == "base"  # The fastest among the measured fw configurations


def test_recommend_quality_chooses_the_best_wer_and_breaks_ties_by_speed():
    # large-v3 ties on WER (0.355) in fw and whispercpp: the faster one wins (whispercpp).
    recs = {r["case"]: r for r in benchmark.recommend(_realistic_table())}
    top = recs["max_quality_transcription"]["choice"]
    assert (top["engine"], top["model"]) == ("whispercpp", "large-v3")


def test_recommend_fine_diarization_requires_word_timestamps():
    # whispercpp has no words: even if its large-v3 is faster, fine diarization
    # falls back to fw large-v3.
    recs = {r["case"]: r for r in benchmark.recommend(_realistic_table())}
    dia = recs["fine_diarization_transcription"]["choice"]
    assert (dia["engine"], dia["model"]) == ("faster-whisper", "large-v3")


def test_recommend_declares_a_case_without_a_candidate_and_gives_a_reason():
    # Hypothetical machine where only whispercpp ran: cases that require fw
    # cases have no candidate AND include a reason; no candidate is fabricated.
    only_wcpp = [_row("whispercpp", "large-v3", 10.66, wer=0.355)]
    recs = {r["case"]: r for r in benchmark.recommend(only_wcpp)}
    assert recs["live_conversation"]["choice"] is None
    assert "no measured config" in recs["live_conversation"]["reason"]
    assert recs["max_quality_transcription"]["choice"] is not None


def test_recommend_ignores_configs_with_errors():
    rows = [
        _row("faster-whisper", "large-v3", None, wer=0.355, error="murio"),
        _row("faster-whisper", "small", 21.15, wer=0.419),
    ]
    recs = {r["case"]: r for r in benchmark.recommend(rows)}
    top = recs["max_quality_transcription"]["choice"]
    assert top["model"] == "small"  # The broken one cannot win because of a good WER


def test_read_table_backward_compatibility_adds_recommendations(tmp_path, monkeypatch):
    # A table measured BEFORE this section gains it when read, without re-measuring.
    import json as _json

    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    old_table = {"schema_version": benchmark.SCHEMA_VERSION, "results": _realistic_table(),
                 "skipped": []}
    benchmark.bench_path().parent.mkdir(parents=True, exist_ok=True)
    benchmark.bench_path().write_text(_json.dumps(old_table), encoding="utf-8")
    table = benchmark.read_table()
    assert table["recommendations"]
    assert table["recommendations"][0]["case"] == "live_conversation"


def test_read_table_migrates_the_old_spanish_keys(tmp_path, monkeypatch):
    # A bench.json written by the code from BEFORE this fix has "recommendations"
    # present but with the old Spanish-language keys: read_table must remeasure them
    # from the intact `results`, not raise a KeyError on last week's file.
    import json as _json

    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    old_table = {
        "schema_version": benchmark.SCHEMA_VERSION, "results": _realistic_table(), "skipped": [],
        "recommendations": [
            # read_table() detects this row by its key spelling ("case" not in recs[0]):
            # translate these keys and the row starts looking like the new format, and
            # the migration branch this test exists to cover never runs again.
            {"caso": "conversacion_en_vivo", "que": "algo", "eleccion": None, "motivo": "algo"},  # spanish-is-data: pre-rename bench.json keys, matched by exact spelling
        ],
    }
    benchmark.bench_path().parent.mkdir(parents=True, exist_ok=True)
    benchmark.bench_path().write_text(_json.dumps(old_table), encoding="utf-8")
    table = benchmark.read_table()
    assert table["recommendations"][0]["case"] == "live_conversation"
    assert table["recommendations"][0]["choice"]["engine"] == "faster-whisper"


def test_read_table_does_not_recalculate_recommendations_that_use_the_new_key(tmp_path, monkeypatch):
    # The opposite of the previous test: a nonempty table already written with the new schema
    # is trusted as-is, without recalculation. If it were recalculated anyway, this stub
    # (deliberately different from what recommend() would actually produce) would not
    # survive being read.
    import json as _json

    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    fresh_table = {
        "schema_version": benchmark.SCHEMA_VERSION, "results": _realistic_table(), "skipped": [],
        "recommendations": [
            {"case": "live_conversation", "description": "d", "choice": None, "reason": "stub"},
        ],
    }
    benchmark.bench_path().parent.mkdir(parents=True, exist_ok=True)
    benchmark.bench_path().write_text(_json.dumps(fresh_table), encoding="utf-8")
    assert benchmark.read_table() == fresh_table
