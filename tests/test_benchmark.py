"""Tests del benchmark multimotor (schema speechtotext.bench/v1). Sin motores reales."""
from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

from speechtotext.core import benchmark
from speechtotext.core.benchmark_child import run_child
from speechtotext.core.enginepin import ENGINE_PIN

ALL_CAPS = ("hotwords", "word_timestamps", "native_signals", "vad")


def _cfg(engine: str, model: str) -> dict:
    return next(
        c for c in benchmark.candidate_configs()
        if c["engine"] == engine and c["model"] == model
    )


# --- candidate_configs -----------------------------------------------------------


def test_candidate_configs_siete_y_capacidades():
    configs = benchmark.candidate_configs()
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
    wer = {(c["engine"], c["model"]): c["wer_ref"] for c in benchmark.candidate_configs()}
    assert wer[("faster-whisper", "small")] == 0.419
    assert wer[("faster-whisper", "large-v3")] == 0.355
    assert wer[("whispercpp", "large-v3")] == 0.355
    assert wer[("faster-whisper", "tiny")] is None
    assert wer[("whispercpp", "small")] is None


# --- available_configs -----------------------------------------------------------


def test_available_configs_exe_ausente(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    viables, skipped = benchmark.available_configs()
    assert all(c["engine"] == "faster-whisper" for c in viables) and len(viables) == 5
    assert [s["model"] for s in skipped] == ["small", "large-v3"]
    assert all(s["engine"] == "whispercpp" and "ausente" in s["reason"] for s in skipped)


def test_available_configs_nvidia_smi_falla(monkeypatch, tmp_path):
    exe = tmp_path / "speechtotext" / "whisper-cpp" / ENGINE_PIN["version"] / "Release" / "whisper-cli.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"fake exe")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    def boom(*_a, **_k):
        raise FileNotFoundError("nvidia-smi no existe")

    monkeypatch.setattr(benchmark.subprocess, "run", boom)
    viables, skipped = benchmark.available_configs()
    assert len(viables) == 5
    assert len(skipped) == 2 and all("nvidia-smi" in s["reason"] for s in skipped)


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
    run, calls = _stub_run("ruido del motor\n" + json.dumps(payload) + "\n")
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


def test_run_config_hijo_con_error():
    run, _ = _stub_run(json.dumps({"error": "CUDA revento"}) + "\n")
    res = benchmark.run_config(_cfg("faster-whisper", "tiny"), "x.wav", 60.0, run=run)
    assert res["error"] == "CUDA revento"
    assert res["load_s"] is None and res["x_realtime"] is None


def test_run_config_salida_basura():
    run, _ = _stub_run("esto no es JSON\n")
    res = benchmark.run_config(_cfg("faster-whisper", "tiny"), "x.wav", 60.0, run=run)
    assert res["error"] is not None and "JSON" in res["error"]


def test_run_config_hijo_muerto_rc():
    run, _ = _stub_run("", returncode=1)
    res = benchmark.run_config(_cfg("faster-whisper", "tiny"), "x.wav", 60.0, run=run)
    assert "rc=1" in res["error"]


# --- run_benchmark ---------------------------------------------------------------


def test_run_benchmark_tabla_completa(monkeypatch, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFFfake")
    fila = {"engine": "faster-whisper", "model": "tiny", "error": None}
    monkeypatch.setattr(benchmark, "run_config", lambda cfg, w, d, **k: dict(fila, model=cfg["model"]))
    monkeypatch.setattr(
        benchmark, "available_configs",
        lambda: ([], [{"engine": "whispercpp", "model": "small", "reason": "exe ausente"}]),
    )
    monkeypatch.setattr(
        benchmark, "machine_info",
        lambda: {"cpu": "fake", "logical_cores": 8, "ram_gb": 16.0, "gpu": None},
    )
    vistos = []
    configs = [_cfg("faster-whisper", "tiny"), _cfg("faster-whisper", "base")]
    table = benchmark.run_benchmark(wav, 60.0, configs, progress=lambda c, r: vistos.append(c["model"]))
    assert table["schema_version"] == "speechtotext.bench/v1"
    assert [r["model"] for r in table["results"]] == ["tiny", "base"]
    assert vistos == ["tiny", "base"]
    assert table["skipped"][0]["reason"] == "exe ausente"
    assert table["audio"]["sha1"] == hashlib.sha1(b"RIFFfake").hexdigest()
    assert table["audio"]["duration_s"] == 60.0
    assert table["machine"]["cpu"] == "fake"


# --- write/read ------------------------------------------------------------------


def test_write_read_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    assert benchmark.bench_path() == tmp_path / "bench.json"
    assert benchmark.read_table() is None
    table = {"schema_version": "speechtotext.bench/v1", "results": [], "skipped": []}
    benchmark.write_table(table)
    assert benchmark.read_table() == table


# --- benchmark_child.run_child ---------------------------------------------------


class _FakeEngine:
    def __init__(self):
        self.opts = None

    def transcribe(self, path, **opts):
        self.opts = opts
        segs = iter([SimpleNamespace(text=" hola "), SimpleNamespace(text="mundo")])
        return segs, SimpleNamespace(language="es")


def test_run_child_con_motor_fake():
    eng = _FakeEngine()
    res = run_child(lambda *a: eng, "faster-whisper", "tiny", "cpu", "int8", "x.wav")
    assert res["error"] is None
    assert res["segments"] == 2 and res["chars"] == 9
    assert res["load_s"] >= 0 and res["transcribe_s"] >= 0
    assert res["peak_ram_mb"] > 0  # el hijo mide SU propio pico, en cualquier OS
    # opts canonicos del contrato
    assert eng.opts["language"] == "es" and eng.opts["beam_size"] == 5
    assert eng.opts["vad_filter"] is True
    assert eng.opts["condition_on_previous_text"] is False


def test_run_child_whispercpp_opts_efectivos():
    eng = _FakeEngine()
    res = run_child(lambda *a: eng, "whispercpp", "small", "cuda", "q5_0", "x.wav")
    assert res["error"] is None
    # bajo whispercpp los opts efectivos apagan vad/word_timestamps (contrato de degradacion)
    assert eng.opts["vad_filter"] is False and eng.opts["word_timestamps"] is False


def test_run_child_factory_revienta():
    def factory(*_a):
        raise RuntimeError("sin modelo")

    res = run_child(factory, "faster-whisper", "tiny", "cpu", "int8", "x.wav")
    assert "sin modelo" in res["error"]
    assert set(res) == {"error"}
