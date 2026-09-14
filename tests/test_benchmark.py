"""Tests del benchmark multimotor (schema speechtotext.bench/v1). Sin motores reales."""
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


def test_candidate_configs_siete_y_capacidades():
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


def test_candidate_configs_fuera_de_win32_etiqueta_native():
    wc = [c for c in benchmark.candidate_configs(platform="darwin") if c["engine"] == "whispercpp"]
    assert wc and all(c["device"] == "native" for c in wc)
    assert all(c["device"] == "cpu" for c in benchmark.candidate_configs(platform="darwin")
               if c["engine"] == "faster-whisper")


# --- available_configs -----------------------------------------------------------


def _maquina(**over):
    base = dict(platform="win32", cpu_count=8, ram_gb=32.0, cuda=False, gpu_name=None,
                vram_free_gb=None, whispercpp=None)
    base.update(over)
    return probe.Machine(**base)


def test_available_configs_sin_binario(monkeypatch):
    monkeypatch.setattr(probe, "machine", lambda: _maquina(cuda=True, vram_free_gb=3.5))
    viables, skipped = benchmark.available_configs()
    assert all(c["engine"] == "faster-whisper" for c in viables) and len(viables) == 5
    assert [s["model"] for s in skipped] == ["small", "large-v3"]
    assert all(s["engine"] == "whispercpp" and "ausente" in s["reason"] for s in skipped)


def test_available_configs_sin_gpu(monkeypatch):
    monkeypatch.setattr(probe, "machine", lambda: _maquina(whispercpp=Path("C:/x/whisper-cli.exe")))
    viables, skipped = benchmark.available_configs()
    assert len(viables) == 5 and all("nvidia-smi" in s["reason"] for s in skipped)


def test_available_configs_con_gpu_y_binario_mide_las_siete(monkeypatch):
    monkeypatch.setattr(probe, "machine",
                        lambda: _maquina(cuda=True, vram_free_gb=3.5, whispercpp=Path("C:/x/whisper-cli.exe")))
    viables, skipped = benchmark.available_configs()
    assert len(viables) == 7 and skipped == []


def test_available_configs_fuera_de_win32_no_exige_nvidia(monkeypatch):
    monkeypatch.setattr(probe, "machine", lambda: _maquina(
        platform="darwin", whispercpp=Path("/opt/homebrew/bin/whisper-cli")))
    viables, skipped = benchmark.available_configs()
    assert skipped == []
    assert [c["device"] for c in viables if c["engine"] == "whispercpp"] == ["native", "native"]


def test_machine_info_es_la_forma_del_schema_v1(monkeypatch):
    monkeypatch.setattr(probe, "machine",
                        lambda: _maquina(gpu_name="GTX 980", cuda=True, vram_free_gb=3.5, ram_gb=31.9))
    info = benchmark.machine_info()
    assert set(info) == {"cpu", "logical_cores", "ram_gb", "gpu"}
    assert (info["logical_cores"], info["ram_gb"], info["gpu"]) == (8, 31.9, "GTX 980")


def test_caps_de_la_tabla_salen_de_los_backends():
    from speechtotext.asr.whispercpp import WhisperCppBackend

    assert benchmark._CAPS["faster-whisper"] == {
        "hotwords": True, "word_timestamps": True, "native_signals": True, "vad": True}
    assert benchmark._CAPS["whispercpp"] == {
        "hotwords": False, "word_timestamps": False, "native_signals": False, "vad": False}
    assert benchmark._CAPS["whispercpp"]["vad"] == (WhisperCppBackend.caps.vad == "honrado")


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
    # recommendations presente: sin la clave, read_table la derivaria (retrocompat)
    # y el roundtrip dejaria de ser identidad.
    table = {"schema_version": "speechtotext.bench/v1", "results": [], "skipped": [],
             "recommendations": []}
    benchmark.write_table(table)
    assert benchmark.read_table() == table


# --- benchmark_child.run_child ---------------------------------------------------


class _FakeBackend:
    def __init__(self, engine="faster-whisper"):
        from speechtotext.asr import Caps

        self.backend_id = engine
        self.caps = (Caps("rechazado", "degradado", "degradado") if engine == "whispercpp"
                     else Caps("honrado", "honrado", "honrado"))
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
        nada = SegmentNativeSignals(None, None, None)
        return TranscriptionResult(
            text="hola mundo", language="es", words=(),
            segments=(TranscriptionSegment(0.0, 1.0, " hola ", (), nada),
                      TranscriptionSegment(1.0, 2.0, "mundo", (), nada)),
            backend=self.backend_id, model="tiny", model_version="1", latency_ms=1,
            native_signals=NativeSignals(None, None, None, None), warnings=(),
        )


@pytest.fixture
def _sin_decodificar(monkeypatch):
    import numpy as np

    from speechtotext.core import benchmark_child

    monkeypatch.setattr(benchmark_child, "load_audio", lambda p: np.zeros(16000, dtype=np.float32))


def test_run_child_con_motor_fake(_sin_decodificar):
    eng = _FakeBackend()
    res = run_child(lambda *a: eng, "faster-whisper", "tiny", "cpu", "int8", "x.wav")
    assert res["error"] is None
    assert res["segments"] == 2 and res["chars"] == 9
    assert res["load_s"] >= 0 and res["transcribe_s"] >= 0
    assert res["peak_ram_mb"] > 0  # el hijo mide SU propio pico, en cualquier OS
    assert eng.warmed == 1
    # peticion canonica del contrato: identica para todo motor, para que los tiempos comparen
    assert eng.request.language == "es" and eng.request.beam_size == 5
    assert eng.request.vad is True and eng.request.word_timestamps is False


def test_run_child_whispercpp_aplica_caps(_sin_decodificar):
    eng = _FakeBackend("whispercpp")
    res = run_child(lambda *a: eng, "whispercpp", "small", "cuda", "q5_0", "x.wav")
    assert res["error"] is None
    assert eng.request.vad is False and eng.request.word_timestamps is False


def test_run_child_factory_revienta(_sin_decodificar):
    def factory(*_a):
        raise RuntimeError("sin modelo")

    res = run_child(factory, "faster-whisper", "tiny", "cpu", "int8", "x.wav")
    assert "sin modelo" in res["error"]
    assert set(res) == {"error"}


# --- recomendaciones por caso de uso ----------------------------------------------------


def _fila(engine, model, x_rt, wer=None, error=None, caps=None):
    if caps is None:
        caps = benchmark._CAPS[engine]
    return {
        "engine": engine, "model": model, "quant": "int8" if engine == "faster-whisper" else "q5_0",
        "device": "cpu" if engine == "faster-whisper" else "cuda",
        "x_realtime": x_rt, "wer_ref": wer, "error": error, "capabilities": dict(caps),
    }


def _tabla_realista():
    # La forma de la tabla medida el 2026-07-27 en la maquina real.
    return [
        _fila("faster-whisper", "base", 53.96),
        _fila("faster-whisper", "small", 21.15, wer=0.419),
        _fila("faster-whisper", "large-v3", 4.89, wer=0.355),
        _fila("whispercpp", "large-v3", 10.66, wer=0.355),
    ]


def test_recommend_cubre_todos_los_casos_declarados():
    recs = benchmark.recommend(_tabla_realista())
    assert [r["caso"] for r in recs] == [c["caso"] for c in benchmark.USE_CASES]
    assert all(r["que"] and r["motivo"] for r in recs)


def test_recommend_conversacion_exige_motor_residente():
    # whispercpp es subprocess (carga el modelo por frase): aunque sea rapido, la
    # conversacion en vivo solo puede elegir faster-whisper.
    recs = {r["caso"]: r for r in benchmark.recommend(_tabla_realista())}
    conv = recs["conversacion_en_vivo"]["eleccion"]
    assert conv["engine"] == "faster-whisper"
    assert conv["model"] == "base"  # la mas rapida entre las fw medidas


def test_recommend_calidad_elige_mejor_wer_y_desempata_por_velocidad():
    # large-v3 empata WER (0.355) en fw y whispercpp: gana el mas rapido (whispercpp).
    recs = {r["caso"]: r for r in benchmark.recommend(_tabla_realista())}
    top = recs["transcripcion_maxima_calidad"]["eleccion"]
    assert (top["engine"], top["model"]) == ("whispercpp", "large-v3")


def test_recommend_diarizacion_fina_exige_word_timestamps():
    # whispercpp no tiene words: aunque su large-v3 sea mas rapido, la diarizacion
    # fina cae al fw large-v3.
    recs = {r["caso"]: r for r in benchmark.recommend(_tabla_realista())}
    dia = recs["transcripcion_con_diarizacion_fina"]["eleccion"]
    assert (dia["engine"], dia["model"]) == ("faster-whisper", "large-v3")


def test_recommend_caso_sin_candidata_se_declara_con_razon():
    # Maquina hipotetica donde solo corrio whispercpp: los casos que exigen
    # capacidades de fw quedan sin candidata Y CON motivo, no inventados.
    solo_wcpp = [_fila("whispercpp", "large-v3", 10.66, wer=0.355)]
    recs = {r["caso"]: r for r in benchmark.recommend(solo_wcpp)}
    assert recs["conversacion_en_vivo"]["eleccion"] is None
    assert "ninguna config" in recs["conversacion_en_vivo"]["motivo"]
    assert recs["transcripcion_maxima_calidad"]["eleccion"] is not None


def test_recommend_ignora_configs_con_error():
    filas = [
        _fila("faster-whisper", "large-v3", None, wer=0.355, error="murio"),
        _fila("faster-whisper", "small", 21.15, wer=0.419),
    ]
    recs = {r["caso"]: r for r in benchmark.recommend(filas)}
    top = recs["transcripcion_maxima_calidad"]["eleccion"]
    assert top["model"] == "small"  # la rota no puede ganar por buen WER


def test_read_table_retrocompat_anade_recommendations(tmp_path, monkeypatch):
    # Una tabla medida ANTES de esta seccion la gana al leerse, sin re-medir.
    import json as _json

    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    vieja = {"schema_version": benchmark.SCHEMA_VERSION, "results": _tabla_realista(),
             "skipped": []}
    benchmark.bench_path().parent.mkdir(parents=True, exist_ok=True)
    benchmark.bench_path().write_text(_json.dumps(vieja), encoding="utf-8")
    table = benchmark.read_table()
    assert table["recommendations"]
    assert table["recommendations"][0]["caso"] == "conversacion_en_vivo"
