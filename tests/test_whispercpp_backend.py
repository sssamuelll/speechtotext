"""WhisperCppBackend: el subprocess jamas corre en tests; se stubbea `run`."""
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


def test_la_fixture_es_sintetica_y_sin_rutas_de_maquina():
    """Guarda de lanzamiento: esta fixture viaja al repo público. Si alguien la regenera
    pegando una salida real de su máquina, se entera aquí y no en el filter-repo."""
    crudo = FIXTURE.read_text(encoding="utf-8")
    assert "\\" not in crudo, "ruta de Windows en la fixture"
    assert "Users" not in crudo, "ruta de máquina en la fixture"
    payload = json.loads(crudo)
    textos = [s["text"].strip() for s in payload["transcription"]]
    assert len(textos) == 6
    assert all(t.startswith("Segmento ") for t in textos), textos


def _run_stub(write="fixture", rc=0, stderr=b""):
    """`write`: 'fixture' copia la fixture sintética; un dict escribe ese JSON; None no
    escribe nada; un str crudo escribe basura. Devuelve (run, seen)."""
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
    # cycle, no iter: test_timeout_proporcional_con_piso llama transcribe() dos veces
    # sobre el mismo backend (4 lecturas de reloj), un iter([...]) de 2 se agotaria.
    ticks = itertools.cycle([10.0, 10.5])
    return WhisperCppBackend(
        "large-v3", exe=Path("C:/wcpp/Release/whisper-cli.exe"),
        model_path=Path("C:/wcpp/models/ggml.bin"), run=run, clock=lambda: next(ticks), **kw,
    )


def _samples(seconds=100.0):
    return np.zeros(int(seconds * 16000), dtype=np.float32)


def test_contrato_caps_e_identidad(monkeypatch):
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


def test_fuera_de_win32_device_native_y_version_sin_pin(monkeypatch):
    # El binario del PATH decide el dispositivo según su build y no lo dice; etiquetar
    # cuda en macOS sería mentir en el JSON. Misma etiqueta que core.probe.choose_route.
    monkeypatch.setattr(sys, "platform", "darwin")
    backend = _backend(lambda *a, **k: None)
    assert backend.device == "native"
    assert backend.engine_version == "whisper.cpp (PATH, sin pin)"
    assert backend.model_version == enginepin.MODELS_PIN["large-v3-q5_0"]["sha256"]  # el ggml sí va pinneado


def test_modelo_no_pinneado_se_rechaza_al_construir():
    with pytest.raises(ValueError, match="is not pinned"):
        WhisperCppBackend("medium")


def test_warm_resuelve_exe_y_modelo_por_el_pin(monkeypatch, tmp_path):
    monkeypatch.setattr(enginepin, "ensure_engine", lambda: tmp_path / "whisper-cli.exe")
    monkeypatch.setattr(enginepin, "ensure_model", lambda name: tmp_path / f"{name}.bin")
    backend = WhisperCppBackend("small")
    backend.warm()
    assert backend._exe == tmp_path / "whisper-cli.exe"
    assert backend._model_path == tmp_path / "small.bin"


def test_transcribe_parsea_la_fixture():
    run, seen = _run_stub()
    result = _backend(run).transcribe(_samples(), TranscriptionRequest(language="es"))
    assert len(result.segments) == 6
    assert (result.segments[0].start, result.segments[0].end) == (0.0, 19.92)
    assert result.segments[0].text == " Segmento uno de la pista de prueba."
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
    assert cmd[cmd.index("-mc") + 1] == "0"  # sin el: loops y 3x tiempo (medido 2026-07-27)
    assert "-np" in cmd and "-ojf" in cmd


def test_el_wav_temporal_es_pcm16_mono_16k_y_se_borra():
    run, seen = _run_stub()
    samples = np.full(16000, 0.5, dtype=np.float32)
    grabado = {}

    def espia(cmd, **kw):
        with wave.open(cmd[cmd.index("-f") + 1]) as w:
            grabado.update(rate=w.getframerate(), ch=w.getnchannels(), width=w.getsampwidth(),
                           n=w.getnframes())
        return run(cmd, **kw)

    _backend(espia).transcribe(samples, TranscriptionRequest())
    assert grabado == {"rate": 16000, "ch": 1, "width": 2, "n": 16000}
    assert seen["wav_existia"] is True
    assert not os.path.exists(seen["wav"])
    assert not os.path.exists(seen["base"]) and not os.path.exists(seen["base"] + ".json")


def test_auto_viaja_como_auto():
    run, seen = _run_stub()
    _backend(run).transcribe(_samples(), TranscriptionRequest(language="auto"))
    assert seen["cmd"][seen["cmd"].index("-l") + 1] == "auto"


def test_timeout_proporcional_con_piso():
    run, seen = _run_stub()
    backend = _backend(run)
    backend.transcribe(_samples(100.0), TranscriptionRequest())
    assert seen["timeout"] == 400  # 4x duracion
    backend.transcribe(_samples(10.0), TranscriptionRequest())
    assert seen["timeout"] == 120  # piso para el JIT frio


def test_rc_no_cero_revienta_con_cola_de_stderr():
    stderr = "\n".join(f"linea {i}" for i in range(20)).encode()
    run, seen = _run_stub(write=None, rc=3, stderr=stderr)
    with pytest.raises(RuntimeError) as ei:
        _backend(run).transcribe(_samples(), TranscriptionRequest())
    msg = str(ei.value)
    assert "whisper-cli rc=3" in msg and "linea 19" in msg and "linea 5" not in msg
    assert not os.path.exists(seen["base"] + ".json")


def test_json_ausente_o_malformado_revienta_con_causa():
    run, _ = _run_stub(write=None)
    with pytest.raises(RuntimeError, match="JSON"):
        _backend(run).transcribe(_samples(), TranscriptionRequest())
    run, seen = _run_stub(write="{esto no es json")
    with pytest.raises(RuntimeError, match="JSON"):
        _backend(run).transcribe(_samples(), TranscriptionRequest())
    assert not os.path.exists(seen["base"] + ".json")


def test_parser_filtra_segmentos_de_texto_vacio():
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


def test_hotwords_se_rechazan_antes_de_correr():
    from speechtotext.asr import AsrError

    backend = _backend(lambda *a, **k: pytest.fail("no debe correr"))
    with pytest.raises(AsrError) as ei:
        backend.transcribe(_samples(1.0), TranscriptionRequest(hotwords=("Bézier",)))
    assert ei.value.code == "unsupported_option"


def test_base_temporal_no_ascii_falla_antes_de_correr(monkeypatch, tmp_path):
    base = tmp_path / "salida-ñ"

    def fake_mkstemp(**kw):
        return os.open(str(base), os.O_CREAT | os.O_RDWR), str(base)

    monkeypatch.setattr(whispercpp.tempfile, "mkstemp", fake_mkstemp)
    with pytest.raises(RuntimeError, match="non-ASCII"):
        _backend(lambda *a, **k: pytest.fail("no debe correr")).transcribe(
            _samples(), TranscriptionRequest(),
        )
    assert not base.exists()
