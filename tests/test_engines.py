import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import speechtotext.core.engines as engines
from speechtotext.core.engines import (
    CAPS,
    ENGINE_FASTER,
    ENGINE_WHISPERCPP,
    ENGINES,
    WhisperCppEngine,
    effective_opts,
    make_engine,
    quant_for,
)

FIXTURE = Path(__file__).parent / "fixtures" / "whispercpp_ojf.json"


# --- contrato de constantes y CAPS ------------------------------------------------

def test_constantes_del_contrato():
    assert ENGINE_FASTER == "faster-whisper"
    assert ENGINE_WHISPERCPP == "whispercpp"
    assert ENGINES == (ENGINE_FASTER, ENGINE_WHISPERCPP)


def test_caps_por_engine():
    assert CAPS[ENGINE_FASTER] == {
        "hotwords": "honrado", "vad_filter": "honrado",
        "word_timestamps": "honrado", "compute_type": "honrado",
    }
    assert CAPS[ENGINE_WHISPERCPP] == {
        "hotwords": "rechazado", "vad_filter": "degradado",
        "word_timestamps": "degradado", "compute_type": "mapeado",
    }


# --- quant_for --------------------------------------------------------------------

def test_quant_for_faster_pasa_tal_cual():
    assert quant_for(ENGINE_FASTER, "int8", "cpu") == "int8"
    assert quant_for(ENGINE_FASTER, "auto", "cuda") == "auto"


def test_quant_for_whispercpp_mapea_auto_y_q5():
    assert quant_for(ENGINE_WHISPERCPP, "auto", "cuda") == "q5_0"
    assert quant_for(ENGINE_WHISPERCPP, "q5_0", "cuda") == "q5_0"


def test_quant_for_whispercpp_rechaza_fp16_citando_la_medicion():
    with pytest.raises(ValueError) as ei:
        quant_for(ENGINE_WHISPERCPP, "float16", "cuda")
    msg = str(ei.value)
    assert "0.53x" in msg and "WDDM" in msg and "2026-07-27" in msg


# --- effective_opts ---------------------------------------------------------------

def _opts(**over):
    base = dict(language="es", beam_size=5, vad_filter=True, hotwords=None, word_timestamps=True)
    base.update(over)
    return base


def test_effective_opts_faster_tal_cual():
    opts = _opts()
    assert effective_opts(ENGINE_FASTER, opts) is opts


def test_effective_opts_whispercpp_apaga_vad_y_word_timestamps_sin_mutar():
    opts = _opts()
    eff = effective_opts(ENGINE_WHISPERCPP, opts)
    assert eff["vad_filter"] is False
    assert eff["word_timestamps"] is False
    assert eff["language"] == "es" and eff["beam_size"] == 5
    # el dict original no se muta: la llave del cache usa lo efectivo, no lo pedido
    assert opts["vad_filter"] is True and opts["word_timestamps"] is True


def test_effective_opts_whispercpp_revienta_con_hotwords():
    with pytest.raises(ValueError, match="hotwords"):
        effective_opts(ENGINE_WHISPERCPP, _opts(hotwords="Bocono"))


# --- make_engine ------------------------------------------------------------------

def test_make_engine_faster_construye_whispermodel_lazy(monkeypatch):
    calls = {}

    class FakeWM:
        def __init__(self, model, device=None, compute_type=None, **kw):
            calls.update(model=model, device=device, compute_type=compute_type, kw=kw)

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWM))
    m = make_engine(ENGINE_FASTER, "small", "cpu", "int8")
    assert isinstance(m, FakeWM)
    assert (calls["model"], calls["device"], calls["compute_type"]) == ("small", "cpu", "int8")
    assert calls["kw"] == {}  # None -> no viajan

    make_engine(ENGINE_FASTER, "small", "cpu", "int8", cpu_threads=3, num_workers=2)
    assert calls["kw"] == {"cpu_threads": 3, "num_workers": 2}


def test_make_engine_whispercpp_resuelve_pin_y_quant(monkeypatch, tmp_path):
    import speechtotext.core.enginepin as enginepin

    exe = tmp_path / "whisper-cli.exe"
    ggml = tmp_path / "ggml-large-v3-q5_0.bin"
    pedidos = []
    monkeypatch.setattr(enginepin, "ensure_engine", lambda: exe)
    monkeypatch.setattr(enginepin, "ensure_model", lambda name: (pedidos.append(name), ggml)[1])
    m = make_engine(ENGINE_WHISPERCPP, "large-v3", "cuda", "auto")
    assert isinstance(m, WhisperCppEngine)
    assert (m.exe_path, m.model_path, m.quant) == (exe, ggml, "q5_0")
    assert pedidos == ["large-v3"]


def test_make_engine_whispercpp_rechaza_compute_type_no_pinneado(monkeypatch, tmp_path):
    import speechtotext.core.enginepin as enginepin

    monkeypatch.setattr(enginepin, "ensure_engine", lambda: tmp_path / "x.exe")
    monkeypatch.setattr(enginepin, "ensure_model", lambda name: tmp_path / "x.bin")
    with pytest.raises(ValueError, match="WDDM"):
        make_engine(ENGINE_WHISPERCPP, "large-v3", "cuda", "float16")


def test_make_engine_desconocido():
    with pytest.raises(ValueError, match="desconocido"):
        make_engine("otro", "small", "cpu", "int8")


# --- adaptador WhisperCppEngine ---------------------------------------------------

def _run_stub(monkeypatch, write="fixture", rc=0, stderr=b""):
    """Stub de subprocess.run: JAMAS corre el exe real. `write` controla que aparece
    en base+'.json': 'fixture' copia la salida real grabada en la 980, un dict escribe
    ese JSON, None no escribe nada, un str crudo escribe basura."""
    seen = {}

    def fake_run(cmd, capture_output=None, timeout=None):
        seen["cmd"] = list(cmd)
        seen["timeout"] = timeout
        base = cmd[cmd.index("-of") + 1]
        seen["base"] = base
        if write == "fixture":
            shutil.copyfile(FIXTURE, base + ".json")
        elif isinstance(write, dict):
            Path(base + ".json").write_text(json.dumps(write), encoding="utf-8")
        elif isinstance(write, str):
            Path(base + ".json").write_text(write, encoding="utf-8")
        return SimpleNamespace(returncode=rc, stdout=b"", stderr=stderr)

    monkeypatch.setattr(engines.subprocess, "run", fake_run)
    return seen


def _engine():
    return WhisperCppEngine("C:/wcpp/Release/whisper-cli.exe", "C:/wcpp/models/ggml.bin", "q5_0")


def _wav(tmp_path):
    wav = tmp_path / "chunk.wav"
    wav.write_bytes(b"RIFF")
    return str(wav)


def test_transcribe_parsea_la_fixture_real(monkeypatch, tmp_path):
    seen = _run_stub(monkeypatch)
    segs, info = _engine().transcribe(
        _wav(tmp_path), language="es", beam_size=5, vad_filter=False, word_timestamps=False
    )
    assert len(segs) == 6
    assert (segs[0].start, segs[0].end) == (0.0, 19.92)
    assert segs[0].text == " Ay, gracias. Gracias por haberme dejado tantos años."
    assert all(s.words is None for s in segs)
    assert info.language == "es"
    assert info.language_probability is None
    # duration la fabrica el orquestador, el adaptador NO la inventa (ley de orquestacion)
    assert not hasattr(info, "duration")
    cmd = seen["cmd"]
    # str(Path(...)): en Windows el adaptador normaliza a backslash
    assert cmd[0] == str(Path("C:/wcpp/Release/whisper-cli.exe"))
    assert cmd[cmd.index("-m") + 1] == str(Path("C:/wcpp/models/ggml.bin"))
    assert cmd[cmd.index("-f") + 1] == _wav(tmp_path)
    assert cmd[cmd.index("-l") + 1] == "es"
    assert cmd[cmd.index("-bs") + 1] == "5"
    assert cmd[cmd.index("-mc") + 1] == "0"  # hardcodeado: sin el, loops y 3x tiempo
    assert "-np" in cmd and "-ojf" in cmd


def test_transcribe_limpia_base_y_json(monkeypatch, tmp_path):
    seen = _run_stub(monkeypatch)
    _engine().transcribe(_wav(tmp_path), language="es", beam_size=5)
    assert not os.path.exists(seen["base"])
    assert not os.path.exists(seen["base"] + ".json")
    # el wav de entrada NO es del adaptador: sigue vivo
    assert os.path.exists(_wav(tmp_path))


def test_transcribe_language_none_viaja_como_auto(monkeypatch, tmp_path):
    seen = _run_stub(monkeypatch)
    _engine().transcribe(_wav(tmp_path), language=None, beam_size=5)
    assert seen["cmd"][seen["cmd"].index("-l") + 1] == "auto"


def test_timeout_proporcional_con_piso_y_techo(monkeypatch, tmp_path):
    seen = _run_stub(monkeypatch)
    eng = _engine()
    eng.transcribe(_wav(tmp_path), language="es", beam_size=5, _duration_s=100)
    assert seen["timeout"] == 400  # 4x duracion
    assert "_duration_s" not in seen["cmd"]  # jamas viaja en argv
    eng.transcribe(_wav(tmp_path), language="es", beam_size=5, _duration_s=10)
    assert seen["timeout"] == 120  # piso para el JIT frio
    eng.transcribe(_wav(tmp_path), language="es", beam_size=5)
    assert seen["timeout"] == 3600  # duracion desconocida


def test_rc_no_cero_revienta_con_cola_de_stderr(monkeypatch, tmp_path):
    stderr = "\n".join(f"linea {i}" for i in range(20)).encode()
    seen = _run_stub(monkeypatch, write=None, rc=3, stderr=stderr)
    with pytest.raises(RuntimeError) as ei:
        _engine().transcribe(_wav(tmp_path), language="es", beam_size=5)
    msg = str(ei.value)
    assert "whisper-cli rc=3" in msg
    assert "linea 19" in msg      # la cola esta
    assert "linea 5" not in msg   # la cabeza no (solo ~10 lineas finales)
    assert not os.path.exists(seen["base"] + ".json")


def test_json_ausente_revienta_con_causa(monkeypatch, tmp_path):
    _run_stub(monkeypatch, write=None, rc=0)
    with pytest.raises(RuntimeError, match="JSON"):
        _engine().transcribe(_wav(tmp_path), language="es", beam_size=5)


def test_json_malformado_revienta_con_causa(monkeypatch, tmp_path):
    seen = _run_stub(monkeypatch, write="{esto no es json", rc=0)
    with pytest.raises(RuntimeError, match="JSON"):
        _engine().transcribe(_wav(tmp_path), language="es", beam_size=5)
    assert not os.path.exists(seen["base"] + ".json")  # cleanup tambien en fallo


def test_parser_filtra_segmentos_de_texto_vacio(monkeypatch, tmp_path):
    payload = {
        "result": {"language": "es"},
        "transcription": [
            {"offsets": {"from": 0, "to": 1000}, "text": "   "},
            {"offsets": {"from": 1000, "to": 2500}, "text": " hola"},
        ],
    }
    _run_stub(monkeypatch, write=payload)
    segs, info = _engine().transcribe(_wav(tmp_path), language="es", beam_size=5)
    assert [(s.start, s.end, s.text) for s in segs] == [(1.0, 2.5, " hola")]


def test_ruta_de_audio_no_ascii_falla_antes_de_correr(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise AssertionError("el subprocess jamas debe correr con ruta no-ASCII")

    monkeypatch.setattr(engines.subprocess, "run", boom)
    with pytest.raises(RuntimeError, match="no ASCII"):
        _engine().transcribe(str(tmp_path / "canción.wav"), language="es", beam_size=5)


def test_base_temporal_no_ascii_falla_antes_de_correr(monkeypatch, tmp_path):
    # %TEMP% con usuario no-ASCII: mkstemp hereda la ruta y whisper-cli la corrompe
    base = tmp_path / "salida-ñ"

    def fake_mkstemp(**kw):
        return os.open(str(base), os.O_CREAT | os.O_RDWR), str(base)

    monkeypatch.setattr(engines.tempfile, "mkstemp", fake_mkstemp)
    monkeypatch.setattr(engines.subprocess, "run", lambda *a, **k: pytest.fail("no debe correr"))
    with pytest.raises(RuntimeError, match="no ASCII"):
        _engine().transcribe(_wav(tmp_path), language="es", beam_size=5)
    assert not base.exists()  # el finally limpia incluso la base rechazada


# --- contrato dual completo: la salida whispercpp por el pipeline entero ----------------


def test_contrato_dual_whispercpp_por_el_pipeline_completo(tmp_path):
    """La salida del parser sobrevive shift -> roundtrip de checkpoint -> assign_segments
    en modo grueso -> write_txt. Sin esto, el contrato duck-typed solo esta probado hasta
    write_json y el resume con checkpoints whispercpp seria fe, no evidencia."""
    from speechtotext.core.chunked import seg_from_dict, seg_to_dict, shift_segments
    from speechtotext.core.formats import write_txt
    from speechtotext.speakers.diarization import assign_segments

    fixture = Path(__file__).parent / "fixtures" / "whispercpp_ojf.json"
    data = json.loads(fixture.read_text(encoding="utf-8"))
    segs, _info = engines._parse_ojf(data)

    # shift (la ruta troceada desplaza al offset global del trozo)
    shifted = shift_segments(segs, 600.0)
    assert shifted[0].start == 600.0 and shifted[0].words is None

    # roundtrip del checkpoint: lo que se escribe es lo que se relee
    revividos = [seg_from_dict(seg_to_dict(s)) for s in shifted]
    assert [(s.start, s.end, s.text) for s in revividos] == \
        [(s.start, s.end, s.text) for s in shifted]
    assert all(s.words is None for s in revividos)  # nadie fabrico words en el camino

    # atribucion GRUESA: sin words cae a un-hablante-por-segmento (rama prevista)
    turns = [(600.0, 640.0, "SPEAKER_00"), (640.0, 700.0, "SPEAKER_01")]
    etiquetados = assign_segments(revividos, turns)
    assert len(etiquetados) == len(revividos)  # cero cortes intra-segmento
    assert {s.speaker for s in etiquetados} <= {"SPEAKER_00", "SPEAKER_01", None}

    out = tmp_path / "salida.txt"
    write_txt(etiquetados, out)
    texto = out.read_text(encoding="utf-8")
    assert "Gracias por haberme dejado" in texto  # el texto real sobrevivio entero


# --- smoke solo-local: el subprocess real contra la instalacion pinneada ----------------


def _instalacion_gpu_disponible() -> bool:
    """Guarda de capacidad NO circular: mide el entorno (exe instalado + driver NVIDIA
    respondiendo), jamas llama al codigo bajo test. En CI (sin GPU, sin instalacion)
    esto es False y el smoke se salta; en la maquina real corre de verdad."""
    import shutil as _shutil
    import subprocess as _sp

    from speechtotext.core import enginepin

    exe = enginepin.install_root() / enginepin.ENGINE_PIN["exe_relpath"]
    modelo = enginepin.install_root() / "models" / "ggml-small.bin"
    if not (exe.exists() and modelo.exists()):
        return False
    smi = _shutil.which("nvidia-smi")
    if smi is None:
        return False
    try:
        return _sp.run([smi, "-L"], capture_output=True, timeout=10).returncode == 0
    except (OSError, _sp.TimeoutExpired):
        return False


@pytest.mark.skipif(not _instalacion_gpu_disponible(),
                    reason="requiere la instalacion pinneada de whisper.cpp y una GPU NVIDIA")
def test_smoke_whispercpp_subprocess_real(tmp_path):
    """Unica prueba que ejercita el exe DE VERDAD: subprocess -> JSON -> parser.
    Audio sintetico de 3 s (tono): el assert es que el camino no revienta y devuelve
    el contrato, no que transcriba algo (un tono no tiene palabras)."""
    import subprocess as _sp

    wav = tmp_path / "tono.wav"
    _sp.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=200:duration=3",
         "-ar", "16000", "-ac", "1", str(wav)],
        check=True, capture_output=True,
    )
    motor = engines.make_engine(engines.ENGINE_WHISPERCPP, "small", "cuda", "auto")
    segs, info = motor.transcribe(str(wav), language="es", beam_size=5,
                                  vad_filter=False, hotwords=None,
                                  condition_on_previous_text=False, word_timestamps=False)
    assert isinstance(segs, list)
    assert info.language == "es"
    assert info.language_probability is None
    assert all(s.words is None for s in segs)
