import logging
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from speechtotext.cli.app import app
from speechtotext.core import chunked
from speechtotext.speakers import registry

runner = CliRunner()


def _seg(start, end, text="hola que tal"):
    return SimpleNamespace(start=start, end=end, text=text)


def _info(duration, language="es", language_probability=1.0):
    return SimpleNamespace(
        duration=duration, language=language, language_probability=language_probability
    )


def _fake_transcribe(monkeypatch, tmp_path, segments, info, boom=None, calls=None):
    """Corta el camino de transcripción justo antes de Whisper: el audio nunca se abre.

    transcribe_file importa chunked dentro de la función, así que parchear el módulo basta.
    calls: lista opcional donde se registra cada (args, kwargs) de run_chunked.
    """
    audio = tmp_path / "charla.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(chunked, "probe_duration", lambda p: info.duration)
    monkeypatch.setattr(chunked, "should_chunk", lambda d, c: True)

    def run(*a, **k):
        if calls is not None:
            calls.append((a, k))
        if boom is not None:
            raise boom
        return segments, info

    monkeypatch.setattr(chunked, "run_chunked", run)
    return audio


def _invoke(audio, tmp_path, *extra, catch=True):
    return runner.invoke(
        app,
        ["transcribe", str(audio), "-f", "txt", "-o", str(tmp_path / "out")] + list(extra),
        catch_exceptions=catch,
    )


def test_voices_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    result = runner.invoke(app, ["voices"])
    assert result.exit_code == 0
    assert "Sin voces" in result.stdout


def test_forget_missing_returns_error(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    result = runner.invoke(app, ["forget", "Nadie"])
    assert result.exit_code == 1


def test_voices_lists_enrolled(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    registry.enroll("Samuel", np.array([1.0, 2.0], dtype=np.float32), seconds=10.0, model="m")
    result = runner.invoke(app, ["voices"])
    assert "Samuel" in result.stdout


def test_transcribe_still_registered():
    # el comando transcribe sigue existiendo como subcomando nombrado
    result = runner.invoke(app, ["transcribe", "--help"])
    assert result.exit_code == 0
    assert "diarize" in result.stdout


# --- 1.1 · cobertura en la línea de resumen -----------------------------------------


def test_resumen_avisa_cuando_se_perdio_audio(tmp_path, monkeypatch):
    # La corrida del hallazgo: 2206 s de audio, ~25% con texto, e imprimía OK a secas.
    segs = [_seg(0.0, 300.0), _seg(600.0, 850.0)]
    audio = _fake_transcribe(monkeypatch, tmp_path, segs, _info(2206.0))
    result = _invoke(audio, tmp_path)
    assert result.exit_code == 0
    assert "25%" in result.stdout
    assert "--no-vad" in result.stdout


def test_resumen_no_avisa_con_cobertura_alta(tmp_path, monkeypatch):
    segs = [_seg(0.0, 900.0)]
    audio = _fake_transcribe(monkeypatch, tmp_path, segs, _info(1000.0))
    result = _invoke(audio, tmp_path)
    assert result.exit_code == 0
    assert "90%" in result.stdout
    assert "--no-vad" not in result.stdout


def test_resumen_sobrevive_duracion_cero(tmp_path, monkeypatch):
    # info.duration == 0 no puede tumbar la línea de resumen por división por cero, y
    # tampoco puede afirmar 0%: sin denominador no hay medida, y el peor caso posible
    # (probe fallido) sería el único que no avisa.
    audio = _fake_transcribe(monkeypatch, tmp_path, [], _info(0.0))
    result = _invoke(audio, tmp_path)
    assert result.exit_code == 0
    assert "desconocida" in result.stdout
    assert "0%" not in result.stdout


def test_no_sugiere_no_vad_a_quien_ya_lo_apago(tmp_path, monkeypatch):
    # Las corridas 2, 3 y 5 del caso real ya iban con --no-vad: repetir el consejo ahí
    # es ruido garantizado. El aviso de pérdida se queda; la sugerencia no.
    segs = [_seg(0.0, 300.0)]
    audio = _fake_transcribe(monkeypatch, tmp_path, segs, _info(2206.0))
    result = _invoke(audio, tmp_path, "--no-vad")
    assert result.exit_code == 0
    assert "se perdió audio" in result.stdout
    assert "--no-vad" not in result.stdout


# --- 1.6 · idioma medido vs forzado -------------------------------------------------


def test_idioma_forzado_no_reporta_probabilidad(tmp_path, monkeypatch):
    # -l es es el default: ahí no se detectó nada, se obedeció al usuario.
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "-l", "es")
    assert "(forzado)" in result.stdout
    assert "prob=" not in result.stdout
    assert "Idioma detectado" not in result.stdout


def test_idioma_auto_reporta_probabilidad(tmp_path, monkeypatch):
    info = _info(10.0, language="en", language_probability=0.87)
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], info)
    result = _invoke(audio, tmp_path, "-l", "auto")
    assert "Idioma detectado" in result.stdout
    assert "prob=0.87" in result.stdout


def test_idioma_auto_omite_probabilidad_desconocida(tmp_path, monkeypatch):
    # La ruta troceada bajo --language auto tiró la probabilidad real: se calla, no inventa.
    info = _info(10.0, language="en", language_probability=None)
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], info)
    result = _invoke(audio, tmp_path, "-l", "auto")
    assert "Idioma detectado" in result.stdout
    assert "prob=" not in result.stdout


# --- 1.8 · logger de faster-whisper -------------------------------------------------


def test_logger_de_faster_whisper_en_info():
    # Sin esto, "VAD filter removed X of audio" nunca sale y Q1 no se puede cerrar.
    assert logging.getLogger("faster_whisper").level == logging.INFO


# --- 1.9 · guarda de memoria --------------------------------------------------------


def test_oom_sale_con_mensaje_accionable(tmp_path, monkeypatch):
    boom = RuntimeError("mkl_malloc: failed to allocate memory")
    audio = _fake_transcribe(monkeypatch, tmp_path, [], _info(2206.0), boom=boom)
    result = _invoke(audio, tmp_path)
    assert result.exit_code == 1
    assert "memoria" in result.stdout
    assert "medium" in result.stdout


def test_runtimeerror_ajeno_se_propaga(tmp_path, monkeypatch):
    # Tragar cualquier RuntimeError convertiría un bug en un mensaje de RAM equivocado.
    boom = RuntimeError("el modelo no existe")
    audio = _fake_transcribe(monkeypatch, tmp_path, [], _info(2206.0), boom=boom)
    with pytest.raises(RuntimeError, match="el modelo no existe"):
        _invoke(audio, tmp_path, catch=False)


# --- multimotor · CAPS, clamp y motor declarado en la salida ------------------------


def test_hotwords_con_whispercpp_rechaza_sin_construir(tmp_path, monkeypatch):
    # --prompt es inerte bajo -mc 0 (medido 2026-07-27): rechazo, no degradación.
    calls = []
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0), calls=calls)
    result = _invoke(audio, tmp_path, "--engine", "whispercpp", "--hotwords", "Aurelius")
    assert result.exit_code == 2
    assert "--hotwords no tiene efecto" in result.stdout
    assert "faster-whisper" in result.stdout
    assert calls == []  # jamás llegó a run_chunked: ni modelo ni caché


def test_engine_invalido_falla(tmp_path, monkeypatch):
    calls = []
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0), calls=calls)
    result = _invoke(audio, tmp_path, "--engine", "chatgpt")
    assert result.exit_code == 2
    assert "no existe" in result.stderr  # BadParameter nuestro, no "no such option"
    assert calls == []


def test_compute_type_no_mapeable_con_whispercpp(tmp_path, monkeypatch):
    # fp16 en la 980 pagina bajo WDDM (0.53x tiempo real): rechazo con la medición.
    calls = []
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0), calls=calls)
    result = _invoke(audio, tmp_path, "--engine", "whispercpp", "--compute-type", "float16")
    assert result.exit_code == 2
    assert "paging WDDM" in result.stderr  # el rechazo cita la medición, no un genérico
    assert calls == []


def test_aviso_vad_con_whispercpp(tmp_path, monkeypatch):
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "--engine", "whispercpp")
    assert result.exit_code == 0
    assert "no trae VAD" in result.stdout


def test_whispercpp_no_sugiere_no_vad(tmp_path, monkeypatch):
    # Con cobertura baja, el consejo "prueba --no-vad" es absurdo bajo whispercpp: el
    # motor no tiene VAD que apagar. El aviso de pérdida se queda; la sugerencia no.
    segs = [_seg(0.0, 300.0)]
    audio = _fake_transcribe(monkeypatch, tmp_path, segs, _info(2206.0))
    result = _invoke(audio, tmp_path, "--engine", "whispercpp")
    assert result.exit_code == 0
    assert "se perdió audio" in result.stdout
    assert "prueba --no-vad" not in result.stdout


def test_aviso_diarize_con_whispercpp(tmp_path, monkeypatch):
    from speechtotext.cli import app as cli_app

    # La diarización real necesita pyannote; aquí solo importa el aviso previo.
    monkeypatch.setattr(cli_app, "_run_diarization", lambda audio, segs, *a: segs)
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "--engine", "whispercpp", "--diarize")
    assert result.exit_code == 0
    assert "atribución por segmento" in result.stdout


def test_clamp_jobs_con_whispercpp_cuda(tmp_path, monkeypatch):
    # 4 subprocesos × 1.28 GB contra 4096 MiB: WDDM pagina 25x en silencio (medido).
    calls = []
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0), calls=calls)
    result = _invoke(audio, tmp_path, "--engine", "whispercpp", "-d", "cuda", "-j", "4")
    assert result.exit_code == 0
    assert "jobs=1" in result.stdout
    ((args, kwargs),) = calls
    assert args[2] == 1  # jobs es el tercer posicional de run_chunked
    assert kwargs["engine"] == "whispercpp"


def test_resumen_incluye_motor_default(tmp_path, monkeypatch):
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path)
    assert result.exit_code == 0
    assert "faster-whisper" in result.stdout  # el resumen declara el motor, siempre
    assert "Motor whisper.cpp" not in result.stdout  # header extra solo si no es default


def test_resumen_y_header_con_whispercpp(tmp_path, monkeypatch):
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "--engine", "whispercpp")
    assert result.exit_code == 0
    assert "whisper.cpp v1.9.1" in result.stdout  # header con versión pinneada
    assert "q5_0" in result.stdout  # la cuantización EFECTIVA, no 'auto'
    assert "whispercpp" in result.stdout  # motor resuelto en el resumen


def test_oom_whispercpp_aconseja_vram(tmp_path, monkeypatch):
    # El stderr de CUDA OOM también trae 'alloc': el consejo de RAM sería un falso amigo.
    boom = RuntimeError("ggml_cuda: failed to allocate 1.28 GB")
    audio = _fake_transcribe(monkeypatch, tmp_path, [], _info(2206.0), boom=boom)
    result = _invoke(audio, tmp_path, "--engine", "whispercpp", "-d", "cuda")
    assert result.exit_code == 1
    assert "VRAM" in result.stdout
    assert "--engine faster-whisper" in result.stdout
    assert "medium" not in result.stdout  # el consejo de RAM de CPU no aparece


def test_oom_faster_whisper_aconseja_ram(tmp_path, monkeypatch):
    # La contraparte: bajo el default el consejo sigue siendo el de RAM de CPU.
    boom = RuntimeError("mkl_malloc: failed to allocate memory")
    audio = _fake_transcribe(monkeypatch, tmp_path, [], _info(2206.0), boom=boom)
    result = _invoke(audio, tmp_path)
    assert result.exit_code == 1
    assert "medium" in result.stdout
    assert "VRAM" not in result.stdout


# --- identidad del motor en el JSON (G2) y device efectivo -----------------------------


def test_json_declara_motor_faster_whisper(tmp_path, monkeypatch):
    # El bloque engine tiene que llegar al ARCHIVO real, no solo existir como kwarg en
    # formats: este test cablea CLI -> write_json de punta a punta.
    import json

    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "-f", "json")
    assert result.exit_code == 0
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    eng = payload["engine"]
    assert eng["name"] == "faster-whisper"
    assert eng["version"].startswith("faster-whisper ")
    assert eng["quant"] == "int8"  # la efectiva de auto+cpu
    assert eng["device"] == "cpu"
    assert eng["selection"] == "explicit"
    assert "diarization" not in eng  # sin --diarize no se afirma nada


def test_json_declara_motor_whispercpp_y_device_cuda(tmp_path, monkeypatch):
    # El binario pinneado es build CUDA y corre en GPU SIEMPRE (medido en el smoke):
    # etiquetar cpu seria mentir. El device efectivo es cuda aunque nadie pase -d.
    import json

    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "-f", "json", "--engine", "whispercpp")
    assert result.exit_code == 0
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    eng = payload["engine"]
    assert eng == {
        "name": "whispercpp", "version": "whisper.cpp v1.9.1", "model": "small",
        "quant": "q5_0", "device": "cuda", "selection": "explicit",
    }
    assert "cuda" in result.stdout  # el header tampoco dice cpu


def test_whispercpp_rechaza_modelo_no_pinneado(tmp_path, monkeypatch):
    # Sin pre-validacion, ensure_model revienta con RuntimeError crudo a mitad de
    # corrida; el error debe llegar antes de construir nada y listar lo disponible.
    calls = []
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0), calls=calls)
    result = _invoke(audio, tmp_path, "--engine", "whispercpp", "-m", "medium")
    assert result.exit_code == 2
    assert "no está pinneado" in result.stderr
    assert "large-v3" in result.stderr and "small" in result.stderr
    assert calls == []  # run_chunked jamas se llamo


def test_whispercpp_avisa_el_remapeo_de_device(tmp_path, monkeypatch):
    # Pisar un -d cpu explicito en silencio seria la sustitucion callada que el
    # contrato prohibe: el remapeo a cuda se avisa siempre que no pidieran cuda.
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "--engine", "whispercpp", "-d", "cpu")
    assert result.exit_code == 0
    assert "corre en la GPU; device=cuda" in result.stdout

    result = _invoke(audio, tmp_path, "--engine", "whispercpp", "-d", "cuda")
    assert result.exit_code == 0
    assert "corre en la GPU" not in result.stdout  # quien pidio cuda no recibe ruido
