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


def _fake_transcribe(monkeypatch, tmp_path, segments, info, boom=None):
    """Corta el camino de transcripción justo antes de Whisper: el audio nunca se abre.

    transcribe_file importa chunked dentro de la función, así que parchear el módulo basta.
    """
    audio = tmp_path / "charla.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(chunked, "probe_duration", lambda p: info.duration)
    monkeypatch.setattr(chunked, "should_chunk", lambda d, c: True)

    def run(*a, **k):
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
