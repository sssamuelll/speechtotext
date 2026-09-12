"""Checks del léxico de hotwords y de los defaults de entorno HF en Windows."""
import os
import re
import sys
from types import SimpleNamespace

from typer.testing import CliRunner

from speechtotext.cli.app import _load_hotwords_file, _resolve_hotwords, app
import numpy as np

runner = CliRunner()


def test_load_hotwords_file_lineas_y_comas(tmp_path):
    p = tmp_path / "lex.txt"
    p.write_text("Táchira\nLa Guaira, Sofitasa\n\n  Boconó  \n", encoding="utf-8")
    assert _load_hotwords_file(p) == "Táchira, La Guaira, Sofitasa, Boconó"


def test_load_hotwords_file_tolera_bom(tmp_path):
    p = tmp_path / "lex.txt"
    p.write_text("Boconó, Cúcuta", encoding="utf-8-sig")  # editor de Windows con BOM
    assert _load_hotwords_file(p) == "Boconó, Cúcuta"


def test_resolve_sin_nada_es_none(tmp_path):
    # Sin flags no hay hotwords: nada de defaults globales que envenenen otro audio.
    assert _resolve_hotwords(None, None) is None
    assert _resolve_hotwords("   ", None) is None


def test_resolve_combina_archivo_e_inline(tmp_path):
    p = tmp_path / "lex.txt"
    p.write_text("Boconó", encoding="utf-8")
    assert _resolve_hotwords("Sofitasa", p) == "Boconó, Sofitasa"


def test_env_defaults_hf_en_windows():
    # Importar el CLI (arriba) ya debió setear estas env vars vía setdefault.
    if sys.platform == "win32":
        assert os.environ.get("HF_HUB_DISABLE_SYMLINKS") == "1"
        assert os.environ.get("HF_HUB_DISABLE_XET") == "1"


# --- 5.2.2 · la prosa dice el mecanismo real y la lista larga avisa --------------------


def _fake_transcribe(monkeypatch, tmp_path):
    """Corta el camino justo antes del motor: el audio nunca se abre."""
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
                           caps=Caps("honrado", "honrado", "honrado"), warm=lambda: None,
                           transcribe=lambda samples, request: result)
    monkeypatch.setattr(core_transcribe, "make_backend", lambda *a, **k: fake)
    return audio


def _invoke(audio, tmp_path, *extra):
    return runner.invoke(
        app,
        ["transcribe", str(audio), "-f", "txt", "-o", str(tmp_path / "out")] + list(extra),
    )


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plana(salida: str) -> str:
    # rich envuelve a 80 columnas bajo CliRunner y, cuando hay color, mete escapes dentro
    # de los tokens. Lo renderizado no es contrato: se limpia antes de asertar.
    return " ".join(_ANSI.sub("", salida).split())


def test_docstring_de_resolve_hotwords_no_dice_sesgo():
    # El defecto de este ciclo FUE un docstring que afirmaba un mecanismo falso
    # ("sesgo probabilístico", §4 del plan): la prosa se testea como el código.
    assert "sesgo" not in _resolve_hotwords.__doc__


def test_help_de_transcribe_no_dice_sesgar():
    result = runner.invoke(app, ["transcribe", "--help"])
    assert result.exit_code == 0
    assert "sesgar" not in result.stdout


def test_lista_larga_de_hotwords_imprime_conteo_y_aviso(tmp_path, monkeypatch):
    # La lista del caso real: 25 términos. Debe salir el conteo y la advertencia con
    # la medición (9 puntos de cobertura perdidos, 2026-08-03).
    audio = _fake_transcribe(monkeypatch, tmp_path)
    lista = ", ".join(f"Término Propio {i:02d}" for i in range(25))
    result = _invoke(audio, tmp_path, "--hotwords", lista)
    assert result.exit_code == 0
    salida = _plana(result.stdout)
    assert "25 términos" in salida
    assert "caracteres" in salida
    assert "degradaron la cobertura 9 puntos" in salida
    assert "texto previo" in salida


def test_lista_corta_de_hotwords_no_avisa(tmp_path, monkeypatch):
    # Con 3 términos la ablación no midió pérdida: el conteo sale, la advertencia no.
    audio = _fake_transcribe(monkeypatch, tmp_path)
    result = _invoke(audio, tmp_path, "--hotwords", "Sofitasa, Boconó, Táchira")
    assert result.exit_code == 0
    salida = _plana(result.stdout)
    assert "3 términos" in salida
    assert "degradaron" not in salida
