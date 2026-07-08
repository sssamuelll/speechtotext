"""Checks del léxico de hotwords y de los defaults de entorno HF en Windows."""
import os
import sys

from speechtotext.cli.app import _load_hotwords_file, _resolve_hotwords


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
