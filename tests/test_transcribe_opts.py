"""Opciones de decodificación de Whisper (item (a) del backlog post-Fase 0)."""
from speechtotext.cli.app import _transcribe_opts


def test_desactiva_condition_on_previous_text():
    # El fix de tuning: sin esto el modelo arrastra el contexto de la ventana previa
    # y alucina hacia frases comunes (ver Fase 0 en docs/benchmark-turboscribe.md).
    assert _transcribe_opts("es", 5, True, None)["condition_on_previous_text"] is False


def test_pasa_los_parametros_de_decodificacion():
    opts = _transcribe_opts("es", 3, False, "Boconó")
    assert opts["language"] == "es"
    assert opts["beam_size"] == 3
    assert opts["vad_filter"] is False
    assert opts["hotwords"] == "Boconó"
