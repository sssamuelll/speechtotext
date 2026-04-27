"""Stubs compartidos: evita descargar modelos pesados durante los tests."""

from __future__ import annotations

import sys
import types


def _install_faster_whisper_stub() -> None:
    if "faster_whisper" in sys.modules:
        return
    module = types.ModuleType("faster_whisper")

    class _Info:
        language = "es"
        language_probability = 0.97
        duration = 12.34

    class WhisperModel:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        def transcribe(self, *_args, **_kwargs):
            class _Seg:
                def __init__(self, start: float, end: float, text: str) -> None:
                    self.start = start
                    self.end = end
                    self.text = text

            segs = [
                _Seg(0.0, 3.5, "Hola mundo."),
                _Seg(3.5, 7.2, "¿Cómo estás?"),
                _Seg(7.2, 12.3, "Probando 1 2 3."),
            ]
            return iter(segs), _Info()

    module.WhisperModel = WhisperModel
    sys.modules["faster_whisper"] = module


_install_faster_whisper_stub()
