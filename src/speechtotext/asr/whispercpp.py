"""Backend whisper.cpp: subprocess sobre whisper-cli (prebuilt CUDA pinneado en enginepin).

El wav de entrada lo escribe este backend desde las muestras (16 kHz mono PCM16) en un
temporal ASCII: el path del audio del usuario jamas viaja en argv, porque whisper-cli es
main(char**) y una ruta no-ASCII llega corrupta en silencio.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Callable

import numpy as np

from speechtotext.asr.base import AsrError, Caps
from speechtotext.asr.types import (
    NativeSignals,
    SegmentNativeSignals,
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptionSegment,
)
from speechtotext.core import enginepin

SAMPLE_RATE = 16000
_NO_SIGNALS = SegmentNativeSignals(None, None, None)


def parse_ojf(data: dict) -> tuple[list[TranscriptionSegment], str | None]:
    """Parser del JSON de `-ojf` (fixture real: tests/fixtures/whispercpp_ojf.json).

    Usa offsets (ms enteros -> segundos) y text; ignora tokens. El texto se conserva tal
    cual (espacio inicial incluido), paridad con faster-whisper. Senales que el motor no
    emite se OMITEN, jamas se rellenan: sin palabras, sin no_speech/avg_logprob.
    """
    segments = []
    for entry in data["transcription"]:
        if not entry["text"].strip():
            continue  # segmentos vacios no aportan texto y ensucian la cobertura
        segments.append(TranscriptionSegment(
            entry["offsets"]["from"] / 1000.0, entry["offsets"]["to"] / 1000.0,
            entry["text"], (), _NO_SIGNALS,
        ))
    return segments, data["result"].get("language")


def _write_wav(path: str, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


class WhisperCppBackend:
    backend_id = "whispercpp"
    # --prompt es INERTE bajo -mc 0 (bit-identico en 7 corridas, medido 2026-07-27);
    # sin -mc 0 contamina la ortografia global. Avisar "degradado" sobre un knob inerte
    # fabricaria un efecto que no ocurrio: rechazo. VAD y palabras: el motor no los trae.
    caps = Caps(hotwords="rechazado", vad="degradado", word_timestamps="degradado")
    quant = "q5_0"

    def __init__(
        self,
        model: str,
        *,
        exe: Path | None = None,
        model_path: Path | None = None,
        run: Callable = subprocess.run,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if model not in enginepin._MODEL_ALIAS:
            raise ValueError(
                f"modelo {model!r} no está pinneado para whispercpp; disponibles: "
                f"{', '.join(sorted(enginepin._MODEL_ALIAS))}"
            )
        self._model = model
        self._exe = exe
        self._model_path = model_path
        self._run = run
        self._clock = clock

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def model_version(self) -> str:
        return enginepin.MODELS_PIN[enginepin._MODEL_ALIAS[self._model]]["sha256"]

    @property
    def engine_version(self) -> str:
        if sys.platform != "win32":
            return "whisper.cpp (PATH, sin pin)"   # binario del sistema: versión no garantizada
        return f"whisper.cpp {enginepin.ENGINE_PIN['version']}"

    @property
    def device(self) -> str:
        # win32: build CUDA pinneado, corre en la GPU siempre. Fuera: el binario del PATH
        # decide según su build (Metal, CUDA, CPU) y no lo dice. Misma etiqueta que core.probe.
        return "cuda" if sys.platform == "win32" else "native"

    def warm(self) -> None:
        if self._exe is None:
            self._exe = enginepin.ensure_engine()
        if self._model_path is None:
            self._model_path = enginepin.ensure_model(self._model)

    def transcribe(self, samples: np.ndarray, request: TranscriptionRequest) -> TranscriptionResult:
        if request.hotwords:
            # No degradacion: avisar sobre un knob inerte fabricaria un efecto que no ocurrio.
            raise AsrError("unsupported_option", False,
                           "--hotwords no tiene efecto con whispercpp (--prompt es inerte con "
                           "-mc 0, medido 2026-07-27); usa --engine faster-whisper")
        self.warm()
        started = self._clock()
        fd, wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        fd, base = tempfile.mkstemp(prefix="wcpp-")  # base SIN extension; el CLI escribe base+".json"
        os.close(fd)
        json_path = base + ".json"
        try:
            for p in (wav, base):
                if not p.isascii():
                    raise RuntimeError(
                        f"ruta con caracteres no ASCII; whisper-cli no la soporta: {p}"
                    )
            _write_wav(wav, samples)
            cmd = [
                str(self._exe),
                "-m", str(self._model_path),
                "-f", wav,
                "-l", "auto" if request.language == "auto" else request.language,
                "-bs", str(request.beam_size),
                # -mc 0 HARDCODEADO: paridad con condition_on_previous_text=False.
                "-mc", "0",
                "-np", "-ojf", "-of", base,
            ]
            # 4x cubre 17 veces el caso caliente medido (19.3x tiempo real); piso 120 s
            # para el JIT frio.
            timeout = max(120, 4 * len(samples) / SAMPLE_RATE)
            proc = self._run(cmd, capture_output=True, timeout=timeout)
            if proc.returncode != 0:
                tail = "\n".join(
                    proc.stderr.decode("utf-8", errors="replace").splitlines()[-10:]
                )
                raise RuntimeError(f"whisper-cli rc={proc.returncode}: {tail}")
            try:
                data = json.loads(Path(json_path).read_text(encoding="utf-8"))
                segments, language = parse_ojf(data)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"whisper-cli termino bien pero su JSON no sirve ({json_path}): {exc}"
                ) from exc
        finally:
            for p in (wav, base, json_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass
        text = "".join(s.text for s in segments).strip()
        warnings = () if text else ("empty_transcript",)
        return TranscriptionResult(
            text=text,
            language=language or ("es" if request.language == "auto" else request.language),
            words=(),
            segments=tuple(segments),
            backend=self.backend_id,
            model=self.model_id,
            model_version=self.model_version,
            latency_ms=round((self._clock() - started) * 1000),
            native_signals=NativeSignals(None, None, None, None),
            warnings=warnings,
        )
