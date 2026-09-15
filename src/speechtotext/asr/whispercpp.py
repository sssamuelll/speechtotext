"""whisper.cpp backend: subprocess over whisper-cli (prebuilt CUDA pinned in enginepin).

This backend writes the input WAV from the samples (16 kHz mono PCM16) to an ASCII
temporary file: the user's audio path never travels in argv because whisper-cli is
main(char**), and a non-ASCII path arrives silently corrupted.
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
    """Parser for `-ojf` JSON (real fixture: tests/fixtures/whispercpp_ojf.json).

    Uses offsets (integer ms -> seconds) and text; ignores tokens. Text is preserved as-is
    (including leading space), for parity with faster-whisper. Native signals the engine
    does not emit are OMITTED, never filled in: no words, no no_speech/avg_logprob.
    """
    segments = []
    for entry in data["transcription"]:
        if not entry["text"].strip():
            continue  # empty segments add no text and muddy coverage
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
    # --prompt is INERT under -mc 0 (bit-identical across 7 runs, measured 2026-07-27);
    # without -mc 0 it contaminates the global spelling. Labeling an inert knob
    # "degraded" would fabricate an effect that never happened: reject it instead.
    # VAD and words: the engine doesn't provide them.
    caps = Caps(hotwords="rejected", vad="degraded", word_timestamps="degraded")
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
                f"model {model!r} is not pinned for whispercpp; available: "
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
            return "whisper.cpp (PATH, unpinned)"   # system binary: version not guaranteed
        return f"whisper.cpp {enginepin.ENGINE_PIN['version']}"

    @property
    def device(self) -> str:
        # win32: pinned CUDA build, always runs on the GPU. Elsewhere: the binary on PATH
        # decides based on its build (Metal, CUDA, CPU) and does not say. Same label as core.probe.
        return "cuda" if sys.platform == "win32" else "native"

    def warm(self) -> None:
        if self._exe is None:
            self._exe = enginepin.ensure_engine()
        if self._model_path is None:
            self._model_path = enginepin.ensure_model(self._model)

    def transcribe(self, samples: np.ndarray, request: TranscriptionRequest) -> TranscriptionResult:
        if request.hotwords:
            # No degradation: warning about an inert knob would fabricate an effect that never happened.
            raise AsrError("unsupported_option", False,
                           "--hotwords has no effect with whispercpp (--prompt is inert with "
                           "-mc 0, measured 2026-07-27); use --engine faster-whisper")
        self.warm()
        started = self._clock()
        fd, wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        fd, base = tempfile.mkstemp(prefix="wcpp-")  # extensionless base; CLI writes base+".json"
        os.close(fd)
        json_path = base + ".json"
        try:
            for p in (wav, base):
                if not p.isascii():
                    raise RuntimeError(
                        f"path has non-ASCII characters; whisper-cli does not support it: {p}"
                    )
            _write_wav(wav, samples)
            cmd = [
                str(self._exe),
                "-m", str(self._model_path),
                "-f", wav,
                "-l", "auto" if request.language == "auto" else request.language,
                "-bs", str(request.beam_size),
                # -mc 0 HARDCODED: parity with condition_on_previous_text=False.
                "-mc", "0",
                "-np", "-ojf", "-of", base,
            ]
            # 4x covers the measured warm case 17 times (19.3x real time); 120 s floor
            # for cold JIT.
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
                    f"whisper-cli exited cleanly but its JSON is unusable ({json_path}): {exc}"
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
