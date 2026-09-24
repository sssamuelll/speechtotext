"""whisper.cpp backend: subprocess over whisper-cli (prebuilt CUDA pinned in enginepin).

This backend writes the input WAV from the samples (16 kHz mono PCM16) to an ASCII
temporary file: the user's audio path never travels in argv because whisper-cli is
main(char**), and a non-ASCII path arrives silently corrupted.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Callable

import numpy as np

from speechtotext.asr.base import AsrError, Caps, raise_if_cancelled
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

# What whisper-cli prints per segment while it decodes, -np included. Measured on v1.9.1
# (2026-09-24): one burst per 30 s window, CRLF on Windows, a blank first line.
_LIVE_LINE = re.compile(
    r"^\[(\d+):(\d{2}):(\d{2})\.(\d{3}) --> (\d+):(\d{2}):(\d{2})\.(\d{3})\]\s?(.*)$"
)


def _seconds(hours: str, minutes: str, secs: str, millis: str) -> float:
    return (((int(hours) * 60 + int(minutes)) * 60 + int(secs)) * 1000 + int(millis)) / 1000


def parse_live_line(line: bytes) -> TranscriptionSegment | None:
    """One stdout line, `[00:00:01.920 --> 00:00:04.060]  text`, as a segment with times
    local to the input. Anything else is None: the blank first line, notices, a segment
    with no text. Only a preview: the result is still read from the -ojf JSON."""
    match = _LIVE_LINE.match(line.decode("utf-8", errors="replace").rstrip("\r\n"))
    if match is None:
        return None
    *stamps, text = match.groups()
    if not text.strip():
        return None
    return TranscriptionSegment(_seconds(*stamps[:4]), _seconds(*stamps[4:]), text, (), _NO_SIGNALS)


def _timeout_for(n_samples: int) -> float:
    # 4x covers the measured warm case 17 times (19.3x real time); 120 s floor for cold JIT.
    return max(120, 4 * n_samples / SAMPLE_RATE)


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


class _Watchdog(threading.Thread):
    """Ends whisper-cli when the caller cancels or the run outlives its timeout. The reader
    blocks on stdout for a whole 30 s window; this thread is what makes cancelling prompt."""

    def __init__(self, proc, cancel: threading.Event | None, timeout: float, poll_s: float) -> None:
        super().__init__(daemon=True)
        self._proc, self._cancel, self._timeout, self._poll_s = proc, cancel, timeout, poll_s
        self.cancelled = False
        self.timed_out = False

    def run(self) -> None:
        deadline = time.monotonic() + self._timeout
        while self._proc.poll() is None:
            if self._cancel is not None and self._cancel.is_set():
                self.cancelled = True
                self._proc.terminate()
                return
            if time.monotonic() > deadline:
                self.timed_out = True
                self._proc.kill()
                return
            time.sleep(self._poll_s)


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
        popen: Callable = subprocess.Popen,
        clock: Callable[[], float] = time.perf_counter,
        poll_s: float = 0.2,
    ) -> None:
        if model not in enginepin._MODEL_ALIAS:
            raise ValueError(
                f"model {model!r} is not pinned for whispercpp; available: "
                f"{', '.join(sorted(enginepin._MODEL_ALIAS))}"
            )
        self._model = model
        self._exe = exe
        self._model_path = model_path
        self._popen = popen
        self._clock = clock
        self._poll_s = poll_s

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

    def transcribe(
        self,
        samples: np.ndarray,
        request: TranscriptionRequest,
        *,
        on_segment: Callable[[TranscriptionSegment], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> TranscriptionResult:
        if request.hotwords:
            # No degradation: warning about an inert knob would fabricate an effect that never happened.
            raise AsrError("unsupported_option", False,
                           "--hotwords has no effect with whispercpp (--prompt is inert with "
                           "-mc 0, measured 2026-07-27); use --engine faster-whisper")
        self.warm()
        raise_if_cancelled(cancel)
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
            self._run_cli(cmd, _timeout_for(len(samples)), on_segment, cancel)
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

    def _run_cli(
        self,
        cmd: list[str],
        timeout: float,
        on_segment: Callable[[TranscriptionSegment], None] | None,
        cancel: threading.Event | None,
    ) -> None:
        """Run whisper-cli, handing each live line to `on_segment`. stderr goes to a file, not
        a pipe: nothing reads it until the end, and a full pipe would stall the process."""
        with tempfile.TemporaryFile() as err:
            proc = self._popen(cmd, stdout=subprocess.PIPE, stderr=err)
            watchdog = _Watchdog(proc, cancel, timeout, self._poll_s)
            watchdog.start()
            try:
                for line in proc.stdout:
                    segment = parse_live_line(line)
                    if segment is not None and on_segment is not None:
                        on_segment(segment)
                returncode = proc.wait()
            finally:
                if proc.poll() is None:      # the callback raised: never leave whisper-cli running
                    proc.kill()
                    proc.wait()
                watchdog.join()
            if watchdog.cancelled:
                raise AsrError("cancelled", True, "transcription cancelled")
            if watchdog.timed_out:
                raise RuntimeError(f"whisper-cli did not finish in {timeout:.0f} s")
            if returncode != 0:
                err.seek(0)
                tail = "\n".join(err.read().decode("utf-8", errors="replace").splitlines()[-10:])
                raise RuntimeError(f"whisper-cli rc={returncode}: {tail}")
