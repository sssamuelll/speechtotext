"""Isolated benchmark measurement child (schema speechtotext.bench/v1).

A process that has already loaded a model contaminates the next one's peak RAM: that is
why each config runs in THIS dedicated child process, which measures its OWN peak at the
end. It emits ONE JSON line to stdout and ALWAYS exits 0 — the parent must not die because
a config dies. No rich, no typer: bare positional argv.

argv: engine model device compute_type wav_path
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from speechtotext.asr.types import TranscriptionRequest
from speechtotext.core.transcribe import _apply_caps, load_audio

# Canonical benchmark request: identical for every config, so timings are comparable
# across engines. Each backend's CAPS degrade it where needed (whispercpp without VAD
# or words), just as on the real code path.
CANON_REQUEST = TranscriptionRequest(language="es", beam_size=5, vad=True, hotwords=(),
                                     word_timestamps=False)


def _peak_ram_mb() -> float:
    """Peak working set of THIS process, in MB."""
    if sys.platform == "win32":
        import ctypes

        # wintypes is imported only INSIDE the win32 branch (house rule).
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32")
        # Explicit restype/argtypes are mandatory: without them ctypes truncates the HANDLE
        # to 32 bits on x64 (lesson measured 2026-07-27).
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = []
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return pmc.PeakWorkingSetSize / (1024 * 1024)
    import resource

    # ponytail: ru_maxrss is in KB on Linux (bytes on macOS); the benchmark runs on
    # win32/linux, so we assume KB. Adjust with sys.platform == "darwin" if ever needed.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_child(backend_factory, engine: str, model: str, device: str, compute_type: str,
              wav_path: str) -> dict:
    """Load, transcribe, and measure. Separate from main() to test it with a fake engine.

    Any exception -> dict with "error"; never propagates. Decoding stays outside both
    wall clocks: load_s is warm(), transcribe_s is only backend.transcribe.
    """
    try:
        t0 = time.perf_counter()
        backend = backend_factory(engine, model, device, compute_type)
        # NOTE whispercpp: warm() only pins/verifies (load_s nearly zero); the model loads
        # inside the executable during transcribe, and the FIRST run pays CUDA JIT (~17 s measured
        # 2026-07-27). The child does NOT warm up: the number is for the machine as-is.
        backend.warm()
        load_s = time.perf_counter() - t0
        samples = load_audio(Path(wav_path))
        request, _ = _apply_caps(backend, CANON_REQUEST)
        t1 = time.perf_counter()
        result = backend.transcribe(samples, request)
        transcribe_s = time.perf_counter() - t1
        return {
            "load_s": round(load_s, 3),
            "transcribe_s": round(transcribe_s, 3),
            "segments": len(result.segments),
            "chars": sum(len(s.text.strip()) for s in result.segments),
            "peak_ram_mb": round(_peak_ram_mb(), 1),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 — the contract requires JSON with an error, not a traceback
        return {"error": f"{type(exc).__name__}: {exc}"}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    try:
        if len(args) != 5:
            raise ValueError(
                f"usage: python -m speechtotext.core.benchmark_child engine model device "
                f"compute_type wav (received {len(args)} args)"
            )
        from speechtotext.core.transcribe import make_backend

        result = run_child(make_backend, *args)
    except Exception as exc:  # noqa: BLE001 — even ImportError is emitted as JSON
        result = {"error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
