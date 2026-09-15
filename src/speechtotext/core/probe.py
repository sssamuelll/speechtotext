"""Machine probe (< 1 s, without loading models) and route selection.

Rules (spec §5.1): the probe NEVER changes the model; explicit `engine`/`device` values are
honored and the probe only fills in what is missing; every remapping is reported in `Route.reason`.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from speechtotext.asr.base import AsrError

ENGINE_FASTER = "faster-whisper"
ENGINE_WHISPERCPP = "whispercpp"
ENGINES = (ENGINE_FASTER, ENGINE_WHISPERCPP)

# ponytail: thresholds measured on ONE machine (5900X + 4 GB GTX 980, 2026-09-11), not
# laws. `bench` is what tunes them; moving them is a deliberate commit with measurement.
VRAM_FW_GB = 5.0     # faster-whisper large-v3 float16 on CUDA
VRAM_WCPP_GB = 2.0   # whisper.cpp large-v3 q5_0: 1.3 GB measured on the 980
RAM_MIN_GB = {"large-v3": 6.0, "small": 3.0}   # ceiling above the measured 3.6 GB peak in int8

# Factor × audio duration, measured on the reference machine (scripts/benchmark_chart.py,
# 2026-09-11: 1.27x, 6.4x, 7.97x, and 15.7x real time). None = unmeasured. This machine's
# bench.json takes precedence over this table (eta_factor).
ETA_FACTORS = {
    (ENGINE_FASTER, "cpu", "int8", "large-v3"): round(1 / 1.27, 3),
    (ENGINE_FASTER, "cpu", "int8", "small"): round(1 / 6.4, 3),
    (ENGINE_WHISPERCPP, "cuda", "q5_0", "large-v3"): round(1 / 7.97, 3),
    (ENGINE_WHISPERCPP, "cuda", "q5_0", "small"): round(1 / 15.7, 3),
}


@dataclass(frozen=True)
class Route:
    engine: str
    device: str
    compute_type: str
    reason: str                       # one sentence to print; empty if there is nothing to report
    eta_factor: float | None = None   # × audio duration; None = unmeasured
    estimated: bool = True            # False if the factor came from THIS machine's bench.json


@dataclass(frozen=True)
class Machine:
    platform: str                 # sys.platform: win32 | darwin | linux
    cpu_count: int
    ram_gb: float | None          # None = could not be measured
    cuda: bool                    # nvidia-smi responds: an NVIDIA GPU with a driver is present
    gpu_name: str | None
    vram_free_gb: float | None
    whispercpp: Path | None       # already installed binary (pinned or on the PATH); never downloads


def nvidia_smi(query: str, timeout_s: float = 10) -> str | None:
    """One nvidia-smi query; None if there is no NVIDIA GPU or the command does not respond."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _ram_gb() -> float | None:
    """Total physical RAM in GB; None if the system does not report it."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes  # only after checking win32 (house rule)

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        kernel32 = ctypes.WinDLL("kernel32")
        # Explicit argtypes/restype: without them ctypes truncates pointers on x64.
        kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
        kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return None
        return round(stat.ullTotalPhys / (1024 ** 3), 2)
    try:
        return round(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 ** 3), 2)
    except (AttributeError, ValueError, OSError):
        return None


def _first_line(raw: str | None) -> str | None:
    # ponytail: with multiple GPUs, the first one wins; choosing the best is bench's job.
    return raw.splitlines()[0].strip() if raw else None


def machine() -> Machine:
    """Probe without loading models: nvidia-smi (name and free VRAM), total RAM, and whether
    the whisper.cpp binary is already present."""
    from speechtotext.core import enginepin   # lazy: enginepin knows where the binary lives

    gpu = _first_line(nvidia_smi("name"))
    free = _first_line(nvidia_smi("memory.free"))   # MiB
    try:
        vram = round(float(free) / 1024, 2) if free is not None else None
    except ValueError:
        vram = None
    return Machine(sys.platform, os.cpu_count() or 1, _ram_gb(), gpu is not None, gpu, vram,
                   enginepin.installed_exe())


def eta_factor(engine: str, device: str, compute_type: str, model: str) -> tuple[float | None, bool]:
    """(factor, estimated). This machine's bench.json if it measured that route; otherwise, the table."""
    from speechtotext.core.benchmark import read_table   # lazy: benchmark imports probe

    try:
        table = read_table()
    except (OSError, ValueError, KeyError):   # a corrupt bench.json does not take down a transcription
        table = None
    for row in (table or {}).get("results", ()):
        if ((row.get("engine"), row.get("device"), row.get("quant"), row.get("model"))
                == (engine, device, compute_type, model) and row.get("x_realtime")):
            return round(1 / row["x_realtime"], 3), False
    return ETA_FACTORS.get((engine, device, compute_type, model)), True


def _fw_device(m: Machine) -> tuple[str, str]:
    """(device, reason) for faster-whisper when the user left device='auto'."""
    free = m.vram_free_gb if m.cuda else None
    if free is not None and free >= VRAM_FW_GB:
        return "cuda", f"GPU with {free:.1f} GB free"
    if free is not None:
        return "cpu", f"GPU with {free:.1f} GB free isn't enough for faster-whisper in float16: CPU"
    return "cpu", "no usable GPU: CPU"


def _auto(m: Machine, model: str, device: str) -> tuple[str, str, str]:
    """The spec §5.1 table for engine='auto'. An explicit device wins (whisper.cpp only
    knows GPU): with -d cpu or -d cuda the engine is faster-whisper. Return
    (engine, device, reason); for whispercpp, choose_route labels the device."""
    if device != "auto":
        return ENGINE_FASTER, device, ""
    from speechtotext.core.enginepin import _MODEL_ALIAS

    free = m.vram_free_gb if m.cuda else None
    if free is not None and free >= VRAM_FW_GB:
        return ENGINE_FASTER, "cuda", f"GPU with {free:.1f} GB free"
    if (free is not None and free >= VRAM_WCPP_GB and model in _MODEL_ALIAS
            and (m.whispercpp is not None or m.platform == "win32")):
        return ENGINE_WHISPERCPP, "auto", f"GPU with {free:.1f} GB free: quantized whisper.cpp"
    if free is not None:
        return ENGINE_FASTER, "cpu", f"GPU with {free:.1f} GB free isn't enough for {model}: CPU"
    return ENGINE_FASTER, "cpu", "no usable GPU: CPU"


def choose_route(m: Machine, model: str, *, engine: str = "auto", device: str = "auto",
                 compute_type: str = "auto") -> Route:
    """Choose engine/device/compute_type for `model` on machine `m`. ValueError for impossible
    flags; AsrError("insufficient_resources") if the model does not fit in RAM — the probe
    NEVER changes the model, reports it, and stops."""
    if engine != "auto" and engine not in ENGINES:
        raise ValueError(f"engine {engine!r} does not exist; available: {', '.join(ENGINES)}")
    need = RAM_MIN_GB.get(model)
    if need is not None and m.ram_gb is not None and m.ram_gb < need:
        advice = "; try -m small" if model != "small" else ""
        raise AsrError(
            "insufficient_resources", False,
            f"{model} needs ~{need:g} GB of RAM and this machine has {m.ram_gb:.1f} GB{advice}",
        )
    reason = ""
    if engine == "auto":
        engine, device, reason = _auto(m, model, device)
    elif engine == ENGINE_FASTER and device == "auto":
        # Explicit engine, free device: the probe fills it in and reports it (spec §5.1).
        device, reason = _fw_device(m)
    if engine == ENGINE_WHISPERCPP:
        from speechtotext.core.enginepin import _MODEL_ALIAS

        if model not in _MODEL_ALIAS:
            raise ValueError(
                f"model {model!r} is not pinned for whispercpp; available: "
                f"{', '.join(sorted(_MODEL_ALIAS))}"
            )
        if compute_type not in ("auto", "q5_0"):
            raise ValueError(
                f"compute_type={compute_type!r} is not supported with whispercpp; use 'auto' or "
                "'q5_0'. Reason: fp16 = 0.53x real time from WDDM paging on the 980 (measured 2026-07-27)."
            )
        # win32: the pinned binary is a CUDA build and ALWAYS runs on the GPU (measured in the
        # smoke test); labeling it cpu would lie in the header, key, and JSON. Outside win32,
        # the whisper-cli on the PATH decides based on its build (Metal, CUDA, or CPU) and does
        # not say: it is labeled native. Silently overriding an explicit -d would be the silent
        # substitution: report it.
        label = "cuda" if m.platform == "win32" else "native"
        if not reason and device != label:
            reason = ("whisper.cpp (CUDA build) runs on the GPU; device=cuda" if label == "cuda"
                      else "the whisper-cli on the PATH decides the device based on its build; device=native")
        factor, estimated = eta_factor(ENGINE_WHISPERCPP, label, "q5_0", model)
        return Route(ENGINE_WHISPERCPP, label, "q5_0", reason, factor, estimated)
    device = "cpu" if device == "auto" else device
    if compute_type == "auto":
        compute_type = "int8" if device == "cpu" else "float16"
    factor, estimated = eta_factor(ENGINE_FASTER, device, compute_type, model)
    return Route(ENGINE_FASTER, device, compute_type, reason, factor, estimated)
