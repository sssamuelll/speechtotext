"""Sondeo de la máquina (< 1 s, sin cargar modelos) y elección de ruta.

Reglas (spec §5.1): el sondeo NUNCA cambia el modelo; `engine`/`device` explícitos se
respetan y el sondeo solo rellena lo que falta; todo remapeo se avisa en `Route.reason`.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ENGINE_FASTER = "faster-whisper"
ENGINE_WHISPERCPP = "whispercpp"
ENGINES = (ENGINE_FASTER, ENGINE_WHISPERCPP)

# ponytail: umbrales medidos en UNA máquina (5900X + GTX 980 de 4 GB, 2026-09-11), no
# leyes. `bench` es quien los afina; moverlos es un commit consciente con medición.
VRAM_FW_GB = 5.0     # faster-whisper large-v3 float16 en CUDA
VRAM_WCPP_GB = 2.0   # whisper.cpp large-v3 q5_0: 1.3 GB medidos en la 980
RAM_MIN_GB = {"large-v3": 6.0, "small": 3.0}   # techo sobre el pico medido de 3.6 GB en int8


@dataclass(frozen=True)
class Machine:
    platform: str                 # sys.platform: win32 | darwin | linux
    cpu_count: int
    ram_gb: float | None          # None = no se pudo medir
    cuda: bool                    # nvidia-smi responde: hay GPU NVIDIA con driver
    gpu_name: str | None
    vram_free_gb: float | None
    whispercpp: Path | None       # binario ya instalado (pinneado o en el PATH); jamás descarga


def nvidia_smi(query: str, timeout_s: float = 10) -> str | None:
    """Una consulta a nvidia-smi; None si no hay GPU NVIDIA o el comando no responde."""
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
    """RAM física total en GB; None si el sistema no la dice."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes  # solo tras comprobar win32 (regla de la casa)

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
        # argtypes/restype explícitos: sin ellos ctypes trunca punteros en x64.
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
    # ponytail: con varias GPUs manda la primera; elegir la mejor es trabajo de bench.
    return raw.splitlines()[0].strip() if raw else None


def machine() -> Machine:
    """Sondea sin cargar modelos: nvidia-smi (nombre y VRAM libre), RAM total y si el
    binario de whisper.cpp ya está presente."""
    from speechtotext.core import enginepin   # perezoso: enginepin sabe dónde vive el binario

    gpu = _first_line(nvidia_smi("name"))
    free = _first_line(nvidia_smi("memory.free"))   # MiB
    try:
        vram = round(float(free) / 1024, 2) if free is not None else None
    except ValueError:
        vram = None
    return Machine(sys.platform, os.cpu_count() or 1, _ram_gb(), gpu is not None, gpu, vram,
                   enginepin.installed_exe())
