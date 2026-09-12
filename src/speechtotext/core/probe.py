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

from speechtotext.asr.base import AsrError

ENGINE_FASTER = "faster-whisper"
ENGINE_WHISPERCPP = "whispercpp"
ENGINES = (ENGINE_FASTER, ENGINE_WHISPERCPP)

# ponytail: umbrales medidos en UNA máquina (5900X + GTX 980 de 4 GB, 2026-09-11), no
# leyes. `bench` es quien los afina; moverlos es un commit consciente con medición.
VRAM_FW_GB = 5.0     # faster-whisper large-v3 float16 en CUDA
VRAM_WCPP_GB = 2.0   # whisper.cpp large-v3 q5_0: 1.3 GB medidos en la 980
RAM_MIN_GB = {"large-v3": 6.0, "small": 3.0}   # techo sobre el pico medido de 3.6 GB en int8

# Factor × duración del audio, medido en la máquina de referencia (scripts/benchmark_chart.py,
# 2026-09-11: 1.27x, 6.4x, 7.97x y 15.7x tiempo real). None = sin medir. El bench.json de
# ESTA máquina manda sobre esta tabla (eta_factor).
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
    reason: str                       # una frase para imprimir; vacía si no hay nada que avisar
    eta_factor: float | None = None   # × duración del audio; None = sin medir
    estimated: bool = True            # False si el factor salió del bench.json de ESTA máquina


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


def eta_factor(engine: str, device: str, compute_type: str, model: str) -> tuple[float | None, bool]:
    """(factor, estimado). El bench.json de esta máquina si midió esa ruta; si no, la tabla."""
    from speechtotext.core.benchmark import read_table   # perezoso: benchmark importa probe

    try:
        table = read_table()
    except (OSError, ValueError, KeyError):   # un bench.json corrupto no tumba una transcripción
        table = None
    for row in (table or {}).get("results", ()):
        if ((row.get("engine"), row.get("device"), row.get("quant"), row.get("model"))
                == (engine, device, compute_type, model) and row.get("x_realtime")):
            return round(1 / row["x_realtime"], 3), False
    return ETA_FACTORS.get((engine, device, compute_type, model)), True


def _auto(m: Machine, model: str, device: str) -> tuple[str, str, str]:
    """La tabla del spec §5.1 para engine='auto'. Un device explícito manda (whisper.cpp
    solo sabe GPU): con -d cpu o -d cuda el motor es faster-whisper. Devuelve
    (engine, device, reason); para whispercpp el device lo etiqueta choose_route."""
    if device != "auto":
        return ENGINE_FASTER, device, ""
    from speechtotext.core.enginepin import _MODEL_ALIAS

    free = m.vram_free_gb if m.cuda else None
    if free is not None and free >= VRAM_FW_GB:
        return ENGINE_FASTER, "cuda", f"GPU con {free:.1f} GB libres"
    if (free is not None and free >= VRAM_WCPP_GB and model in _MODEL_ALIAS
            and (m.whispercpp is not None or m.platform == "win32")):
        return ENGINE_WHISPERCPP, "auto", f"GPU con {free:.1f} GB libres: whisper.cpp cuantizado"
    if free is not None:
        return ENGINE_FASTER, "cpu", f"GPU con {free:.1f} GB libres no alcanza para {model}: CPU"
    return ENGINE_FASTER, "cpu", "sin GPU utilizable: CPU"


def choose_route(m: Machine, model: str, *, engine: str = "auto", device: str = "auto",
                 compute_type: str = "auto") -> Route:
    """Elige motor/device/compute_type para `model` en la máquina `m`. ValueError con flags
    imposibles; AsrError("insufficient_resources") si el modelo no cabe en RAM — el sondeo
    NUNCA cambia el modelo, lo dice y para."""
    if engine != "auto" and engine not in ENGINES:
        raise ValueError(f"engine {engine!r} no existe; disponibles: {', '.join(ENGINES)}")
    need = RAM_MIN_GB.get(model)
    if need is not None and m.ram_gb is not None and m.ram_gb < need:
        consejo = "; prueba -m small" if model != "small" else ""
        raise AsrError(
            "insufficient_resources", False,
            f"{model} necesita ~{need:g} GB de RAM y esta máquina tiene {m.ram_gb:.1f} GB{consejo}",
        )
    reason = ""
    if engine == "auto":
        engine, device, reason = _auto(m, model, device)
    if engine == ENGINE_WHISPERCPP:
        from speechtotext.core.enginepin import _MODEL_ALIAS

        if model not in _MODEL_ALIAS:
            raise ValueError(
                f"modelo {model!r} no está pinneado para whispercpp; disponibles: "
                f"{', '.join(sorted(_MODEL_ALIAS))}"
            )
        if compute_type not in ("auto", "q5_0"):
            raise ValueError(
                f"compute_type={compute_type!r} no soportado con whispercpp; usa 'auto' o 'q5_0'. "
                "Motivo: fp16 = 0.53x tiempo real por paging WDDM en la 980 (medido 2026-07-27)."
            )
        # win32: el binario pinneado es build CUDA y corre en la GPU SIEMPRE (medido en el
        # smoke); etiquetar cpu sería mentir en el header, la llave y el JSON. Fuera de
        # win32 el whisper-cli del PATH decide según su build (Metal, CUDA o CPU) y no lo
        # dice: se etiqueta native. Pisar un -d explícito en silencio sería la sustitución
        # callada: se avisa.
        label = "cuda" if m.platform == "win32" else "native"
        if not reason and device != label:
            reason = ("whisper.cpp (build CUDA) corre en la GPU; device=cuda" if label == "cuda"
                      else "el whisper-cli del PATH decide el dispositivo según su build; device=native")
        factor, estimated = eta_factor(ENGINE_WHISPERCPP, label, "q5_0", model)
        return Route(ENGINE_WHISPERCPP, label, "q5_0", reason, factor, estimated)
    device = "cpu" if device == "auto" else device
    if compute_type == "auto":
        compute_type = "int8" if device == "cpu" else "float16"
    factor, estimated = eta_factor(ENGINE_FASTER, device, compute_type, model)
    return Route(ENGINE_FASTER, device, compute_type, reason, factor, estimated)
