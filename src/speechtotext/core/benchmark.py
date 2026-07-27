"""Benchmark de configs ASR viables en ESTA maquina -> tabla bench.json.

Schema "speechtotext.bench/v1": una fila por config con tiempos, picos de memoria,
capacidades y WER de referencia, para que aurelius (via MCP) elija config segun lo
que necesite (rapidez, calidad, hotwords...). Cada config se mide en un subproceso
hijo dedicado (benchmark_child): un proceso que ya cargo un modelo contamina el
pico de RAM del siguiente.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

# Se reutiliza el _home privado de finder: misma frontera core/, mismo dueño, y el
# bench vive al lado del index bajo SPEECHTOTEXT_HOME. Duplicarlo seria un segundo
# punto de verdad para el mismo directorio.
from speechtotext.core.finder import _home

SCHEMA_VERSION = "speechtotext.bench/v1"

# Contrato de degradacion ya mergeado (docs/plan-multimotor.md seccion 4): whispercpp
# no honra hotwords/word_timestamps/vad ni emite señales nativas.
_CAPS = {
    "faster-whisper": {"hotwords": True, "word_timestamps": True, "native_signals": True, "vad": True},
    "whispercpp": {"hotwords": False, "word_timestamps": False, "native_signals": False, "vad": False},
}

# WER medido 2026-07-27 contra la referencia curada (sesion multimotor); estatico
# porque el WER es del par (motor, modelo), no de la maquina. El resto: null = no medido.
_WER_REF = {
    ("faster-whisper", "small"): 0.419,
    ("faster-whisper", "large-v3"): 0.355,
    ("whispercpp", "large-v3"): 0.355,
}

_FW_MODELS = ("tiny", "base", "small", "medium", "large-v3")
_WCPP_MODELS = ("small", "large-v3")


def bench_path() -> Path:
    return _home() / "bench.json"


def _config(engine: str, model: str, quant: str, device: str) -> dict:
    return {
        "engine": engine,
        "model": model,
        "quant": quant,
        "device": device,
        "capabilities": dict(_CAPS[engine]),
        "wer_ref": _WER_REF.get((engine, model)),
    }


def candidate_configs() -> list[dict]:
    """Las 7 candidatas fijas; available_configs las filtra por maquina."""
    configs = [_config("faster-whisper", m, "int8", "cpu") for m in _FW_MODELS]
    configs += [_config("whispercpp", m, "q5_0", "cuda") for m in _WCPP_MODELS]
    return configs


def _nvidia_smi(query: str, timeout_s: float = 10) -> str | None:
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


def _pinned_exe() -> Path | None:
    """Path del whisper-cli.exe pinneado SIN descargarlo ni reventar fuera de win32."""
    from speechtotext.core import enginepin

    try:
        return enginepin.install_root() / enginepin.ENGINE_PIN["exe_relpath"]
    except KeyError:
        # LOCALAPPDATA ausente (no-win32): no hay exe pinneado posible. Guardas de
        # entorno jamas dependen del entorno del que protegen.
        return None


def available_configs() -> tuple[list[dict], list[dict]]:
    """(viables, skipped): filtra whispercpp si falta el exe pinneado o nvidia-smi calla."""
    exe = _pinned_exe()
    if exe is None or not exe.exists():
        wcpp_reason = f"whisper-cli.exe pinneado ausente ({exe})"
    elif _nvidia_smi("name") is None:
        wcpp_reason = "nvidia-smi no responde (sin GPU NVIDIA utilizable)"
    else:
        wcpp_reason = None
    viables: list[dict] = []
    skipped: list[dict] = []
    for cfg in candidate_configs():
        if cfg["engine"] == "whispercpp" and wcpp_reason:
            skipped.append({"engine": cfg["engine"], "model": cfg["model"], "reason": wcpp_reason})
        else:
            viables.append(cfg)
    return viables, skipped


def _ram_gb() -> float | None:
    if sys.platform != "win32":
        # ponytail: el bench corre en esta maquina win32; leer /proc/meminfo si migra.
        return None
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
    # argtypes/restype explicitos: sin ellos ctypes trunca punteros en x64.
    kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        return None
    return round(stat.ullTotalPhys / (1024 ** 3), 2)


def machine_info() -> dict:
    import os
    import platform

    return {
        "cpu": platform.processor() or platform.machine(),
        "logical_cores": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "gpu": _nvidia_smi("name"),
    }


class _VramPoller:
    """Pico de VRAM por polling de nvidia-smi cada 0.5 s durante la corrida.

    ponytail: aproximado por muestreo (puede perder picos < 0.5 s) y mide la GPU
    entera, no el proceso hijo; suficiente para decidir config. NVML por proceso
    si algun dia hace falta precision.
    """

    def __init__(self):
        self.peak_mb: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            raw = _nvidia_smi("memory.used", timeout_s=2)
            if raw is not None:
                try:
                    mb = float(raw.splitlines()[0])
                except ValueError:
                    mb = None
                if mb is not None:
                    self.peak_mb = max(self.peak_mb or 0.0, mb)
            self._stop.wait(0.5)

    def start(self):
        self._thread.start()

    def stop(self) -> float | None:
        self._stop.set()
        self._thread.join(timeout=5)
        return self.peak_mb


def run_config(config: dict, wav_path, duration_s: float, *, run=None,
               timeout_s: int = 1800) -> dict:
    """Mide UNA config en su subproceso hijo. Config que muere -> fila con error,
    numeros en null, JAMAS excepcion. `run` inyectable para tests."""
    run = run or subprocess.run
    result = {
        "engine": config["engine"],
        "model": config["model"],
        "quant": config["quant"],
        "device": config["device"],
        "load_s": None,
        "transcribe_s": None,
        "x_realtime": None,
        "peak_ram_mb": None,
        "peak_vram_mb": None,
        "segments": None,
        "chars": None,
        "capabilities": dict(config["capabilities"]),
        "wer_ref": config["wer_ref"],
        "error": None,
    }
    argv = [
        sys.executable, "-m", "speechtotext.core.benchmark_child",
        config["engine"], config["model"], config["device"], config["quant"], str(wav_path),
    ]
    poller = _VramPoller() if config["device"] == "cuda" else None
    if poller:
        poller.start()
    try:
        proc = run(argv, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        result["error"] = f"timeout: la config no termino en {timeout_s} s"
        return result
    except OSError as exc:
        result["error"] = f"no se pudo lanzar el hijo: {exc}"
        return result
    finally:
        if poller:
            result["peak_vram_mb"] = poller.stop()
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").splitlines()[-5:])
        result["error"] = f"hijo murio con rc={proc.returncode}: {tail}"
        return result
    # El hijo emite UNA linea JSON al final; lo anterior en stdout (si lo hay) es ruido
    # de las libs del motor.
    lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
    try:
        payload = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        result["error"] = f"salida del hijo no es JSON: {(proc.stdout or '')[-200:]!r}"
        return result
    if payload.get("error"):
        result["error"] = payload["error"]
        return result
    result["load_s"] = payload.get("load_s")
    result["transcribe_s"] = payload.get("transcribe_s")
    result["peak_ram_mb"] = payload.get("peak_ram_mb")
    result["segments"] = payload.get("segments")
    result["chars"] = payload.get("chars")
    if result["transcribe_s"]:
        result["x_realtime"] = round(duration_s / result["transcribe_s"], 2)
    return result


def _sha1(path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_benchmark(wav_path, duration_s: float, configs: list[dict], *, progress=None) -> dict:
    """Corre las configs dadas y arma la tabla completa del schema.

    `skipped` sale de available_configs() en el momento de la corrida: la tabla
    documenta POR QUE faltan filas, no solo cuales corrieron.
    `progress`: callable(config, resultado) por config; None lo apaga.
    """
    _viables, skipped = available_configs()
    results = []
    for cfg in configs:
        res = run_config(cfg, wav_path, duration_s)
        results.append(res)
        if progress:
            progress(cfg, res)
    return {
        "schema_version": SCHEMA_VERSION,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "machine": machine_info(),
        "audio": {"source": str(wav_path), "duration_s": duration_s, "sha1": _sha1(wav_path)},
        "results": results,
        "skipped": skipped,
    }


def write_table(table: dict, path: Path | None = None) -> Path:
    path = path or bench_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_table(path: Path | None = None) -> dict | None:
    path = path or bench_path()
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
