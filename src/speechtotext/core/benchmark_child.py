"""Hijo de medicion aislada del benchmark (schema speechtotext.bench/v1).

Un proceso que ya cargo un modelo contamina el pico de RAM del siguiente: por eso
cada config corre en ESTE proceso hijo dedicado, que mide su PROPIO pico al final.
Emite UNA linea JSON a stdout y sale 0 SIEMPRE — el padre no debe morir porque una
config muera. Sin rich, sin typer: argv posicional pelado.

argv: engine model device compute_type wav_path
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from speechtotext.asr.types import TranscriptionRequest
from speechtotext.core.transcribe import _apply_caps, load_audio

# Peticion canonica del benchmark: identica para toda config, para que los tiempos sean
# comparables entre motores. Los CAPS de cada backend la degradan donde toque (whispercpp
# sin VAD ni palabras), igual que en la ruta real.
CANON_REQUEST = TranscriptionRequest(language="es", beam_size=5, vad=True, hotwords=(),
                                     word_timestamps=False)


def _peak_ram_mb() -> float:
    """Pico de working set de ESTE proceso, en MB."""
    if sys.platform == "win32":
        import ctypes

        # wintypes solo se importa DENTRO de la rama win32 (regla de la casa).
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
        # restype/argtypes explicitos u obligatorios: sin ellos ctypes trunca el HANDLE
        # a 32 bits en x64 (leccion medida 2026-07-27).
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

    # ponytail: ru_maxrss viene en KB en Linux (bytes en macOS); el bench corre en
    # win32/linux, asumimos KB. Ajustar con sys.platform == "darwin" si algun dia toca.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_child(backend_factory, engine: str, model: str, device: str, compute_type: str,
              wav_path: str) -> dict:
    """Carga, transcribe y mide. Separada de main() para testearla con un motor fake.

    Cualquier excepcion -> dict con "error"; jamas propaga. La decodificacion queda fuera
    de los dos relojes: load_s es warm(), transcribe_s es solo backend.transcribe.
    """
    try:
        t0 = time.perf_counter()
        backend = backend_factory(engine, model, device, compute_type)
        # OJO whispercpp: warm() solo pinnea/verifica (load_s casi cero); el modelo se carga
        # dentro del exe en transcribe, y la PRIMERA corrida paga JIT CUDA (~17 s medido
        # 2026-07-27). El hijo NO calienta: el numero es el de la maquina tal cual.
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
    except Exception as exc:  # noqa: BLE001 — el contrato exige JSON con error, no traceback
        return {"error": f"{type(exc).__name__}: {exc}"}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    try:
        if len(args) != 5:
            raise ValueError(
                f"uso: python -m speechtotext.core.benchmark_child engine model device "
                f"compute_type wav (recibidos {len(args)} args)"
            )
        from speechtotext.core.transcribe import make_backend

        result = run_child(make_backend, *args)
    except Exception as exc:  # noqa: BLE001 — hasta el ImportError sale como JSON
        result = {"error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
