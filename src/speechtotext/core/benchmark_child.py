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

# Opts canonicos del benchmark: identicos para toda config, para que los tiempos sean
# comparables entre motores. Para whispercpp pasan por effective_opts (degradacion ya
# mergeada, docs/plan-multimotor.md seccion 4).
CANON_OPTS = {
    "language": "es",
    "beam_size": 5,
    "vad_filter": True,
    "hotwords": None,
    "condition_on_previous_text": False,
    "word_timestamps": False,
}


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


def run_child(engine_factory, engine: str, model: str, device: str, compute_type: str,
              wav_path: str) -> dict:
    """Carga, transcribe y mide. Separada de main() para testearla con un motor fake.

    Cualquier excepcion -> dict con "error"; jamas propaga.
    """
    try:
        from speechtotext.core.engines import effective_opts

        t0 = time.perf_counter()
        eng = engine_factory(engine, model, device, compute_type)
        load_s = time.perf_counter() - t0
        # OJO whispercpp: make_engine solo pinnea/verifica (load_s casi cero); el modelo
        # se carga dentro del exe en transcribe, y la PRIMERA corrida paga JIT CUDA
        # (~17 s medido 2026-07-27). El hijo NO calienta: el numero es el de la maquina
        # tal cual, JIT incluido en transcribe_s.
        opts = effective_opts(engine, dict(CANON_OPTS))
        t1 = time.perf_counter()
        segments, _info = eng.transcribe(wav_path, **opts)
        n_segments = 0
        n_chars = 0
        # faster-whisper devuelve generador perezoso: consumir ES transcribir.
        for seg in segments:
            n_segments += 1
            n_chars += len(seg.text.strip())
        transcribe_s = time.perf_counter() - t1
        return {
            "load_s": round(load_s, 3),
            "transcribe_s": round(transcribe_s, 3),
            "segments": n_segments,
            "chars": n_chars,
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
        from speechtotext.core.engines import make_engine

        result = run_child(make_engine, *args)
    except Exception as exc:  # noqa: BLE001 — hasta el ImportError sale como JSON
        result = {"error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
