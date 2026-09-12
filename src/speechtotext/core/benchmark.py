"""Benchmark de configs ASR viables en ESTA maquina -> tabla bench.json.

Schema "speechtotext.bench/v1": una fila por config con tiempos, picos de memoria,
capacidades y WER de referencia, para que quien consuma la tabla elija config según lo
que necesite (rapidez, calidad, hotwords...). Cada config se mide en un subproceso
hijo dedicado (benchmark_child): un proceso que ya cargo un modelo contamina el
pico de RAM del siguiente.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from speechtotext.asr.faster_whisper import FasterWhisperBackend
from speechtotext.asr.whispercpp import WhisperCppBackend
from speechtotext.core import probe

# Se reutiliza el _home privado de finder: misma frontera core/, mismo dueño, y el
# bench vive al lado del index bajo SPEECHTOTEXT_HOME. Duplicarlo seria un segundo
# punto de verdad para el mismo directorio.
from speechtotext.core.finder import _home

SCHEMA_VERSION = "speechtotext.bench/v1"


def _caps(backend_cls, native_signals: bool) -> dict:
    """La tabla de capacidades sale del Caps del backend: un solo punto de verdad.
    native_signals no es un knob de Caps (no se pide, se emite): literal aquí, medido."""
    c = backend_cls.caps
    return {"hotwords": c.hotwords == "honrado", "word_timestamps": c.word_timestamps == "honrado",
            "native_signals": native_signals, "vad": c.vad == "honrado"}


_CAPS = {
    "faster-whisper": _caps(FasterWhisperBackend, True),
    "whispercpp": _caps(WhisperCppBackend, False),
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


def available_configs() -> tuple[list[dict], list[dict]]:
    """(viables, skipped): whispercpp solo con binario instalado y GPU NVIDIA que responda."""
    m = probe.machine()
    if m.whispercpp is None:
        wcpp_reason = "whisper-cli ausente (ni pinneado ni en el PATH)"
    elif not m.cuda:
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


def machine_info() -> dict:
    """La forma del schema v1; los datos los pone probe.machine()."""
    m = probe.machine()
    return {
        "cpu": platform.processor() or platform.machine(),
        "logical_cores": m.cpu_count,
        "ram_gb": m.ram_gb,
        "gpu": m.gpu_name,
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
            raw = probe.nvidia_smi("memory.used", timeout_s=2)
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


# Casos de uso del ecosistema: la tabla no solo mide, RECOMIENDA. Cada caso declara
# requisitos duros (capacidades/engine) y un criterio; la eleccion sale de lo MEDIDO,
# asi cambia sola si cambia la maquina (GPU nueva, exe ausente, config que revienta).
#
# El requisito engine=faster-whisper en conversacion/dictado no es capricho: un asistente
# por voz mantiene el motor CARGADO en su proceso y transcribe frase a frase; whispercpp
# es un subprocess que carga el modelo en cada invocacion — pagar la carga por frase lo
# descarta por arquitectura, no por velocidad.
USE_CASES = (
    {
        "caso": "conversacion_en_vivo",
        "que": "Conversación por voz: motor residente, una frase corta cada vez",
        "requisitos": {"engine": "faster-whisper"},
        "criterio": "mas_rapido",
    },
    {
        "caso": "dictado_por_voz",
        "que": "Dictado: mas precision que conversacion con latencia todavia comoda",
        "requisitos": {"engine": "faster-whisper"},
        "criterio": "equilibrio",
    },
    {
        "caso": "transcripcion_maxima_calidad",
        "que": "Transcribir archivos con la mejor calidad disponible",
        "requisitos": {},
        "criterio": "mejor_calidad",
    },
    {
        "caso": "transcripcion_con_diarizacion_fina",
        "que": "Quien-dijo-que palabra a palabra (--diarize fino, corta en el cambio de voz)",
        "requisitos": {"word_timestamps": True},
        "criterio": "mejor_calidad",
    },
    {
        "caso": "audio_con_nombres_propios",
        "que": ("Audio lleno de nombres/jerga: --hotwords existe, pero medidos produjeron apagones "
                "en bloque (n=3, 2026-09-11); compara contra una corrida sin ellos"),
        "requisitos": {"hotwords": True},
        "criterio": "mejor_calidad",
    },
    {
        "caso": "borrador_rapido",
        "que": "Texto aproximado lo antes posible, la calidad es secundaria",
        "requisitos": {},
        "criterio": "mas_rapido",
    },
)


def _cumple(r: dict, requisitos: dict) -> bool:
    for clave, valor in requisitos.items():
        if clave == "engine":
            if r["engine"] != valor:
                return False
        elif not (r.get("capabilities") or {}).get(clave):
            return False
    return True


def _elegir(candidatas: list[dict], criterio: str) -> tuple[dict | None, str]:
    """(ganadora, motivo). Los motivos citan numeros MEDIDOS: la recomendacion debe
    poder defenderse sola ante quien lea la tabla."""
    if not candidatas:
        return None, "ninguna config medida cumple los requisitos en esta maquina"
    rapida = max(candidatas, key=lambda r: r["x_realtime"])
    con_wer = [r for r in candidatas if r.get("wer_ref") is not None]
    if criterio == "mas_rapido":
        return rapida, f"la mas rapida que cumple: {rapida['x_realtime']}x tiempo real"
    if criterio == "mejor_calidad":
        if not con_wer:
            return rapida, (
                f"sin WER medido entre las candidatas; se elige la mas rapida "
                f"({rapida['x_realtime']}x)"
            )
        mejor = min(con_wer, key=lambda r: (r["wer_ref"], -r["x_realtime"]))
        return mejor, (
            f"mejor WER medido ({mejor['wer_ref']}) a {mejor['x_realtime']}x tiempo real"
        )
    if criterio == "equilibrio":
        # ponytail: "comodo" = >= 10x tiempo real; umbral a ojo sobre lo medido hoy,
        # subelo si el dictado se siente lento.
        comodas = [r for r in con_wer if r["x_realtime"] >= 10.0]
        if comodas:
            mejor = min(comodas, key=lambda r: r["wer_ref"])
            return mejor, (
                f"mejor WER ({mejor['wer_ref']}) manteniendo >= 10x tiempo real "
                f"({mejor['x_realtime']}x)"
            )
        if con_wer:
            mejor = min(con_wer, key=lambda r: r["wer_ref"])
            return mejor, f"mejor WER medido ({mejor['wer_ref']}); ninguna llega a 10x"
        return rapida, f"sin WER medido; la mas rapida ({rapida['x_realtime']}x)"
    return None, f"criterio desconocido: {criterio}"


def recommend(results: list[dict]) -> list[dict]:
    """Una recomendacion por caso de uso, derivada de las filas medidas SIN error.

    Un caso sin candidata viable se declara con su razon (p.ej. diarizacion fina en
    una maquina donde solo corrio whispercpp): la ausencia explicada vale mas que
    una recomendacion inventada.
    """
    vivas = [r for r in results if not r.get("error") and r.get("x_realtime")]
    out = []
    for caso in USE_CASES:
        eleccion, motivo = _elegir(
            [r for r in vivas if _cumple(r, caso["requisitos"])], caso["criterio"]
        )
        out.append({
            "caso": caso["caso"],
            "que": caso["que"],
            "eleccion": None if eleccion is None else {
                "engine": eleccion["engine"], "model": eleccion["model"],
                "quant": eleccion["quant"], "device": eleccion["device"],
            },
            "motivo": motivo,
        })
    return out


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
        "recommendations": recommend(results),
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
    table = json.loads(path.read_text(encoding="utf-8"))
    if "recommendations" not in table:
        # Retrocompat: las tablas medidas antes de esta seccion la ganan al leerse,
        # sin re-medir nada — las recomendaciones son derivadas, la medicion manda.
        table["recommendations"] = recommend(table.get("results") or [])
    return table
