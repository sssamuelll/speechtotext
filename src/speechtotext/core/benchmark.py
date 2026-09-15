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
    return {"hotwords": c.hotwords == "honored", "word_timestamps": c.word_timestamps == "honored",
            "native_signals": native_signals, "vad": c.vad == "honored"}


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


def _wcpp_device(platform: str) -> str:
    # Misma etiqueta que core.probe.choose_route y WhisperCppBackend.device: cuda es el
    # build pinneado de win32; fuera, el whisper-cli del PATH decide y se etiqueta native.
    return "cuda" if platform == "win32" else "native"


def candidate_configs(platform: str = sys.platform) -> list[dict]:
    """Las 7 candidatas fijas; available_configs las filtra por máquina."""
    configs = [_config("faster-whisper", m, "int8", "cpu") for m in _FW_MODELS]
    configs += [_config("whispercpp", m, "q5_0", _wcpp_device(platform)) for m in _WCPP_MODELS]
    return configs


def available_configs() -> tuple[list[dict], list[dict]]:
    """(viables, skipped): whispercpp solo con binario instalado y GPU NVIDIA que responda."""
    m = probe.machine()
    if m.whispercpp is None:
        wcpp_reason = "whisper-cli missing (not pinned, not on the PATH)"
    elif m.platform == "win32" and not m.cuda:
        # El build pinneado es CUDA: sin nvidia-smi no corre. Fuera de win32 el binario
        # del PATH decide (Metal/CUDA/CPU) y basta con que exista.
        wcpp_reason = "nvidia-smi is not responding (no usable NVIDIA GPU)"
    else:
        wcpp_reason = None
    viables: list[dict] = []
    skipped: list[dict] = []
    for cfg in candidate_configs(m.platform):
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
        result["error"] = f"timeout: the config did not finish in {timeout_s} s"
        return result
    except OSError as exc:
        result["error"] = f"could not launch the child process: {exc}"
        return result
    finally:
        if poller:
            result["peak_vram_mb"] = poller.stop()
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").splitlines()[-5:])
        result["error"] = f"child process died with rc={proc.returncode}: {tail}"
        return result
    # El hijo emite UNA linea JSON al final; lo anterior en stdout (si lo hay) es ruido
    # de las libs del motor.
    lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
    try:
        payload = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        result["error"] = f"child process output is not JSON: {(proc.stdout or '')[-200:]!r}"
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
        "case": "live_conversation",
        "description": "Voice conversation: resident engine, one short sentence at a time",
        "requirements": {"engine": "faster-whisper"},
        "criterion": "fastest",
    },
    {
        "case": "voice_dictation",
        "description": "Dictation: more accurate than conversation, latency still comfortable",
        "requirements": {"engine": "faster-whisper"},
        "criterion": "balanced",
    },
    {
        "case": "max_quality_transcription",
        "description": "Transcribing files at the best quality available",
        "requirements": {},
        "criterion": "best_quality",
    },
    {
        "case": "fine_diarization_transcription",
        "description": "Who-said-what, word by word (fine --diarize, cuts on the speaker change)",
        "requirements": {"word_timestamps": True},
        "criterion": "best_quality",
    },
    {
        "case": "audio_with_proper_nouns",
        "description": ("Audio full of names/jargon: --hotwords exists, but measured runs produced "
                "blackouts (n=3, 2026-09-11); compare against a run without them"),
        "requirements": {"hotwords": True},
        "criterion": "best_quality",
    },
    {
        "case": "fast_draft",
        "description": "Rough text as fast as possible, quality is secondary",
        "requirements": {},
        "criterion": "fastest",
    },
)


def _meets(r: dict, requirements: dict) -> bool:
    for key, value in requirements.items():
        if key == "engine":
            if r["engine"] != value:
                return False
        elif not (r.get("capabilities") or {}).get(key):
            return False
    return True


def _choose(candidates: list[dict], criterion: str) -> tuple[dict | None, str]:
    """(winner, reason). Reasons cite MEASURED numbers: the recommendation has to be
    able to stand on its own for whoever reads the table."""
    if not candidates:
        return None, "no measured config meets the requirements on this machine"
    fastest = max(candidates, key=lambda r: r["x_realtime"])
    with_wer = [r for r in candidates if r.get("wer_ref") is not None]
    if criterion == "fastest":
        return fastest, f"the fastest that qualifies: {fastest['x_realtime']}x real time"
    if criterion == "best_quality":
        if not with_wer:
            return fastest, (
                f"no measured WER among the candidates; picking the fastest "
                f"({fastest['x_realtime']}x)"
            )
        best = min(with_wer, key=lambda r: (r["wer_ref"], -r["x_realtime"]))
        return best, (
            f"best measured WER ({best['wer_ref']}) at {best['x_realtime']}x real time"
        )
    if criterion == "balanced":
        # ponytail: "comodo" = >= 10x tiempo real; umbral a ojo sobre lo medido hoy,
        # subelo si el dictado se siente lento.
        comfortable = [r for r in with_wer if r["x_realtime"] >= 10.0]
        if comfortable:
            best = min(comfortable, key=lambda r: r["wer_ref"])
            return best, (
                f"best WER ({best['wer_ref']}) while staying >= 10x real time "
                f"({best['x_realtime']}x)"
            )
        if with_wer:
            best = min(with_wer, key=lambda r: r["wer_ref"])
            return best, f"best measured WER ({best['wer_ref']}); none reaches 10x"
        return fastest, f"no measured WER; the fastest ({fastest['x_realtime']}x)"
    return None, f"unknown criterion: {criterion}"


def recommend(results: list[dict]) -> list[dict]:
    """Una recomendacion por caso de uso, derivada de las filas medidas SIN error.

    Un caso sin candidata viable se declara con su razon (p.ej. diarizacion fina en
    una maquina donde solo corrio whispercpp): la ausencia explicada vale mas que
    una recomendacion inventada.
    """
    viable = [r for r in results if not r.get("error") and r.get("x_realtime")]
    out = []
    for case in USE_CASES:
        choice, reason = _choose(
            [r for r in viable if _meets(r, case["requirements"])], case["criterion"]
        )
        out.append({
            "case": case["case"],
            "description": case["description"],
            "choice": None if choice is None else {
                "engine": choice["engine"], "model": choice["model"],
                "quant": choice["quant"], "device": choice["device"],
            },
            "reason": reason,
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
    recs = table.get("recommendations")
    if recs is None or (recs and "case" not in recs[0]):
        # Retrocompat: a table measured before this section existed, or written with
        # the old Spanish recommendation keys (caso/que/motivo/eleccion), regains a
        # fresh block on read without re-measuring anything — recommendations are
        # derived from `results` (untouched by the key rename), the measurement rules.
        table["recommendations"] = recommend(table.get("results") or [])
    return table
