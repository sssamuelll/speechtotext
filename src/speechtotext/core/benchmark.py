"""Benchmark viable ASR configs on THIS machine -> bench.json table.

Schema "speechtotext.bench/v1": one row per config with timings, memory peaks,
capabilities, and reference WER, so consumers of the table can choose a config based
on their needs (speed, quality, hotwords...). Each config is measured in a dedicated
child subprocess (benchmark_child): a process that has already loaded a model
contaminates the next one's peak RAM.
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

# Reuse finder's private _home: same core/ boundary, same owner, and the benchmark
# lives beside the index under SPEECHTOTEXT_HOME. Duplicating it would create a second
# source of truth for the same directory.
from speechtotext.core.finder import _home

SCHEMA_VERSION = "speechtotext.bench/v1"


def _caps(backend_cls, native_signals: bool) -> dict:
    """The capabilities table comes from the backend's Caps: a single source of truth.
    native_signals is not a Caps knob (it is not requested, it is emitted): literal here, measured."""
    c = backend_cls.caps
    return {"hotwords": c.hotwords == "honored", "word_timestamps": c.word_timestamps == "honored",
            "native_signals": native_signals, "vad": c.vad == "honored"}


_CAPS = {
    "faster-whisper": _caps(FasterWhisperBackend, True),
    "whispercpp": _caps(WhisperCppBackend, False),
}

# WER measured 2026-07-27 against the curated reference (multi-engine session); static
# because WER belongs to the (engine, model) pair, not the machine. The rest: null = unmeasured.
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
    # Same label as core.probe.choose_route and WhisperCppBackend.device: cuda is the
    # pinned win32 build; elsewhere, the whisper-cli on the PATH decides and is labeled native.
    return "cuda" if platform == "win32" else "native"


def candidate_configs(platform: str = sys.platform) -> list[dict]:
    """The 7 fixed candidates; available_configs filters them by machine."""
    configs = [_config("faster-whisper", m, "int8", "cpu") for m in _FW_MODELS]
    configs += [_config("whispercpp", m, "q5_0", _wcpp_device(platform)) for m in _WCPP_MODELS]
    return configs


def available_configs() -> tuple[list[dict], list[dict]]:
    """(viable, skipped): whispercpp only with an installed binary and a responsive NVIDIA GPU."""
    m = probe.machine()
    if m.whispercpp is None:
        wcpp_reason = "whisper-cli missing (not pinned, not on the PATH)"
    elif m.platform == "win32" and not m.cuda:
        # The pinned build is CUDA: it does not run without nvidia-smi. Outside win32, the
        # binary on the PATH decides (Metal/CUDA/CPU), and merely existing is enough.
        wcpp_reason = "nvidia-smi is not responding (no usable NVIDIA GPU)"
    else:
        wcpp_reason = None
    viable_configs: list[dict] = []
    skipped: list[dict] = []
    for cfg in candidate_configs(m.platform):
        if cfg["engine"] == "whispercpp" and wcpp_reason:
            skipped.append({"engine": cfg["engine"], "model": cfg["model"], "reason": wcpp_reason})
        else:
            viable_configs.append(cfg)
    return viable_configs, skipped


def machine_info() -> dict:
    """The shape of the v1 schema; probe.machine() supplies the data."""
    m = probe.machine()
    return {
        "cpu": platform.processor() or platform.machine(),
        "logical_cores": m.cpu_count,
        "ram_gb": m.ram_gb,
        "gpu": m.gpu_name,
    }


class _VramPoller:
    """Peak VRAM from polling nvidia-smi every 0.5 s during the run.

    ponytail: approximate by sampling (may miss peaks < 0.5 s) and measures the entire
    GPU, not the child process; enough to choose a config. Per-process NVML if precision
    is ever needed.
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
    """Measure ONE config in its child subprocess. Config that dies -> row with error,
    numbers as null, NEVER an exception. Injectable `run` for tests."""
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
    # The child emits ONE JSON line at the end; any preceding stdout is noise from
    # the engine libraries.
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


# Ecosystem use cases: the table does not just measure, it RECOMMENDS. Each case declares
# hard requirements (capabilities/engine) and a criterion; the choice comes from WHAT THE
# MEASUREMENTS SAY, so it changes automatically if the machine changes (new GPU, missing
# executable, config that blows up).
#
# The engine=faster-whisper requirement for conversation/dictation is not arbitrary: a voice
# assistant keeps the engine LOADED in its process and transcribes sentence by sentence;
# whispercpp is a subprocess that loads the model on every invocation — paying the load cost
# per sentence rules it out architecturally, not for speed.
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
        # ponytail: "comfortable" = >= 10x real time; eyeballed threshold based on today's
        # measurements, raise it if dictation feels slow.
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
    """One recommendation per use case, derived from measured rows WITHOUT errors.

    A case without a viable candidate is declared with its reason (e.g. fine diarization
    on a machine where only whispercpp ran): an explained absence is worth more than an
    invented recommendation.
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
    """Run the given configs and build the complete schema table.

    `skipped` comes from available_configs() at run time: the table documents WHY rows
    are missing, not just which ones ran.
    `progress`: callable(config, result) per config; None disables it.
    """
    _viable_configs, skipped = available_configs()
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
        # Backward compatibility: a table measured before this section existed, or written
        # with the old Spanish-language recommendation keys, regains a fresh block on read
        # without re-measuring anything — recommendations are derived from `results`
        # (untouched by the key rename), the measurement rules.
        table["recommendations"] = recommend(table.get("results") or [])
    return table
