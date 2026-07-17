from __future__ import annotations

import ctypes
import json
import os
import platform
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from speechtotext.evaluation.privacy import protected_ref

PACKAGES = (
    "speechtotext",
    "av",
    "ctranslate2",
    "faster-whisper",
    "huggingface-hub",
    "numpy",
    "onnxruntime",
    "scikit-learn",
    "scipy",
    "tokenizers",
)


def _parse_linux_statm(payload: str, *, page_size: int) -> int:
    fields = payload.split()
    if len(fields) < 2 or not fields[1].isdecimal() or page_size <= 0:
        raise ValueError("/proc/self/statm invalido")
    return int(fields[1]) * page_size


def process_memory_bytes() -> dict[str, int]:
    if sys.platform == "win32":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
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

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        )
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
        return {
            "rss": int(counters.WorkingSetSize),
            "peak_rss": int(counters.PeakWorkingSetSize),
        }

    import resource

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        peak *= 1024
    if sys.platform.startswith("linux"):
        rss = _parse_linux_statm(
            Path("/proc/self/statm").read_text(encoding="ascii"),
            page_size=os.sysconf("SC_PAGE_SIZE"),
        )
        return {"rss": rss, "peak_rss": max(peak, rss)}
    return {"rss": peak, "peak_rss": peak}


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _git_revision(repo: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _git_status(repo: Path) -> str:
    proc = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def collect_environment(repo: Path, *, ref_key: bytes) -> dict[str, object]:
    protected_ref(ref_key, "key-check", "")
    if _git_status(repo):
        raise ValueError("el reporte reproducible exige un worktree limpio")
    return {
        "schema_version": "speechtotext.environment/v1",
        "git_ref": protected_ref(ref_key, "git-revision", _git_revision(repo)),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "executable_name": Path(sys.executable).name,
        "memory": process_memory_bytes(),
        "packages": {name: _package_version(name) for name in PACKAGES},
    }


def write_environment_report(
    repo: Path,
    output: Path,
    *,
    ref_key: bytes,
) -> dict[str, object]:
    report = collect_environment(repo, ref_key=ref_key)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return report
