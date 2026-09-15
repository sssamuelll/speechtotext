"""core.probe: machine probing and route selection. No GPU, binaries, or network."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from speechtotext.asr.base import AsrError
from speechtotext.core import enginepin, probe


# --- nvidia_smi ------------------------------------------------------------------------

def test_nvidia_smi_without_binary_or_with_error_returns_none(monkeypatch):
    def no_binary(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(probe.subprocess, "run", no_binary)
    assert probe.nvidia_smi("name") is None

    monkeypatch.setattr(probe.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], 1, "", "driver down"))
    assert probe.nvidia_smi("name") is None


def test_nvidia_smi_returns_stripped_stdout(monkeypatch):
    observed = {}

    def run(cmd, **kw):
        observed["cmd"], observed["timeout"] = cmd, kw.get("timeout")
        return subprocess.CompletedProcess(cmd, 0, "  3541 \n", "")

    monkeypatch.setattr(probe.subprocess, "run", run)
    assert probe.nvidia_smi("memory.free", timeout_s=2) == "3541"
    assert observed["cmd"] == ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    assert observed["timeout"] == 2


# --- _ram_gb ----------------------------------------------------------------------------

def test_ram_gb_outside_win32_uses_sysconf(monkeypatch):
    monkeypatch.setattr(probe.sys, "platform", "linux")
    values = {"SC_PHYS_PAGES": 4_000_000, "SC_PAGE_SIZE": 4096}
    monkeypatch.setattr(probe.os, "sysconf", lambda k: values[k], raising=False)
    assert probe._ram_gb() == round(4_000_000 * 4096 / 1024 ** 3, 2)   # 15.26


def test_ram_gb_outside_win32_without_sysconf_is_none(monkeypatch):
    monkeypatch.setattr(probe.sys, "platform", "darwin")

    def boom(k):
        raise ValueError(k)

    monkeypatch.setattr(probe.os, "sysconf", boom, raising=False)
    assert probe._ram_gb() is None


@pytest.mark.skipif(sys.platform != "win32", reason="GlobalMemoryStatusEx es de win32")
def test_ram_gb_on_win32_measures_something_reasonable():
    ram = probe._ram_gb()
    assert ram is not None and 1.0 < ram < 4096.0


# --- machine() (without the conftest stub: real_machine) ---------------------------------

@pytest.mark.real_machine
def test_machine_without_gpu_or_binary(monkeypatch):
    monkeypatch.setattr(probe, "nvidia_smi", lambda q, timeout_s=10: None)
    monkeypatch.setattr(probe, "_ram_gb", lambda: 31.9)
    monkeypatch.setattr(enginepin, "installed_exe", lambda: None)
    m = probe.machine()
    assert m == probe.Machine(sys.platform, os.cpu_count() or 1, 31.9, False, None, None, None)


@pytest.mark.real_machine
def test_machine_with_gpu_reads_name_and_free_vram_from_the_first_one(monkeypatch):
    responses = {"name": "NVIDIA GeForce GTX 980\nNVIDIA T400", "memory.free": "3541\n1800"}
    monkeypatch.setattr(probe, "nvidia_smi", lambda q, timeout_s=10: responses[q])
    monkeypatch.setattr(probe, "_ram_gb", lambda: 64.0)
    exe = Path("C:/x/Release/whisper-cli.exe")
    monkeypatch.setattr(enginepin, "installed_exe", lambda: exe)
    m = probe.machine()
    assert (m.cuda, m.gpu_name, m.vram_free_gb, m.whispercpp) == (
        True, "NVIDIA GeForce GTX 980", 3.46, exe)


@pytest.mark.real_machine
def test_machine_with_unreadable_vram_does_not_blow_up(monkeypatch):
    responses = {"name": "GPU rara", "memory.free": "[N/A]"}
    monkeypatch.setattr(probe, "nvidia_smi", lambda q, timeout_s=10: responses[q])
    monkeypatch.setattr(probe, "_ram_gb", lambda: None)
    monkeypatch.setattr(enginepin, "installed_exe", lambda: None)
    m = probe.machine()
    assert m.cuda is True and m.vram_free_gb is None and m.ram_gb is None


def test_the_conftest_sets_a_machine_without_gpu():
    # Every test without @pytest.mark.real_machine sees this machine: the 'auto' route
    # is the same here and in CI.
    m = probe.machine()
    assert (m.platform, m.cuda, m.whispercpp) == ("win32", False, None)


# --- installed_exe (enginepin) ----------------------------------------------------------

def test_installed_exe_on_win32_only_if_the_pinned_one_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(enginepin.sys, "platform", "win32")
    monkeypatch.setattr(enginepin, "install_root", lambda: tmp_path)
    assert enginepin.installed_exe() is None
    exe = tmp_path / "Release" / "whisper-cli.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"fake")
    assert enginepin.installed_exe() == exe


def test_installed_exe_outside_win32_searches_the_path(monkeypatch):
    monkeypatch.setattr(enginepin.sys, "platform", "linux")
    monkeypatch.setattr(enginepin.shutil, "which", lambda name: None)
    assert enginepin.installed_exe() is None
    monkeypatch.setattr(
        enginepin.shutil, "which",
        lambda name: "/opt/homebrew/bin/whisper-cli" if name == "whisper-cli" else None,
    )
    assert enginepin.installed_exe() == Path("/opt/homebrew/bin/whisper-cli")


# --- choose_route: the table from spec §5.1 --------------------------------------------


def _m(**over):
    base = dict(platform="win32", cpu_count=12, ram_gb=32.0, cuda=False, gpu_name=None,
                vram_free_gb=None, whispercpp=None)
    base.update(over)
    return probe.Machine(**base)


def _route_tuple(r):
    return (r.engine, r.device, r.compute_type)


def test_without_gpu_the_route_goes_to_cpu_int8():
    r = probe.choose_route(_m(), "large-v3")
    assert _route_tuple(r) == ("faster-whisper", "cpu", "int8")
    assert r.reason == "no usable GPU: CPU"


def test_gpu_with_headroom_goes_to_faster_whisper_cuda_float16():
    r = probe.choose_route(_m(cuda=True, gpu_name="RTX 3060", vram_free_gb=11.2), "large-v3")
    assert _route_tuple(r) == ("faster-whisper", "cuda", "float16")
    assert r.reason == "GPU with 11.2 GB free"


def test_tight_gpu_goes_to_whispercpp_if_it_is_installed_or_on_win32():
    r = probe.choose_route(_m(cuda=True, gpu_name="GTX 980", vram_free_gb=3.5), "large-v3")
    assert _route_tuple(r) == ("whispercpp", "cuda", "q5_0")
    assert r.reason == "GPU with 3.5 GB free: quantized whisper.cpp"
    # Linux with the binary on PATH: also, labeled native
    r = probe.choose_route(
        _m(platform="linux", cuda=True, vram_free_gb=3.5, whispercpp=Path("/usr/bin/whisper-cli")),
        "large-v3",
    )
    assert _route_tuple(r) == ("whispercpp", "native", "q5_0")


def test_tight_gpu_without_binary_outside_win32_falls_back_to_cpu_and_says_so():
    r = probe.choose_route(_m(platform="linux", cuda=True, vram_free_gb=3.5), "large-v3")
    assert _route_tuple(r) == ("faster-whisper", "cpu", "int8")
    assert r.reason == "GPU with 3.5 GB free isn't enough for large-v3: CPU"


def test_tight_gpu_with_unpinned_model_falls_back_to_cpu_and_says_so():
    r = probe.choose_route(_m(cuda=True, vram_free_gb=3.5), "medium")
    assert _route_tuple(r) == ("faster-whisper", "cpu", "int8")
    assert "isn't enough for medium" in r.reason


def test_route_thresholds_are_exact():
    assert probe.choose_route(_m(cuda=True, vram_free_gb=5.0), "large-v3").engine == "faster-whisper"
    assert probe.choose_route(_m(cuda=True, vram_free_gb=4.99), "large-v3").engine == "whispercpp"
    assert probe.choose_route(_m(cuda=True, vram_free_gb=2.0), "large-v3").engine == "whispercpp"
    assert probe.choose_route(_m(cuda=True, vram_free_gb=1.99), "large-v3").device == "cpu"


def test_gpu_without_readable_vram_goes_to_cpu():
    r = probe.choose_route(_m(cuda=True, gpu_name="rara", vram_free_gb=None), "large-v3")
    assert _route_tuple(r) == ("faster-whisper", "cpu", "int8") and r.reason == "no usable GPU: CPU"


def test_explicit_device_overrides_the_table():
    m = _m(cuda=True, vram_free_gb=3.5)
    assert _route_tuple(probe.choose_route(m, "large-v3", device="cpu")) == ("faster-whisper", "cpu", "int8")
    assert _route_tuple(probe.choose_route(m, "large-v3", device="cuda")) == ("faster-whisper", "cuda", "float16")
    assert probe.choose_route(m, "large-v3", device="cuda", compute_type="int8").compute_type == "int8"
    assert probe.choose_route(m, "large-v3", device="cpu").reason == ""


def test_explicit_engine_is_honored_and_auto_device_is_probed():
    m = _m(cuda=True, vram_free_gb=11.0)
    r = probe.choose_route(m, "large-v3", engine="faster-whisper")
    assert _route_tuple(r) == ("faster-whisper", "cuda", "float16") and r.reason == "GPU with 11.0 GB free"
    r = probe.choose_route(_m(cuda=True, vram_free_gb=3.5), "large-v3", engine="faster-whisper")
    assert _route_tuple(r) == ("faster-whisper", "cpu", "int8")
    assert r.reason == "GPU with 3.5 GB free isn't enough for faster-whisper in float16: CPU"
    r = probe.choose_route(_m(), "large-v3", engine="faster-whisper")
    assert _route_tuple(r) == ("faster-whisper", "cpu", "int8") and r.reason == "no usable GPU: CPU"
    assert probe.choose_route(m, "large-v3", engine="faster-whisper", device="cpu").reason == ""
    r = probe.choose_route(_m(), "large-v3", engine="whispercpp")
    assert _route_tuple(r) == ("whispercpp", "cuda", "q5_0")
    assert r.reason == "whisper.cpp (CUDA build) runs on the GPU; device=cuda"
    assert probe.choose_route(_m(), "large-v3", engine="whispercpp", device="cuda").reason == ""


def test_whispercpp_outside_win32_is_labeled_native_and_warns():
    r = probe.choose_route(_m(platform="darwin"), "large-v3", engine="whispercpp")
    assert (r.device, r.eta_factor) == ("native", None)
    assert "device=native" in r.reason
    assert probe.choose_route(_m(platform="darwin"), "large-v3", engine="whispercpp",
                              device="native").reason == ""


def test_the_probe_never_changes_the_model():
    with pytest.raises(AsrError) as ei:
        probe.choose_route(_m(ram_gb=4.0), "large-v3")
    assert ei.value.code == "insufficient_resources" and ei.value.recoverable is False
    assert "large-v3 needs ~6 GB" in str(ei.value) and "-m small" in str(ei.value)
    with pytest.raises(AsrError) as ei:
        probe.choose_route(_m(ram_gb=2.0), "small")
    assert "-m small" not in str(ei.value)
    # without a RAM measurement nothing is rejected; neither is a model outside the table
    assert probe.choose_route(_m(ram_gb=None), "large-v3").engine == "faster-whisper"
    assert probe.choose_route(_m(ram_gb=1.0), "tiny").engine == "faster-whisper"


def test_impossible_flags_are_rejected():
    with pytest.raises(ValueError, match="does not exist"):
        probe.choose_route(_m(), "large-v3", engine="chatgpt")
    with pytest.raises(ValueError, match="WDDM paging"):
        probe.choose_route(_m(), "large-v3", engine="whispercpp", compute_type="float16")
    with pytest.raises(ValueError, match="is not pinned"):
        probe.choose_route(_m(), "medium", engine="whispercpp")


# --- ETA ---------------------------------------------------------------------------------

def test_eta_from_the_table_is_estimated():
    r = probe.choose_route(_m(), "large-v3")
    assert (r.eta_factor, r.estimated) == (round(1 / 1.27, 3), True)
    assert probe.choose_route(_m(), "medium").eta_factor is None


def test_eta_from_bench_json_takes_precedence_and_is_not_estimated():
    from speechtotext.core import benchmark

    benchmark.write_table({"schema_version": "speechtotext.bench/v1", "results": [
        {"engine": "faster-whisper", "model": "large-v3", "quant": "int8", "device": "cpu",
         "x_realtime": 2.0, "error": None, "capabilities": {}, "wer_ref": None},
    ], "skipped": [], "recommendations": []})
    r = probe.choose_route(_m(), "large-v3")
    assert (r.eta_factor, r.estimated) == (0.5, False)
    assert probe.choose_route(_m(), "small").estimated is True   # that route was not measured


def test_corrupt_bench_json_does_not_bring_down_the_route():
    from speechtotext.core import benchmark

    p = benchmark.bench_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{no es json", encoding="utf-8")
    assert probe.choose_route(_m(), "large-v3").estimated is True
