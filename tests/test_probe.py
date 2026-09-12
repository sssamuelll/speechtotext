"""core.probe: sondeo de la máquina y elección de ruta. Sin GPU, sin binarios, sin red."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from speechtotext.core import enginepin, probe


# --- nvidia_smi ------------------------------------------------------------------------

def test_nvidia_smi_sin_binario_o_con_error_devuelve_none(monkeypatch):
    def sin_binario(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(probe.subprocess, "run", sin_binario)
    assert probe.nvidia_smi("name") is None

    monkeypatch.setattr(probe.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], 1, "", "driver caído"))
    assert probe.nvidia_smi("name") is None


def test_nvidia_smi_devuelve_stdout_limpio(monkeypatch):
    visto = {}

    def run(cmd, **kw):
        visto["cmd"], visto["timeout"] = cmd, kw.get("timeout")
        return subprocess.CompletedProcess(cmd, 0, "  3541 \n", "")

    monkeypatch.setattr(probe.subprocess, "run", run)
    assert probe.nvidia_smi("memory.free", timeout_s=2) == "3541"
    assert visto["cmd"] == ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    assert visto["timeout"] == 2


# --- _ram_gb ----------------------------------------------------------------------------

def test_ram_gb_fuera_de_win32_usa_sysconf(monkeypatch):
    monkeypatch.setattr(probe.sys, "platform", "linux")
    valores = {"SC_PHYS_PAGES": 4_000_000, "SC_PAGE_SIZE": 4096}
    monkeypatch.setattr(probe.os, "sysconf", lambda k: valores[k], raising=False)
    assert probe._ram_gb() == round(4_000_000 * 4096 / 1024 ** 3, 2)   # 15.26


def test_ram_gb_fuera_de_win32_sin_sysconf_es_none(monkeypatch):
    monkeypatch.setattr(probe.sys, "platform", "darwin")

    def boom(k):
        raise ValueError(k)

    monkeypatch.setattr(probe.os, "sysconf", boom, raising=False)
    assert probe._ram_gb() is None


@pytest.mark.skipif(sys.platform != "win32", reason="GlobalMemoryStatusEx es de win32")
def test_ram_gb_en_win32_mide_algo_razonable():
    ram = probe._ram_gb()
    assert ram is not None and 1.0 < ram < 4096.0


# --- machine() (sin el stub del conftest: real_machine) ----------------------------------

@pytest.mark.real_machine
def test_machine_sin_gpu_ni_binario(monkeypatch):
    monkeypatch.setattr(probe, "nvidia_smi", lambda q, timeout_s=10: None)
    monkeypatch.setattr(probe, "_ram_gb", lambda: 31.9)
    monkeypatch.setattr(enginepin, "installed_exe", lambda: None)
    m = probe.machine()
    assert m == probe.Machine(sys.platform, os.cpu_count() or 1, 31.9, False, None, None, None)


@pytest.mark.real_machine
def test_machine_con_gpu_lee_nombre_y_vram_libre_de_la_primera(monkeypatch):
    respuestas = {"name": "NVIDIA GeForce GTX 980\nNVIDIA T400", "memory.free": "3541\n1800"}
    monkeypatch.setattr(probe, "nvidia_smi", lambda q, timeout_s=10: respuestas[q])
    monkeypatch.setattr(probe, "_ram_gb", lambda: 64.0)
    exe = Path("C:/x/Release/whisper-cli.exe")
    monkeypatch.setattr(enginepin, "installed_exe", lambda: exe)
    m = probe.machine()
    assert (m.cuda, m.gpu_name, m.vram_free_gb, m.whispercpp) == (
        True, "NVIDIA GeForce GTX 980", 3.46, exe)


@pytest.mark.real_machine
def test_machine_con_vram_ilegible_no_revienta(monkeypatch):
    respuestas = {"name": "GPU rara", "memory.free": "[N/A]"}
    monkeypatch.setattr(probe, "nvidia_smi", lambda q, timeout_s=10: respuestas[q])
    monkeypatch.setattr(probe, "_ram_gb", lambda: None)
    monkeypatch.setattr(enginepin, "installed_exe", lambda: None)
    m = probe.machine()
    assert m.cuda is True and m.vram_free_gb is None and m.ram_gb is None


def test_el_conftest_fija_una_maquina_sin_gpu():
    # Todo test sin @pytest.mark.real_machine ve esta máquina: la ruta 'auto' es la misma
    # aquí y en CI.
    m = probe.machine()
    assert (m.platform, m.cuda, m.whispercpp) == ("win32", False, None)


# --- installed_exe (enginepin) ----------------------------------------------------------

def test_installed_exe_win32_solo_si_el_pinneado_existe(monkeypatch, tmp_path):
    monkeypatch.setattr(enginepin.sys, "platform", "win32")
    monkeypatch.setattr(enginepin, "install_root", lambda: tmp_path)
    assert enginepin.installed_exe() is None
    exe = tmp_path / "Release" / "whisper-cli.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"fake")
    assert enginepin.installed_exe() == exe


def test_installed_exe_fuera_de_win32_busca_en_el_path(monkeypatch):
    monkeypatch.setattr(enginepin.sys, "platform", "linux")
    monkeypatch.setattr(enginepin.shutil, "which", lambda name: None)
    assert enginepin.installed_exe() is None
    monkeypatch.setattr(
        enginepin.shutil, "which",
        lambda name: "/opt/homebrew/bin/whisper-cli" if name == "whisper-cli" else None,
    )
    assert enginepin.installed_exe() == Path("/opt/homebrew/bin/whisper-cli")
