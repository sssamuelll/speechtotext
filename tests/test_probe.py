"""core.probe: sondeo de la máquina y elección de ruta. Sin GPU, sin binarios, sin red."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from speechtotext.asr.base import AsrError
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


# --- choose_route: la tabla del spec §5.1 --------------------------------------------


def _m(**over):
    base = dict(platform="win32", cpu_count=12, ram_gb=32.0, cuda=False, gpu_name=None,
                vram_free_gb=None, whispercpp=None)
    base.update(over)
    return probe.Machine(**base)


def _ruta(r):
    return (r.engine, r.device, r.compute_type)


def test_sin_gpu_va_a_cpu_int8():
    r = probe.choose_route(_m(), "large-v3")
    assert _ruta(r) == ("faster-whisper", "cpu", "int8")
    assert r.reason == "sin GPU utilizable: CPU"


def test_gpu_holgada_va_a_faster_whisper_cuda_float16():
    r = probe.choose_route(_m(cuda=True, gpu_name="RTX 3060", vram_free_gb=11.2), "large-v3")
    assert _ruta(r) == ("faster-whisper", "cuda", "float16")
    assert r.reason == "GPU con 11.2 GB libres"


def test_gpu_justa_va_a_whispercpp_si_esta_instalado_o_es_win32():
    r = probe.choose_route(_m(cuda=True, gpu_name="GTX 980", vram_free_gb=3.5), "large-v3")
    assert _ruta(r) == ("whispercpp", "cuda", "q5_0")
    assert r.reason == "GPU con 3.5 GB libres: whisper.cpp cuantizado"
    # linux con el binario en el PATH: también, etiquetado native
    r = probe.choose_route(
        _m(platform="linux", cuda=True, vram_free_gb=3.5, whispercpp=Path("/usr/bin/whisper-cli")),
        "large-v3",
    )
    assert _ruta(r) == ("whispercpp", "native", "q5_0")


def test_gpu_justa_sin_binario_fuera_de_win32_cae_a_cpu_y_lo_dice():
    r = probe.choose_route(_m(platform="linux", cuda=True, vram_free_gb=3.5), "large-v3")
    assert _ruta(r) == ("faster-whisper", "cpu", "int8")
    assert r.reason == "GPU con 3.5 GB libres no alcanza para large-v3: CPU"


def test_gpu_justa_con_modelo_no_pinneado_cae_a_cpu_y_lo_dice():
    r = probe.choose_route(_m(cuda=True, vram_free_gb=3.5), "medium")
    assert _ruta(r) == ("faster-whisper", "cpu", "int8")
    assert "no alcanza para medium" in r.reason


def test_umbrales_exactos():
    assert probe.choose_route(_m(cuda=True, vram_free_gb=5.0), "large-v3").engine == "faster-whisper"
    assert probe.choose_route(_m(cuda=True, vram_free_gb=4.99), "large-v3").engine == "whispercpp"
    assert probe.choose_route(_m(cuda=True, vram_free_gb=2.0), "large-v3").engine == "whispercpp"
    assert probe.choose_route(_m(cuda=True, vram_free_gb=1.99), "large-v3").device == "cpu"


def test_gpu_sin_vram_legible_va_a_cpu():
    r = probe.choose_route(_m(cuda=True, gpu_name="rara", vram_free_gb=None), "large-v3")
    assert _ruta(r) == ("faster-whisper", "cpu", "int8") and r.reason == "sin GPU utilizable: CPU"


def test_device_explicito_manda_sobre_la_tabla():
    m = _m(cuda=True, vram_free_gb=3.5)
    assert _ruta(probe.choose_route(m, "large-v3", device="cpu")) == ("faster-whisper", "cpu", "int8")
    assert _ruta(probe.choose_route(m, "large-v3", device="cuda")) == ("faster-whisper", "cuda", "float16")
    assert probe.choose_route(m, "large-v3", device="cuda", compute_type="int8").compute_type == "int8"
    assert probe.choose_route(m, "large-v3", device="cpu").reason == ""


def test_engine_explicito_se_respeta_y_el_device_auto_se_sondea():
    m = _m(cuda=True, vram_free_gb=11.0)
    r = probe.choose_route(m, "large-v3", engine="faster-whisper")
    assert _ruta(r) == ("faster-whisper", "cuda", "float16") and r.reason == "GPU con 11.0 GB libres"
    r = probe.choose_route(_m(cuda=True, vram_free_gb=3.5), "large-v3", engine="faster-whisper")
    assert _ruta(r) == ("faster-whisper", "cpu", "int8")
    assert r.reason == "GPU con 3.5 GB libres no alcanza para faster-whisper en float16: CPU"
    r = probe.choose_route(_m(), "large-v3", engine="faster-whisper")
    assert _ruta(r) == ("faster-whisper", "cpu", "int8") and r.reason == "sin GPU utilizable: CPU"
    assert probe.choose_route(m, "large-v3", engine="faster-whisper", device="cpu").reason == ""
    r = probe.choose_route(_m(), "large-v3", engine="whispercpp")
    assert _ruta(r) == ("whispercpp", "cuda", "q5_0")
    assert r.reason == "whisper.cpp (build CUDA) corre en la GPU; device=cuda"
    assert probe.choose_route(_m(), "large-v3", engine="whispercpp", device="cuda").reason == ""


def test_whispercpp_fuera_de_win32_se_etiqueta_native_y_avisa():
    r = probe.choose_route(_m(platform="darwin"), "large-v3", engine="whispercpp")
    assert (r.device, r.eta_factor) == ("native", None)
    assert "device=native" in r.reason
    assert probe.choose_route(_m(platform="darwin"), "large-v3", engine="whispercpp",
                              device="native").reason == ""


def test_el_sondeo_nunca_cambia_el_modelo():
    with pytest.raises(AsrError) as ei:
        probe.choose_route(_m(ram_gb=4.0), "large-v3")
    assert ei.value.code == "insufficient_resources" and ei.value.recoverable is False
    assert "large-v3 necesita ~6 GB" in str(ei.value) and "-m small" in str(ei.value)
    with pytest.raises(AsrError) as ei:
        probe.choose_route(_m(ram_gb=2.0), "small")
    assert "-m small" not in str(ei.value)
    # sin medida de RAM no se corta nada; un modelo fuera de la tabla tampoco
    assert probe.choose_route(_m(ram_gb=None), "large-v3").engine == "faster-whisper"
    assert probe.choose_route(_m(ram_gb=1.0), "tiny").engine == "faster-whisper"


def test_flags_imposibles():
    with pytest.raises(ValueError, match="no existe"):
        probe.choose_route(_m(), "large-v3", engine="chatgpt")
    with pytest.raises(ValueError, match="paging WDDM"):
        probe.choose_route(_m(), "large-v3", engine="whispercpp", compute_type="float16")
    with pytest.raises(ValueError, match="no está pinneado"):
        probe.choose_route(_m(), "medium", engine="whispercpp")


# --- ETA ---------------------------------------------------------------------------------

def test_eta_de_la_tabla_es_estimada():
    r = probe.choose_route(_m(), "large-v3")
    assert (r.eta_factor, r.estimated) == (round(1 / 1.27, 3), True)
    assert probe.choose_route(_m(), "medium").eta_factor is None


def test_eta_de_bench_json_manda_y_no_es_estimada():
    from speechtotext.core import benchmark

    benchmark.write_table({"schema_version": "speechtotext.bench/v1", "results": [
        {"engine": "faster-whisper", "model": "large-v3", "quant": "int8", "device": "cpu",
         "x_realtime": 2.0, "error": None, "capabilities": {}, "wer_ref": None},
    ], "skipped": [], "recommendations": []})
    r = probe.choose_route(_m(), "large-v3")
    assert (r.eta_factor, r.estimated) == (0.5, False)
    assert probe.choose_route(_m(), "small").estimated is True   # esa ruta no se midió


def test_bench_json_corrupto_no_tumba_la_ruta():
    from speechtotext.core import benchmark

    p = benchmark.bench_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{no es json", encoding="utf-8")
    assert probe.choose_route(_m(), "large-v3").estimated is True
