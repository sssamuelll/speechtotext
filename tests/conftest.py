"""Fixtures globales: la suite entera corre sin GPU, sin red, sin modelos y sin tocar la
máquina.

Cada test recibe un SPEECHTOTEXT_HOME vacío (voces, chunks, índice, bench.json y modelos
viven ahí) y una máquina sondeada FIJA (sin GPU, sin whisper.cpp, platform="win32" en
todos los sistemas) para que la ruta 'auto' sea la misma en la máquina de desarrollo y en
CI. Un test que quiera el sondeo real se marca @pytest.mark.real_machine y stubbea él
mismo nvidia-smi y compañía (tests/test_probe.py)."""
import pytest

from speechtotext.core import probe

CPU_MACHINE = probe.Machine(platform="win32", cpu_count=8, ram_gb=32.0, cuda=False,
                            gpu_name=None, vram_free_gb=None, whispercpp=None)


@pytest.fixture(autouse=True)
def _hermetico(request, monkeypatch, tmp_path):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path / "home"))
    if request.node.get_closest_marker("real_machine") is None:
        monkeypatch.setattr(probe, "machine", lambda: CPU_MACHINE)
