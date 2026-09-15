"""Global fixtures: the entire suite runs without a GPU, network, or models and without
touching the machine.

Each test gets an empty SPEECHTOTEXT_HOME (voices, chunks, index, bench.json, and models
live there) and a FIXED probed machine (no GPU, no whisper.cpp, platform="win32" on all
systems) so the 'auto' route is the same on the development machine and in CI. A test
that needs the real probe is marked @pytest.mark.real_machine and stubs nvidia-smi and
related tools itself (tests/test_probe.py)."""
import pytest

from speechtotext.core import probe

CPU_MACHINE = probe.Machine(platform="win32", cpu_count=8, ram_gb=32.0, cuda=False,
                            gpu_name=None, vram_free_gb=None, whispercpp=None)


@pytest.fixture(autouse=True)
def _hermetic(request, monkeypatch, tmp_path):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path / "home"))
    if request.node.get_closest_marker("real_machine") is None:
        monkeypatch.setattr(probe, "machine", lambda: CPU_MACHINE)
