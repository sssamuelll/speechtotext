"""Las cuatro herramientas MCP, probadas sin el SDK.

El extra [mcp] no está en [dev] a propósito: las herramientas son funciones planas que
no lo tocan, y serve() —lo único que lo importa— se prueba sembrando un doble en
sys.modules, el mismo patrón que test_models.py usa con huggingface_hub.
"""
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from speechtotext.cli import mcp_server
from speechtotext.core import finder
from speechtotext.core import transcribe as core_transcribe
from speechtotext.core.segments import LabeledSegment
from speechtotext.core.transcribe import EngineInfo
from speechtotext.speakers import registry


def _transcript(segments):
    return SimpleNamespace(
        segments=segments,
        language="es",
        language_probability=0.97,
        duration=12.5,
        speech_s=11.0,
        gaps=[],
        engine=EngineInfo("whispercpp", "whisper.cpp v1.9.1", "large-v3", "q5_0", "cuda"),
        request=None,
        warnings=("whispercpp: empty_transcript",),
        diarization=None,
    )


def test_transcribe_escribe_el_json_al_lado_y_devuelve_el_texto(monkeypatch, tmp_path):
    audio = tmp_path / "reunion.mp4"
    audio.write_bytes(b"no importa: transcribe va stubbeado")
    visto = {}

    def falso(ruta, **kw):
        visto.update(ruta=ruta, kw=kw)
        return _transcript([LabeledSegment(0.0, 3.0, " hola"),
                            LabeledSegment(3.0, 6.0, " qué tal")])

    monkeypatch.setattr(core_transcribe, "transcribe", falso)

    salida = mcp_server.transcribe(str(audio))

    assert salida["text"] == "hola\nqué tal"
    assert salida["json"] == str(tmp_path / "reunion.json")
    assert salida["language"] == "es"
    assert salida["duration"] == 12.5
    assert salida["warnings"] == ["whispercpp: empty_transcript"]
    # el JSON existe de verdad y lo escribió el mismo writer que el CLI
    data = json.loads(Path(salida["json"]).read_text(encoding="utf-8"))
    assert data["language"] == "es" and len(data["segments"]) == 2
    assert data["engine"]["name"] == "whispercpp"
    # sin progreso: un callback que imprima rompe el protocolo stdio
    assert visto["kw"]["on_progress"] is None
    assert visto["kw"]["model"] == "large-v3" and visto["kw"]["language"] == "auto"
    assert visto["kw"]["diarize"] is False


def test_find_devuelve_las_regiones_sin_extraer(monkeypatch, tmp_path):
    audio = tmp_path / "largo.m4a"
    audio.write_bytes(b"x")
    monkeypatch.setattr(finder, "load_or_build_index",
                        lambda a, m, r=False: ([{"start": 0.0, "end": 1.0, "text": "presupuesto"}], True))
    monkeypatch.setattr(finder, "search",
                        lambda segs, q, **kw: [finder.Region(10.0, 70.0, 3, 3, "…presupuesto…")])

    salida = mcp_server.find(str(audio), "presupuesto")

    assert salida["index"] == "caché"
    assert salida["regions"] == [
        {"start": 10.0, "end": 70.0, "hits": 3, "matches": 3, "snippet": "…presupuesto…"}
    ]


def test_voices_lista_el_registro(monkeypatch):
    monkeypatch.setattr(registry, "list_voices",
                        lambda model=None: [{"name": "Voz 1", "model": "pyannote", "seconds": 30}])
    assert mcp_server.voices() == {
        "voices": [{"name": "Voz 1", "model": "pyannote", "seconds": 30}]
    }


def test_probe_devuelve_maquina_y_ruta():
    # conftest fija la máquina: win32, sin GPU, sin whisper.cpp, 32 GB
    salida = mcp_server.probe()
    assert salida["machine"] == {
        "platform": "win32", "cpu_count": 8, "ram_gb": 32.0, "cuda": False,
        "gpu_name": None, "vram_free_gb": None, "whispercpp": None,
    }
    assert salida["route"]["engine"] == "faster-whisper"
    assert salida["route"]["device"] == "cpu"
    assert salida["route"]["compute_type"] == "int8"
    assert "error" not in salida


def test_probe_dice_el_motivo_cuando_el_modelo_no_cabe(monkeypatch):
    from speechtotext.core import probe as core_probe

    chica = core_probe.Machine("linux", 2, 2.0, False, None, None, None)
    monkeypatch.setattr(core_probe, "machine", lambda: chica)

    salida = mcp_server.probe()

    assert salida["route"] is None
    assert "RAM" in salida["error"]
    assert salida["machine"]["ram_gb"] == 2.0


def test_serve_sin_el_sdk_dice_como_instalarlo(monkeypatch):
    monkeypatch.setitem(sys.modules, "mcp", None)   # fuerza ImportError en el import
    with pytest.raises(SystemExit) as ei:
        mcp_server.serve()
    assert "speechtotext[mcp]" in str(ei.value)


def test_serve_registra_las_cuatro_herramientas_y_sirve(monkeypatch):
    registradas = []
    corrido = []

    class ServidorDoble:
        def __init__(self, nombre):
            registradas.append(("nombre", nombre))

        def tool(self):
            def decorador(fn):
                registradas.append(fn.__name__)
                return fn
            return decorador

        def run(self):
            corrido.append(True)

    paquete = ModuleType("mcp")
    submodulo = ModuleType("mcp.server")
    submodulo.MCPServer = ServidorDoble
    paquete.server = submodulo
    monkeypatch.setitem(sys.modules, "mcp", paquete)
    monkeypatch.setitem(sys.modules, "mcp.server", submodulo)

    mcp_server.serve()

    assert registradas[0] == ("nombre", "speechtotext")
    assert registradas[1:] == ["transcribe", "find", "voices", "probe"]
    assert corrido == [True]


def test_el_modulo_no_importa_el_sdk_al_cargarse():
    """Si mcp_server importara `mcp` arriba, `speechtotext --help` exigiría el extra."""
    fuente = Path(mcp_server.__file__).read_text(encoding="utf-8")
    arriba = fuente.split("def serve", 1)[0]
    assert "import mcp" not in arriba and "from mcp" not in arriba
