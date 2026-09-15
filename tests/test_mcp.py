"""Las cuatro herramientas MCP, probadas sin el SDK.

El extra [mcp] no está en [dev] a propósito: las herramientas son funciones planas que
no lo tocan, y serve() —lo único que lo importa— se prueba sembrando un doble en
sys.modules, el mismo patrón que test_models.py usa con huggingface_hub.

El doble, sin embargo, lo escribió quien escribió serve(): valida coherencia consigo
mismo, no compatibilidad con el SDK. Por eso al final hay un test que corre contra el
paquete de verdad y se salta cuando no está; el CI lo instala en un job para que alguien
lo ejecute siempre.
"""
import importlib.util
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

SIN_SDK = importlib.util.find_spec("mcp") is None


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
    # el audio que se transcribe es el que pidieron. `salida["json"]` se deriva del
    # argumento del propio tool, así que sin esto una copia que le pasara otra ruta al
    # núcleo aprobaría los once asserts de arriba.
    assert visto["ruta"] == Path(str(audio))


def test_find_devuelve_las_regiones_sin_extraer(monkeypatch, tmp_path):
    audio = tmp_path / "largo.m4a"
    audio.write_bytes(b"x")
    # Los dobles anotan con qué los llamaron: si find() dejara de pasar el query o la
    # ruta, devolver la forma correcta no bastaría para aprobar.
    visto = {}
    segmentos = [{"start": 0.0, "end": 1.0, "text": "presupuesto"}]

    def indice_falso(ruta, scan_model, rebuild=False):
        visto.update(ruta=ruta, scan_model=scan_model, rebuild=rebuild)
        return segmentos, True

    def busca_falso(segs, consulta, **kw):
        visto.update(segs=segs, consulta=consulta)
        return [finder.Region(10.0, 70.0, 3, 3, "…presupuesto…")]

    monkeypatch.setattr(finder, "load_or_build_index", indice_falso)
    monkeypatch.setattr(finder, "search", busca_falso)

    salida = mcp_server.find(str(audio), "presupuesto")

    assert salida["index"] == "cached"
    assert salida["regions"] == [
        {"start": 10.0, "end": 70.0, "hits": 3, "matches": 3, "snippet": "…presupuesto…"}
    ]
    assert visto["ruta"] == Path(str(audio))
    assert visto["scan_model"] == "tiny"       # el índice va con el modelo barato
    assert visto["consulta"] == "presupuesto"  # el query llega tal cual, sin tocar
    assert visto["segs"] is segmentos          # busca sobre lo que devolvió el índice


def test_voices_lista_el_registro(monkeypatch):
    visto = {}

    def lista_falsa(model=None):
        visto["model"] = model
        return [{"name": "Voz 1", "model": "pyannote", "seconds": 30}]

    monkeypatch.setattr(registry, "list_voices", lista_falsa)
    assert mcp_server.voices() == {
        "voices": [{"name": "Voz 1", "model": "pyannote", "seconds": 30}]
    }
    # sin filtrar: la herramienta no expone `model`, así que tiene que listarlas todas
    assert visto["model"] is None


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


@pytest.mark.skipif(SIN_SDK, reason="el extra [mcp] no está instalado")
def test_serve_registra_contra_el_sdk_real():
    """Lo único que el doble no puede probar: que el API del SDK siga siendo el que
    `serve()` asume. Si `mcp` 2.x mueve `MCPServer`, `.tool()` o la derivación del
    esquema desde las anotaciones, el resto de la suite sigue verde y el comando
    revienta en la máquina del usuario. Esto lo caza donde el extra esté instalado."""
    import asyncio

    from mcp.server import MCPServer

    servidor = MCPServer(mcp_server.NAME)
    for fn in mcp_server.TOOLS:
        servidor.tool()(fn)

    herramientas = asyncio.run(servidor.list_tools())
    assert [t.name for t in herramientas] == ["transcribe", "find", "voices", "probe"]

    esquemas = {t.name: t.input_schema for t in herramientas}
    # los esquemas salen de las anotaciones: lo que no tiene default es obligatorio
    assert esquemas["transcribe"]["required"] == ["path"]
    assert sorted(esquemas["transcribe"]["properties"]) == [
        "diarize", "language", "model", "path",
    ]
    assert sorted(esquemas["find"]["required"]) == ["path", "query"]
    assert not esquemas["voices"].get("properties")
    assert not esquemas["probe"].get("properties")
