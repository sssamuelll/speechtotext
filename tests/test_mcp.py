"""The four MCP tools, tested without the SDK.

The [mcp] extra is intentionally absent from [dev]: the tools are plain functions that
do not touch it, and serve()—the only code that imports it—is tested by seeding a stub
in sys.modules, the same pattern test_models.py uses with huggingface_hub.

However, the stub was written by the same person who wrote serve(): it validates
internal consistency, not SDK compatibility. That is why the final test runs against
the real package and is skipped when it is unavailable; CI installs it in one job so
that someone always runs it.
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

WITHOUT_SDK = importlib.util.find_spec("mcp") is None


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


def test_transcribe_writes_the_json_next_to_the_audio_and_returns_the_text(monkeypatch, tmp_path):
    audio = tmp_path / "reunion.mp4"
    audio.write_bytes(b"doesn't matter: transcribe is stubbed")
    seen = {}

    def fake_transcribe(path, **kw):
        seen.update(path=path, kw=kw)
        return _transcript([LabeledSegment(0.0, 3.0, " hello"),
                            LabeledSegment(3.0, 6.0, " there")])

    monkeypatch.setattr(core_transcribe, "transcribe", fake_transcribe)

    result = mcp_server.transcribe(str(audio))

    assert result["text"] == "hello\nthere"
    assert result["json"] == str(tmp_path / "reunion.json")
    assert result["language"] == "es"
    assert result["duration"] == 12.5
    assert result["warnings"] == ["whispercpp: empty_transcript"]
    # The JSON actually exists and was written by the same writer as the CLI.
    data = json.loads(Path(result["json"]).read_text(encoding="utf-8"))
    assert data["language"] == "es" and len(data["segments"]) == 2
    assert data["engine"]["name"] == "whispercpp"
    # No progress: a callback that prints would break the stdio protocol.
    assert seen["kw"]["on_progress"] is None
    assert seen["kw"]["model"] == "large-v3" and seen["kw"]["language"] == "auto"
    assert seen["kw"]["diarize"] is False
    # The audio being transcribed is the requested file. `result["json"]` is derived
    # from the tool's own argument, so without this check a copy that passed another
    # path to the core would satisfy all eleven assertions above.
    assert seen["path"] == Path(str(audio))


def test_find_returns_the_regions_without_extracting_them(monkeypatch, tmp_path):
    audio = tmp_path / "largo.m4a"
    audio.write_bytes(b"x")
    # The stubs record what they were called with: if find() stopped passing the query
    # or path, returning the correct shape would not be enough to pass.
    seen = {}
    segments = [{"start": 0.0, "end": 1.0, "text": "presupuesto"}]

    def fake_index(path, scan_model, rebuild=False):
        seen.update(path=path, scan_model=scan_model, rebuild=rebuild)
        return segments, True

    def fake_search(segs, query, **kw):
        seen.update(segs=segs, query=query)
        return [finder.Region(10.0, 70.0, 3, 3, "…presupuesto…")]

    monkeypatch.setattr(finder, "load_or_build_index", fake_index)
    monkeypatch.setattr(finder, "search", fake_search)

    result = mcp_server.find(str(audio), "presupuesto")

    assert result["index"] == "cached"
    assert result["regions"] == [
        {"start": 10.0, "end": 70.0, "hits": 3, "matches": 3, "snippet": "…presupuesto…"}
    ]
    assert seen["path"] == Path(str(audio))
    assert seen["scan_model"] == "tiny"       # The index uses the inexpensive model.
    assert seen["query"] == "presupuesto"     # The query arrives exactly as given.
    assert seen["segs"] is segments            # Search uses what the index returned.


def test_voices_lists_the_voice_registry(monkeypatch):
    seen = {}

    def fake_list(model=None):
        seen["model"] = model
        return [{"name": "Voice 1", "model": "pyannote", "seconds": 30}]

    monkeypatch.setattr(registry, "list_voices", fake_list)
    assert mcp_server.voices() == {
        "voices": [{"name": "Voice 1", "model": "pyannote", "seconds": 30}]
    }
    # Unfiltered: the tool does not expose `model`, so it must list every voice.
    assert seen["model"] is None


def test_probe_returns_the_machine_and_route():
    # conftest fixes the machine: win32, no GPU, no whisper.cpp, 32 GB.
    result = mcp_server.probe()
    assert result["machine"] == {
        "platform": "win32", "cpu_count": 8, "ram_gb": 32.0, "cuda": False,
        "gpu_name": None, "vram_free_gb": None, "whispercpp": None,
    }
    assert result["route"]["engine"] == "faster-whisper"
    assert result["route"]["device"] == "cpu"
    assert result["route"]["compute_type"] == "int8"
    assert "error" not in result


def test_probe_reports_the_reason_when_the_model_does_not_fit(monkeypatch):
    from speechtotext.core import probe as core_probe

    small_machine = core_probe.Machine("linux", 2, 2.0, False, None, None, None)
    monkeypatch.setattr(core_probe, "machine", lambda: small_machine)

    result = mcp_server.probe()

    assert result["route"] is None
    assert "RAM" in result["error"]
    assert result["machine"]["ram_gb"] == 2.0


def test_serve_without_the_sdk_explains_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "mcp", None)   # Force ImportError on import.
    with pytest.raises(SystemExit) as ei:
        mcp_server.serve()
    assert "speechtotext[mcp]" in str(ei.value)


def test_serve_registers_the_four_tools_and_serves(monkeypatch):
    registered = []
    ran = []

    class StubServer:
        def __init__(self, name):
            registered.append(("name", name))

        def tool(self):
            def decorator(fn):
                registered.append(fn.__name__)
                return fn
            return decorator

        def run(self):
            ran.append(True)

    package = ModuleType("mcp")
    submodule = ModuleType("mcp.server")
    submodule.MCPServer = StubServer
    package.server = submodule
    monkeypatch.setitem(sys.modules, "mcp", package)
    monkeypatch.setitem(sys.modules, "mcp.server", submodule)

    mcp_server.serve()

    assert registered[0] == ("name", "speechtotext")
    assert registered[1:] == ["transcribe", "find", "voices", "probe"]
    assert ran == [True]


def test_the_module_does_not_import_the_sdk_when_loaded():
    """If mcp_server imported `mcp` at the top, `speechtotext --help` would require the extra."""
    source = Path(mcp_server.__file__).read_text(encoding="utf-8")
    before_serve = source.split("def serve", 1)[0]
    assert "import mcp" not in before_serve and "from mcp" not in before_serve


@pytest.mark.skipif(WITHOUT_SDK, reason="the [mcp] extra is not installed")
def test_serve_registers_against_the_real_sdk():
    """The one thing the stub cannot test: that the SDK API remains what `serve()`
    assumes. If `mcp` 2.x moves `MCPServer`, `.tool()`, or schema derivation from
    annotations, the rest of the suite stays green while the command fails on the
    user's machine. This catches that wherever the extra is installed."""
    import asyncio

    from mcp.server import MCPServer

    server = MCPServer(mcp_server.NAME)
    for fn in mcp_server.TOOLS:
        server.tool()(fn)

    tools = asyncio.run(server.list_tools())
    assert [t.name for t in tools] == ["transcribe", "find", "voices", "probe"]

    schemas = {t.name: t.input_schema for t in tools}
    # Schemas come from annotations: anything without a default is required.
    assert schemas["transcribe"]["required"] == ["path"]
    assert sorted(schemas["transcribe"]["properties"]) == [
        "diarize", "language", "model", "path",
    ]
    assert sorted(schemas["find"]["required"]) == ["path", "query"]
    assert not schemas["voices"].get("properties")
    assert not schemas["probe"].get("properties")
