"""MCP server over stdio: four thin tools over the core.

The four are plain annotated functions (the SDK derives the schema from that) and they
don't touch `mcp`: that way they're tested without the extra installed, and
`speechtotext --help` doesn't require it. `serve()` is the only thing that imports it.

Nothing here prints. A stdio server that writes to stdout breaks the protocol: hence
`on_progress=None` instead of the CLI's rich callback.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

NAME = "speechtotext"


def transcribe(path: str, language: str = "auto", model: str = "large-v3",
               diarize: bool = False, diarizer: str = "pyannote") -> dict:
    """Transcribe a local audio or video file and write the JSON next to it.

    diarize marks who speaks. diarizer is "pyannote" (default; names enrolled voices) or
    "nemotron" (about 30x faster on CPU, no names; needs the [nemotron] extra)."""
    from speechtotext.core.formats import write_json
    from speechtotext.core import transcribe as core_transcribe

    audio = Path(path)
    t = core_transcribe.transcribe(
        audio, model=model, language=language, diarize=diarize, diarizer=diarizer,
        on_progress=None,
    )
    dest = audio.with_suffix(".json")   # same default as the CLI without -o
    info = SimpleNamespace(language=t.language,
                           language_probability=t.language_probability,
                           duration=t.duration)
    write_json(t.segments, info, dest, engine_info=t.engine.to_dict(),
               speech_s=t.speech_s, gaps=t.gaps)
    return {
        "text": "\n".join(s.text.strip() for s in t.segments),
        "json": str(dest),
        "language": t.language,
        "duration": t.duration,
        "warnings": list(t.warnings),
    }


def find(path: str, query: str) -> dict:
    """Search a long audio file for words and return the regions where they appear."""
    from speechtotext.core import finder

    segments, cached = finder.load_or_build_index(Path(path), "tiny")
    return {
        "index": "cached" if cached else "built",
        "regions": [
            {"start": r.start, "end": r.end, "hits": r.hits,
             "matches": r.matches, "snippet": r.snippet}
            for r in finder.search(segments, query)
        ],
    }


def voices() -> dict:
    """List enrolled voices used to identify speakers."""
    from speechtotext.speakers import registry

    return {"voices": registry.list_voices()}


def probe() -> dict:
    """Probe this machine and report the route `transcribe` would choose for large-v3."""
    from speechtotext.asr import AsrError
    from speechtotext.core import probe as probe_mod

    m = probe_mod.machine()
    output: dict = {"machine": {
        "platform": m.platform, "cpu_count": m.cpu_count, "ram_gb": m.ram_gb,
        "cuda": m.cuda, "gpu_name": m.gpu_name, "vram_free_gb": m.vram_free_gb,
        "whispercpp": str(m.whispercpp) if m.whispercpp is not None else None,
    }}
    try:
        r = probe_mod.choose_route(m, "large-v3")
    except AsrError as e:
        # The probe never changes the model: if it doesn't fit, it says so and stops (spec §5.1).
        output["route"] = None
        output["error"] = str(e)
        return output
    output["route"] = {"engine": r.engine, "device": r.device,
                       "compute_type": r.compute_type, "reason": r.reason,
                       "eta_factor": r.eta_factor, "estimated": r.estimated}
    return output


TOOLS = (transcribe, find, voices, probe)


def serve() -> None:
    """Registers the four tools and serves over stdio. Blocks until they disconnect."""
    try:
        from mcp.server import MCPServer
    except ImportError as e:
        raise SystemExit(
            'The MCP server needs the official SDK. Install it with:\n'
            '    pip install "speechtotext[mcp]"'
        ) from e

    server = MCPServer(NAME)
    for fn in TOOLS:
        server.tool()(fn)
    server.run()     # stdio by default
