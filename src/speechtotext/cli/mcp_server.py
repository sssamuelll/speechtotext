"""Servidor MCP sobre stdio: cuatro herramientas delgadas sobre el núcleo.

Las cuatro son funciones planas con anotaciones (el SDK deriva el esquema de ahí) y no
tocan `mcp`: asi se prueban sin el extra instalado, y `speechtotext --help` no lo exige.
`serve()` es lo unico que lo importa.

Nada aqui imprime. Un servidor stdio que escriba en stdout rompe el protocolo: por eso
`on_progress=None` en vez del callback rich del CLI.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

NOMBRE = "speechtotext"


def transcribe(path: str, language: str = "auto", model: str = "large-v3",
               diarize: bool = False) -> dict:
    """Transcribe a local audio or video file and write the JSON next to it."""
    from speechtotext.core.formats import write_json
    from speechtotext.core import transcribe as nucleo

    audio = Path(path)
    t = nucleo.transcribe(audio, model=model, language=language, diarize=diarize,
                          on_progress=None)
    destino = audio.with_suffix(".json")   # el mismo default que el CLI sin -o
    info = SimpleNamespace(language=t.language,
                           language_probability=t.language_probability,
                           duration=t.duration)
    write_json(t.segments, info, destino, engine_info=t.engine.to_dict(),
               speech_s=t.speech_s, gaps=t.gaps)
    return {
        "text": "\n".join(s.text.strip() for s in t.segments),
        "json": str(destino),
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
    from speechtotext.core import probe as sondeo

    m = sondeo.machine()
    salida: dict = {"machine": {
        "platform": m.platform, "cpu_count": m.cpu_count, "ram_gb": m.ram_gb,
        "cuda": m.cuda, "gpu_name": m.gpu_name, "vram_free_gb": m.vram_free_gb,
        "whispercpp": str(m.whispercpp) if m.whispercpp is not None else None,
    }}
    try:
        r = sondeo.choose_route(m, "large-v3")
    except AsrError as e:
        # El sondeo nunca cambia el modelo: si no cabe, lo dice y para (spec §5.1).
        salida["route"] = None
        salida["error"] = str(e)
        return salida
    salida["route"] = {"engine": r.engine, "device": r.device,
                       "compute_type": r.compute_type, "reason": r.reason,
                       "eta_factor": r.eta_factor, "estimated": r.estimated}
    return salida


HERRAMIENTAS = (transcribe, find, voices, probe)


def serve() -> None:
    """Registra las cuatro herramientas y sirve por stdio. Bloquea hasta que cierren."""
    try:
        from mcp.server import MCPServer
    except ImportError as e:
        raise SystemExit(
            'The MCP server needs the official SDK. Install it with:\n'
            '    pip install "speechtotext[mcp]"'
        ) from e

    servidor = MCPServer(NOMBRE)
    for fn in HERRAMIENTAS:
        servidor.tool()(fn)
    servidor.run()     # stdio por defecto
