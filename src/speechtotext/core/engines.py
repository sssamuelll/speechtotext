"""Multimotor v1: factory make_engine, contrato de capacidades CAPS y adaptador whisper.cpp.

Vive en core/ (no en asr/) porque la frontera C-7 del plan de calidad prohibe el primer
import de asr/ dentro de core/, y el multimotor pertenece al CLI batch (plan 2.1).
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

ENGINE_FASTER = "faster-whisper"
ENGINE_WHISPERCPP = "whispercpp"
ENGINES = (ENGINE_FASTER, ENGINE_WHISPERCPP)

# Contrato de capacidades por motor (plan 2.2). Se consulta ANTES de construir modelo
# alguno. Regla madre: degradar con aviso cuando el resultado sigue siendo lo pedido con
# menos precision; rechazar cuando el knob seria inerte; jamas silencio ni sustitucion.
CAPS: dict[str, dict[str, str]] = {
    ENGINE_FASTER: {
        "hotwords": "honrado",
        "vad_filter": "honrado",
        "word_timestamps": "honrado",
        "compute_type": "honrado",
    },
    ENGINE_WHISPERCPP: {
        # --prompt es INERTE bajo -mc 0 (bit-identico en 7 corridas, medido 2026-07-27);
        # sin -mc 0 contamina la ortografia global. Avisar "degradado" sobre un knob
        # inerte fabricaria un efecto que no ocurrio: rechazo.
        "hotwords": "rechazado",
        "vad_filter": "degradado",
        "word_timestamps": "degradado",
        "compute_type": "mapeado",
    },
}

# ponytail: tabla de dos entradas, no registro de motores; con dos motores el dict ES la politica.
_WCPP_QUANT = {"auto": "q5_0", "q5_0": "q5_0"}


def quant_for(engine: str, compute_type: str, device: str) -> str:
    """La cuantizacion EFECTIVA: lo que entra a la llave del cache y al bloque engine del JSON."""
    if engine != ENGINE_WHISPERCPP:
        return compute_type
    quant = _WCPP_QUANT.get(compute_type)
    if quant is None:
        raise ValueError(
            f"compute_type={compute_type!r} no soportado con whispercpp; usa 'auto' o 'q5_0'. "
            "Motivo: fp16 = 0.53x tiempo real por paging WDDM en la 980 (medido 2026-07-27)."
        )
    return quant


def effective_opts(engine: str, opts: dict) -> dict:
    """Los opts EFECTIVOS post-mapeo: lo que entra a la llave del cache (plan 2.3).

    Bajo whispercpp, vad_filter y word_timestamps efectivos son False: --vad y --no-vad
    producen el MISMO digest porque la salida del motor es identica.
    """
    if engine != ENGINE_WHISPERCPP:
        return opts
    if opts.get("hotwords"):
        # No deberia llegar aqui: la cli valida CAPS antes. Doble candado, no cortesia:
        # hotwords jamas entra al join del cache bajo whispercpp.
        raise ValueError("--hotwords no tiene efecto con whisper.cpp; usa --engine faster-whisper")
    eff = dict(opts)
    eff["vad_filter"] = False
    eff["word_timestamps"] = False
    return eff


def make_engine(engine: str, model_name: str, device: str, compute_type: str, *,
                cpu_threads: int | None = None, num_workers: int | None = None):
    """Los DOS puntos de construccion del modelo (thunk de run_chunked y ruta directa)
    pasan por aqui. Devuelve un objeto duck-typed con .transcribe(path, **opts)."""
    if engine == ENGINE_FASTER:
        from faster_whisper import WhisperModel  # perezoso: no pagar el import si no toca

        kwargs = {}
        if cpu_threads is not None:
            kwargs["cpu_threads"] = cpu_threads
        if num_workers is not None:
            kwargs["num_workers"] = num_workers
        return WhisperModel(model_name, device=device, compute_type=compute_type, **kwargs)
    if engine == ENGINE_WHISPERCPP:
        # Import perezoso del pin: la ruta faster-whisper jamas toca la descarga/verificacion.
        from speechtotext.core import enginepin

        quant = quant_for(engine, compute_type, device)
        return WhisperCppEngine(enginepin.ensure_engine(), enginepin.ensure_model(model_name), quant)
    raise ValueError(f"engine desconocido: {engine!r}; disponibles: {ENGINES}")


def _parse_ojf(data: dict):
    """Parser del JSON de `-ojf` (fixture real: tests/fixtures/whispercpp_ojf.json).

    Usa offsets (ms enteros -> segundos) y text; ignora tokens en v1. El texto se
    conserva tal cual (espacio inicial incluido), paridad con faster-whisper.
    Senales que el motor no emite se OMITEN, jamas se rellenan (ley G5): words=None,
    language_probability=None, y SIN duration — esa la fabrica el orquestador.
    """
    segments = []
    for entry in data["transcription"]:
        if not entry["text"].strip():
            continue  # segmentos vacios no aportan texto y ensucian la cobertura
        segments.append(SimpleNamespace(
            start=entry["offsets"]["from"] / 1000.0,
            end=entry["offsets"]["to"] / 1000.0,
            text=entry["text"],
            words=None,
        ))
    info = SimpleNamespace(language=data["result"].get("language"), language_probability=None)
    return segments, info


class WhisperCppEngine:
    """Adaptador subprocess sobre whisper-cli.exe (prebuilt CUDA pinneado, plan 2.8).

    Habla el dialecto duck-typed de WhisperModel: transcribe(wav, **opts) -> (segmentos,
    info). El wav de entrada llega SIEMPRE 16k mono ya transcodificado (la ruta troceada
    lo garantiza; la directa transcodifica antes): el path del audio original del usuario
    jamas viaja en argv.
    """

    def __init__(self, exe_path, model_path, quant: str):
        self.exe_path = Path(exe_path)
        self.model_path = Path(model_path)
        self.quant = quant

    def transcribe(self, audio_path, **opts):
        # Inyectada por el orquestador para el timeout proporcional; jamas viaja en argv.
        duration_s = opts.pop("_duration_s", None)
        fd, base = tempfile.mkstemp(prefix="wcpp-")  # base SIN extension; el CLI escribe base+".json"
        os.close(fd)
        json_path = base + ".json"
        try:
            # whisper-cli es main(char**): argv viaja en ANSI y una ruta no-ASCII llega
            # corrupta en silencio. Fallar ANTES de correr, con causa.
            for p in (str(audio_path), base):
                if not p.isascii():
                    raise RuntimeError(
                        f"ruta con caracteres no ASCII; whisper-cli no la soporta: {p}"
                    )
            cmd = [
                str(self.exe_path),
                "-m", str(self.model_path),
                "-f", str(audio_path),
                # None = --language auto; whisper-cli lo detecta con "auto"
                "-l", str(opts.get("language") or "auto"),
                "-bs", str(opts["beam_size"]),
                # -mc 0 HARDCODEADO: paridad con condition_on_previous_text=False.
                # Sin el: loops de repeticion y 3x el tiempo (medido 2026-07-27).
                "-mc", "0",
                "-np", "-ojf", "-of", base,
            ]
            # 4x cubre 17 veces el caso caliente medido (19.3x tiempo real); piso 120 s
            # para el JIT frio (H6 lo congela). Sin duracion conocida: techo generoso.
            timeout = max(120, 4 * duration_s) if duration_s else 3600
            proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
            if proc.returncode != 0:
                tail = "\n".join(
                    proc.stderr.decode("utf-8", errors="replace").splitlines()[-10:]
                )
                raise RuntimeError(f"whisper-cli rc={proc.returncode}: {tail}")
            try:
                # encoding explicito: Python en Windows abre cp1252 por default
                data = json.loads(Path(json_path).read_text(encoding="utf-8"))
                return _parse_ojf(data)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"whisper-cli termino bien pero su JSON no sirve ({json_path}): {exc}"
                ) from exc
        finally:
            # El wav no es nuestro (lo creo el orquestador): no se toca. La base de
            # mkstemp y el .json si (puede no existir si el exe murio antes).
            for p in (base, json_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass
