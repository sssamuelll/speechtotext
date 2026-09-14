"""El mapa se rompe con ruido: si `docs/api.md` y el código se separan, esto lo dice.

Dos direcciones, ninguna con parser de firmas:

  (a) mecánica — todo nombre exportado en un `__all__` público aparece en api.md.
  (b) curada   — toda ruta de CONTRATO importa y aparece en api.md.

La (a) caza el modo de fallo que de verdad ocurrió (exportar algo y no documentarlo:
llegaron a ser 17 de 20 en `audio`). La (b) caza el inverso, documentar lo que ya no
existe. Un parser de firmas Markdown se pudriría más rápido que lo que vigila.

`core/` y `speakers/` no tienen `__all__` — se importan por submódulo — así que su
superficie pública vive en CONTRATO, nombre a nombre. Añadir algo ahí es declararlo
contrato: sale en el CHANGELOG cuando cambie.

Límite conocido, a propósito. La dirección (b) compara el nombre DESNUDO contra los
tokens del documento entero, sin atarlo a su módulo, y eso deja dos huecos. Uno: dos
rutas con el mismo último segmento (`core.models.remove` y `speakers.registry.remove`)
se tapan entre sí — si una pierde su documentación, el token de la otra la sigue
cubriendo. Dos: un token puede venir de una mención que no documenta nada; durante un
commit, `speakers.diarization.diarize` pasó en verde porque el documento nombraba la
flag `--diarize` y la etapa de progreso `diarize`, no la función. Exigir la mención
calificada arreglaría ambos y rompería las menciones legítimas del propio documento, así
que se queda el chequeo desnudo y el aviso escrito.
"""
import importlib
import re
from pathlib import Path

import pytest

API_MD = Path(__file__).resolve().parents[1] / "docs" / "api.md"

MODULOS_CON_ALL = ("speechtotext.asr", "speechtotext.audio")

CONTRATO = (
    "speechtotext.core.transcribe.transcribe",
    "speechtotext.core.transcribe.Transcript",
    "speechtotext.core.transcribe.Progress",
    "speechtotext.core.transcribe.EngineInfo",
    "speechtotext.core.transcribe.DiarizationReport",
    "speechtotext.core.transcribe.load_audio",
    "speechtotext.core.probe.machine",
    "speechtotext.core.probe.choose_route",
    "speechtotext.core.probe.Machine",
    "speechtotext.core.probe.Route",
    "speechtotext.core.models.data_dir",
    "speechtotext.core.models.installed",
    "speechtotext.core.models.ensure",
    "speechtotext.core.models.remove",
    "speechtotext.core.models.remote_size",
    "speechtotext.core.models.ModelInfo",
    "speechtotext.core.formats.write_json",
    "speechtotext.core.formats.is_suspect",
    "speechtotext.core.segments.native_signals",
    "speechtotext.speakers.registry.enroll",
    "speechtotext.speakers.registry.list_voices",
    "speechtotext.speakers.registry.get_embeddings",
    "speechtotext.speakers.registry.remove",
    "speechtotext.speakers.identify.assign_names",
    "speechtotext.speakers.diarization.diarize",
    "speechtotext.speakers.diarization.embed_voice",
    "speechtotext.asr.base.AsrError",
    "speechtotext.asr.faster_whisper.FasterWhisperBackend",
    "speechtotext.asr.whispercpp.WhisperCppBackend",
)

_BLOQUE = re.compile(r"```.*?```", re.S)


def _nombrados(texto: str) -> set[str]:
    """Identificadores que api.md nombra en código o en un encabezado.

    Cuenta los bloques cercados, los `code span` sueltos y los encabezados; la prosa
    no cuenta, porque nombrar algo de pasada no es documentarlo. Se tokeniza en vez
    de buscar la subcadena: si no, `AudioView` pasaría gratis porque existe
    `AudioViewName`.
    """
    bloques = _BLOQUE.findall(texto)
    resto = _BLOQUE.sub("\n", texto)
    trozos = (bloques
              + re.findall(r"`([^`\n]+)`", resto)
              + re.findall(r"^#{1,6}\s+(.+)$", resto, re.M))
    return {t for trozo in trozos for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", trozo)}


@pytest.fixture(scope="module")
def nombrados() -> set[str]:
    return _nombrados(API_MD.read_text(encoding="utf-8"))


@pytest.mark.parametrize("modulo", MODULOS_CON_ALL)
def test_todo_lo_exportado_esta_documentado(modulo, nombrados):
    exportados = importlib.import_module(modulo).__all__
    faltan = sorted(n for n in exportados if n not in nombrados)
    assert not faltan, (
        f"{modulo}.__all__ exporta {len(faltan)} nombres que docs/api.md no menciona: "
        f"{', '.join(faltan)}. Documéntalos, o sácalos de __all__ si no son contrato."
    )


@pytest.mark.parametrize("ruta", CONTRATO)
def test_lo_documentado_existe_y_se_importa(ruta, nombrados):
    modulo, _, nombre = ruta.rpartition(".")
    objeto = importlib.import_module(modulo)
    assert hasattr(objeto, nombre), (
        f"docs/api.md promete {ruta} y no existe. Si se renombró, renómbralo también "
        f"en el documento y en el CHANGELOG (rompe)."
    )
    assert nombre in nombrados, f"{ruta} es contrato y docs/api.md no lo nombra"


def test_el_tokenizador_no_regala_prefijos():
    """La trampa que hay que evitar: `AudioViewName` no puede documentar `AudioView`."""
    tokens = _nombrados("Ver `AudioViewName` para las vistas.")
    assert "AudioViewName" in tokens
    assert "AudioView" not in tokens


def test_la_prosa_no_documenta():
    """Nombrar algo en una frase no es documentarlo: tiene que estar en código."""
    assert _nombrados("AudioClip es la entrada por clip.") == set()
    assert "AudioClip" in _nombrados("- **`AudioClip(started_at, ...)`** — la entrada.")
