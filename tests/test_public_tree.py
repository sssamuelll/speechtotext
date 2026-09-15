"""The public tree speaks English. This file is the ratchet.

It does not grade English or judge style. It finds Spanish that nobody translated,
using three cheap signals:

  1. Letters English does not write: n-tilde, inverted marks, accented vowels.
     One is enough to flag the line.
  2. Spanish function words, at a threshold of TWO distinct words per file. The
     threshold is the point: `con`, `los` and `solo` are English words too, and a
     one-hit rule would grow an exception list until nobody reads it. No ten-word
     Spanish sentence contains only one function word.
  3. The project's own Spanish vocabulary -- the left column of the plan's
     glossary. `trozo`, `hueco`, `hablante` and the rest are not English in any
     reading, so one hit is enough.

What this cannot see, said plainly: a three-word comment with no accent in it.
`# sube el pin` trips nothing. The complement is the sampled read that section 8
of the design asks for, not more regex.

PENDING lists the files nobody has translated yet. Each task of the plan deletes
its line. When the list empties, the test stops forgiving -- and that, not
anyone's opinion, is what "done" means here.

Deliberately out of scope: `docs/superpowers/` and `.superpowers/`. Section 9.5
strips both from the published tree, so translating them is wasted work -- and a
sweep that includes them ends up editing the design document to silence a grep,
which is exactly what happened during plan 2c.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).relative_to(ROOT).as_posix()

# What ships. `docs/superpowers/` is not here and never will be.
WATCHED = (
    "src",
    "tests",
    "scripts",
    ".github",
    "docs/api.md",
    "docs/README.md",
    "docs/design.md",
    "README.md",
    "CHANGELOG.md",
    "pyproject.toml",
    ".gitignore",
)

# Still untranslated. One line per task; they come off in order.
PENDING = (
    "src/",                          # task 4
    "tests/",                        # task 5
    "README.md",                     # task 6
    "docs/api.md",                   # task 7
    "CHANGELOG.md",                  # task 7
    "scripts/benchmark_chart.py",    # task 7
    "docs/README.md",                # task 8
)

# Terms English writes with an accent. Not leftover Spanish.
ALLOWED = ("Bézier",)

ACCENTS = re.compile("[ñÑ¿¡áéíóúüÁÉÍÓÚÜ]")

# Spanish function words that are not English words. `en` and `es` are left out on
# purpose: they are language codes and appear everywhere (`language="es"`). So are
# `todo` and friends -- `# TODO:` would trip them.
#
# `del` stays in despite being a Python keyword: it is the densest single signal in
# this repo's Spanish (68 occurrences in comments) against one file that uses `del`
# as a statement. If a fully translated file ever fails this check with `del` among
# the words listed, that is the reason -- and it still needs a second, real hit to
# fail at all.
#
# The boundary is a lookaround, not `\b`, because `\b` treats `_` as a word
# character: `test_mide_voz_ruido` hides every word in it from `\b`, and test names
# are 414 of the strings this file exists to find.
FUNCTION_WORDS = re.compile(
    r"(?<![A-Za-z0-9])(que|para|del|las|los|una|uno|este|esta|esto|estos|estas"
    r"|como|cuando|donde|porque|pero|aunque|desde|hasta|entre|sobre|cada|nada"
    r"|algo|hay|tiene|tienen|puede|pueden|debe|deben|dice|dicen|hace|hacen"
    r"|mismo|misma|otro|otra|sino|antes|siempre|nunca|menos|ahora|luego"
    r"|entonces|ser|estar|tener|hacer|por|nos|les|sus|muy|con|solo)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# The glossary's left column: this project's Spanish nouns and verbs. One hit is
# enough -- none of these is an English word. `motor` and `vista` belong to the
# glossary but are left out here on purpose: English writes both.
GLOSSARY = re.compile(
    r"(?<![A-Za-z0-9])(trozo|trozos|trocear|troceado|troceada|costura|costuras"
    r"|hueco|huecos|sondeo|sondear|ruta|rutas|huella|huellas|senal|senales"
    r"|hablante|hablantes|muestra|muestras|reloj|medido|medida|medidas|apagon"
    r"|apagones|cuantizacion|ganancia|recorte|recortar|pista|pistas|honrado"
    r"|degradado|rechazado|aparato|archivo|archivos|modelo|modelos|salida"
    r"|salidas|entrada|entradas|prueba|pruebas|palabra|palabras|mensaje"
    r"|mensajes|nombre|nombres|tiempo|silencio|voz|voces|calidad|idioma"
    r"|texto|textos|habla|transcripcion|segmento|segmentos|tabla|tablas"
    r"|caso|casos|bajo|sola)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def _files():
    """Every text file under WATCHED, skipping PENDING and this file."""
    for entry in WATCHED:
        root = ROOT / entry
        if not root.exists():
            continue
        candidates = sorted(root.rglob("*")) if root.is_dir() else [root]
        for path in candidates:
            if not path.is_file():
                continue
            # The suffix filter guards directory walks only. An entry that names
            # a file names it on purpose, and `.gitignore` has no suffix at all:
            # Path(".gitignore").suffix is "".
            if root.is_dir() and path.suffix not in (".py", ".md", ".toml", ".yml"):
                continue
            rel = path.relative_to(ROOT).as_posix()
            if rel == SELF or any(rel == p or rel.startswith(p) for p in PENDING):
                continue
            yield rel


def _read(rel: str) -> str:
    text = (ROOT / rel).read_text(encoding="utf-8")
    for allowed in ALLOWED:
        text = text.replace(allowed, "")
    return text


FILES = sorted(_files())


@pytest.mark.parametrize("rel", FILES)
def test_no_spanish_letters(rel):
    hits = [
        f"{rel}:{n}: {line.strip()[:90]}"
        for n, line in enumerate(_read(rel).splitlines(), 1)
        if ACCENTS.search(line)
    ]
    assert not hits, "untranslated Spanish:\n" + "\n".join(hits)


@pytest.mark.parametrize("rel", FILES)
def test_no_spanish_function_words(rel):
    found = {m.group(0).lower() for m in FUNCTION_WORDS.finditer(_read(rel))}
    assert len(found) < 2, (
        f"{rel} uses {len(found)} distinct Spanish function words "
        f"({', '.join(sorted(found))}): that is Spanish prose, not a false positive"
    )


@pytest.mark.parametrize("rel", FILES)
def test_no_spanish_vocabulary(rel):
    found = {m.group(0).lower() for m in GLOSSARY.finditer(_read(rel))}
    assert not found, f"{rel} still uses the Spanish glossary: {', '.join(sorted(found))}"


def test_pending_only_names_things_that_exist():
    """A PENDING entry pointing at nothing is a task that forgot to delete its
    line: the file left the watch list and nobody noticed."""
    orphans = [p for p in PENDING if not (ROOT / p).exists()]
    assert not orphans, f"PENDING names paths that do not exist: {orphans}"


# Real people. The repo ships with the history of a private project attached to
# it; these two are the ones who leaked into fixtures and examples. There is no
# PENDING for this one -- a real name in a published tree is not a translation
# that got delayed.
REAL_NAMES = re.compile(r"\b(samuel|simon|ale)\b", re.IGNORECASE)


@pytest.mark.parametrize("rel", FILES)
def test_no_real_names(rel):
    hits = [
        f"{rel}:{n}: {line.strip()[:90]}"
        for n, line in enumerate((ROOT / rel).read_text(encoding="utf-8").splitlines(), 1)
        if REAL_NAMES.search(line)
    ]
    assert not hits, "a real person's name in a public file:\n" + "\n".join(hits)
