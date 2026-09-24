"""The front page and `docs/` are one document split across files, and every
link between them is a promise. GitHub keeps none of those promises for you:
a moved file is a 404, a renamed heading drops the reader at the top of a
long page in silence, and a Python example that imports a name that no
longer exists looks exactly like one that works. This file is what fails
instead.

`tests/test_public_tree.py` already watches the README's anchors into
`docs/api.md`. This generalizes it: every page, every relative link, every
anchor, in every direction, plus the ``python`` blocks. The slug rule is the
one GitHub applies to headings (lowercase, drop what is not a letter, digit,
space, hyphen or underscore, each space to a hyphen, `-N` on a repeat), so
what resolves here is what resolves there.
"""
import ast
import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _pages(root: Path) -> list[str]:
    """Every markdown page that ships: the top level, `docs/` without the
    working documents under `docs/superpowers/`, and `.github/`."""
    found = list(root.glob("*.md"))
    found += [p for p in (root / "docs").rglob("*.md") if "superpowers" not in p.relative_to(root).parts]
    found += list((root / ".github").rglob("*.md"))
    return sorted(p.relative_to(root).as_posix() for p in found)


PAGES = _pages(ROOT)

FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^#{1,6}\s+(\S.*)")
# `[text](target)` and `[text](target "title")`; the target ends at whitespace
# or the closing parenthesis. `![alt](image.svg)` matches the same way.
MD_LINK = re.compile(r"\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
# What the README's `<picture>` blocks and inline HTML point at.
HTML_REF = re.compile(r"""\b(?:src|srcset|href)\s*=\s*["']([^"']+)["']""")
HTML_ID = re.compile(r"""\b(?:id|name)\s*=\s*["']([^"']+)["']""")
INLINE_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
PY_OPEN = re.compile(r"^\s*```python\s*$")
EXTERNAL = ("http://", "https://", "mailto:")


def _prose(text: str):
    """(line number, line) for every line outside a fenced block: a `#` in
    a shell block is a comment, and a `[x](y)` there is not a link."""
    fenced = False
    for n, line in enumerate(text.splitlines(), 1):
        if FENCE.match(line):
            fenced = not fenced
            continue
        if not fenced:
            yield n, line


def _slug(heading: str) -> str:
    """GitHub's heading id: link text stands in for the link, punctuation
    goes (so do backticks and slashes), the rest is lowercased and each
    space becomes a hyphen."""
    text = INLINE_LINK.sub(r"\1", heading.strip())
    text = re.sub(r"[^\w\s-]", "", text).lower()
    return text.replace(" ", "-")


def _anchors(text: str) -> set[str]:
    """Every id a page exposes: its headings, numbered `-1`, `-2` on a
    repeat the way GitHub numbers them, plus explicit `id=`/`name=` in HTML."""
    seen: dict[str, int] = {}
    found: set[str] = set()
    for _, line in _prose(text):
        m = HEADING.match(line)
        if m:
            base = _slug(m.group(1))
            n = seen.get(base, 0)
            seen[base] = n + 1
            found.add(base if n == 0 else f"{base}-{n}")
        found.update(HTML_ID.findall(line))
    return found


def _links(text: str):
    """(line number, target) for every relative link; external ones are
    never fetched here."""
    for n, line in _prose(text):
        for target in MD_LINK.findall(line) + HTML_REF.findall(line):
            if not target.startswith(EXTERNAL):
                yield n, target


def broken_links(root: Path, page: str) -> list[str]:
    """Every link in `page` (relative to `root`) that leads nowhere, as
    `page:line: target -- why`, in the order they appear."""
    here = root / page
    out = []
    for n, target in _links(here.read_text(encoding="utf-8")):
        path, _, anchor = target.partition("#")
        dest = here.parent / path if path else here
        if not dest.exists():
            out.append(f"{page}:{n}: {target} -- no such file")
            continue
        if anchor:
            if dest.suffix != ".md":
                out.append(f"{page}:{n}: {target} -- an anchor into a file with no headings")
            elif anchor not in _anchors(dest.read_text(encoding="utf-8")):
                rel = dest.resolve().relative_to(root.resolve()).as_posix()
                out.append(f"{page}:{n}: {target} -- no heading in {rel} produces #{anchor}")
    return out


def _python_blocks(text: str):
    """(line number of the first code line, code) per ```python block."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if PY_OPEN.match(lines[i]):
            j = i + 1
            while j < len(lines) and not FENCE.match(lines[j]):
                j += 1
            yield i + 2, "\n".join(lines[i + 1:j])
            i = j + 1
        else:
            i += 1


def example_problems(root: Path, page: str) -> list[str]:
    """Every ```python block in `page` that would not run as written: it
    does not parse, or it imports a `speechtotext` name that is not there.
    Nothing is executed; the block is parsed and its imports are resolved."""
    out = []
    for start, code in _python_blocks((root / page).read_text(encoding="utf-8")):
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            out.append(f"{page}:{start + (exc.lineno or 1) - 1}: does not parse: {exc.msg}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            line = start + node.lineno - 1
            if isinstance(node, ast.ImportFrom):
                if not node.module or node.module.partition(".")[0] != "speechtotext":
                    continue
                try:
                    mod = importlib.import_module(node.module)
                except ImportError as exc:
                    out.append(f"{page}:{line}: from {node.module} import ... -- {exc}")
                    continue
                for alias in node.names:
                    if hasattr(mod, alias.name):
                        continue
                    # `from speechtotext.speakers import registry` names a
                    # submodule, which Python imports on demand.
                    try:
                        importlib.import_module(f"{node.module}.{alias.name}")
                    except ImportError:
                        out.append(f"{page}:{line}: from {node.module} import {alias.name} -- no such name")
            else:
                for alias in node.names:
                    if alias.name.partition(".")[0] != "speechtotext":
                        continue
                    try:
                        importlib.import_module(alias.name)
                    except ImportError as exc:
                        out.append(f"{page}:{line}: import {alias.name} -- {exc}")
    return out


def test_the_page_list_is_not_empty():
    assert "README.md" in PAGES and "docs/api.md" in PAGES


@pytest.mark.parametrize("page", PAGES)
def test_every_relative_link_resolves(page):
    problems = broken_links(ROOT, page)
    assert not problems, "links that lead nowhere:\n" + "\n".join(problems)


@pytest.mark.parametrize("page", PAGES)
def test_python_examples_import_names_that_exist(page):
    problems = example_problems(ROOT, page)
    assert not problems, "examples that would not run:\n" + "\n".join(problems)


def test_broken_links_catches_each_way_a_link_can_lie(tmp_path):
    """Proof, not trust: against a throwaway tree. One good link of each
    kind, and one bad one of each kind; the checker has to name exactly the
    bad ones. The heading with the code span and the slash is there because
    GitHub drops both when it builds the slug, and the repeated heading is
    there because GitHub numbers the second one."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "img").mkdir()
    (tmp_path / "docs" / "img" / "ok.svg").write_text("<svg/>", encoding="utf-8")
    (tmp_path / "docs" / "b.md").write_text(
        "# The `audio/` layer: no verdict\n\n## Limits\n\n## Limits\n\n"
        "```bash\n# not a heading\n```\n\n[up](../a.md#top)\n",
        encoding="utf-8",
    )
    (tmp_path / "a.md").write_text(
        "# Top\n\n"
        "[good](docs/b.md#the-audio-layer-no-verdict) [second](docs/b.md#limits-1)\n"
        '<img src="docs/img/ok.svg"> [self](#top)\n'
        "[dead anchor](docs/b.md#nowhere)\n"
        "[missing file](docs/c.md)\n"
        '<source srcset="docs/img/missing.svg">\n'
        "[fenced](docs/b.md#not-a-heading)\n"
        "[external](https://example.invalid/never-fetched)\n"
        "```text\n[in a code block](docs/never.md)\n```\n",
        encoding="utf-8",
    )

    assert broken_links(tmp_path, "docs/b.md") == []
    assert broken_links(tmp_path, "a.md") == [
        "a.md:5: docs/b.md#nowhere -- no heading in docs/b.md produces #nowhere",
        "a.md:6: docs/c.md -- no such file",
        "a.md:7: docs/img/missing.svg -- no such file",
        "a.md:8: docs/b.md#not-a-heading -- no heading in docs/b.md produces #not-a-heading",
    ]


def test_example_problems_catches_a_name_that_does_not_exist(tmp_path):
    """Same discipline for the examples: a block that imports a real name
    passes, a block that imports a name that is not there is named with its
    line, and a block that does not parse is named too."""
    (tmp_path / "ok.md").write_text(
        "```python\nfrom speechtotext.core.transcribe import transcribe\n```\n", encoding="utf-8"
    )
    (tmp_path / "bad.md").write_text(
        "text\n\n```python\nfrom speechtotext.core.transcribe import no_such_thing\n```\n\n"
        "```python\ndef broken(:\n```\n",
        encoding="utf-8",
    )

    assert example_problems(tmp_path, "ok.md") == []
    problems = example_problems(tmp_path, "bad.md")
    assert problems[0] == "bad.md:4: from speechtotext.core.transcribe import no_such_thing -- no such name"
    assert problems[1].startswith("bad.md:8: does not parse:")
