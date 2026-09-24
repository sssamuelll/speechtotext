"""The map fails loudly: if `docs/api.md` and the code diverge, this says so.

Two directions, neither with a signature parser:

  (a) mechanical — every name exported by a public `__all__` appears in api.md.
  (b) curated    — every path in CONTRACT imports and appears in api.md.

Direction (a) catches the failure mode that actually occurred (exporting something without
documenting it: only 17 of 20 names in `audio` were documented). Direction (b) catches the
inverse, documenting something that no longer exists. A Markdown signature parser would rot
faster than what it monitors.

`core/` and `speakers/` have no `__all__` — they are imported by submodule — so their public
surface lives in CONTRACT, name by name. Adding something there declares it part of the
contract: it goes in the CHANGELOG when it changes.

Known limitation, by design. Direction (b) compares the BARE name against tokens from the
entire document without tying it to its module, leaving two gaps. First, two paths with the
same final segment (`core.models.remove` and `speakers.registry.remove`) mask each other — if
one loses its documentation, the other's token still covers it. Second, a token can come from
a mention that documents nothing; during one commit, `speakers.diarization.diarize` passed
because the document named the `--diarize` flag and the `diarize` progress stage, not the
function. Requiring a qualified mention would fix both problems and break legitimate mentions
within the document itself, so the bare check and this written warning remain.
"""
import importlib
import re
from pathlib import Path

import pytest

API_MD = Path(__file__).resolve().parents[1] / "docs" / "api.md"

MODULES_WITH_ALL = ("speechtotext.asr", "speechtotext.audio")

CONTRACT = (
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
    "speechtotext.speakers.nemotron.diarize",
    "speechtotext.speakers.nemotron.missing",
    "speechtotext.asr.base.AsrError",
    "speechtotext.asr.faster_whisper.FasterWhisperBackend",
    "speechtotext.asr.whispercpp.WhisperCppBackend",
)

_BLOCK = re.compile(r"```.*?```", re.S)


def _named_identifiers(text: str) -> set[str]:
    """Identifiers that api.md names in code or in a heading.

    Count fenced blocks, individual `code spans`, and headings; prose does not count,
    because mentioning something in passing does not document it. Tokenize instead of
    searching for a substring; otherwise, `AudioView` would pass for free because
    `AudioViewName` exists.
    """
    blocks = _BLOCK.findall(text)
    remainder = _BLOCK.sub("\n", text)
    pieces = (blocks
              + re.findall(r"`([^`\n]+)`", remainder)
              + re.findall(r"^#{1,6}\s+(.+)$", remainder, re.M))
    return {token for piece in pieces
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", piece)}


@pytest.fixture(scope="module")
def named_identifiers() -> set[str]:
    return _named_identifiers(API_MD.read_text(encoding="utf-8"))


@pytest.mark.parametrize("module", MODULES_WITH_ALL)
def test_every_exported_name_is_documented(module, named_identifiers):
    exported_names = importlib.import_module(module).__all__
    missing = sorted(name for name in exported_names if name not in named_identifiers)
    assert not missing, (
        f"{module}.__all__ exports {len(missing)} names that docs/api.md does not mention: "
        f"{', '.join(missing)}. Document them, or remove them from __all__ if they are not "
        f"part of the contract."
    )


@pytest.mark.parametrize("dotted_path", CONTRACT)
def test_every_documented_name_exists_and_imports(dotted_path, named_identifiers):
    module, _, name = dotted_path.rpartition(".")
    target = importlib.import_module(module)
    assert hasattr(target, name), (
        f"docs/api.md promises {dotted_path}, but it does not exist. If it was renamed, "
        f"rename it in the document and the CHANGELOG as well (breaking change)."
    )
    assert name in named_identifiers, (
        f"{dotted_path} is part of the contract, but docs/api.md does not name it"
    )


def test_the_tokenizer_does_not_accept_prefixes_for_free():
    """The trap to avoid: `AudioViewName` cannot document `AudioView`."""
    tokens = _named_identifiers("See `AudioViewName` for the views.")
    assert "AudioViewName" in tokens
    assert "AudioView" not in tokens


def test_prose_does_not_count_as_documentation():
    """Naming something in a sentence does not document it: it must be in code."""
    assert _named_identifiers("AudioClip is the entry for a clip.") == set()
    assert "AudioClip" in _named_identifiers("- **`AudioClip(started_at, ...)`** — the entry.")


def test_no_code_span_crosses_a_line_break():
    """A span split across two lines is not seen by `[^`\\n]+`: the symbol it names no
    longer counts as documented and, worse, it throws off matching for the rest of the
    line. This is how `TranscriptionRequest` and `Route` stopped counting from their own
    signatures and passed only because they were named elsewhere. Monitor this here, not
    in the mind of whoever edits the document."""
    inside_code_block = False
    split_spans = []
    for line_number, line in enumerate(API_MD.read_text(encoding="utf-8").split("\n"), 1):
        if line.lstrip().startswith("```"):
            inside_code_block = not inside_code_block
        elif not inside_code_block and line.count("`") % 2:
            split_spans.append(line_number)
    assert not split_spans, (
        f"docs/api.md has code spans that cross a line break, on lines {split_spans}. "
        f"Adjust the break so the entire span fits on one line: a split span does not "
        f"count as documentation."
    )


def test_the_document_does_not_call_monitored_code_internal():
    """A contradiction that has already appeared twice in this document: api.md
    declaring a submodule internal while CONTRACT monitors one of its symbols. Both
    times, someone wrote it carefully; that is why a test monitors it now."""
    text = API_MD.read_text(encoding="utf-8")
    assert "The rest of `core/` is internal" in text, (
        "the sentence that marks the list of internal core/ submodules changed; "
        "adjust this test to the new sentence or the warning will cease to exist"
    )
    paragraph = text.split("The rest of `core/` is internal", 1)[1].split("\n\n", 1)[0]
    internal_modules = set(re.findall(r"`([a-z_]+)`", paragraph))
    monitored_modules = {
        dotted_path.split(".")[2]
        for dotted_path in CONTRACT
        if dotted_path.startswith("speechtotext.core.")
    }
    conflict = internal_modules & monitored_modules
    assert not conflict, (
        f"docs/api.md calls submodules internal while CONTRACT monitors one of their "
        f"symbols: {sorted(conflict)}. Decide which statement is true."
    )
