"""Herramientas reproducibles de evaluacion; dependencias pesadas son opcionales."""

from speechtotext.evaluation.manifest import (
    CorpusEntry,
    CorpusManifest,
    CorpusAsset,
    parse_corpus_manifest_bytes,
    load_corpus_manifest,
)

__all__ = [
    "CorpusEntry",
    "CorpusManifest",
    "CorpusAsset",
    "parse_corpus_manifest_bytes",
    "load_corpus_manifest",
]
