"""Herramientas reproducibles de evaluacion; dependencias pesadas son opcionales."""

from speechtotext.evaluation.manifest import (
    CorpusEntry,
    CorpusManifest,
    CorpusAsset,
    parse_corpus_manifest_bytes,
    load_corpus_manifest,
)
from speechtotext.evaluation.splits import DatasetSplit, split_by_recording_day
from speechtotext.evaluation.training import (
    LabeledFeatureExample,
    LabeledFeaturePartition,
    fit_segment_usable_calibrator,
)
from speechtotext.evaluation.metrics import (
    ErrorRate,
    RiskCoveragePoint,
    brier_score,
    character_error,
    cluster_bootstrap_error_upper,
    cluster_bootstrap_percentile_upper,
    expected_calibration_error,
    normalize_transcript,
    one_sided_error_upper,
    one_sided_success_lower,
    percentile,
    risk_coverage_curve,
    word_error,
)

__all__ = [
    "CorpusEntry",
    "CorpusManifest",
    "CorpusAsset",
    "parse_corpus_manifest_bytes",
    "load_corpus_manifest",
]

__all__ += ["DatasetSplit", "split_by_recording_day"]

__all__ += [
    "LabeledFeatureExample",
    "LabeledFeaturePartition",
    "fit_segment_usable_calibrator",
]

__all__ += [
    "ErrorRate",
    "RiskCoveragePoint",
    "brier_score",
    "character_error",
    "cluster_bootstrap_error_upper",
    "cluster_bootstrap_percentile_upper",
    "expected_calibration_error",
    "normalize_transcript",
    "one_sided_error_upper",
    "one_sided_success_lower",
    "percentile",
    "risk_coverage_curve",
    "word_error",
]
