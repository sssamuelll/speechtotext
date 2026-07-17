from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from speechtotext.statistics import (
    one_sided_error_upper,
    one_sided_success_lower as one_sided_success_lower,
)


def normalize_transcript(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    without_punctuation = "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in normalized
    )
    return " ".join(without_punctuation.split())


def _edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, ref_item in enumerate(reference, start=1):
        current = [row]
        for column, hyp_item in enumerate(hypothesis, start=1):
            substitution = previous[column - 1] + int(ref_item != hyp_item)
            deletion = previous[column] + 1
            insertion = current[column - 1] + 1
            current.append(min(substitution, deletion, insertion))
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class ErrorRate:
    errors: int
    reference_units: int
    rate: float


def _error_rate(reference: Sequence[str], hypothesis: Sequence[str]) -> ErrorRate:
    errors = _edit_distance(reference, hypothesis)
    denominator = len(reference)
    if denominator == 0:
        rate = 0.0 if not hypothesis else 1.0
    else:
        rate = errors / denominator
    return ErrorRate(errors, denominator, rate)


def word_error(reference: str, hypothesis: str) -> ErrorRate:
    return _error_rate(
        normalize_transcript(reference).split(),
        normalize_transcript(hypothesis).split(),
    )


def character_error(reference: str, hypothesis: str) -> ErrorRate:
    return _error_rate(
        list(normalize_transcript(reference).replace(" ", "")),
        list(normalize_transcript(hypothesis).replace(" ", "")),
    )


def cluster_bootstrap_error_upper(
    counts: Sequence[tuple[int, int]],
    confidence: float = 0.95,
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> float:
    if len(counts) < 3:
        raise ValueError("se requieren al menos tres bloques independientes")
    if resamples < 1000 or not 0.0 < confidence < 1.0:
        raise ValueError("resamples/confidence invalidos")
    array = np.asarray(counts, dtype=np.int64)
    if np.any(array < 0) or np.any(array[:, 1] == 0):
        raise ValueError("conteos de error invalidos")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(array), size=(resamples, len(array)))
    block_rates = array[:, 0] / array[:, 1]
    bootstrap_macro = block_rates[sampled].mean(axis=1)
    bootstrap_upper = float(
        np.quantile(bootstrap_macro, confidence, method="higher")
    )
    simultaneous_confidence = 1.0 - (1.0 - confidence) / len(array)
    block_uppers = [
        one_sided_error_upper(int(errors), int(units), simultaneous_confidence)
        if errors <= units
        else float(errors / units)
        for errors, units in array
    ]
    simultaneous_macro_upper = float(np.mean(block_uppers))
    return max(
        float(np.mean(block_rates)),
        bootstrap_upper,
        simultaneous_macro_upper,
    )


def cluster_bootstrap_percentile_upper(
    blocks: Sequence[Sequence[float]],
    q: float = 95.0,
    confidence: float = 0.95,
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> float:
    if len(blocks) < 3:
        raise ValueError("se requieren al menos tres bloques independientes")
    if (
        resamples < 1000
        or not 0.0 <= q <= 100.0
        or not 0.0 < confidence < 1.0
    ):
        raise ValueError("q/resamples/confidence invalidos")
    arrays = tuple(np.asarray(block, dtype=np.float64) for block in blocks)
    if any(array.size == 0 or not np.isfinite(array).all() for array in arrays):
        raise ValueError("los bloques de latencia deben ser no vacios y finitos")
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.integers(0, len(arrays), size=len(arrays))
        values = np.concatenate([arrays[block_index] for block_index in sampled])
        estimates[index] = np.percentile(values, q)
    return float(np.quantile(estimates, confidence, method="higher"))


def _validate_binary(
    probabilities: Sequence[float],
    labels: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("probabilities y labels deben ser no vacios e igual longitud")
    if any(isinstance(value, (bool, np.bool_)) for value in probabilities):
        raise ValueError("probabilidades fuera de rango o no finitas")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in labels
    ):
        raise ValueError("labels deben ser enteros binarios")
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int8)
    if (
        not np.isfinite(probs).all()
        or np.any((probs < 0.0) | (probs > 1.0))
        or np.any((truth != 0) & (truth != 1))
    ):
        raise ValueError("probabilidades o labels fuera de rango")
    return probs, truth


def brier_score(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    probs, truth = _validate_binary(probabilities, labels)
    return float(np.mean(np.square(probs - truth)))


def expected_calibration_error(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    bins: int = 10,
) -> float:
    if bins < 1:
        raise ValueError("bins debe ser >= 1")
    probs, truth = _validate_binary(probabilities, labels)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = len(probs)
    error = 0.0
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = (
            (probs >= lower) & (probs <= upper)
            if index == bins - 1
            else (probs >= lower) & (probs < upper)
        )
        if np.any(mask):
            error += (
                float(np.sum(mask))
                / total
                * abs(float(np.mean(probs[mask])) - float(np.mean(truth[mask])))
            )
    return error


@dataclass(frozen=True)
class RiskCoveragePoint:
    threshold: float
    coverage: float
    risk: float
    errors: int
    error_upper_95: float
    accepted: int
    total: int


def risk_coverage_curve(
    probabilities: Sequence[float],
    labels: Sequence[int],
) -> tuple[RiskCoveragePoint, ...]:
    probs, truth = _validate_binary(probabilities, labels)
    predictions = (probs >= 0.5).astype(np.int8)
    order = np.argsort(-probs, kind="stable")
    points: list[RiskCoveragePoint] = []
    mistakes = 0
    for rank, index in enumerate(order, start=1):
        mistakes += int(predictions[index] != truth[index])
        next_probability = probs[order[rank]] if rank < len(order) else None
        if next_probability is None or next_probability != probs[index]:
            points.append(
                RiskCoveragePoint(
                    threshold=float(probs[index]),
                    coverage=rank / len(order),
                    risk=mistakes / rank,
                    errors=mistakes,
                    error_upper_95=one_sided_error_upper(mistakes, rank),
                    accepted=rank,
                    total=len(order),
                )
            )
    return tuple(points)


def percentile(values: Sequence[float], q: float) -> float:
    if not values or not 0.0 <= q <= 100.0:
        raise ValueError("values no vacio y q en [0, 100]")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))
