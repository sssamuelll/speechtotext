from __future__ import annotations

import math

_BISECTION_STEPS = 80


def _logaddexp(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    high, low = max(left, right), min(left, right)
    return high + math.log1p(math.exp(low - high))


def _log_binomial_cdf(errors: int, trials: int, probability: float) -> float:
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 0.0 if errors == trials else -math.inf
    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    total = -math.inf
    for value in range(errors + 1):
        term = (
            math.lgamma(trials + 1)
            - math.lgamma(value + 1)
            - math.lgamma(trials - value + 1)
            + value * log_p
            + (trials - value) * log_q
        )
        total = _logaddexp(total, term)
    return min(0.0, total)


def one_sided_error_upper(
    errors: int,
    trials: int,
    confidence: float = 0.95,
) -> float:
    if trials <= 0 or not 0 <= errors <= trials:
        raise ValueError("errors/trials invalidos")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence debe estar en (0, 1)")
    if errors == trials:
        return 1.0
    alpha_log = math.log1p(-confidence)
    if errors == 0:
        return -math.expm1(alpha_log / trials)
    low, high = 0.0, 1.0
    for _ in range(_BISECTION_STEPS):
        midpoint = (low + high) / 2.0
        if _log_binomial_cdf(errors, trials, midpoint) > alpha_log:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def one_sided_success_lower(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> float:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("successes/trials invalidos")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence debe estar en (0, 1)")
    if successes == 0:
        return 0.0
    return 1.0 - one_sided_error_upper(
        trials - successes,
        trials,
        confidence,
    )
