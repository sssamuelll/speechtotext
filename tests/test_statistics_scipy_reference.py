import pytest

scipy_stats = pytest.importorskip("scipy.stats")

from speechtotext.statistics import one_sided_success_lower

# Este archivo pertenece al extra `evaluation`: usa SciPy solo como oraculo
# independiente del limite inferior Clopper-Pearson unilateral. Runtime ni el
# trainer productivo lo importan.

CONFIDENCE = 0.95
ALPHA = 1.0 - CONFIDENCE
TOLERANCE = 2e-12


def _grid(n):
    return sorted({0, 1, n // 2, n - 1, n})


@pytest.mark.parametrize("n", [3, 10, 100, 3000])
def test_lower_bound_coincide_con_beta_ppf(n):
    for k in _grid(n):
        actual = one_sided_success_lower(k, n, CONFIDENCE)
        if k == 0:
            assert actual == 0.0
            continue
        expected = float(scipy_stats.beta.ppf(ALPHA, k, n - k + 1))
        assert actual == pytest.approx(expected, abs=TOLERANCE)
