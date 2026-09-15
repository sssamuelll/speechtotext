import numpy as np
import pytest
from hypothesis import given, strategies as st

from speechtotext.audio.level import apply_fixed_gain


def test_fixed_gain_applies_a_constant_factor():
    result = apply_fixed_gain(np.array([0.01, -0.02], dtype=np.float32), 6.0)
    ratio = result.samples / np.array([0.01, -0.02], dtype=np.float32)
    assert ratio[0] == pytest.approx(ratio[1], rel=1e-6)
    assert result.applied_gain_db == pytest.approx(6.0)
    assert result.limited is False
    assert result.samples.flags.writeable is False


def test_fixed_gain_reduces_gain_to_respect_the_peak_limit():
    result = apply_fixed_gain(np.array([0.9, -0.5], dtype=np.float32), 18.0)
    peak_limit = 10 ** (-1.0 / 20.0)
    assert np.max(np.abs(result.samples)) <= peak_limit + 1e-6
    assert result.applied_gain_db < 18.0
    assert result.limited is True


def test_fixed_gain_rejects_gain_above_the_maximum():
    with pytest.raises(ValueError, match="max_gain_db"):
        apply_fixed_gain(np.array([0.1], dtype=np.float32), 18.1)


def test_fixed_gain_does_not_create_a_signal_from_silence():
    result = apply_fixed_gain(np.zeros(4, dtype=np.float32), 18.0)
    assert result.samples.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert result.applied_gain_db == 18.0


@given(
    st.lists(
        st.floats(
            min_value=-1.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
        min_size=1,
        max_size=2000,
    ),
    st.floats(
        min_value=-12.0,
        max_value=18.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_fixed_gain_is_always_finite_and_respects_the_peak_limit(samples, gain_db):
    result = apply_fixed_gain(np.asarray(samples, dtype=np.float32), gain_db)
    assert np.isfinite(result.samples).all()
    assert np.max(np.abs(result.samples)) <= 10 ** (-1.0 / 20.0) + 1e-6
