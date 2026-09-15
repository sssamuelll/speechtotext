"""Postprocessing of transcribed text: time normalization (backlog item #3)."""
from speechtotext.core.postprocess import normalize_hours


def test_converts_time_with_period_to_colon():
    assert normalize_hours("it started at 8.33 in the morning") == "it started at 8:33 in the morning"


def test_converts_multiple_times_on_one_line():
    assert normalize_hours("8.13 and 8.33") == "8:13 and 8:33"


def test_preserves_seismic_magnitudes():
    # A single decimal = seismic magnitude, not a time: it remains a digit.
    assert normalize_hours("a tremor of 7.2 and another of 7.5") == "a tremor of 7.2 and another of 7.5"


def test_invalid_minutes_are_left_unchanged():
    # 8.99 is not a valid minute value (00-59): it is not a time, so it is preserved.
    assert normalize_hours("measured value 8.99") == "measured value 8.99"


def test_two_digit_hours_are_converted():
    assert normalize_hours("at 18.45") == "at 18:45"


def test_text_without_numbers_is_unchanged():
    assert normalize_hours("hello world") == "hello world"
