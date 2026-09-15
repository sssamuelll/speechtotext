"""DSP voice evidence: deterministic signal measurements, not a model's opinion.
The question is not "what did it say" but "was there a human voice source here."

Measured (2026-09-11) on real phone audio: Whisper's `no_speech` had the sign
backward (it cut speech and let fabrications through); voice-band energy separated
conversation from an empty room with 0% interquartile overlap.

The tests here came from an adversarial review in which 24 of 33 mutations
SURVIVED the previous suite. Each new test names the mutation it kills.
"""
import math
import subprocess
import sys
import tracemalloc

import numpy as np
import pytest

from speechtotext.audio import VoiceEvidence, compute_voice_evidence
from speechtotext.audio.evidence import _normalized_autocorrelation

SR = 16000


def _synthetic_voice(f0=140.0, seconds=1.0, sr=SR, amplitude=0.2):
    """Harmonic series at F0 with energy concentrated where vowels live.

    Five harmonics weighted toward 400-1200 Hz (F1/F2 of an open vowel),
    with a slow envelope so it is not a laboratory tone.
    """
    t = np.arange(int(seconds * sr)) / sr
    weights = {1: 0.5, 2: 0.9, 3: 1.0, 4: 0.8, 5: 0.5, 6: 0.3, 8: 0.2}
    x = sum(w * np.sin(2 * np.pi * f0 * k * t) for k, w in weights.items())
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t)   # syllabic rate
    return (amplitude * x / np.max(np.abs(x)) * envelope).astype(np.float32)


def _white_noise(seconds=1.0, sr=SR, seed=7, amplitude=0.05):
    rng = np.random.default_rng(seed)
    return (amplitude * rng.standard_normal(int(seconds * sr))).astype(np.float32)


def _tone(hz, seconds=1.0, sr=SR, amplitude=0.2):
    t = np.arange(int(seconds * sr)) / sr
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)


# ----------------------------------------------------------------------------
# what a voice leaves behind, and what it does not
# ----------------------------------------------------------------------------

def test_a_synthetic_voice_leaves_voice_evidence():
    e = compute_voice_evidence(_synthetic_voice(f0=140.0), SR)
    assert e.frames > 0
    assert e.voice_band_ratio > 0.6
    assert e.voiced_ratio > 0.8
    assert e.spectral_flatness < 0.1


def test_white_noise_does_NOT_leave_voice_evidence():
    e = compute_voice_evidence(_white_noise(), SR)
    assert e.voiced_ratio < 0.2
    # flat noise: the 300-3400 band of a 0-8000 spectrum weighs ~0.39
    assert e.voice_band_ratio < 0.5
    assert e.spectral_flatness > 0.5


def test_a_tonal_hum_below_the_band_is_NOT_voice():
    """Classic HNR rewarded a refrigerator hum because it is MORE harmonic than
    speech. Here the hum falls outside the voice band and the F0 range, so it
    does not count as voiced."""
    e = compute_voice_evidence(_tone(50.0), SR)
    assert e.voice_band_ratio < 0.05
    assert e.voiced_ratio < 0.1


def test_a_pure_tone_INSIDE_the_band_gives_a_voice_profile_and_that_is_declared():
    """Known limitation, not a bug: a doorbell, alarm, or 1 kHz feedback gives
    band=1 and voiced=1 with an invented subharmonic F0. Autocorrelation has
    peaks at every multiple of the period, and the 70-350 Hz range catches them.
    What DOES reveal it is flatness: a single spectral line. The caller combines
    all three measurements; this library does not issue verdicts."""
    e = compute_voice_evidence(_tone(1000.0), SR)
    assert e.voice_band_ratio > 0.99
    assert e.voiced_ratio > 0.99
    assert e.f0_median_hz is not None and e.f0_median_hz < 350.0
    assert e.spectral_flatness < 1e-6


# ----------------------------------------------------------------------------
# invariances: the ones the review found broken
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("gain_db", [-20.0, -40.0])
def test_the_measurements_do_NOT_depend_on_gain(gain_db):
    """The review measured spectral_flatness moving 4e9x between 0 and -40 dB
    because of an absolute EPS. Real phone audio lives at -48..-60 LUFS: right
    there. A descriptor that changes with volume does not describe the spectrum."""
    x = _synthetic_voice()
    a = compute_voice_evidence(x, SR)
    b = compute_voice_evidence(x * 10 ** (gain_db / 20), SR)
    assert b.voice_band_ratio == pytest.approx(a.voice_band_ratio, rel=1e-6)
    assert b.spectral_flatness == pytest.approx(a.spectral_flatness, rel=1e-3)
    assert b.voiced_ratio == pytest.approx(a.voiced_ratio, abs=0.02)
    assert b.f0_median_hz == pytest.approx(a.f0_median_hz, rel=1e-6)


def test_flatness_describes_the_spectrum_and_NOT_the_pause_fraction():
    """Unweighted mean across frames: flatness followed the silence fraction
    (0.00 -> 0.25 -> 0.49 with a 0/50/90% dithered pause). Frames are weighted
    by their energy, as voice_band_ratio already was."""
    voice = _synthetic_voice(seconds=1.0)
    pause = _white_noise(seconds=1.0, amplitude=1e-5)       # dither, not exact zeros
    alone = compute_voice_evidence(voice, SR).spectral_flatness
    with_pause = compute_voice_evidence(np.concatenate([voice, pause]), SR).spectral_flatness
    assert with_pause == pytest.approx(alone, abs=0.01)


# ----------------------------------------------------------------------------
# F0: the arithmetic, with no slack for an off-by-one to hide in
# ----------------------------------------------------------------------------

def test_praat_normalization_is_one_at_the_period_even_when_the_lag_is_long():
    """Dividing by r[0] penalizes long lags: at 70 Hz (lag 229 of 1024) it gives
    0.78 and biases F0 upward. Using the energy of each half gives ~1. This kills
    the "r[0] instead of Praat" mutation that the previous suite let through."""
    frame = round(SR * 0.064)
    x = _tone(70.0, seconds=frame / SR)[None, :].astype(np.float64)
    r = _normalized_autocorrelation(x)
    lag = round(SR / 70.0)
    assert r[0, lag] > 0.99
    assert r[0, lag] > r[0, 0] * 0.99   # and it does NOT fall with the lag


@pytest.mark.parametrize("f0", [90.0, 140.0, 220.0, 300.0])
def test_f0_is_EXACTLY_the_nearest_integer_lag(f0):
    """Without parabolic interpolation, the correct F0 is sr/round(sr/f0). A 3%
    relative tolerance let an extra lag (+1) through: at 300 Hz that is 290.9 Hz
    and nobody noticed. It also kills the octave drop (without octave cost,
    300 -> 100)."""
    e = compute_voice_evidence(_synthetic_voice(f0=f0), SR)
    assert e.f0_median_hz == pytest.approx(SR / round(SR / f0), rel=1e-9)


def test_voiced_ratio_is_a_FRACTION_not_a_maximum():
    """Every signal in the previous suite gave exactly 0.0 or 1.0, so
    `mean -> max` survived. Half voice, half silence: ~0.5."""
    x = np.concatenate([_synthetic_voice(seconds=0.5), np.zeros(SR // 2, dtype=np.float32)])
    e = compute_voice_evidence(x, SR)
    assert 0.35 < e.voiced_ratio < 0.65


# ----------------------------------------------------------------------------
# the band edges are the descriptor's definition
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("hz, inside", [(250.0, False), (350.0, True), (3300.0, True), (3600.0, False)])
def test_the_voice_band_edges_are_300_and_3400(hz, inside):
    """No test pinned 300/3400; moving the band to 200-4000 passed the suite.
    A tone 50 Hz from either edge falls entirely on one side."""
    e = compute_voice_evidence(_tone(hz), SR)
    assert (e.voice_band_ratio > 0.98) is inside
    assert (e.voice_band_ratio < 0.02) is (not inside)


# ----------------------------------------------------------------------------
# "could not measure" is not "there is no voice"
# ----------------------------------------------------------------------------

def test_silence_does_not_invent_evidence():
    e = compute_voice_evidence(np.zeros(SR, dtype=np.float32), SR)
    assert e.frames > 0
    assert e.voice_band_ratio is None
    assert e.spectral_flatness is None
    assert e.voiced_ratio is None
    assert e.f0_median_hz is None


def test_too_low_to_measure_is_None_for_ALL_measurements():
    """Previously voice_band_ratio was None only for exact digital zeros
    (-190 dBFS), while voiced_ratio was 0.0 both without voice and with voice at
    -100 dB. A 0.0 that means two things is a hidden verdict. There is ONE rule:
    if no frame exceeds SILENCE_RMS, nothing was measured."""
    e = compute_voice_evidence(_synthetic_voice(amplitude=1e-5), SR)   # -100 dBFS
    assert e.frames > 0
    assert e == VoiceEvidence(None, None, None, None, e.frames)


def test_audible_noise_without_voice_is_measured_ZERO_not_None():
    e = compute_voice_evidence(_white_noise(), SR)
    assert e.voiced_ratio == 0.0
    assert e.voice_band_ratio is not None


def test_audio_shorter_than_one_frame_has_no_evidence():
    e = compute_voice_evidence(np.zeros(100, dtype=np.float32), SR)
    assert e == VoiceEvidence(
        voice_band_ratio=None, voiced_ratio=None, f0_median_hz=None,
        spectral_flatness=None, frames=0,
    )


@pytest.mark.parametrize(
    "samples, sr",
    [
        (np.zeros((2, SR), dtype=np.float32), SR),          # stereo
        (np.array([0.0, np.nan], dtype=np.float32), SR),     # NaN
        (np.zeros(SR, dtype=np.float32), 0),                 # invalid sr
        (np.zeros(SR, dtype=np.float32), 300),               # sr below the F0 range
    ],
)
def test_invalid_input_is_rejected(samples, sr):
    with pytest.raises(ValueError):
        compute_voice_evidence(samples, sr)


@pytest.mark.parametrize(
    "fields",
    [
        dict(voice_band_ratio=-0.1), dict(voice_band_ratio=1.5), dict(voiced_ratio=2.0),
        dict(f0_median_hz=-1.0), dict(f0_median_hz=float("nan")),
        dict(spectral_flatness=float("inf")), dict(frames=-1),
    ],
)
def test_the_dataclass_validates_like_its_neighbors(fields):
    base = dict(voice_band_ratio=0.5, voiced_ratio=0.5, f0_median_hz=120.0,
                spectral_flatness=0.1, frames=10)
    with pytest.raises(ValueError):
        VoiceEvidence(**{**base, **fields})


# ----------------------------------------------------------------------------
# actual determinism, and bounded memory
# ----------------------------------------------------------------------------

# Pinned values for `_synthetic_voice()` at 16 kHz (numpy pocketfft, 2026-09-11,
# on Windows x86-64). If they change without a change to the measurement definition,
# the backend or arithmetic changed — and that is what this test exists to flag.
#
# But they are NOT compared bit for bit. The first CI run outside Windows made that
# clear: pocketfft dispatches SIMD by machine, and the last bit does not carry over.
# Measured on 2026-09-14 with these same values: Linux x86-64 gives `band` 1 ULP from
# the Windows value, and macOS arm64 gives `flat` 3 ULP away. This is not a regression;
# exact cross-platform float equality was never a property this code could promise.
# The tolerance below admits that noise while still flagging any meaningful change:
# a different backend or formula moves digits, not bits.
#
# What is compared bit for bit is the other process against this one: same machine,
# same build, where exact equality is required. That is the half of the test that
# truly checks determinism and catches a mutant that depends on time.time_ns().
GOLDEN = {
    "band": 0.6558522702500179,
    "voiced": 1.0,
    "f0": 140.35087719298247,   # 16000 / 114
    "flat": 2.1362539235000618e-09,
}
TOLERANCE = 1e-12   # ~6000 times the measured platform noise

_OTHER_PROCESS_CODE = (
    "import sys; sys.path.insert(0, 'tests');"
    "from test_audio_evidence import _synthetic_voice, SR;"
    "from speechtotext.audio import compute_voice_evidence;"
    "e = compute_voice_evidence(_synthetic_voice(), SR);"
    "print(e.voice_band_ratio.hex(), e.voiced_ratio.hex(), "
    "e.f0_median_hz.hex(), e.spectral_flatness.hex(), e.frames)"
)


def test_evidence_is_deterministic_ACROSS_PROCESSES_and_pinned():
    """Comparing two calls in the same process is a tautology: the review built
    a mutant that depended on time.time_ns() and passed it. Here another process
    computes the same result, and the bytes must match a pinned value — which also
    detects a change to the FFT backend."""
    e = compute_voice_evidence(_synthetic_voice(), SR)
    other = subprocess.run([sys.executable, "-c", _OTHER_PROCESS_CODE],
                           capture_output=True, text=True, check=True)
    assert other.stdout.split() == [
        e.voice_band_ratio.hex(), e.voiced_ratio.hex(), e.f0_median_hz.hex(),
        e.spectral_flatness.hex(), str(e.frames),
    ]
    measured = {"band": e.voice_band_ratio, "voiced": e.voiced_ratio,
                "f0": e.f0_median_hz, "flat": e.spectral_flatness}
    far = {
        key: (value, GOLDEN[key])
        for key, value in measured.items()
        if not math.isclose(value, GOLDEN[key], rel_tol=TOLERANCE)
    }
    assert not far, f"measurement moved farther than the platform noise: {far}"


def _peak_mb(x):
    tracemalloc.start()
    try:
        compute_voice_evidence(x, SR)
        return tracemalloc.get_traced_memory()[1] / 1e6
    finally:
        tracemalloc.stop()


def test_memory_is_BOUNDED_and_does_not_grow_with_duration():
    """The review measured 88-90x the buffer: 330 MB for 60 s, 19 GB for one hour,
    in a library that claims to process hours-long recordings. Frames are a view
    (without a copy) and are processed in batches: the peak does not depend on duration."""
    short = _peak_mb(_synthetic_voice(seconds=20.0))
    long = _peak_mb(_synthetic_voice(seconds=120.0))
    assert short < 80.0, f"peak {short:.0f} MB for 20 s"
    assert long < short * 1.5, f"20 s: {short:.0f} MB, 120 s: {long:.0f} MB"
