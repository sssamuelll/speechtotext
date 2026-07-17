import numpy as np
import pytest

from speechtotext.audio import SpeechRegion
from speechtotext.audio.quality import compute_audio_quality


def test_quality_mide_voz_ruido_y_snr():
    sample_rate = 1000
    capture = np.full(1000, 0.01, dtype=np.float32)
    capture[200:700] = 0.1
    report = compute_audio_quality(
        capture,
        capture,
        sample_rate,
        (SpeechRegion(0.2, 0.7),),
        requested_gain_db=0.0,
        applied_gain_db=0.0,
    )
    assert report.duration_ms == 1000
    assert report.effective_voice_ms == 500
    assert report.noise_floor_dbfs == pytest.approx(-40.0, abs=0.1)
    assert report.snr_db == pytest.approx(20.0, abs=0.1)
    assert report.clipping_ratio == 0.0


def test_quality_reporta_raw_y_procesado_por_separado():
    raw = np.full(100, 0.01, dtype=np.float32)
    processed = np.full(100, 0.1, dtype=np.float32)
    report = compute_audio_quality(
        raw,
        processed,
        1000,
        (SpeechRegion(0.0, 0.1),),
        requested_gain_db=20.0,
        applied_gain_db=20.0,
        dropped_frames=2,
        discontinuities=1,
    )
    assert report.input_rms_dbfs == pytest.approx(-40.0, abs=0.1)
    assert report.processed_rms_dbfs == pytest.approx(-20.0, abs=0.1)
    assert report.dropped_frames == 2
    assert report.discontinuities == 1


def test_quality_detecta_clipping_y_silencio():
    samples = np.array([1.0, -1.0, 0.0, 0.0], dtype=np.float32)
    report = compute_audio_quality(
        samples,
        samples,
        4,
        (),
        requested_gain_db=0.0,
        applied_gain_db=0.0,
    )
    assert report.effective_voice_ms == 0
    assert report.clipping_ratio == pytest.approx(0.5)
    assert report.snr_db is None
    assert "no_speech_regions" in report.warnings
