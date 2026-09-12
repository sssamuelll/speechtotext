"""core.transcribe: ruta, fabrica de backends, decodificacion y la orquestacion."""
import wave
from pathlib import Path

import numpy as np
import pytest

from speechtotext.asr.faster_whisper import FasterWhisperBackend
from speechtotext.asr.whispercpp import WhisperCppBackend
from speechtotext.audio.io import AudioDecodeError
from speechtotext.core import transcribe as core
from speechtotext.core.transcribe import EngineInfo, Route, load_audio, make_backend, resolve_route


# --- ruta ----------------------------------------------------------------------------

def test_ruta_auto_es_faster_whisper_en_cpu_int8():
    assert resolve_route() == Route("faster-whisper", "cpu", "int8", "")


def test_ruta_cuda_explicita_usa_float16():
    assert resolve_route(device="cuda") == Route("faster-whisper", "cuda", "float16", "")
    assert resolve_route(device="cuda", compute_type="int8").compute_type == "int8"


def test_ruta_whispercpp_corre_en_cuda_q5_0_y_lo_avisa():
    ruta = resolve_route(engine="whispercpp")
    assert (ruta.engine, ruta.device, ruta.compute_type) == ("whispercpp", "cuda", "q5_0")
    assert "device=cuda" in ruta.reason
    assert resolve_route(engine="whispercpp", device="cuda").reason == ""


def test_ruta_rechaza_engine_desconocido_y_compute_type_no_mapeable():
    with pytest.raises(ValueError, match="no existe"):
        resolve_route(engine="chatgpt")
    with pytest.raises(ValueError, match="paging WDDM"):
        resolve_route(engine="whispercpp", compute_type="float16")


def test_ruta_rechaza_modelo_no_pinneado_bajo_whispercpp():
    with pytest.raises(ValueError, match="no está pinneado"):
        resolve_route(engine="whispercpp", model="medium")
    assert resolve_route(engine="whispercpp", model="small").engine == "whispercpp"


# --- fabrica -------------------------------------------------------------------------

def test_make_backend_construye_el_tipo_y_la_config():
    fw = make_backend("faster-whisper", "large-v3", "cpu", "int8")
    assert isinstance(fw, FasterWhisperBackend)
    assert fw.model_id == "large-v3"
    assert (fw.config.device, fw.config.compute_type) == ("cpu", "int8")
    wc = make_backend("whispercpp", "small", "cuda", "q5_0")
    assert isinstance(wc, WhisperCppBackend)


# --- decodificacion ------------------------------------------------------------------

def _wav(path: Path, seconds: float, rate: int = 8000) -> Path:
    n = int(seconds * rate)
    t = np.arange(n) / rate
    pcm = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path


def test_load_audio_decodifica_a_16k_mono_float32(tmp_path):
    samples = load_audio(_wav(tmp_path / "tono.wav", 0.5, rate=8000))
    assert samples.dtype == np.float32 and samples.ndim == 1
    assert abs(len(samples) - 8000) <= 160  # 0,5 s a 16 kHz, con tolerancia del resampler
    assert 0.3 < float(np.abs(samples).max()) <= 1.0


def test_load_audio_rechaza_lo_que_no_es_audio(tmp_path):
    basura = tmp_path / "x.wav"
    basura.write_bytes(b"RIFF esto no es un wav")
    with pytest.raises(AudioDecodeError):
        load_audio(basura)


# --- tipos ---------------------------------------------------------------------------

def test_engine_info_omite_diarization_si_no_aplica():
    info = EngineInfo("faster-whisper", "faster-whisper 1.2", "large-v3", "int8", "cpu")
    assert info.to_dict() == {
        "name": "faster-whisper", "version": "faster-whisper 1.2", "model": "large-v3",
        "quant": "int8", "device": "cpu", "selection": "explicit",
    }
    assert EngineInfo("whispercpp", "v", "small", "q5_0", "cuda", diarization="segment").to_dict()["diarization"] == "segment"
