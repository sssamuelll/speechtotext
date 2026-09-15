from types import SimpleNamespace

import numpy as np
import pytest

from speechtotext.asr import AsrBackend, Caps, TranscriptionRequest
from speechtotext.asr.faster_whisper import FasterWhisperBackend, FasterWhisperConfig

_INFO = SimpleNamespace(language="es", language_probability=0.98)


def _backend(model, segments, info, calls, **kwargs):
    class FakeModel:
        def transcribe(self, audio, **opts):
            calls["audio"] = audio
            calls["kwargs"] = opts
            return iter(segments), info

    def factory(path, **factory_kwargs):
        calls["factory_path"] = path
        calls["factory_kwargs"] = factory_kwargs
        return FakeModel()

    ticks = iter([10.0, 10.125])
    return FasterWhisperBackend(
        model, FasterWhisperConfig(), model_factory=factory,
        clock=lambda: next(ticks), **kwargs,
    )


def _samples(seconds: float = 1.0) -> np.ndarray:
    return np.zeros(int(seconds * 16000), dtype=np.float32)


def test_backend_extrae_palabras_senales_y_opciones():
    words = [SimpleNamespace(start=0.1, end=0.4, word=" hola", probability=0.91)]
    segments = [
        SimpleNamespace(start=0.0, end=1.0, text=" hola", words=words,
                        no_speech_prob=0.10, avg_logprob=-0.20, compression_ratio=1.1),
        SimpleNamespace(start=1.0, end=3.0, text=" mundo", words=None,
                        no_speech_prob=0.30, avg_logprob=-0.40, compression_ratio=1.4),
    ]
    calls = {}
    backend = _backend("large-v3", segments, _INFO, calls)
    assert isinstance(backend, AsrBackend)
    assert backend.model_id == "large-v3"
    assert backend.model_version == "unpinned"
    samples = _samples()
    result = backend.transcribe(
        samples,
        TranscriptionRequest(language="es", hotwords=("Bézier",), context="reunión"),
    )
    assert result.text == "hola mundo"
    assert result.segments[0].text == " hola"          # crudo: el espacio inicial es del motor
    assert result.words[0].text == " hola"
    assert calls["audio"] is samples
    assert calls["kwargs"]["vad_filter"] is False
    assert result.model == "large-v3"
    assert result.model_version == "unpinned"
    assert result.words[0].confidence == pytest.approx(0.91)
    assert result.native_signals.no_speech == pytest.approx(0.30)
    assert result.native_signals.avg_logprob == pytest.approx((-0.2 + -0.8) / 3)
    assert result.native_signals.compression_ratio == pytest.approx(1.4)
    assert result.native_signals.language_probability == pytest.approx(0.98)
    assert result.latency_ms == 125
    assert calls["kwargs"]["condition_on_previous_text"] is False
    assert calls["kwargs"]["hotwords"] == "Bézier"
    assert calls["kwargs"]["initial_prompt"] == "reunión"


def test_nombre_de_modelo_deja_que_faster_whisper_lo_resuelva():
    calls = {}
    _backend("large-v3", [], _INFO, calls).warm()
    assert calls["factory_path"] == "large-v3"
    assert calls["factory_kwargs"]["local_files_only"] is False
    assert calls["factory_kwargs"]["device"] == "cpu"
    assert calls["factory_kwargs"]["compute_type"] == "int8"


def test_ruta_a_directorio_se_carga_solo_local(tmp_path):
    calls = {}
    backend = _backend(tmp_path, [], _INFO, calls)
    assert backend.model_id == tmp_path.name
    backend.warm()
    assert calls["factory_path"] == str(tmp_path)
    assert calls["factory_kwargs"]["local_files_only"] is True


def test_model_version_es_del_llamador():
    revision = "0123456789abcdef0123456789abcdef01234567"
    backend = _backend("large-v3", [], _INFO, {}, model_version=revision)
    assert backend.model_version == revision
    assert backend.transcribe(_samples(), TranscriptionRequest()).model_version == revision


def test_warm_carga_una_sola_vez():
    calls = {"n": 0}

    def factory(path, **kwargs):
        calls["n"] += 1
        return SimpleNamespace(transcribe=lambda audio, **opts: (iter([]), _INFO))

    backend = FasterWhisperBackend("small", model_factory=factory)
    backend.warm()
    backend.warm()
    backend.transcribe(_samples(), TranscriptionRequest())
    assert calls["n"] == 1
    assert backend.config == FasterWhisperConfig()


def test_backend_vacio_no_inventa_senales():
    info = SimpleNamespace(language="es", language_probability=0.8)
    result = _backend("small", [], info, {}).transcribe(_samples(), TranscriptionRequest())
    assert result.text == ""
    assert result.native_signals.no_speech is None
    assert result.native_signals.avg_logprob is None
    assert result.warnings == ("empty_transcript",)


def test_modelo_invalido_se_rechaza_al_construir():
    with pytest.raises(TypeError, match="model"):
        FasterWhisperBackend(123)
    with pytest.raises(ValueError, match="model"):
        FasterWhisperBackend("   ")
    with pytest.raises(ValueError, match="model_version"):
        FasterWhisperBackend("small", model_version="")


def test_config_fingerprint_liga_todos_los_parametros_efectivos():
    base = FasterWhisperConfig()
    assert base.fingerprint == FasterWhisperConfig().fingerprint
    assert base.fingerprint != FasterWhisperConfig(compute_type="float32").fingerprint
    assert base.fingerprint != FasterWhisperConfig(cpu_threads=2).fingerprint
    with pytest.raises(ValueError, match="cpu_threads"):
        FasterWhisperConfig(cpu_threads=True)


def test_backend_declara_caps_y_version_del_motor():
    backend = _backend("large-v3", [], _INFO, {})
    assert isinstance(backend, AsrBackend)
    assert backend.caps == Caps("honored", "honored", "honored")
    assert backend.engine_version.startswith("faster-whisper")
    assert (backend.quant, backend.device) == ("int8", "cpu")


def test_vad_y_auto_viajan_al_motor():
    calls = {}
    _backend("large-v3", [], _INFO, calls).transcribe(
        _samples(), TranscriptionRequest(language="auto", vad=True),
    )
    assert calls["kwargs"]["vad_filter"] is True
    assert calls["kwargs"]["language"] is None
