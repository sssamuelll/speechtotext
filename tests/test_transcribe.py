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

def test_resolve_route_sondea_y_delega(monkeypatch):
    visto = {}
    monkeypatch.setattr(core.probe, "machine", lambda: "MAQUINA")

    def elegir(m, model, **kw):
        visto.update(m=m, model=model, **kw)
        return Route("faster-whisper", "cpu", "int8", "")

    monkeypatch.setattr(core.probe, "choose_route", elegir)
    assert resolve_route(engine="auto", device="cpu", compute_type="int8", model="small").device == "cpu"
    assert visto == {"m": "MAQUINA", "model": "small", "engine": "auto", "device": "cpu",
                     "compute_type": "int8"}


def test_ruta_por_defecto_en_la_maquina_de_pruebas_es_cpu_int8():
    # conftest fija una máquina sin GPU: 'auto' resuelve a faster-whisper en CPU int8.
    r = resolve_route()
    assert (r.engine, r.device, r.compute_type, r.reason) == (
        "faster-whisper", "cpu", "int8", "sin GPU utilizable: CPU")
    assert r.eta_factor == round(1 / 1.27, 3) and r.estimated is True


# --- fabrica -------------------------------------------------------------------------

def test_make_backend_construye_el_tipo_y_la_config():
    fw = make_backend("faster-whisper", "large-v3", "cpu", "int8")
    assert isinstance(fw, FasterWhisperBackend)
    assert fw.model_id == "large-v3"
    assert (fw.config.device, fw.config.compute_type) == ("cpu", "int8")
    wc = make_backend("whispercpp", "small", "cuda", "q5_0")
    assert isinstance(wc, WhisperCppBackend)


def test_make_backend_reparte_hilos_al_trocear():
    import os

    solo = make_backend("faster-whisper", "large-v3", "cpu", "int8")
    assert (solo.config.cpu_threads, solo.config.num_workers) == (0, 1)
    cuatro = make_backend("faster-whisper", "large-v3", "cpu", "int8", jobs=4)
    assert cuatro.config.num_workers == 4
    assert cuatro.config.cpu_threads == max(1, (os.cpu_count() or 1) // 4)


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


# --- la orquestacion -----------------------------------------------------------------

import json
import threading

from speechtotext.asr import AsrError, Caps
from speechtotext.asr.types import (
    NativeSignals, SegmentNativeSignals, TranscriptionResult, TranscriptionSegment,
    TranscriptionWord,
)
from speechtotext.core import chunked


class FakeBackend:
    """Motor de mentira: devuelve los segmentos dados (tiempos LOCALES al trozo)."""

    def __init__(self, segments=((1.0, 2.0, " hola"),), *, language="es", probability=0.9,
                 boom=None, wrap=False, backend_id="faster-whisper",
                 caps=Caps("honrado", "honrado", "honrado")):
        self.segments, self.language, self.probability, self.boom = segments, language, probability, boom
        self.wrap = wrap
        self.backend_id, self.caps = backend_id, caps
        self.quant = "q5_0" if backend_id == "whispercpp" else "int8"
        self.device = "cuda" if backend_id == "whispercpp" else "cpu"
        self.model_id, self.model_version, self.engine_version = "large-v3", "unpinned", "fake 1.0"
        self.calls, self.warmed = [], 0

    def warm(self):
        self.warmed += 1

    def transcribe(self, samples, request):
        self.calls.append((len(samples), request))
        if self.boom is not None:
            if self.wrap:
                raise AsrError("backend_failed", True, str(self.boom))
            raise self.boom
        segs = tuple(
            TranscriptionSegment(s, e, t, (TranscriptionWord(t, s, e, None),) if request.word_timestamps else (),
                                 SegmentNativeSignals(None, None, None))
            for s, e, t in self.segments
        )
        return TranscriptionResult(
            text="".join(t for _, _, t in self.segments).strip(), language=self.language,
            words=(), segments=segs, backend=self.backend_id, model=self.model_id,
            model_version=self.model_version, latency_ms=1,
            native_signals=NativeSignals(None, None, None, self.probability), warnings=(),
        )


def _zeros(seconds):
    return np.zeros(int(seconds * 16000), dtype=np.float32)


def test_un_trozo_produce_el_transcript_completo():
    backend = FakeBackend([(1.0, 2.0, " son las 8.30"), (2.0, 20.0, " y tal")])
    t = core.transcribe(_zeros(30.0), backend=backend, language="es", chunk=False)
    assert backend.warmed == 1 and len(backend.calls) == 1
    assert t.duration == 30.0
    assert [s.text for s in t.segments] == [" son las 8:30", " y tal"]  # normalize_hours
    assert (t.language, t.language_probability) == ("es", 1.0)  # forzado -> 1.0
    assert t.speech_s == 19.0
    assert t.gaps == [[20.0, 30.0]]
    assert t.engine.to_dict() == {
        "name": "faster-whisper", "version": "fake 1.0", "model": "large-v3",
        "quant": "int8", "device": "cpu", "selection": "explicit",
    }
    assert t.request.vad is False and t.warnings == () and t.diarization is None


def test_idioma_auto_toma_el_detectado_y_su_probabilidad():
    t = core.transcribe(_zeros(5.0), backend=FakeBackend(language="en", probability=0.7), chunk=False)
    assert (t.language, t.language_probability) == ("en", 0.7)


def test_progreso_por_callback_y_decodificacion_una_vez(tmp_path, monkeypatch):
    llamadas = []
    monkeypatch.setattr(core, "load_audio", lambda p: (llamadas.append(p), _zeros(10.0))[1])
    eventos = []
    t = core.transcribe(tmp_path / "a.wav", backend=FakeBackend(), chunk=False,
                        on_progress=eventos.append)
    assert llamadas == [tmp_path / "a.wav"]
    # dos eventos decode: antes (indeterminado) y después, con done = total = duración,
    # que es lo que el CLI necesita para la ETA
    assert [e.stage for e in eventos] == ["decode", "decode", "load", "transcribe"]
    assert (eventos[0].done, eventos[0].total) == (0, None)
    assert (eventos[1].done, eventos[1].total, eventos[1].detail) == (10.0, 10.0, "a.wav")
    assert eventos[-1].done == 1 and eventos[-1].total == 1 and "(nuevo)" in eventos[-1].detail
    assert t.duration == 10.0


def test_acepta_la_ruta_como_str(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "load_audio",
                        lambda p: _zeros(3.0) if isinstance(p, Path) else pytest.fail("str crudo"))
    t = core.transcribe(str(tmp_path / "a.wav"), backend=FakeBackend(), chunk=False)
    assert t.duration == 3.0


def test_flags_imposibles_cortan_antes_de_decodificar(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "load_audio", lambda p: pytest.fail("decodificó antes de validar"))
    with pytest.raises(ValueError, match="no existe"):
        core.transcribe(tmp_path / "a.wav", engine="chatgpt")


def test_hotwords_rechazado_corta_antes_de_cargar_el_modelo():
    backend = FakeBackend(backend_id="whispercpp", caps=Caps("rechazado", "degradado", "degradado"))
    with pytest.raises(AsrError) as ei:
        core.transcribe(_zeros(5.0), backend=backend, hotwords=("Bézier",), chunk=False)
    assert ei.value.code == "unsupported_option" and ei.value.recoverable is False
    assert "--hotwords no tiene efecto" in str(ei.value) and "faster-whisper" in str(ei.value)
    assert backend.warmed == 0 and backend.calls == []


def test_caps_degradado_avisa_y_apaga_el_knob():
    backend = FakeBackend(backend_id="whispercpp", caps=Caps("rechazado", "degradado", "degradado"))
    t = core.transcribe(_zeros(5.0), backend=backend, vad=True, word_timestamps=True, chunk=False)
    assert any("no trae VAD" in w for w in t.warnings)
    assert any("atribución por segmento" in w for w in t.warnings)
    assert t.request.vad is False and t.request.word_timestamps is False
    assert backend.calls[0][1].vad is False


def test_oom_se_traduce_con_consejo_por_motor():
    with pytest.raises(AsrError) as ei:
        core.transcribe(_zeros(5.0), chunk=False,
                        backend=FakeBackend(boom=RuntimeError("mkl_malloc: failed to allocate memory")))
    assert ei.value.code == "out_of_memory" and "-m medium" in str(ei.value) and "VRAM" not in str(ei.value)
    with pytest.raises(AsrError) as ei:
        core.transcribe(_zeros(5.0), chunk=False,
                        backend=FakeBackend(backend_id="whispercpp",
                                            boom=RuntimeError("ggml_cuda: failed to allocate 1.28 GB")))
    assert "VRAM" in str(ei.value) and "--engine faster-whisper" in str(ei.value) and "medium" not in str(ei.value)


def test_runtimeerror_ajeno_se_propaga_crudo_con_un_trozo():
    with pytest.raises(RuntimeError, match="^cable suelto$"):
        core.transcribe(_zeros(5.0), chunk=False, backend=FakeBackend(boom=RuntimeError("cable suelto")))


def test_cancelacion_antes_de_empezar():
    backend = FakeBackend()
    parar = threading.Event()
    parar.set()
    with pytest.raises(AsrError) as ei:
        core.transcribe(_zeros(5.0), backend=backend, cancel=parar, chunk=False)
    assert ei.value.code == "cancelled" and backend.calls == []


def test_backend_dado_no_construye_otro(monkeypatch):
    monkeypatch.setattr(core, "make_backend", lambda *a, **k: pytest.fail("no debe construir"))
    core.transcribe(_zeros(5.0), backend=FakeBackend(), chunk=False)


def test_el_nucleo_clampa_jobs_para_whispercpp_y_los_reparte_para_faster(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    monkeypatch.setattr(core, "load_audio", lambda p: _zeros(1300.0))
    visto = []

    def fabrica(engine, model, device, compute_type, jobs=1):
        visto.append((engine, jobs))
        return FakeBackend(backend_id=engine, caps=Caps("rechazado", "degradado", "degradado")
                           if engine == "whispercpp" else Caps("honrado", "honrado", "honrado"))

    monkeypatch.setattr(core, "make_backend", fabrica)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    core.transcribe(audio, engine="whispercpp", model="small", chunk=True, jobs=4)   # 3 trozos fijos
    core.transcribe(audio, engine="faster-whisper", chunk=True, jobs=4)
    assert visto == [("whispercpp", 1), ("faster-whisper", 3)]


def test_varios_trozos_reensamblan_en_orden_con_tiempos_globales(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    audio = tmp_path / "largo.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(core, "load_audio", lambda p: _zeros(1200.0))
    monkeypatch.setattr(core, "plan_chunks", lambda path, dur: [(0.0, 600.0), (600.0, 1200.0)])
    backend = FakeBackend([(1.0, 2.0, " t")])
    eventos = []
    t = core.transcribe(audio, backend=backend, chunk=True, jobs=2, on_progress=eventos.append)
    assert [(s.start, s.end) for s in t.segments] == [(1.0, 2.0), (601.0, 602.0)]
    assert sorted(n for n, _ in backend.calls) == [600 * 16000, 600 * 16000]
    detalles = [e.detail for e in eventos if e.stage == "transcribe"]
    assert len(detalles) == 2 and all("(nuevo)" in d for d in detalles)
    assert t.language_probability is None  # varios trozos: nadie midio una sola probabilidad


def test_checkpoint_evita_el_motor_y_se_recorta_al_leer(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(core, "load_audio", lambda p: _zeros(600.0))
    monkeypatch.setattr(core, "plan_chunks", lambda path, dur: [(0.0, 600.0)])
    backend = FakeBackend()
    request, _ = core._apply_caps(backend, core._request(language="auto", vad=False, hotwords=(),
                                                         beam_size=5, word_timestamps=False))
    p = chunked.chunk_path(core._identity(audio, backend, request), 0.0, 600.0)
    p.write_text(json.dumps({"language": "es", "segments": [
        {"start": 1.0, "end": 2.0, "text": " cache"},
        {"start": 599.9, "end": 629.9, "text": " Gracias por ver el video."},  # fantasma sobre el relleno
    ]}), encoding="utf-8")
    eventos = []
    t = core.transcribe(audio, backend=backend, chunk=True, on_progress=eventos.append)
    assert backend.calls == [] and backend.warmed == 1
    assert [s.text for s in t.segments] == [" cache"]
    assert any("(cache)" in e.detail for e in eventos)
    assert t.language == "es"


def test_el_trozo_nuevo_deja_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(core, "load_audio", lambda p: _zeros(600.0))
    monkeypatch.setattr(core, "plan_chunks", lambda path, dur: [(0.0, 600.0)])
    core.transcribe(audio, backend=FakeBackend(), chunk=True)
    escritos = list((tmp_path / "chunks").glob("*.json"))
    assert len(escritos) == 1
    assert json.loads(escritos[0].read_text(encoding="utf-8"))["segments"][0]["text"] == " hola"


def test_muestras_sin_archivo_trocean_sin_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    backend = FakeBackend()
    t = core.transcribe(_zeros(1300.0), backend=backend, chunk=True)
    assert len(backend.calls) == 3  # pick_cuts sin silencios: 600 + 600 + 100
    assert not (tmp_path / "chunks").exists()
    assert len(t.segments) == 3


def test_primer_fallo_cancela_los_pendientes_y_nombra_el_trozo(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    monkeypatch.setattr(core, "load_audio", lambda p: _zeros(1200.0))
    monkeypatch.setattr(core, "plan_chunks", lambda path, dur: [(0.0, 600.0), (600.0, 1200.0)])
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    with pytest.raises(AsrError) as ei:
        core.transcribe(audio, backend=FakeBackend(boom=RuntimeError("cable")), chunk=True, jobs=1)
    assert ei.value.code == "backend_failed" and "fallo en el trozo 1/2" in str(ei.value)


def test_oom_envuelto_por_el_backend_real_tambien_se_traduce():
    boom = RuntimeError("mkl_malloc: failed to allocate memory")
    with pytest.raises(AsrError) as ei:
        core.transcribe(_zeros(5.0), chunk=False, backend=FakeBackend(boom=boom, wrap=True))
    assert ei.value.code == "out_of_memory" and "-m medium" in str(ei.value)


def test_fallo_envuelto_en_varios_trozos_nombra_el_trozo(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    monkeypatch.setattr(core, "load_audio", lambda p: _zeros(1200.0))
    monkeypatch.setattr(core, "plan_chunks", lambda path, dur: [(0.0, 600.0), (600.0, 1200.0)])
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    with pytest.raises(AsrError) as ei:
        core.transcribe(audio, backend=FakeBackend(boom=RuntimeError("cable"), wrap=True), chunk=True, jobs=1)
    assert ei.value.code == "backend_failed" and "fallo en el trozo 1/2" in str(ei.value)


def test_cancelacion_envuelta_no_se_reetiqueta():
    parar = threading.Event()
    parar.set()
    with pytest.raises(AsrError) as ei:
        core.transcribe(_zeros(5.0), backend=FakeBackend(), cancel=parar, chunk=False)
    assert ei.value.code == "cancelled"


def test_diariza_sobre_las_mismas_muestras_y_mide_antes(monkeypatch):
    from speechtotext.speakers import diarization, registry

    visto = {}

    def fake_diarize(samples, sample_rate, num_speakers=None):
        visto.update(n=len(samples), sr=sample_rate, k=num_speakers)
        return [(0.0, 30.0, "SPEAKER_00")], {"SPEAKER_00": np.array([1.0, 0.0])}

    monkeypatch.setattr(diarization, "diarize", fake_diarize)
    monkeypatch.setattr(registry, "get_embeddings", lambda model: {"Alice": np.array([1.0, 0.0])})
    backend = FakeBackend([(0.0, 30.0, " Gracias.")])
    t = core.transcribe(_zeros(30.0), backend=backend, diarize=True, speakers=1, chunk=False)
    assert visto == {"n": 30 * 16000, "sr": 16000, "k": 1}
    assert backend.calls[0][1].word_timestamps is True  # diarizar pide palabras
    assert t.segments[0].speaker == "Alice"
    assert t.segments[0].src_dur == 30.0            # el span que el ASR emitio, para is_suspect
    assert t.speech_s == 30.0 and t.gaps == []      # medido ANTES de diarizar
    assert t.engine.diarization == "word"
    assert t.diarization.speakers == 1 and t.diarization.identified == 1 and t.diarization.auto is False


def test_diarizacion_sin_extra_es_un_error_con_codigo(monkeypatch):
    from speechtotext.speakers import diarization

    def sin_pyannote(samples, sample_rate, num_speakers=None):
        raise ImportError("No module named 'pyannote'")

    monkeypatch.setattr(diarization, "diarize", sin_pyannote)
    with pytest.raises(AsrError) as ei:
        core.transcribe(_zeros(5.0), backend=FakeBackend(), diarize=True, chunk=False)
    assert ei.value.code == "diarize_unavailable" and "[diarize]" in str(ei.value)


# --- avisos del motor y cancelación a mitad del pool ------------------------------------

def test_los_avisos_del_motor_llegan_al_transcript_sin_repetirse(tmp_path, monkeypatch):
    from dataclasses import replace as dc_replace

    class Avisa(FakeBackend):
        def transcribe(self, samples, request):
            return dc_replace(super().transcribe(samples, request), warnings=("empty_transcript",))

    t = core.transcribe(_zeros(5.0), backend=Avisa(), chunk=False)
    assert t.warnings == ("faster-whisper: empty_transcript",)

    monkeypatch.setattr(core, "load_audio", lambda p: _zeros(1200.0))
    monkeypatch.setattr(core, "plan_chunks", lambda path, dur: [(0.0, 600.0), (600.0, 1200.0)])
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    t = core.transcribe(audio, backend=Avisa(), chunk=True, jobs=1)
    assert t.warnings == ("faster-whisper: empty_transcript",)   # dos trozos, un aviso


def test_los_avisos_de_caps_van_antes_que_los_del_motor():
    from dataclasses import replace as dc_replace

    class Avisa(FakeBackend):
        def transcribe(self, samples, request):
            return dc_replace(super().transcribe(samples, request), warnings=("empty_transcript",))

    backend = Avisa(backend_id="whispercpp", caps=Caps("rechazado", "degradado", "degradado"))
    t = core.transcribe(_zeros(5.0), backend=backend, vad=True, chunk=False)
    assert "no trae VAD" in t.warnings[0] and t.warnings[-1] == "whispercpp: empty_transcript"


def test_route_dada_no_vuelve_a_sondear_y_marca_selection(monkeypatch):
    monkeypatch.setattr(core.probe, "machine", lambda: pytest.fail("sondeó dos veces"))
    visto = []
    monkeypatch.setattr(core, "make_backend", lambda *a, **k: (visto.append(a), FakeBackend())[1])
    ruta = Route("faster-whisper", "cpu", "int8", "")
    t = core.transcribe(_zeros(5.0), route=ruta, chunk=False)                      # engine default: auto
    assert visto[0][:4] == ("faster-whisper", "large-v3", "cpu", "int8")
    assert t.engine.selection == "auto"
    t = core.transcribe(_zeros(5.0), route=ruta, engine="faster-whisper", chunk=False)
    assert t.engine.selection == "explicit"


def test_selection_auto_solo_cuando_el_sondeo_eligio(monkeypatch):
    monkeypatch.setattr(core, "make_backend", lambda *a, **k: FakeBackend())
    assert core.transcribe(_zeros(5.0), chunk=False).engine.selection == "auto"
    assert core.transcribe(_zeros(5.0), engine="faster-whisper", chunk=False).engine.selection == "explicit"
    assert core.transcribe(_zeros(5.0), backend=FakeBackend(), chunk=False).engine.selection == "explicit"


def test_cancelacion_a_mitad_del_pool_no_lanza_mas_trozos(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "load_audio", lambda p: _zeros(1800.0))
    monkeypatch.setattr(core, "plan_chunks",
                        lambda path, dur: [(0.0, 600.0), (600.0, 1200.0), (1200.0, 1800.0)])
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    parar = threading.Event()

    class Cancelador(FakeBackend):
        def transcribe(self, samples, request):
            parar.set()          # el primer trozo pide parar desde dentro
            return super().transcribe(samples, request)

    backend = Cancelador()
    with pytest.raises(AsrError) as ei:
        core.transcribe(audio, backend=backend, cancel=parar, chunk=True, jobs=1)
    assert ei.value.code == "cancelled"
    assert len(backend.calls) == 1     # los trozos 2 y 3 jamás llegaron al motor
