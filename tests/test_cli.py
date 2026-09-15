import logging
import re
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from speechtotext.cli.app import app
from speechtotext.speakers import registry

runner = CliRunner()


def _seg(start, end, text="hola que tal"):
    return SimpleNamespace(start=start, end=end, text=text)


def _info(duration, language="es", language_probability=1.0):
    return SimpleNamespace(
        duration=duration, language=language, language_probability=language_probability
    )


def _result(segments, info):
    """TranscriptionResult a partir de los segmentos de mentira (SimpleNamespace) de los
    tests: dialecto crudo de faster-whisper (no_speech_prob) y palabras opcionales."""
    from speechtotext.asr.types import (
        NativeSignals, SegmentNativeSignals, TranscriptionResult, TranscriptionSegment,
        TranscriptionWord,
    )

    segs = []
    for s in segments:
        words = tuple(TranscriptionWord(w.word, w.start, w.end, None)
                      for w in (getattr(s, "words", None) or ()))
        segs.append(TranscriptionSegment(s.start, s.end, s.text, words, SegmentNativeSignals(
            getattr(s, "no_speech_prob", None), getattr(s, "avg_logprob", None),
            getattr(s, "compression_ratio", None))))
    return TranscriptionResult(
        text="".join(s.text for s in segments).strip(), language=info.language, words=(),
        segments=tuple(segs), backend="fake", model="fake", model_version="1", latency_ms=1,
        native_signals=NativeSignals(None, None, None, info.language_probability), warnings=(),
    )


def _fake_transcribe(monkeypatch, tmp_path, segments, info, boom=None, calls=None):
    """Corta el camino justo antes del motor: el audio nunca se abre y el backend es de
    mentira. calls: lista opcional donde se registra cada (samples, request)."""
    from speechtotext.asr import Caps
    from speechtotext.core import transcribe as core_transcribe

    audio = tmp_path / "charla.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(core_transcribe, "load_audio",
                        lambda p: np.zeros(int(info.duration * 16000), dtype=np.float32))
    monkeypatch.setattr(core_transcribe, "should_chunk", lambda d, c: False)

    class FakeBackend:
        def __init__(self, engine, model, device, compute_type, jobs=1):
            self.backend_id, self.model_id, self.device, self.quant = engine, model, device, compute_type
            self.model_version = "1"
            self.engine_version = "whisper.cpp v1.9.1" if engine == "whispercpp" else "faster-whisper 1.2.0"
            self.caps = (Caps("rechazado", "degradado", "degradado") if engine == "whispercpp"
                         else Caps("honrado", "honrado", "honrado"))

        def warm(self):
            pass

        def transcribe(self, samples, request):
            if calls is not None:
                calls.append((samples, request))
            if boom is not None:
                raise boom
            return _result(segments, info)

    monkeypatch.setattr(core_transcribe, "make_backend", FakeBackend)
    return audio


def _invoke(audio, tmp_path, *extra, catch=True):
    return runner.invoke(
        app,
        ["transcribe", str(audio), "-f", "txt", "-o", str(tmp_path / "out")] + list(extra),
        catch_exceptions=catch,
    )


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plana(salida: str) -> str:
    """Normaliza lo que rich renderiza para poder asertar contra ello.

    Dos cosas fuera de nuestro control: envuelve a 80 columnas bajo CliRunner, y cuando
    hay color colorea el primer guion de un flag aparte del resto —
    `'\\x1b[1m-\\x1b[0m\\x1b[1m-threshold\\x1b[0m'` — con lo que `--threshold` deja de
    existir como substring. Local corre sin color y CI con color, así que asertar sobre
    el texto crudo pasa aquí y falla allá. Lo renderizado no es un contrato de nadie:
    se limpia antes de mirarlo.
    """
    return " ".join(_ANSI.sub("", salida).split())


def test_voices_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    result = runner.invoke(app, ["voices"])
    assert result.exit_code == 0
    assert "Sin voces" in result.stdout


def test_forget_missing_returns_error(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    result = runner.invoke(app, ["forget", "Nadie"])
    assert result.exit_code == 1


def test_voices_lists_enrolled(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    registry.enroll("Alice", np.array([1.0, 2.0], dtype=np.float32), seconds=10.0, model="m")
    result = runner.invoke(app, ["voices"])
    assert "Alice" in result.stdout


def test_transcribe_still_registered():
    # el comando transcribe sigue existiendo como subcomando nombrado
    result = runner.invoke(app, ["transcribe", "--help"])
    assert result.exit_code == 0
    assert "diarize" in result.stdout


# --- 1.1 · cobertura en la línea de resumen -----------------------------------------


def test_resumen_lista_los_huecos(tmp_path, monkeypatch):
    # La corrida del hallazgo: 2206 s de audio, texto en 0-300 y 600-850.
    # el consejo "prueba --no-vad" solo aplica con VAD puesto; ya no es el default
    segs = [_seg(0.0, 300.0), _seg(600.0, 850.0)]
    audio = _fake_transcribe(monkeypatch, tmp_path, segs, _info(2206.0))
    result = _invoke(audio, tmp_path, "--vad")
    assert result.exit_code == 0
    assert "25%" in result.stdout
    assert "2 huecos sin texto: 05:00-10:00 (300 s), 14:10-36:46 (1356 s)" in result.stdout
    assert "prueba --no-vad" in result.stdout


def test_cobertura_alta_con_hueco_real_lo_lista(tmp_path, monkeypatch):
    # 90%: el número se ve sano y hay 100 s sin texto. Es LA banda que el plan señala
    # (§1.3: "70-90%, exactamente donde se pierde contenido caro") y el caso que el
    # umbral borrado nunca disparaba. Bajo el código viejo esta corrida imprimía "90%"
    # y nada más; si esta línea desaparece, la Fase 1 perdió su razón de existir.
    segs = [_seg(0.0, 900.0)]
    audio = _fake_transcribe(monkeypatch, tmp_path, segs, _info(1000.0))
    result = _invoke(audio, tmp_path)
    assert result.exit_code == 0
    assert "90%" in result.stdout
    assert "1 hueco sin texto: 15:00-16:40 (100 s)" in result.stdout


def test_resumen_sin_huecos_lo_dice_explicitamente(tmp_path, monkeypatch):
    # Un solo segmento contiguo: la ausencia de huecos se afirma, no se calla.
    segs = [_seg(0.0, 900.0)]
    audio = _fake_transcribe(monkeypatch, tmp_path, segs, _info(900.0))
    result = _invoke(audio, tmp_path)
    assert result.exit_code == 0
    assert "100%" in result.stdout
    assert "sin huecos de 5 s o más" in result.stdout
    assert "--no-vad" not in result.stdout


def test_no_sugiere_no_vad_a_quien_ya_lo_apago(tmp_path, monkeypatch):
    # Las corridas 2, 3 y 5 del caso real ya iban con --no-vad: repetir el consejo ahí
    # es ruido garantizado. La línea de huecos se queda; la sugerencia no.
    segs = [_seg(0.0, 300.0)]
    audio = _fake_transcribe(monkeypatch, tmp_path, segs, _info(2206.0))
    result = _invoke(audio, tmp_path, "--no-vad")
    assert result.exit_code == 0
    assert "1 hueco sin texto: 05:00-36:46 (1906 s)" in result.stdout
    assert "--no-vad" not in result.stdout


# --- 5.1.3 · una sola cantidad, calculada antes de diarizar --------------------------


def _diarize_recomprimiendo(monkeypatch, salida):
    """Lo que hace la diarización real: cada span se recomprime a la extensión de sus
    palabras. Medir después publica otro número con el mismo nombre — C-13."""
    from speechtotext.core import transcribe as core_transcribe
    from speechtotext.core.segments import LabeledSegment
    from speechtotext.core.transcribe import DiarizationReport

    labeled = [LabeledSegment(s.start, s.end, s.text) for s in salida]
    monkeypatch.setattr(core_transcribe, "_diarize",
                        lambda samples, segs, *a: (labeled, DiarizationReport(1, 0, 0, 0, None, True)))


def test_json_mide_sobre_lo_que_el_asr_emitio_no_sobre_lo_diarizado(tmp_path, monkeypatch):
    # El cableado que arregla C-13. Sin este test la regresión pasa en verde: basta con
    # mover `cov = sum(...)` debajo de _diarize y los otros 604 siguen pasando.
    import json

    _diarize_recomprimiendo(monkeypatch, [_seg(0.0, 1.0), _seg(600.0, 601.0)])
    segs = [_seg(0.0, 300.0), _seg(600.0, 850.0)]
    audio = _fake_transcribe(monkeypatch, tmp_path, segs, _info(2206.0))
    result = _invoke(audio, tmp_path, "-f", "json", "--diarize")
    assert result.exit_code == 0
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert payload["speech_s"] == 550.0  # 300 + 250; post-diarización daría 2.0
    assert payload["gaps"] == [[300.0, 600.0], [850.0, 2206.0]]
    # Y es exactamente lo que la consola dijo: una cantidad, dos canales.
    assert "2 huecos sin texto: 05:00-10:00 (300 s), 14:10-36:46 (1356 s)" in result.stdout


# --- 5.2.1 · los contratos de rango los valida typer, no la prosa del --help -----------


def test_threshold_fuera_de_rango_sale_con_exit_2(tmp_path, monkeypatch):
    # Con 2.0 la identificación de voz quedaba desactivada en silencio (assign_names
    # rompe el bucle en el primer candidato y devuelve {}); con -1 nombraba todo.
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "--threshold", "2.0")
    assert result.exit_code == 2
    assert "--threshold" in _plana(result.stderr)


def test_speakers_cero_sale_con_exit_2(tmp_path, monkeypatch):
    # 0 es falsy y speakers/diarization.py lo reinterpretaba como "auto" sin aviso:
    # exactamente la sustitución callada que el contrato de capacidades prohíbe.
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "--speakers", "0")
    assert result.exit_code == 2
    assert "--speakers" in _plana(result.stderr)


def test_beam_size_cero_sale_con_exit_2(tmp_path, monkeypatch):
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "--beam-size", "0")
    assert result.exit_code == 2
    assert "--beam-size" in _plana(result.stderr)


def test_find_threshold_fuera_de_rango_sale_con_exit_2(tmp_path, monkeypatch):
    # find reexpone los mismos flags: arreglar solo transcribe dejaba una puerta abierta.
    from speechtotext.core import finder

    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    audio = tmp_path / "programa.wav"
    audio.write_bytes(b"x" * 100)
    # Si la validación falta, el callback corre: que termine rápido y con exit 0.
    monkeypatch.setattr(finder, "load_or_build_index", lambda *a, **k: ([], True))
    result = runner.invoke(app, ["find", str(audio), "sismica", "--threshold", "2.0"])
    assert result.exit_code == 2
    assert "--threshold" in _plana(result.stderr)


def test_find_speakers_cero_sale_con_exit_2(tmp_path, monkeypatch):
    from speechtotext.core import finder

    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    audio = tmp_path / "programa.wav"
    audio.write_bytes(b"x" * 100)
    monkeypatch.setattr(finder, "load_or_build_index", lambda *a, **k: ([], True))
    result = runner.invoke(app, ["find", str(audio), "sismica", "--speakers", "0"])
    assert result.exit_code == 2
    assert "--speakers" in _plana(result.stderr)


# --- 5.2.3 · la marca [?] sobrevive a --diarize ----------------------------------------


def test_diarize_marca_sospechoso_igual_que_sin_diarizar(tmp_path, monkeypatch):
    # El caso canónico del plan: un segmento ASR de 30 s con una sola palabra de 1 s.
    # Sin --diarize el gate mide los 30 s del span; con --diarize el span se recomprime
    # a 1 s y solo src_dur salva la marca. Ruta real de punta a punta: diarize y el
    # registro de voces stubbeados, assign_segments/apply_names/post-proceso reales.
    from speechtotext.speakers import diarization

    palabra = SimpleNamespace(start=0.4, end=1.4, word=" Gracias.")
    seg = SimpleNamespace(start=0.0, end=30.0, text=" Gracias.", words=[palabra])
    audio = _fake_transcribe(monkeypatch, tmp_path, [seg], _info(30.0))
    monkeypatch.setattr(
        diarization, "diarize",
        lambda samples, sample_rate, num_speakers=None: (
            [(0.0, 30.0, "SPEAKER_00")], {"SPEAKER_00": np.array([1.0])}
        ),
    )
    monkeypatch.setattr(registry, "get_embeddings", lambda model: {})

    sin = _invoke(audio, tmp_path)
    assert sin.exit_code == 0
    assert "[?] Gracias." in (tmp_path / "out.txt").read_text(encoding="utf-8")

    con = _invoke(audio, tmp_path, "--diarize")
    assert con.exit_code == 0
    texto = (tmp_path / "out.txt").read_text(encoding="utf-8")
    assert "[?] Gracias." in texto  # con el span recomprimido a 1 s, solo src_dur la marca
    assert "Hablante 1" in texto


# --- 5.2.4 · reporte de calidad de la diarización --------------------------------------


def _fake_diarization(monkeypatch, tmp_path, turns, clusters, enrolled):
    """Stub de la frontera con modelos: diarize y el registro de voces. El resto de la
    ruta (assign_segments, assign_names, apply_names, el reporte) corre de verdad."""
    from speechtotext.speakers import diarization

    monkeypatch.setattr(
        diarization, "diarize", lambda samples, sample_rate, num_speakers=None: (turns, clusters)
    )
    monkeypatch.setattr(
        registry, "get_embeddings",
        # El espacio vectorial importa: si el CLI pide el de otro modelo, no hay voces.
        lambda model: enrolled if model == diarization.EMBEDDING_MODEL else {},
    )


def test_reporte_diarizacion_sin_voces_registradas(tmp_path, monkeypatch):
    # Dos clusters y ninguna voz registrada: sale el conteo y no sale la línea de score.
    clusters = {"SPEAKER_00": np.array([1.0, 0.0]), "SPEAKER_01": np.array([0.0, 1.0])}
    turns = [(0.0, 5.0, "SPEAKER_00"), (5.0, 9.0, "SPEAKER_01")]
    _fake_diarization(monkeypatch, tmp_path, turns, clusters, {})
    segs = [_seg(0.0, 5.0), _seg(5.0, 9.0)]
    audio = _fake_transcribe(monkeypatch, tmp_path, segs, _info(9.0))
    result = _invoke(audio, tmp_path, "--diarize")
    assert result.exit_code == 0
    salida = _plana(result.stdout)
    assert "2 hablantes · 0% sin atribuir" in salida
    assert "voces identificadas" not in salida  # sin registro, la cláusula no aplica
    assert "mejor score" not in salida


def test_reporte_diarizacion_mejor_score_bajo_umbral(tmp_path, monkeypatch):
    # Una voz registrada que no alcanza el umbral: sale el mejor score y el umbral, o
    # la identificación falla a oscuras (el caso real del plan: 0.38 < 0.50 sin aviso).
    clusters = {"SPEAKER_00": np.array([1.0, 3.0]), "SPEAKER_01": np.array([0.0, 1.0])}
    turns = [(0.0, 5.0, "SPEAKER_00"), (5.0, 9.0, "SPEAKER_01")]
    enrolled = {"Alice": np.array([1.0, 0.0])}  # coseno con SPEAKER_00: 1/sqrt(10) = 0.32
    _fake_diarization(monkeypatch, tmp_path, turns, clusters, enrolled)
    segs = [_seg(0.0, 5.0), _seg(5.0, 9.0)]
    audio = _fake_transcribe(monkeypatch, tmp_path, segs, _info(9.0))
    result = _invoke(audio, tmp_path, "--diarize")
    assert result.exit_code == 0
    salida = _plana(result.stdout)
    assert "0 de 1 voces identificadas" in salida
    assert "mejor score 0.32 < 0.50" in salida


def test_reporte_diarizacion_sugiere_speakers_cuando_el_automatico_se_dispara(
    tmp_path, monkeypatch
):
    clusters = {f"SPEAKER_{i:02d}": np.array([1.0, 0.0]) for i in range(6)}
    turns = [(float(i), float(i + 1), f"SPEAKER_{i:02d}") for i in range(6)]
    _fake_diarization(monkeypatch, tmp_path, turns, clusters, {})
    segs = [_seg(float(i), float(i + 1)) for i in range(6)]
    audio = _fake_transcribe(monkeypatch, tmp_path, segs, _info(6.0))

    result = _invoke(audio, tmp_path, "--diarize")
    assert result.exit_code == 0
    assert "6 hablantes" in _plana(result.stdout)
    assert "--speakers N" in result.stdout

    # Con el número fijado por el usuario la sugerencia no aplica.
    result = _invoke(audio, tmp_path, "--diarize", "--speakers", "6")
    assert result.exit_code == 0
    assert "--speakers N" not in result.stdout


# --- 1.6 · idioma medido vs forzado -------------------------------------------------


def test_idioma_forzado_no_reporta_probabilidad(tmp_path, monkeypatch):
    # -l explícito: ahí no se detectó nada, se obedeció al usuario (el default es auto).
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "-l", "es")
    assert "(forzado)" in result.stdout
    assert "prob=" not in result.stdout
    assert "Idioma detectado" not in result.stdout


def test_idioma_auto_reporta_probabilidad(tmp_path, monkeypatch):
    info = _info(10.0, language="en", language_probability=0.87)
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], info)
    result = _invoke(audio, tmp_path, "-l", "auto")
    assert "Idioma detectado" in result.stdout
    assert "prob=0.87" in result.stdout


def test_idioma_auto_omite_probabilidad_desconocida(tmp_path, monkeypatch):
    # La ruta troceada bajo --language auto tiró la probabilidad real: se calla, no inventa.
    info = _info(10.0, language="en", language_probability=None)
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], info)
    result = _invoke(audio, tmp_path, "-l", "auto")
    assert "Idioma detectado" in result.stdout
    assert "prob=" not in result.stdout


# --- 1.8 · logger de faster-whisper -------------------------------------------------


def test_logger_de_faster_whisper_en_info():
    # Sin esto, "VAD filter removed X of audio" nunca sale y Q1 no se puede cerrar.
    assert logging.getLogger("faster_whisper").level == logging.INFO


# --- 1.9 · guarda de memoria --------------------------------------------------------


def test_oom_sale_con_mensaje_accionable(tmp_path, monkeypatch):
    boom = RuntimeError("mkl_malloc: failed to allocate memory")
    audio = _fake_transcribe(monkeypatch, tmp_path, [], _info(2206.0), boom=boom)
    result = _invoke(audio, tmp_path)
    assert result.exit_code == 1
    assert "memoria" in result.stdout
    assert "medium" in result.stdout


def test_runtimeerror_ajeno_se_propaga(tmp_path, monkeypatch):
    # Tragar cualquier RuntimeError convertiría un bug en un mensaje de RAM equivocado.
    boom = RuntimeError("el modelo no existe")
    audio = _fake_transcribe(monkeypatch, tmp_path, [], _info(2206.0), boom=boom)
    with pytest.raises(RuntimeError, match="el modelo no existe"):
        _invoke(audio, tmp_path, catch=False)


# --- multimotor · CAPS, clamp y motor declarado en la salida ------------------------


def test_hotwords_con_whispercpp_rechaza_sin_construir(tmp_path, monkeypatch):
    # --prompt es inerte bajo -mc 0 (medido 2026-07-27): rechazo, no degradación.
    calls = []
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0), calls=calls)
    result = _invoke(audio, tmp_path, "--engine", "whispercpp", "--hotwords", "Bézier")
    assert result.exit_code == 2
    assert "--hotwords no tiene efecto" in result.stdout
    assert "faster-whisper" in result.stdout
    assert calls == []  # jamás llegó a el backend: ni modelo ni caché


def test_engine_invalido_falla(tmp_path, monkeypatch):
    calls = []
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0), calls=calls)
    result = _invoke(audio, tmp_path, "--engine", "chatgpt")
    assert result.exit_code == 2
    assert "no existe" in result.stderr  # BadParameter nuestro, no "no such option"
    assert calls == []


def test_compute_type_no_mapeable_con_whispercpp(tmp_path, monkeypatch):
    # fp16 en la 980 pagina bajo WDDM (0.53x tiempo real): rechazo con la medición.
    calls = []
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0), calls=calls)
    result = _invoke(audio, tmp_path, "--engine", "whispercpp", "--compute-type", "float16")
    assert result.exit_code == 2
    assert "paging WDDM" in result.stderr  # el rechazo cita la medición, no un genérico
    assert calls == []


def test_aviso_vad_con_whispercpp(tmp_path, monkeypatch):
    # --vad explícito: el default ya es False, así que sin pedirlo no hay nada que degradar.
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "--engine", "whispercpp", "--vad")
    assert result.exit_code == 0
    assert "no trae VAD" in result.stdout


def test_whispercpp_no_sugiere_no_vad(tmp_path, monkeypatch):
    # Con huecos en la línea de tiempo, el consejo "prueba --no-vad" es absurdo bajo
    # whispercpp: el motor no tiene VAD que apagar. La línea de huecos se queda; la
    # sugerencia no.
    segs = [_seg(0.0, 300.0)]
    audio = _fake_transcribe(monkeypatch, tmp_path, segs, _info(2206.0))
    result = _invoke(audio, tmp_path, "--engine", "whispercpp")
    assert result.exit_code == 0
    assert "1 hueco sin texto: 05:00-36:46 (1906 s)" in result.stdout
    assert "prueba --no-vad" not in result.stdout


def test_aviso_diarize_con_whispercpp(tmp_path, monkeypatch):
    from speechtotext.core import transcribe as core_transcribe
    from speechtotext.core.transcribe import DiarizationReport

    # La diarización real necesita pyannote; aquí solo importa el aviso previo.
    monkeypatch.setattr(core_transcribe, "_diarize",
                        lambda samples, segs, *a: ([], DiarizationReport(0, 0, 0, 0, None, True)))
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "--engine", "whispercpp", "--diarize")
    assert result.exit_code == 0
    assert "atribución por segmento" in result.stdout


def test_clamp_jobs_con_whispercpp_cuda(tmp_path, monkeypatch):
    # 4 subprocesos × 1.28 GB contra 4096 MiB: WDDM pagina 25x en silencio (medido).
    calls = []
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0), calls=calls)
    result = _invoke(audio, tmp_path, "--engine", "whispercpp", "-d", "cuda", "-j", "4")
    assert result.exit_code == 0
    assert "jobs=1" in result.stdout
    assert len(calls) == 1


def test_resumen_incluye_motor_default(tmp_path, monkeypatch):
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path)
    assert result.exit_code == 0
    assert "faster-whisper" in result.stdout  # el resumen declara el motor, siempre
    assert "Motor whisper.cpp" not in result.stdout  # header extra solo si no es default


def test_resumen_y_header_con_whispercpp(tmp_path, monkeypatch):
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "--engine", "whispercpp")
    assert result.exit_code == 0
    assert "whisper.cpp v1.9.1" in result.stdout  # header con versión pinneada
    assert "q5_0" in result.stdout  # la cuantización EFECTIVA, no 'auto'
    assert "whispercpp" in result.stdout  # motor resuelto en el resumen


def test_oom_whispercpp_aconseja_vram(tmp_path, monkeypatch):
    # El stderr de CUDA OOM también trae 'alloc': el consejo de RAM sería un falso amigo.
    boom = RuntimeError("ggml_cuda: failed to allocate 1.28 GB")
    audio = _fake_transcribe(monkeypatch, tmp_path, [], _info(2206.0), boom=boom)
    result = _invoke(audio, tmp_path, "--engine", "whispercpp", "-d", "cuda")
    assert result.exit_code == 1
    assert "VRAM" in result.stdout
    assert "--engine faster-whisper" in result.stdout
    assert "medium" not in result.stdout  # el consejo de RAM de CPU no aparece


def test_oom_faster_whisper_aconseja_ram(tmp_path, monkeypatch):
    # La contraparte: bajo el default el consejo sigue siendo el de RAM de CPU.
    boom = RuntimeError("mkl_malloc: failed to allocate memory")
    audio = _fake_transcribe(monkeypatch, tmp_path, [], _info(2206.0), boom=boom)
    result = _invoke(audio, tmp_path)
    assert result.exit_code == 1
    assert "medium" in result.stdout
    assert "VRAM" not in result.stdout


# --- identidad del motor en el JSON (G2) y device efectivo -----------------------------


def test_json_declara_motor_faster_whisper(tmp_path, monkeypatch):
    # El bloque engine tiene que llegar al ARCHIVO real, no solo existir como kwarg en
    # formats: este test cablea CLI -> write_json de punta a punta.
    import json

    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "-f", "json")
    assert result.exit_code == 0
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    eng = payload["engine"]
    assert eng["name"] == "faster-whisper"
    assert eng["version"].startswith("faster-whisper ")
    assert eng["quant"] == "int8"  # la efectiva de auto+cpu
    assert eng["device"] == "cpu"
    assert eng["selection"] == "auto"  # --engine auto (default): lo eligió el sondeo
    assert "diarization" not in eng  # sin --diarize no se afirma nada


def test_json_declara_motor_whispercpp_y_device_cuda(tmp_path, monkeypatch):
    # El binario pinneado es build CUDA y corre en GPU SIEMPRE (medido en el smoke):
    # etiquetar cpu seria mentir. El device efectivo es cuda aunque nadie pase -d.
    import json

    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "-f", "json", "--engine", "whispercpp")
    assert result.exit_code == 0
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    eng = payload["engine"]
    assert eng == {
        "name": "whispercpp", "version": "whisper.cpp v1.9.1", "model": "large-v3",
        "quant": "q5_0", "device": "cuda", "selection": "explicit",
    }
    assert "cuda" in result.stdout  # el header tampoco dice cpu


def test_whispercpp_rechaza_modelo_no_pinneado(tmp_path, monkeypatch):
    # Sin pre-validacion, ensure_model revienta con RuntimeError crudo a mitad de
    # corrida; el error debe llegar antes de construir nada y listar lo disponible.
    calls = []
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0), calls=calls)
    result = _invoke(audio, tmp_path, "--engine", "whispercpp", "-m", "medium")
    assert result.exit_code == 2
    assert "no está pinneado" in result.stderr
    assert "large-v3" in result.stderr and "small" in result.stderr
    assert calls == []  # el backend jamas se llamo


# --- bench · tabla de configs medidas (núcleo SIEMPRE stubbeado, jamás motores) ------


def _bench_row(**over):
    row = {
        "engine": "faster-whisper", "model": "small", "quant": "int8", "device": "cpu",
        "load_s": 1.2, "transcribe_s": 7.0, "x_realtime": 8.5,
        "peak_ram_mb": 900.0, "peak_vram_mb": None, "segments": 12, "chars": 800,
        "capabilities": {"hotwords": True, "word_timestamps": True,
                         "native_signals": True, "vad": True},
        "wer_ref": 0.419, "error": None,
    }
    row.update(over)
    return row


def _bench_table(results, skipped=()):
    return {
        "schema_version": "speechtotext.bench/v1",
        "measured_at": "2026-07-27T00:00:00+00:00",
        "machine": {"cpu": "x", "logical_cores": 8, "ram_gb": 16.0, "gpu": None},
        "audio": {"source": "a.wav", "duration_s": 60.0, "sha1": "0" * 40},
        "results": results,
        "skipped": list(skipped),
    }


def test_bench_show_sin_tabla(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    result = runner.invoke(app, ["bench", "--show"])
    assert result.exit_code == 1
    assert "speechtotext bench" in result.stdout  # el mensaje dice CÓMO medirla


def test_bench_show_con_tabla(tmp_path, monkeypatch):
    from speechtotext.core import benchmark

    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    benchmark.write_table(_bench_table([_bench_row()]))
    result = runner.invoke(app, ["bench", "--show"])
    assert result.exit_code == 0
    assert "faster-whisper" in result.stdout
    assert "8.5" in result.stdout  # x_realtime visible


def test_bench_audio_escribe_tabla_y_reporta_skipped(tmp_path, monkeypatch):
    from speechtotext.cli import app as cli_app
    from speechtotext.core import audio as core_audio
    from speechtotext.core import benchmark

    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    src = tmp_path / "charla.mp3"
    src.write_bytes(b"ID3")
    wav = tmp_path / "t.wav"
    wav.write_bytes(b"RIFF")
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")
    monkeypatch.setattr(core_audio, "transcode_to_wav", lambda b, **k: wav)
    monkeypatch.setattr(cli_app, "_trim_wav", lambda w, s: clip)
    monkeypatch.setattr(cli_app, "_wav_seconds", lambda p: 42.0)

    viable = {"engine": "faster-whisper", "model": "tiny", "quant": "int8", "device": "cpu",
              "capabilities": _bench_row()["capabilities"], "wer_ref": None}
    monkeypatch.setattr(benchmark, "available_configs", lambda: ([viable], []))

    def fake_run_benchmark(wav_path, duration_s, configs, *, progress=None):
        results = []
        for cfg in configs:
            res = _bench_row(engine=cfg["engine"], model=cfg["model"])
            results.append(res)
            if progress:
                progress(cfg, res)
        return _bench_table(
            results,
            skipped=[{"engine": "whispercpp", "model": "small", "reason": "exe ausente"}],
        )

    monkeypatch.setattr(benchmark, "run_benchmark", fake_run_benchmark)
    result = runner.invoke(app, ["bench", str(src)])
    assert result.exit_code == 0
    assert (tmp_path / "bench.json").exists()  # write_table real, home sembrado
    assert "bench.json" in result.stdout  # el path se imprime
    assert "exe ausente" in result.stdout  # las saltadas se explican
    assert "tiny" in result.stdout  # fila de progreso por config


def test_bench_quick_salta_los_lentos(tmp_path, monkeypatch):
    from speechtotext.cli import app as cli_app
    from speechtotext.core import audio as core_audio
    from speechtotext.core import benchmark

    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    src = tmp_path / "charla.mp3"
    src.write_bytes(b"ID3")
    wav = tmp_path / "t.wav"
    wav.write_bytes(b"RIFF")
    monkeypatch.setattr(core_audio, "transcode_to_wav", lambda b, **k: wav)
    monkeypatch.setattr(cli_app, "_trim_wav", lambda w, s: wav)
    monkeypatch.setattr(cli_app, "_wav_seconds", lambda p: 42.0)
    caps = _bench_row()["capabilities"]
    viables = [
        {"engine": "faster-whisper", "model": m, "quant": "int8", "device": "cpu",
         "capabilities": caps, "wer_ref": None}
        for m in ("tiny", "medium", "large-v3")
    ]
    monkeypatch.setattr(benchmark, "available_configs", lambda: (viables, []))
    seen = {}

    def fake_run_benchmark(wav_path, duration_s, configs, *, progress=None):
        seen["models"] = [c["model"] for c in configs]
        return _bench_table([])

    monkeypatch.setattr(benchmark, "run_benchmark", fake_run_benchmark)
    result = runner.invoke(app, ["bench", str(src), "--quick"])
    assert result.exit_code == 0
    assert seen["models"] == ["tiny"]  # medium y large-v3 de fw quedan fuera


def test_bench_config_con_error_sale_marcada(tmp_path, monkeypatch):
    from speechtotext.core import benchmark

    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    rota = _bench_row(model="medium", load_s=None, transcribe_s=None, x_realtime=None,
                      peak_ram_mb=None, segments=None, chars=None, error="hijo murio rc=1")
    benchmark.write_table(_bench_table([_bench_row(), rota]))
    result = runner.invoke(app, ["bench", "--show"])
    assert result.exit_code == 0
    assert "hijo murio" in result.stdout  # la fila rota se ve, no se oculta
    assert "1 con error" in result.stdout


def test_whispercpp_avisa_el_remapeo_de_device(tmp_path, monkeypatch):
    # Pisar un -d cpu explicito en silencio seria la sustitucion callada que el
    # contrato prohibe: el remapeo a cuda se avisa siempre que no pidieran cuda.
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "--engine", "whispercpp", "-d", "cpu")
    assert result.exit_code == 0
    assert "corre en la GPU; device=cuda" in result.stdout

    result = _invoke(audio, tmp_path, "--engine", "whispercpp", "-d", "cuda")
    assert result.exit_code == 0
    assert "corre en la GPU" not in result.stdout  # quien pidio cuda no recibe ruido


def test_bench_quick_documenta_las_saltadas_en_skipped(tmp_path, monkeypatch):
    # Una tabla con filas ausentes sin razon haria que el consumidor de la tabla eligiera
    # sin saber que faltan candidatas: las quick-saltadas van a skipped con su motivo.
    import json

    from speechtotext.core import benchmark

    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path / "home"))
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    from speechtotext.cli import app as app_mod

    monkeypatch.setattr(app_mod, "_trim_wav", lambda wav, s: wav)
    monkeypatch.setattr(
        "speechtotext.core.audio.transcode_to_wav", lambda b: tmp_path / "t.wav"
    )
    (tmp_path / "t.wav").write_bytes(b"RIFF")
    monkeypatch.setattr("speechtotext.cli.app._wav_seconds", lambda p: 60.0)
    monkeypatch.setattr(
        benchmark, "available_configs",
        lambda: ([
            {"engine": "faster-whisper", "model": "small"},
            {"engine": "faster-whisper", "model": "medium"},
            {"engine": "faster-whisper", "model": "large-v3"},
        ], []),
    )
    monkeypatch.setattr(
        benchmark, "run_benchmark",
        lambda clip, d, configs, progress=None: {
            "results": [], "skipped": [], "machine": {}, "audio": {},
        },
    )
    result = runner.invoke(app, ["bench", str(audio), "--quick"], catch_exceptions=False)
    assert result.exit_code == 0
    tabla = json.loads((tmp_path / "home" / "bench.json").read_text(encoding="utf-8"))
    razones = {(s["engine"], s["model"]): s["reason"] for s in tabla["skipped"]}
    assert razones[("faster-whisper", "medium")] == "saltada por --quick"
    assert razones[("faster-whisper", "large-v3")] == "saltada por --quick"


def test_bench_ffmpeg_roto_sale_con_mensaje(tmp_path, monkeypatch):
    # La ruta de error del recorte: mensaje rojo y exit 1, no traceback crudo.
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(
        "speechtotext.core.audio.transcode_to_wav", lambda b: tmp_path / "t.wav"
    )
    (tmp_path / "t.wav").write_bytes(b"RIFF")
    from speechtotext.cli import app as app_mod

    def boom(wav, s):
        raise RuntimeError("ffmpeg no pudo recortar el audio: pista corrupta")

    monkeypatch.setattr(app_mod, "_trim_wav", boom)
    result = runner.invoke(app, ["bench", str(audio)])
    assert result.exit_code == 1
    assert "No se pudo recortar" in result.stdout


def test_troceado_anuncia_y_lista_cada_trozo_fuera_de_tty(tmp_path, monkeypatch):
    from speechtotext.core import transcribe as core_transcribe

    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(1.0, 2.0)], _info(1200.0))
    monkeypatch.setattr(core_transcribe, "should_chunk", lambda d, c: True)
    monkeypatch.setattr(core_transcribe, "plan_chunks", lambda path, dur: [(0.0, 600.0), (600.0, 1200.0)])
    monkeypatch.setenv("SPEECHTOTEXT_HOME", str(tmp_path))
    result = _invoke(audio, tmp_path, "-j", "2")
    assert result.exit_code == 0, result.stdout
    salida = _plana(result.stdout)
    assert "Troceado (jobs=2)" in salida
    assert "[1/2]" in salida and "[2/2]" in salida
    assert "(nuevo)" in salida


# --- defaults del spec §5.2, ETA e idioma dudoso ------------------------------------------


def test_defaults_del_cli_son_los_del_spec(tmp_path, monkeypatch):
    import json

    calls = []
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0), calls=calls)
    result = _invoke(audio, tmp_path, "-f", "json")
    assert result.exit_code == 0, result.stdout
    (_, request), = calls
    assert (request.language, request.vad, request.beam_size, request.hotwords) == ("auto", False, 5, ())
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert payload["engine"]["model"] == "large-v3"
    assert payload["engine"]["device"] == "cpu"      # conftest: máquina sin GPU
    assert payload["engine"]["selection"] == "auto"
    assert "Idioma detectado" in result.stdout
    assert "motor faster-whisper" in _plana(result.stdout)


def test_eta_se_imprime_tras_decodificar(tmp_path, monkeypatch):
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(600.0))
    result = _invoke(audio, tmp_path)
    assert result.exit_code == 0, result.stdout
    assert "Duración 10.0 min · ETA ~8 min (estimado)" in _plana(result.stdout)


def test_eta_medida_con_bench_no_dice_estimado(tmp_path, monkeypatch):
    from speechtotext.core import benchmark

    benchmark.write_table({"schema_version": "speechtotext.bench/v1", "results": [
        {"engine": "faster-whisper", "model": "large-v3", "quant": "int8", "device": "cpu",
         "x_realtime": 2.0, "error": None, "capabilities": {}, "wer_ref": None}],
        "skipped": [], "recommendations": []})
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(600.0))
    result = _invoke(audio, tmp_path)
    assert "ETA ~5 min (medido con bench)" in _plana(result.stdout)


def test_ruta_sin_eta_lo_dice(tmp_path, monkeypatch):
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(60.0))
    result = _invoke(audio, tmp_path, "-m", "medium")
    assert "ETA sin medir para esta ruta" in _plana(result.stdout)


def test_idioma_dudoso_sugiere_fijarlo(tmp_path, monkeypatch):
    info = _info(10.0, language="pt", language_probability=0.41)
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], info)
    result = _invoke(audio, tmp_path)
    assert "prob=0.41" in result.stdout and "fíjalo con -l" in result.stdout
    info = _info(10.0, language="pt", language_probability=0.9)
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], info)
    assert "fíjalo con -l" not in _invoke(audio, tmp_path).stdout


def test_modelo_que_no_cabe_en_ram_corta_sin_cambiarlo(tmp_path, monkeypatch):
    from speechtotext.core import probe

    calls = []
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0), calls=calls)
    monkeypatch.setattr(probe, "machine", lambda: probe.Machine("win32", 4, 4.0, False, None, None, None))
    result = _invoke(audio, tmp_path)
    assert result.exit_code == 1
    assert "large-v3 necesita ~6 GB" in result.stdout and "-m small" in result.stdout
    assert calls == []


def test_la_ruta_auto_avisa_y_anuncia_la_descarga_de_whispercpp(tmp_path, monkeypatch):
    import json

    from speechtotext.core import probe

    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    monkeypatch.setattr(probe, "machine",
                        lambda: probe.Machine("win32", 12, 32.0, True, "GTX 980", 3.5, None))
    result = _invoke(audio, tmp_path, "-f", "json")
    assert result.exit_code == 0, result.stdout
    salida = _plana(result.stdout)
    assert "GPU con 3.5 GB libres: whisper.cpp cuantizado" in salida
    assert "whisper.cpp v1.9.1 no está instalado: se descarga ahora (~646 MB, una sola vez)" in salida
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert (payload["engine"]["name"], payload["engine"]["device"]) == ("whispercpp", "cuda")


def test_whispercpp_ya_instalado_no_anuncia_descarga(tmp_path, monkeypatch):
    from pathlib import Path

    from speechtotext.core import probe

    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    monkeypatch.setattr(probe, "machine", lambda: probe.Machine(
        "win32", 12, 32.0, True, "GTX 980", 3.5, Path("C:/x/whisper-cli.exe")))
    result = _invoke(audio, tmp_path)
    assert result.exit_code == 0 and "se descarga ahora" not in result.stdout


# --- probe y models ------------------------------------------------------------------------


def test_probe_imprime_maquina_y_rutas(monkeypatch):
    from speechtotext.core import probe

    monkeypatch.setattr(probe, "machine",
                        lambda: probe.Machine("win32", 12, 31.9, True, "GTX 980", 3.46, None))
    result = runner.invoke(app, ["probe"])
    assert result.exit_code == 0, result.stdout
    salida = _plana(result.stdout)
    # _plana colapsa cualquier corrida de espacios a uno solo (" ".join(texto.split())): el
    # alineado de columnas del comando real usa varios espacios, pero aquí solo sobrevive uno.
    assert "platform win32" in salida and "ram_gb 31.9" in salida
    assert "gpu GTX 980" in salida and "vram_free 3.46 GB" in salida
    assert "whispercpp no instalado" in salida
    assert ("large-v3 whispercpp · cuda · q5_0 · ~8.0x tiempo real (estimado) · "
            "GPU con 3.5 GB libres: whisper.cpp cuantizado") in salida
    assert "small whispercpp · cuda · q5_0 · ~15.6x tiempo real (estimado)" in salida


def test_probe_dice_cuando_el_modelo_no_cabe(monkeypatch):
    from speechtotext.core import probe

    monkeypatch.setattr(probe, "machine", lambda: probe.Machine("linux", 4, 4.0, False, None, None, None))
    result = runner.invoke(app, ["probe"])
    assert result.exit_code == 0, result.stdout
    salida = _plana(result.stdout)
    assert "large-v3 large-v3 necesita ~6 GB" in salida
    assert "small faster-whisper · cpu · int8 · ~6.4x tiempo real (estimado) · sin GPU utilizable: CPU" in salida
    assert "ram_gb 4.0" in salida


def test_probe_sin_medidas_no_inventa(monkeypatch):
    from speechtotext.core import probe

    monkeypatch.setattr(probe, "machine", lambda: probe.Machine("darwin", 8, None, False, None, None, None))
    salida = _plana(runner.invoke(app, ["probe"]).stdout)
    assert "ram_gb sin medir" in salida and "vram_free -" in salida and "gpu -" in salida


def _models_doble(monkeypatch, instalados=(), size=None, boom=None):
    from pathlib import Path

    from speechtotext.core import models

    visto = {}
    monkeypatch.setattr(models, "installed",
                        lambda engine=None: [m for m in instalados if engine in (None, m.engine)])
    monkeypatch.setattr(models, "remote_size", lambda engine, name: size)

    def ensure(engine, name, on_progress=None):
        if boom:
            raise boom
        visto["ensure"] = (engine, name)
        return Path("C:/hf/snap")

    def remove(engine, name):
        if boom:
            raise boom
        visto["remove"] = (engine, name)

    monkeypatch.setattr(models, "ensure", ensure)
    monkeypatch.setattr(models, "remove", remove)
    return visto


def test_models_lista_vacia_sugiere_pull(monkeypatch):
    _models_doble(monkeypatch)
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0, result.stdout
    assert "models pull large-v3" in _plana(result.stdout)


def test_models_lista_tabla(monkeypatch, tmp_path):
    from speechtotext.core.models import ModelInfo

    _models_doble(monkeypatch, [
        ModelInfo("faster-whisper", "large-v3", tmp_path / "x", 3_090_839_273, False),
        ModelInfo("whispercpp", "small", tmp_path / "y.bin", 487_601_967, True),
    ])
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0, result.stdout
    salida = _plana(result.stdout)
    assert "large-v3" in salida and "2.9 GB" in salida and "465 MB" in salida and "sí" in salida
    assert "Datos en" in salida


def test_models_pull_anuncia_tamano_y_baja(monkeypatch):
    visto = _models_doble(monkeypatch, size=3_090_839_273)
    result = runner.invoke(app, ["models", "pull", "large-v3"])
    assert result.exit_code == 0, result.stdout
    assert "Descargando large-v3 (faster-whisper, ~2.9 GB)" in _plana(result.stdout)
    assert visto["ensure"] == ("faster-whisper", "large-v3")
    result = runner.invoke(app, ["models", "pull", "small", "--engine", "whispercpp"])
    assert result.exit_code == 0 and visto["ensure"] == ("whispercpp", "small")


def test_models_pull_sin_tamano_no_inventa(monkeypatch):
    _models_doble(monkeypatch, size=None)
    result = runner.invoke(app, ["models", "pull", "small"])
    assert "Descargando small (faster-whisper)..." in _plana(result.stdout)


def test_models_pull_fallo_de_descarga_sale_1(monkeypatch):
    _models_doble(monkeypatch, boom=RuntimeError("sha256 de ggml-small.bin no cuadra con el pin"))
    result = runner.invoke(app, ["models", "pull", "small", "--engine", "whispercpp"])
    assert result.exit_code == 1 and "sha256" in result.stdout


def test_models_rm(monkeypatch):
    visto = _models_doble(monkeypatch)
    result = runner.invoke(app, ["models", "rm", "small"])
    assert result.exit_code == 0 and visto["remove"] == ("faster-whisper", "small")
    _models_doble(monkeypatch, boom=FileNotFoundError("small (faster-whisper) no está instalado"))
    result = runner.invoke(app, ["models", "rm", "small"])
    assert result.exit_code == 1 and "no está instalado" in result.stdout


# --- I3 · el aviso de jobs=1 solo bajo --engine explícito -----------------------------


def test_ruta_auto_a_whispercpp_no_regana_por_jobs(tmp_path, monkeypatch):
    from pathlib import Path

    from speechtotext.core import probe

    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    monkeypatch.setattr(probe, "machine", lambda: probe.Machine(
        "win32", 12, 32.0, True, "GTX 980", 3.5, Path("C:/x/whisper-cli.exe")))
    result = _invoke(audio, tmp_path)
    assert result.exit_code == 0, result.stdout
    assert "paraleliza" not in result.stdout


def test_engine_whispercpp_explicito_con_jobs_avisa(tmp_path, monkeypatch):
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    result = _invoke(audio, tmp_path, "--engine", "whispercpp", "-j", "4")
    assert result.exit_code == 0, result.stdout
    assert "la GPU no paraleliza; jobs=1" in result.stdout


# --- I4/I5 · una sola sonda, selection fiel a quién eligió el motor -------------------


def test_el_cli_sondea_una_sola_vez(tmp_path, monkeypatch):
    from speechtotext.core import probe

    fija = probe.machine()
    veces = []
    monkeypatch.setattr(probe, "machine", lambda: (veces.append(1), fija)[1])
    audio = _fake_transcribe(monkeypatch, tmp_path, [_seg(0.0, 9.0)], _info(10.0))
    assert _invoke(audio, tmp_path).exit_code == 0
    assert len(veces) == 1


def test_mcp_llama_a_serve(monkeypatch):
    from speechtotext.cli import mcp_server

    llamado = []
    monkeypatch.setattr(mcp_server, "serve", lambda: llamado.append(True))
    resultado = runner.invoke(app, ["mcp"])
    assert resultado.exit_code == 0, resultado.output
    assert llamado == [True]
