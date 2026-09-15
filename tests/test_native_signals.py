"""Señales nativas por segmento (no_speech, avg_logprob, compression_ratio).

Fase 2 de plan-calidad-transcripcion / PR-2 de plan-multimotor: faster-whisper ya las
emite gratis en cada Segment, pero la ruta del CLI las tiraba. `is_suspect`
(core/formats.py) leía `no_speech` de un atributo que nadie seteaba: rama muerta.

Ley G5 (core/engines.py): señal que el motor no emite se OMITE, jamás se rellena.
whisper.cpp no las emite -> None de punta a punta, y el JSON omite la clave.
"""
import json
from types import SimpleNamespace

from speechtotext.core.chunked import (
    TimedSegment,
    TimedWord,
    seg_from_dict,
    seg_to_dict,
    shift_segments,
)
from speechtotext.core.formats import is_suspect, write_json
from speechtotext.core.segments import LabeledSegment, native_signals
from speechtotext.speakers.diarization import apply_names, assign_segments


def _raw(start, end, text="hola que tal", **kw):
    """Segment crudo de faster-whisper: el nombre del campo allá es `no_speech_prob`."""
    return SimpleNamespace(start=start, end=end, text=text, words=None, **kw)


def _info(duration=60.0):
    return SimpleNamespace(duration=duration, language="es", language_probability=1.0)


# --- native_signals: un solo lector para los dos dialectos ---


def test_native_signals_lee_el_dialecto_de_faster_whisper():
    seg = _raw(0.0, 2.0, no_speech_prob=0.8, avg_logprob=-0.4, compression_ratio=1.7)
    assert native_signals(seg) == (0.8, -0.4, 1.7)


def test_native_signals_lee_el_dialecto_propio():
    seg = TimedSegment(0.0, 2.0, "hola", None, no_speech=0.8,
                       avg_logprob=-0.4, compression_ratio=1.7)
    assert native_signals(seg) == (0.8, -0.4, 1.7)


def test_native_signals_sin_senales_es_none_no_cero():
    """whisper.cpp (_parse_ojf) no emite ninguna: cero sería una medida inventada."""
    assert native_signals(_raw(0.0, 2.0)) == (None, None, None)


# --- shift_segments: el punto de extracción de la ruta troceada ---


def test_shift_segments_copia_las_senales():
    out = shift_segments(
        [_raw(1.0, 3.0, no_speech_prob=0.9, avg_logprob=-0.2, compression_ratio=2.1)],
        10.0,
    )
    assert (out[0].no_speech, out[0].avg_logprob, out[0].compression_ratio) == (0.9, -0.2, 2.1)
    assert (out[0].start, out[0].end) == (11.0, 13.0)  # el offset sigue aplicándose


def test_shift_segments_sin_senales_deja_none():
    out = shift_segments([_raw(0.0, 2.0)], 0.0)
    assert (out[0].no_speech, out[0].avg_logprob, out[0].compression_ratio) == (None, None, None)


# --- checkpoint: round-trip y compatibilidad con caché vieja ---


def test_checkpoint_round_trip_conserva_las_senales():
    seg = TimedSegment(0.0, 2.0, "hola", [TimedWord(0.0, 1.0, "hola")],
                       no_speech=0.7, avg_logprob=-0.3, compression_ratio=1.9)
    back = seg_from_dict(json.loads(json.dumps(seg_to_dict(seg))))
    assert (back.no_speech, back.avg_logprob, back.compression_ratio) == (0.7, -0.3, 1.9)
    assert back.words[0].word == "hola"


def test_checkpoint_viejo_sin_senales_no_revienta():
    """Los .json de ~/.speechtotext/chunks escritos antes de esto no tienen las claves:
    degradan a None (= comportamiento de hoy), no a KeyError."""
    back = seg_from_dict({"start": 0.0, "end": 2.0, "text": "hola"})
    assert (back.no_speech, back.avg_logprob, back.compression_ratio) == (None, None, None)


def test_checkpoint_omite_las_senales_ausentes():
    d = seg_to_dict(TimedSegment(0.0, 2.0, "hola"))
    assert "no_speech" not in d and "avg_logprob" not in d and "compression_ratio" not in d


# --- is_suspect: la rama que estaba muerta ---


def test_is_suspect_enciende_por_no_speech():
    """Segmento corto y denso: hoy NO dispara por la heurística de densidad. Sólo
    no_speech puede marcarlo, y hasta ahora nadie lo seteaba."""
    seg = LabeledSegment(0.0, 2.0, "hola que tal como estas", no_speech=0.75)
    assert is_suspect(seg) is True


def test_is_suspect_no_enciende_con_no_speech_bajo():
    seg = LabeledSegment(0.0, 2.0, "hola que tal como estas", no_speech=0.5)
    assert is_suspect(seg) is False


def test_is_suspect_sin_senal_cae_a_la_densidad_como_siempre():
    denso = LabeledSegment(0.0, 2.0, "hola que tal como estas")
    ralo = LabeledSegment(0.0, 12.0, "eh")
    assert is_suspect(denso) is False
    assert is_suspect(ralo) is True


# --- diarización: las señales sobreviven al reparto por hablante ---


def test_assign_segments_grueso_propaga_las_senales():
    raw = TimedSegment(0.0, 4.0, "hola que tal", None,
                       no_speech=0.8, avg_logprob=-0.5, compression_ratio=2.0)
    out = assign_segments([raw], [(0.0, 4.0, "SPEAKER_00")])
    assert len(out) == 1
    assert (out[0].no_speech, out[0].avg_logprob, out[0].compression_ratio) == (0.8, -0.5, 2.0)


def test_assign_segments_por_palabras_propaga_a_cada_run():
    """Los N runs de un segmento heredan sus señales, igual que src_dur: vienen de la
    misma ventana de decodificación."""
    words = [TimedWord(0.0, 1.0, "hola"), TimedWord(2.0, 3.0, "adios")]
    raw = TimedSegment(0.0, 3.0, "hola adios", words,
                       no_speech=0.8, avg_logprob=-0.5, compression_ratio=2.0)
    out = assign_segments([raw], [(0.0, 1.5, "SPEAKER_00"), (1.5, 3.0, "SPEAKER_01")])
    assert len(out) == 2
    for seg in out:
        assert (seg.no_speech, seg.avg_logprob, seg.compression_ratio) == (0.8, -0.5, 2.0)


def test_apply_names_conserva_las_senales():
    labeled = [LabeledSegment(0.0, 2.0, "hola", "SPEAKER_00",
                              no_speech=0.8, avg_logprob=-0.5, compression_ratio=2.0)]
    out = apply_names(labeled, {"SPEAKER_00": "Alice"})
    assert out[0].speaker == "Alice"
    assert (out[0].no_speech, out[0].avg_logprob, out[0].compression_ratio) == (0.8, -0.5, 2.0)


# --- JSON: el artefacto de máquina las expone ---


def test_write_json_emite_las_senales(tmp_path):
    path = tmp_path / "out.json"
    segs = [LabeledSegment(0.0, 2.0, "hola que tal", no_speech=0.75,
                           avg_logprob=-0.3, compression_ratio=1.8)]
    write_json(segs, _info(), path)
    seg = json.loads(path.read_text(encoding="utf-8"))["segments"][0]
    assert seg["no_speech"] == 0.75
    assert seg["avg_logprob"] == -0.3
    assert seg["compression_ratio"] == 1.8
    assert seg["suspect"] is True  # la marca sale de la señal, no de la densidad


def test_write_json_omite_las_senales_ausentes(tmp_path):
    path = tmp_path / "out.json"
    write_json([LabeledSegment(0.0, 2.0, "hola que tal")], _info(), path)
    seg = json.loads(path.read_text(encoding="utf-8"))["segments"][0]
    assert "no_speech" not in seg
    assert "avg_logprob" not in seg
    assert "compression_ratio" not in seg


# --- el CLI de punta a punta: la ruta directa también las propaga ---


def test_cli_transcribe_lleva_las_senales_al_json(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from speechtotext.cli.app import app

    audio = tmp_path / "charla.wav"
    audio.write_bytes(b"RIFF")
    info = _info(duration=60.0)
    segs = [_raw(0.0, 2.0, no_speech_prob=0.75, avg_logprob=-0.3, compression_ratio=1.8)]
    from speechtotext.asr import Caps
    from speechtotext.asr.types import (
        NativeSignals, SegmentNativeSignals, TranscriptionResult, TranscriptionSegment,
    )
    from speechtotext.core import transcribe as core_transcribe

    import numpy as np

    monkeypatch.setattr(core_transcribe, "load_audio", lambda p: np.zeros(60 * 16000, dtype=np.float32))
    monkeypatch.setattr(core_transcribe, "should_chunk", lambda d, c: False)
    result = TranscriptionResult(
        text="hola que tal", language="es", words=(),
        segments=tuple(TranscriptionSegment(s.start, s.end, s.text, (), SegmentNativeSignals(
            s.no_speech_prob, s.avg_logprob, s.compression_ratio)) for s in segs),
        backend="faster-whisper", model="small", model_version="1", latency_ms=1,
        native_signals=NativeSignals(None, None, None, 1.0), warnings=(),
    )
    fake = SimpleNamespace(backend_id="faster-whisper", model_id="small", device="cpu", quant="int8",
                           model_version="1", engine_version="faster-whisper 1.2.0",
                           caps=Caps("honored", "honored", "honored"), warm=lambda: None,
                           transcribe=lambda samples, request: result)
    monkeypatch.setattr(core_transcribe, "make_backend", lambda *a, **k: fake)

    # -o sobre una ruta inexistente y sin separador final es un BASE path, no una carpeta
    # (_resolve_output_base, cli/app.py:70): el JSON sale en out.json, no en out/charla.json.
    result = CliRunner().invoke(
        app, ["transcribe", str(audio), "-f", "json", "-o", str(tmp_path / "out")]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    seg = payload["segments"][0]
    assert seg["no_speech"] == 0.75
    assert seg["suspect"] is True


# --- valores patológicos: el motor puede emitir basura, el artefacto no ---


def test_native_signals_descarta_no_finitos():
    """NaN/inf no son medidas: entran como None (G5), no como número. Sin esto,
    json.dumps escribe los tokens NaN/Infinity — JSON inválido por RFC 8259, que
    Python relee pero jq y JSON.parse rechazan — y encima `NaN > 0.6` es False,
    así que la señal más rota de todas sería la única que no se marca."""
    seg = _raw(0.0, 2.0, no_speech_prob=float("nan"),
               avg_logprob=float("-inf"), compression_ratio=float("inf"))
    assert native_signals(seg) == (None, None, None)


def test_write_json_no_emite_tokens_invalidos(tmp_path):
    path = tmp_path / "out.json"
    segs = shift_segments([_raw(0.0, 2.0, no_speech_prob=float("nan"))], 0.0)
    write_json([LabeledSegment(s.start, s.end, s.text, no_speech=s.no_speech)
                for s in segs], _info(), path)
    crudo = path.read_text(encoding="utf-8")
    assert "NaN" not in crudo and "Infinity" not in crudo
    json.loads(crudo)  # relee: si hubiera tokens inválidos, el propio repo los toleraría


def test_write_json_redondea_las_senales(tmp_path):
    """float32 de faster-whisper llega como -0.30000001192092896; el resto del payload
    (start/end/language_probability) ya se redondea, estas no eran la excepción."""
    path = tmp_path / "out.json"
    segs = [LabeledSegment(0.0, 2.0, "hola que tal", no_speech=0.1234567,
                           avg_logprob=-0.30000001192092896, compression_ratio=1.23456789)]
    write_json(segs, _info(), path)
    seg = json.loads(path.read_text(encoding="utf-8"))["segments"][0]
    assert seg["no_speech"] == 0.1235
    assert seg["avg_logprob"] == -0.3
    assert seg["compression_ratio"] == 1.2346
