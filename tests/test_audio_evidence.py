"""Evidencia de voz por DSP: medidas deterministas de senal, no la opinion de un
modelo. La pregunta no es «que dijo» sino «habia una fuente de voz humana aqui».

Medido en aurelius (2026-09-11) sobre audio real del telefono: el `no_speech`
de whisper salio con el signo al reves (cortaba habla y dejaba pasar inventos);
la energia en banda de voz y la fraccion de tramos con F0 separaron
conversacion de cuarto vacio con 0 % de solape intercuartilico.
"""
import numpy as np
import pytest

from speechtotext.audio import VoiceEvidence, compute_voice_evidence

SR = 16000


def _voz_sintetica(f0=140.0, segundos=1.0, sr=SR):
    """Serie armonica en F0 con energia concentrada donde viven las vocales.

    Cinco armonicos ponderados hacia 400-1200 Hz (F1/F2 de una vocal abierta),
    con envolvente lenta para que no sea un tono de laboratorio.
    """
    t = np.arange(int(segundos * sr)) / sr
    pesos = {1: 0.5, 2: 0.9, 3: 1.0, 4: 0.8, 5: 0.5, 6: 0.3, 8: 0.2}
    x = sum(w * np.sin(2 * np.pi * f0 * k * t) for k, w in pesos.items())
    envolvente = 0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t)   # tasa silabica
    return (0.2 * x / np.max(np.abs(x)) * envolvente).astype(np.float32)


def _ruido_blanco(segundos=1.0, sr=SR, semilla=7):
    return (0.05 * np.random.default_rng(semilla).standard_normal(int(segundos * sr))).astype(np.float32)


def test_la_evidencia_es_DETERMINISTA():
    """Mismo audio, mismos numeros, siempre. Es la propiedad que `no_speech`
    no tiene y por la que existe este modulo."""
    x = _voz_sintetica()
    assert compute_voice_evidence(x, SR) == compute_voice_evidence(x.copy(), SR)


def test_una_voz_sintetica_deja_evidencia_de_voz():
    e = compute_voice_evidence(_voz_sintetica(f0=140.0), SR)
    assert e.frames > 0
    assert e.voice_band_ratio > 0.6
    assert e.voiced_ratio > 0.8
    assert e.f0_median_hz == pytest.approx(140.0, abs=5.0)
    assert e.spectral_flatness < 0.1


def test_el_ruido_blanco_NO_deja_evidencia_de_voz():
    e = compute_voice_evidence(_ruido_blanco(), SR)
    assert e.voiced_ratio < 0.2
    assert e.f0_median_hz is None or e.voiced_ratio < 0.2
    # ruido plano: la banda 300-3400 de un espectro de 0-8000 pesa ~0.39
    assert e.voice_band_ratio < 0.5
    assert e.spectral_flatness > 0.5


def test_un_zumbido_tonal_por_debajo_de_la_banda_NO_es_voz():
    """El HNR clasico premiaba el zumbido de la nevera porque es MAS armonico
    que el habla. Aqui el zumbido cae fuera de la banda de voz y fuera del
    rango de F0, y no cuenta como sonoro."""
    t = np.arange(SR) / SR
    zumbido = (0.2 * np.sin(2 * np.pi * 50.0 * t)).astype(np.float32)
    e = compute_voice_evidence(zumbido, SR)
    assert e.voice_band_ratio < 0.05
    assert e.voiced_ratio < 0.1


def test_el_silencio_no_inventa_evidencia():
    e = compute_voice_evidence(np.zeros(SR, dtype=np.float32), SR)
    assert e.frames > 0
    assert e.voice_band_ratio is None
    assert e.spectral_flatness is None
    assert e.voiced_ratio == 0.0
    assert e.f0_median_hz is None


def test_audio_mas_corto_que_un_tramo_no_tiene_evidencia():
    e = compute_voice_evidence(np.zeros(100, dtype=np.float32), SR)
    assert e == VoiceEvidence(
        voice_band_ratio=None, voiced_ratio=0.0, f0_median_hz=None,
        spectral_flatness=None, frames=0,
    )


@pytest.mark.parametrize(
    "muestras, sr",
    [
        (np.zeros((2, SR), dtype=np.float32), SR),          # estereo
        (np.array([0.0, np.nan], dtype=np.float32), SR),     # NaN
        (np.zeros(SR, dtype=np.float32), 0),                 # sr invalido
    ],
)
def test_la_entrada_invalida_se_rechaza(muestras, sr):
    with pytest.raises(ValueError):
        compute_voice_evidence(muestras, sr)


@pytest.mark.parametrize("f0", [90.0, 140.0, 220.0, 300.0])
def test_el_f0_no_cae_una_octava_en_una_serie_armonica(f0):
    """Una serie armonica pura tiene autocorrelacion ~1 en su periodo Y en sus
    multiplos. Sin un coste de octava, el argmax cae en el subarmonico por
    ruido de redondeo: a 300 Hz salia 100."""
    e = compute_voice_evidence(_voz_sintetica(f0=f0), SR)
    assert e.f0_median_hz == pytest.approx(f0, rel=0.03)
