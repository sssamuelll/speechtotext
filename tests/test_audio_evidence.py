"""Evidencia de voz por DSP: medidas deterministas de senal, no la opinion de un
modelo. La pregunta no es «que dijo» sino «habia una fuente de voz humana aqui».

Medido (2026-09-11) sobre audio real del telefono: el `no_speech`
de whisper salio con el signo al reves (cortaba habla y dejaba pasar inventos);
la energia en banda de voz separo conversacion de cuarto vacio con 0 % de
solape intercuartilico.

Los tests de aqui nacieron de una revision adversarial en la que 24 de 33
mutaciones SOBREVIVIERON a la suite anterior. Cada test nuevo nombra la
mutacion que mata.
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


def _voz_sintetica(f0=140.0, segundos=1.0, sr=SR, amplitud=0.2):
    """Serie armonica en F0 con energia concentrada donde viven las vocales.

    Cinco armonicos ponderados hacia 400-1200 Hz (F1/F2 de una vocal abierta),
    con envolvente lenta para que no sea un tono de laboratorio.
    """
    t = np.arange(int(segundos * sr)) / sr
    pesos = {1: 0.5, 2: 0.9, 3: 1.0, 4: 0.8, 5: 0.5, 6: 0.3, 8: 0.2}
    x = sum(w * np.sin(2 * np.pi * f0 * k * t) for k, w in pesos.items())
    envolvente = 0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t)   # tasa silabica
    return (amplitud * x / np.max(np.abs(x)) * envolvente).astype(np.float32)


def _ruido_blanco(segundos=1.0, sr=SR, semilla=7, amplitud=0.05):
    rng = np.random.default_rng(semilla)
    return (amplitud * rng.standard_normal(int(segundos * sr))).astype(np.float32)


def _tono(hz, segundos=1.0, sr=SR, amplitud=0.2):
    t = np.arange(int(segundos * sr)) / sr
    return (amplitud * np.sin(2 * np.pi * hz * t)).astype(np.float32)


# ----------------------------------------------------------------------------
# lo que una voz deja, lo que no
# ----------------------------------------------------------------------------

def test_una_voz_sintetica_deja_evidencia_de_voz():
    e = compute_voice_evidence(_voz_sintetica(f0=140.0), SR)
    assert e.frames > 0
    assert e.voice_band_ratio > 0.6
    assert e.voiced_ratio > 0.8
    assert e.spectral_flatness < 0.1


def test_el_ruido_blanco_NO_deja_evidencia_de_voz():
    e = compute_voice_evidence(_ruido_blanco(), SR)
    assert e.voiced_ratio < 0.2
    # ruido plano: la banda 300-3400 de un espectro de 0-8000 pesa ~0.39
    assert e.voice_band_ratio < 0.5
    assert e.spectral_flatness > 0.5


def test_un_zumbido_tonal_por_debajo_de_la_banda_NO_es_voz():
    """El HNR clasico premiaba el zumbido de la nevera porque es MAS armonico
    que el habla. Aqui el zumbido cae fuera de la banda de voz y fuera del
    rango de F0, y no cuenta como sonoro."""
    e = compute_voice_evidence(_tono(50.0), SR)
    assert e.voice_band_ratio < 0.05
    assert e.voiced_ratio < 0.1


def test_un_tono_puro_DENTRO_de_la_banda_da_perfil_de_voz_y_eso_se_declara():
    """Limite conocido, no bug: un timbre, una alarma o un acople a 1 kHz da
    banda=1 y sonoro=1 con un F0 subarmonico inventado. La autocorrelacion
    tiene picos en todos los multiplos del periodo y el rango 70-350 Hz los
    recoge. Lo que SI lo delata es la planitud: una linea espectral sola.
    El que llama combina las tres medidas; esta libreria no da veredictos."""
    e = compute_voice_evidence(_tono(1000.0), SR)
    assert e.voice_band_ratio > 0.99
    assert e.voiced_ratio > 0.99
    assert e.f0_median_hz is not None and e.f0_median_hz < 350.0
    assert e.spectral_flatness < 1e-6


# ----------------------------------------------------------------------------
# invariancias: las que la revision encontro rotas
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("ganancia_db", [-20.0, -40.0])
def test_las_medidas_NO_dependen_de_la_ganancia(ganancia_db):
    """La revision midio spectral_flatness moviendose 4e9x entre 0 y -40 dB por
    un EPS absoluto. El audio real del telefono vive a -48..-60 LUFS: justo
    ahi. Un descriptor que cambia con el volumen no describe el espectro."""
    x = _voz_sintetica()
    a = compute_voice_evidence(x, SR)
    b = compute_voice_evidence(x * 10 ** (ganancia_db / 20), SR)
    assert b.voice_band_ratio == pytest.approx(a.voice_band_ratio, rel=1e-6)
    assert b.spectral_flatness == pytest.approx(a.spectral_flatness, rel=1e-3)
    assert b.voiced_ratio == pytest.approx(a.voiced_ratio, abs=0.02)
    assert b.f0_median_hz == pytest.approx(a.f0_median_hz, rel=1e-6)


def test_la_planitud_describe_el_espectro_y_NO_la_fraccion_de_pausa():
    """Media sin ponderar sobre tramos: la planitud seguia la fraccion de
    silencio (0.00 -> 0.25 -> 0.49 con 0/50/90 % de pausa con dither). Los
    tramos pesan por su energia, como ya hacia voice_band_ratio."""
    voz = _voz_sintetica(segundos=1.0)
    pausa = _ruido_blanco(segundos=1.0, amplitud=1e-5)       # dither, no ceros exactos
    sola = compute_voice_evidence(voz, SR).spectral_flatness
    con_pausa = compute_voice_evidence(np.concatenate([voz, pausa]), SR).spectral_flatness
    assert con_pausa == pytest.approx(sola, abs=0.01)


# ----------------------------------------------------------------------------
# F0: la aritmetica, sin holgura donde se esconde un off-by-one
# ----------------------------------------------------------------------------

def test_la_normalizacion_de_praat_da_uno_en_el_periodo_aunque_el_lag_sea_largo():
    """Dividir por r[0] castiga los lags largos: a 70 Hz (lag 229 de 1024) da
    0.78 y sesga el F0 hacia arriba. Con la energia de cada mitad da ~1.
    Mata la mutacion «r[0] en vez de Praat», que la suite anterior dejaba pasar."""
    frame = round(SR * 0.064)
    x = _tono(70.0, segundos=frame / SR)[None, :].astype(np.float64)
    r = _normalized_autocorrelation(x)
    lag = round(SR / 70.0)
    assert r[0, lag] > 0.99
    assert r[0, lag] > r[0, 0] * 0.99   # y NO cae con el lag


@pytest.mark.parametrize("f0", [90.0, 140.0, 220.0, 300.0])
def test_el_f0_es_EXACTAMENTE_el_lag_entero_mas_cercano(f0):
    """Sin interpolacion parabolica, el F0 correcto es sr/round(sr/f0). Una
    tolerancia relativa del 3 % dejaba pasar un lag de mas (+1): a 300 Hz eso
    son 290.9 Hz y nadie lo veia. Tambien mata la caida de octava (sin coste
    de octava, 300 -> 100)."""
    e = compute_voice_evidence(_voz_sintetica(f0=f0), SR)
    assert e.f0_median_hz == pytest.approx(SR / round(SR / f0), rel=1e-9)


def test_voiced_ratio_es_una_FRACCION_no_un_maximo():
    """Todas las senales de la suite anterior daban 0.0 o 1.0 exactos, asi que
    `mean -> max` sobrevivia. Media voz, media silencio: ~0.5."""
    x = np.concatenate([_voz_sintetica(segundos=0.5), np.zeros(SR // 2, dtype=np.float32)])
    e = compute_voice_evidence(x, SR)
    assert 0.35 < e.voiced_ratio < 0.65


# ----------------------------------------------------------------------------
# los bordes de la banda son la definicion del descriptor
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("hz, dentro", [(250.0, False), (350.0, True), (3300.0, True), (3600.0, False)])
def test_los_bordes_de_la_banda_de_voz_son_300_y_3400(hz, dentro):
    """Ningun test fijaba 300/3400; mover la banda a 200-4000 pasaba la suite.
    Un tono a 50 Hz de cada borde cae entero de un lado."""
    e = compute_voice_evidence(_tono(hz), SR)
    assert (e.voice_band_ratio > 0.98) is dentro
    assert (e.voice_band_ratio < 0.02) is (not dentro)


# ----------------------------------------------------------------------------
# «no pude medir» no es «no hay voz»
# ----------------------------------------------------------------------------

def test_el_silencio_no_inventa_evidencia():
    e = compute_voice_evidence(np.zeros(SR, dtype=np.float32), SR)
    assert e.frames > 0
    assert e.voice_band_ratio is None
    assert e.spectral_flatness is None
    assert e.voiced_ratio is None
    assert e.f0_median_hz is None


def test_demasiado_bajo_para_medir_es_None_en_TODAS_las_medidas():
    """Antes voice_band_ratio era None solo con ceros digitales exactos
    (-190 dBFS) y voiced_ratio salia 0.0 tanto sin voz como con voz a -100 dB.
    Un 0.0 que significa dos cosas es un veredicto escondido. La regla es UNA:
    si ningun tramo supera SILENCE_RMS, no se midio nada."""
    e = compute_voice_evidence(_voz_sintetica(amplitud=1e-5), SR)   # -100 dBFS
    assert e.frames > 0
    assert e == VoiceEvidence(None, None, None, None, e.frames)


def test_ruido_audible_sin_voz_es_CERO_medido_no_None():
    e = compute_voice_evidence(_ruido_blanco(), SR)
    assert e.voiced_ratio == 0.0
    assert e.voice_band_ratio is not None


def test_audio_mas_corto_que_un_tramo_no_tiene_evidencia():
    e = compute_voice_evidence(np.zeros(100, dtype=np.float32), SR)
    assert e == VoiceEvidence(
        voice_band_ratio=None, voiced_ratio=None, f0_median_hz=None,
        spectral_flatness=None, frames=0,
    )


@pytest.mark.parametrize(
    "muestras, sr",
    [
        (np.zeros((2, SR), dtype=np.float32), SR),          # estereo
        (np.array([0.0, np.nan], dtype=np.float32), SR),     # NaN
        (np.zeros(SR, dtype=np.float32), 0),                 # sr invalido
        (np.zeros(SR, dtype=np.float32), 300),               # sr por debajo del rango de F0
    ],
)
def test_la_entrada_invalida_se_rechaza(muestras, sr):
    with pytest.raises(ValueError):
        compute_voice_evidence(muestras, sr)


@pytest.mark.parametrize(
    "campos",
    [
        dict(voice_band_ratio=-0.1), dict(voice_band_ratio=1.5), dict(voiced_ratio=2.0),
        dict(f0_median_hz=-1.0), dict(f0_median_hz=float("nan")),
        dict(spectral_flatness=float("inf")), dict(frames=-1),
    ],
)
def test_el_dataclass_valida_como_sus_vecinos(campos):
    base = dict(voice_band_ratio=0.5, voiced_ratio=0.5, f0_median_hz=120.0,
                spectral_flatness=0.1, frames=10)
    with pytest.raises(ValueError):
        VoiceEvidence(**{**base, **campos})


# ----------------------------------------------------------------------------
# determinismo de verdad, y memoria acotada
# ----------------------------------------------------------------------------

# Valores clavados de `_voz_sintetica()` a 16 kHz (numpy pocketfft, 2026-09-11, en
# Windows x86-64). Si cambian sin que cambie la definicion de la medida, cambio el
# backend o la aritmetica — y eso es lo que este test existe para gritar.
#
# Pero NO se comparan bit a bit. La primera corrida de CI fuera de Windows lo dejo
# claro: pocketfft despacha por SIMD segun la maquina y el ultimo bit no viaja. Medido
# 2026-09-14 con estos mismos valores: Linux x86-64 da `band` a 1 ULP del valor de
# Windows, y macOS arm64 da `flat` a 3 ULP. No es una regresion; es que la igualdad
# exacta de floats entre plataformas nunca fue una propiedad que este codigo pudiera
# prometer. La tolerancia de abajo deja pasar ese ruido y sigue gritando ante cualquier
# cambio que importe: un backend o una formula distintos mueven digitos, no bits.
#
# Lo que si se compara bit a bit es el otro proceso contra este: misma maquina, misma
# build, y ahi la igualdad exacta es exigible. Esa es la mitad del test que de verdad
# comprueba determinismo, y la que caza un mutante que dependa de time.time_ns().
GOLDEN = {
    "band": 0.6558522702500179,
    "voiced": 1.0,
    "f0": 140.35087719298247,   # 16000 / 114
    "flat": 2.1362539235000618e-09,
}
TOLERANCIA = 1e-12   # ~6000 veces el ruido de plataforma medido

_CODIGO_OTRO_PROCESO = (
    "import sys; sys.path.insert(0, 'tests');"
    "from test_audio_evidence import _voz_sintetica, SR;"
    "from speechtotext.audio import compute_voice_evidence;"
    "e = compute_voice_evidence(_voz_sintetica(), SR);"
    "print(e.voice_band_ratio.hex(), e.voiced_ratio.hex(), "
    "e.f0_median_hz.hex(), e.spectral_flatness.hex(), e.frames)"
)


def test_la_evidencia_es_determinista_ENTRE_PROCESOS_y_esta_clavada():
    """Comparar dos llamadas en el mismo proceso es una tautologia: la revision
    construyo un mutante que dependia de time.time_ns() y lo pasaba. Aqui otro
    proceso calcula lo mismo y los bytes tienen que coincidir con un valor
    clavado — que ademas detecta un cambio de backend de FFT."""
    e = compute_voice_evidence(_voz_sintetica(), SR)
    otro = subprocess.run([sys.executable, "-c", _CODIGO_OTRO_PROCESO],
                          capture_output=True, text=True, check=True)
    assert otro.stdout.split() == [
        e.voice_band_ratio.hex(), e.voiced_ratio.hex(), e.f0_median_hz.hex(),
        e.spectral_flatness.hex(), str(e.frames),
    ]
    medido = {"band": e.voice_band_ratio, "voiced": e.voiced_ratio,
              "f0": e.f0_median_hz, "flat": e.spectral_flatness}
    lejos = {
        clave: (valor, GOLDEN[clave])
        for clave, valor in medido.items()
        if not math.isclose(valor, GOLDEN[clave], rel_tol=TOLERANCIA)
    }
    assert not lejos, f"la medida se movio mas que el ruido de plataforma: {lejos}"


def _pico_mb(x):
    tracemalloc.start()
    try:
        compute_voice_evidence(x, SR)
        return tracemalloc.get_traced_memory()[1] / 1e6
    finally:
        tracemalloc.stop()


def test_la_memoria_esta_ACOTADA_y_no_crece_con_la_duracion():
    """La revision midio 88-90x el buffer: 330 MB por 60 s, 19 GB por una hora,
    en una libreria que dice procesar grabaciones de horas. Los tramos son una
    vista (sin copia) y se procesan por lotes: el pico no depende de la duracion."""
    corto = _pico_mb(_voz_sintetica(segundos=20.0))
    largo = _pico_mb(_voz_sintetica(segundos=120.0))
    assert corto < 80.0, f"pico {corto:.0f} MB por 20 s"
    assert largo < corto * 1.5, f"20 s: {corto:.0f} MB, 120 s: {largo:.0f} MB"
