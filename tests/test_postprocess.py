"""Post-proceso del texto transcrito: normalización de horas (item #3 del backlog)."""
from speechtotext.core.postprocess import normalize_hours


def test_convierte_hora_con_punto_a_dos_puntos():
    assert normalize_hours("empezó a las 8.33 de la mañana") == "empezó a las 8:33 de la mañana"


def test_convierte_varias_horas_en_una_linea():
    assert normalize_hours("8.13 y 8.33") == "8:13 y 8:33"


def test_preserva_magnitudes_sismicas():
    # Un solo decimal = magnitud sísmica, no hora: se conserva como dígito.
    assert normalize_hours("un sismo de 7.2 y otro de 7.5") == "un sismo de 7.2 y otro de 7.5"


def test_minutos_invalidos_no_se_tocan():
    # 8.99 no son minutos válidos (00-59): no es hora, se conserva.
    assert normalize_hours("valor 8.99 medido") == "valor 8.99 medido"


def test_horas_de_dos_digitos():
    assert normalize_hours("a las 18.45") == "a las 18:45"


def test_texto_sin_numeros_intacto():
    assert normalize_hours("hola mundo") == "hola mundo"
