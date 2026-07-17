import pytest

from speechtotext.evaluation.metrics import (
    brier_score,
    character_error,
    cluster_bootstrap_error_upper,
    cluster_bootstrap_percentile_upper,
    expected_calibration_error,
    normalize_transcript,
    one_sided_error_upper,
    one_sided_success_lower,
    percentile,
    risk_coverage_curve,
    word_error,
)


def test_normalizador_preserva_acentos_y_elimina_puntuacion():
    assert normalize_transcript(
        "  ¡Táchira, SÍ!  "
    ) == "táchira sí"


def test_wer_y_cer_reportan_numerador_y_denominador():
    wer = word_error("hola mundo", "hola cruel mundo")
    assert (wer.errors, wer.reference_units) == (1, 2)
    assert wer.rate == pytest.approx(0.5)
    cer = character_error("abc", "adc")
    assert (cer.errors, cer.reference_units, cer.rate) == (1, 3, pytest.approx(1 / 3))


def test_limite_cero_fallos_coincide_con_regla_exacta():
    upper = one_sided_error_upper(0, 3000)
    assert upper == pytest.approx(1.0 - 0.05 ** (1.0 / 3000), rel=1e-9)
    assert upper < 0.001
    assert one_sided_success_lower(3000, 3000) == pytest.approx(1.0 - upper)


def test_metricas_de_calibracion_y_riesgo_cobertura():
    probabilities = [0.9, 0.8, 0.2, 0.1]
    labels = [1, 1, 0, 0]
    assert brier_score(probabilities, labels) == pytest.approx(0.025)
    assert expected_calibration_error(probabilities, labels, bins=2) == pytest.approx(0.15)
    curve = risk_coverage_curve(probabilities, labels)
    assert curve[0].coverage == pytest.approx(0.25)
    assert curve[0].risk == 0.0
    assert curve[0].errors == 0
    assert curve[0].error_upper_95 == pytest.approx(0.95)
    assert curve[-1].coverage == 1.0
    assert curve[-1].risk == 0.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_metricas_de_calibracion_rechazan_probabilidad_no_finita(bad):
    with pytest.raises(ValueError, match="probabilidades"):
        brier_score([bad], [1])


@pytest.mark.parametrize("bad", [True, 0.5, "1"])
def test_metricas_de_calibracion_rechazan_label_no_binario(bad):
    with pytest.raises(ValueError, match="labels"):
        expected_calibration_error([0.5], [bad])


def test_percentile_interpola_linealmente():
    assert percentile([100.0, 200.0, 300.0], 95.0) == pytest.approx(290.0)


def test_bootstrap_de_error_es_determinista_y_exige_tres_bloques():
    counts = [(1, 100), (2, 100), (3, 100)]
    first = cluster_bootstrap_error_upper(counts, resamples=1000, seed=7)
    assert first == cluster_bootstrap_error_upper(counts, resamples=1000, seed=7)
    assert first >= 0.02
    with pytest.raises(ValueError, match="tres bloques"):
        cluster_bootstrap_error_upper(counts[:2])


def test_upper_es_macro_y_cero_fallos_conserva_incertidumbre():
    imbalanced = [(0, 10_000), (1, 1), (0, 10_000)]
    assert cluster_bootstrap_error_upper(
        imbalanced, resamples=1000, seed=7
    ) >= 1 / 3
    zero = cluster_bootstrap_error_upper(
        [(0, 1000), (0, 1000), (0, 1000)], resamples=1000, seed=7
    )
    assert 0.0 < zero < 0.01


def test_bloque_con_wer_mayor_a_uno_no_se_recorta_y_fuerza_fallo():
    # Un bloque alucinado (mas errores que palabras de referencia) conserva su
    # tasa >1; recortar la tasa/limite por bloque a <=1.0 enmascararia el fallo.
    upper = cluster_bootstrap_error_upper(
        [(6, 2), (0, 100), (0, 100)], resamples=1000, seed=7
    )
    # El codigo real devuelve ~2.0; un mutante que recorte la tasa a <=1.0
    # colapsa a ~0.667, por debajo del umbral, y el test lo mata.
    assert upper >= 1.0


def test_upper_del_p95_resamplea_dias_completos_y_exige_tres_bloques():
    blocks = ([100.0, 110.0], [200.0, 220.0], [300.0, 330.0])
    upper = cluster_bootstrap_percentile_upper(
        blocks,
        q=95.0,
        resamples=1000,
        seed=7,
    )
    assert upper == cluster_bootstrap_percentile_upper(
        blocks,
        q=95.0,
        resamples=1000,
        seed=7,
    )
    assert upper >= percentile([value for block in blocks for value in block], 95.0)
    with pytest.raises(ValueError, match="tres bloques"):
        cluster_bootstrap_percentile_upper(blocks[:2])
