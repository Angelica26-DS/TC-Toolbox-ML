"""
Unit tests for toolbox_ml.eda.core.
"""

# ══════════════════════════════════════════════
# Tests de plot_features_num_regression
# ══════════════════════════════════════════════

def test_plot_features_num_regression_devuelve_lista(monkeypatch):
    """Caso correcto: devuelve una lista con columnas correladas."""
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    df = pd.DataFrame({
        "target": [1, 2, 3, 4, 5],
        "alta_corr": [2, 4, 6, 8, 10],
        "baja_corr": [5, 1, 5, 1, 5],
    })
    resultado = plot_features_num_regression(
        df, target_col="target", umbral_corr=0.8
    )
    assert isinstance(resultado, list)
    assert "alta_corr" in resultado
    assert "baja_corr" not in resultado


def test_plot_features_num_regression_retorna_none_df_invalido():
    """Caso de error: df no es DataFrame → retorna None."""
    assert plot_features_num_regression("no es un df", target_col="target") is None


def test_plot_features_num_regression_retorna_none_target_no_existe():
    """Caso de error: target_col no existe en el DataFrame → retorna None."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert plot_features_num_regression(df, target_col="no_existe") is None


def test_plot_features_num_regression_retorna_none_target_no_numerico():
    """Caso de error: target_col no es numérica → retorna None."""
    df = pd.DataFrame({"target": ["a", "b", "c"], "x": [1, 2, 3]})
    assert plot_features_num_regression(df, target_col="target") is None


def test_plot_features_num_regression_retorna_none_umbral_invalido():
    """Caso de error: umbral_corr fuera de rango → retorna None."""
    df = pd.DataFrame({"target": [1, 2, 3], "x": [1, 2, 3]})
    assert plot_features_num_regression(df, target_col="target", umbral_corr=1.5) is None


def test_plot_features_num_regression_lista_vacia_si_nada_correlado(monkeypatch):
    """Si ninguna columna supera el umbral, devuelve lista vacía."""
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    df = pd.DataFrame({
        "target": [1, 2, 3, 4, 5],
        "ruido":  [5, 3, 1, 4, 2],
    })
    resultado = plot_features_num_regression(
        df, target_col="target", umbral_corr=0.99
    )
    assert resultado == []
