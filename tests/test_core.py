"""
Unit tests for toolbox_ml.eda.core.
"""
import pytest
import pandas as pd
from toolbox_ml.eda.core import (
    describe_df,
    tipifica_variables,
    plot_features_num_regression,
    get_features_num_regression,
    plot_features_cat_regression,
)


# ══════════════════════════════════════════════
# Tests de describe_df
# ══════════════════════════════════════════════

def test_describe_df_devuelve_dataframe():
    df = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
    resultado = describe_df(df)

    assert isinstance(resultado, pd.DataFrame)


def test_describe_df_columnas_correctas():
    df = pd.DataFrame({"a": [1, 2, 3]})
    resultado = describe_df(df)

    assert set(resultado.columns) == {
        "tipo",
        "porcentaje_nulos",
        "valores_unicos",
        "porcentaje_cardinalidad",
    }


def test_describe_df_porcentaje_nulos_correcto():
    df = pd.DataFrame({"a": [1, None, None, None]})
    resultado = describe_df(df)

    assert resultado.loc["a", "porcentaje_nulos"] == pytest.approx(75.0, abs=0.01)


def test_describe_df_valores_unicos_correcto():
    df = pd.DataFrame({"col": [1, 2, 2, None]})
    resultado = describe_df(df)

    assert resultado.loc["col", "valores_unicos"] == 2


def test_describe_df_retorna_none_con_input_invalido():
    assert describe_df("esto no es un dataframe") is None
    assert describe_df([1, 2, 3]) is None
    assert describe_df(None) is None


def test_describe_df_retorna_none_con_df_vacio():
    assert describe_df(pd.DataFrame()) is None


# ══════════════════════════════════════════════
# Tests de tipifica_variables
# ══════════════════════════════════════════════

def test_tipifica_variables_devuelve_dataframe():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
    resultado = tipifica_variables(df, umbral_categoria=10, umbral_continua=50.0)

    assert isinstance(resultado, pd.DataFrame)


def test_tipifica_variables_columnas_correctas():
    df = pd.DataFrame({"a": [1, 2]})
    resultado = tipifica_variables(df, umbral_categoria=10, umbral_continua=50.0)

    assert set(resultado.columns) == {"nombre_variable", "tipo_sugerido"}


def test_tipifica_variables_detecta_binaria():
    df = pd.DataFrame({"sexo": ["hombre", "mujer", "hombre", "mujer"]})
    resultado = tipifica_variables(df, umbral_categoria=10, umbral_continua=50.0)
    fila = resultado[resultado["nombre_variable"] == "sexo"]

    assert fila["tipo_sugerido"].values[0] == "Binaria"


def test_tipifica_variables_detecta_categorica():
    df = pd.DataFrame({"color": ["rojo", "azul", "verde", "rojo", "azul", "verde"]})
    resultado = tipifica_variables(df, umbral_categoria=10, umbral_continua=50.0)
    fila = resultado[resultado["nombre_variable"] == "color"]

    assert fila["tipo_sugerido"].values[0] == "Categórica"


def test_tipifica_variables_detecta_numerica_continua():
    df = pd.DataFrame({"precio": list(range(100))})
    resultado = tipifica_variables(df, umbral_categoria=10, umbral_continua=50.0)
    fila = resultado[resultado["nombre_variable"] == "precio"]

    assert fila["tipo_sugerido"].values[0] == "Numérica Continua"


def test_tipifica_variables_retorna_none_si_df_invalido():
    assert tipifica_variables("no es un df", 10, 50.0) is None


def test_tipifica_variables_retorna_none_si_umbral_categoria_invalido():
    df = pd.DataFrame({"a": [1, 2, 3]})

    assert tipifica_variables(df, -5, 50.0) is None
    assert tipifica_variables(df, 0, 50.0) is None


def test_tipifica_variables_retorna_none_si_umbral_continua_invalido():
    df = pd.DataFrame({"a": [1, 2, 3]})

    assert tipifica_variables(df, 10, -1.0) is None
    assert tipifica_variables(df, 10, 150.0) is None


# ══════════════════════════════════════════════
# Tests de plot_features_num_regression
# ══════════════════════════════════════════════

def test_plot_features_num_regression_devuelve_lista(monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    df = pd.DataFrame({
        "target": [1, 2, 3, 4, 5],
        "alta_corr": [2, 4, 6, 8, 10],
        "baja_corr": [5, 1, 5, 1, 5],
    })

    resultado = plot_features_num_regression(
        df,
        target_col="target",
        umbral_corr=0.8,
    )

    assert isinstance(resultado, list)
    assert "alta_corr" in resultado
    assert "baja_corr" not in resultado


def test_plot_features_num_regression_retorna_none_df_invalido():
    assert plot_features_num_regression("no es un df", target_col="target") is None


def test_plot_features_num_regression_retorna_none_target_no_existe():
    df = pd.DataFrame({"a": [1, 2, 3]})

    assert plot_features_num_regression(df, target_col="no_existe") is None


def test_plot_features_num_regression_retorna_none_target_no_numerico():
    df = pd.DataFrame({"target": ["a", "b", "c"], "x": [1, 2, 3]})

    assert plot_features_num_regression(df, target_col="target") is None


def test_plot_features_num_regression_retorna_none_umbral_invalido():
    df = pd.DataFrame({"target": [1, 2, 3], "x": [1, 2, 3]})

    assert plot_features_num_regression(
        df,
        target_col="target",
        umbral_corr=1.5,
    ) is None


def test_plot_features_num_regression_lista_vacia_si_nada_correlado(monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    df = pd.DataFrame({
        "target": [1, 2, 3, 4, 5],
        "ruido": [5, 3, 1, 4, 2],
    })

    resultado = plot_features_num_regression(
        df,
        target_col="target",
        umbral_corr=0.99,
    )

    assert resultado == []


# ══════════════════════════════════════════════
# Tests de get_features_num_regression
# ══════════════════════════════════════════════

def test_get_features_num_regression_detects_correlated_features():
    df = pd.DataFrame({
        "target": [1, 2, 3, 4, 5],
        "feature_corr": [2, 4, 6, 8, 10],
        "feature_no_corr": [5, 3, 1, 3, 5],
        "category": ["a", "b", "a", "b", "a"],
    })

    result = get_features_num_regression(
        df=df,
        target_col="target",
        umbral_corr=0.8,
    )

    assert "feature_corr" in result
    assert "feature_no_corr" not in result
    assert "category" not in result


def test_get_features_num_regression_returns_none_if_target_missing():
    df = pd.DataFrame({"x": [1, 2, 3]})

    result = get_features_num_regression(
        df=df,
        target_col="target",
        umbral_corr=0.3,
    )

    assert result is None


def test_get_features_num_regression_returns_none_if_target_not_numeric():
    df = pd.DataFrame({
        "target": ["a", "b", "c"],
        "x": [1, 2, 3],
    })

    result = get_features_num_regression(
        df=df,
        target_col="target",
        umbral_corr=0.3,
    )

    assert result is None


def test_get_features_num_regression_ignores_constant_features():
    df = pd.DataFrame({
        "target": [1, 2, 3, 4, 5],
        "x_constant": [1, 1, 1, 1, 1],
        "x_corr": [10, 20, 30, 40, 50],
    })

    result = get_features_num_regression(
        df=df,
        target_col="target",
        umbral_corr=0.5,
    )

    assert "x_constant" not in result
    assert "x_corr" in result


def test_get_features_num_regression_filters_by_pvalue():
    df = pd.DataFrame({
        "target": [1, 2, 3, 4, 5, 6, 7, 8],
        "x_corr": [2, 4, 6, 8, 10, 12, 14, 16],
        "x_random": [8, 1, 6, 3, 7, 2, 5, 4],
    })

    result = get_features_num_regression(
        df=df,
        target_col="target",
        umbral_corr=0.8,
        pvalue=0.05,
    )

    assert "x_corr" in result


def test_get_features_num_regression_returns_none_if_df_is_not_dataframe():
    result = get_features_num_regression(
        df=[1, 2, 3, 4],
        target_col="target",
        umbral_corr=0.3,
    )

    assert result is None


def test_get_features_num_regression_returns_none_if_umbral_corr_invalid():
    df = pd.DataFrame({
        "target": [1, 2, 3],
        "x": [2, 4, 6],
    })

    result = get_features_num_regression(
        df=df,
        target_col="target",
        umbral_corr=1.5,
    )

    assert result is None


def test_get_features_num_regression_returns_none_if_pvalue_invalid():
    df = pd.DataFrame({
        "target": [1, 2, 3],
        "x": [2, 4, 6],
    })

    result = get_features_num_regression(
        df=df,
        target_col="target",
        umbral_corr=0.3,
        pvalue=1.5,
    )

    assert result is None

# ══════════════════════════════════════════════
# Tests de plot_features_cat_regression
# ══════════════════════════════════════════════

def test_plot_features_cat_regression_devuelve_lista(monkeypatch):
    """Caso correcto: devuelve una lista con columnas categóricas."""
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    df = pd.DataFrame({
        "target": [1, 2, 3, 4, 5, 6],
        "categoria": ["a", "b", "a", "b", "a", "b"],
    })
    resultado = plot_features_cat_regression(df, target_col="target", columns=["categoria"])
    assert isinstance(resultado, list)
    assert "categoria" in resultado


def test_plot_features_cat_regression_retorna_none_df_invalido():
    """Caso de error: df no es DataFrame → retorna None."""
    assert plot_features_cat_regression("no es un df", target_col="target") is None


def test_plot_features_cat_regression_retorna_none_df_vacio():
    """Caso de error: DataFrame vacío → retorna None."""
    assert plot_features_cat_regression(pd.DataFrame(), target_col="target") is None


def test_plot_features_cat_regression_retorna_none_target_no_existe():
    """Caso de error: target_col no existe → retorna None."""
    df = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
    assert plot_features_cat_regression(df, target_col="no_existe") is None


def test_plot_features_cat_regression_retorna_none_target_no_numerico():
    """Caso de error: target_col no es numérica → retorna None."""
    df = pd.DataFrame({"target": ["a", "b"], "x": ["c", "d"]})
    assert plot_features_cat_regression(df, target_col="target") is None


def test_plot_features_cat_regression_retorna_none_pvalue_invalido():
    """Caso de error: pvalue fuera de rango → retorna None."""
    df = pd.DataFrame({"target": [1, 2, 3], "cat": ["a", "b", "a"]})
    assert plot_features_cat_regression(df, target_col="target", pvalue=1.5) is None
