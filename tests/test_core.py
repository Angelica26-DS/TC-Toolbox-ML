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
    get_features_cat_regression,
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
# Tests de get_features_cat_regression
# ══════════════════════════════════════════════

def test_get_features_cat_regression_detects_binary_categorical_feature():
    df = pd.DataFrame({
        "target": [10, 11, 12, 13, 100, 101, 102, 103],
        "cat_binary": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "cat_no_relation": ["X", "Y", "X", "Y", "X", "Y", "X", "Y"],
        "num_col": [1, 2, 3, 4, 5, 6, 7, 8],
    })

    result = get_features_cat_regression(
        df=df,
        target_col="target",
        pvalue=0.05,
    )

    assert "cat_binary" in result
    assert "num_col" not in result


def test_get_features_cat_regression_detects_multicategory_feature():
    df = pd.DataFrame({
        "target": [10, 11, 12, 50, 51, 52, 100, 101, 102],
        "cat_multi": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "cat_no_relation": ["X", "Y", "Z", "X", "Y", "Z", "X", "Y", "Z"],
    })

    result = get_features_cat_regression(
        df=df,
        target_col="target",
        pvalue=0.05,
    )

    assert "cat_multi" in result


def test_get_features_cat_regression_returns_none_if_df_is_not_dataframe():
    result = get_features_cat_regression(
        df=[1, 2, 3],
        target_col="target",
        pvalue=0.05,
    )

    assert result is None


def test_get_features_cat_regression_returns_none_if_target_missing():
    df = pd.DataFrame({
        "cat": ["A", "B", "A"],
        "x": [1, 2, 3],
    })

    result = get_features_cat_regression(
        df=df,
        target_col="target",
        pvalue=0.05,
    )

    assert result is None


def test_get_features_cat_regression_returns_none_if_target_not_numeric():
    df = pd.DataFrame({
        "target": ["alto", "bajo", "medio"],
        "cat": ["A", "B", "A"],
    })

    result = get_features_cat_regression(
        df=df,
        target_col="target",
        pvalue=0.05,
    )

    assert result is None


def test_get_features_cat_regression_returns_none_if_pvalue_invalid():
    df = pd.DataFrame({
        "target": [1, 2, 3],
        "cat": ["A", "B", "A"],
    })

    result = get_features_cat_regression(
        df=df,
        target_col="target",
        pvalue=1.5,
    )

    assert result is None


def test_get_features_cat_regression_ignores_constant_categorical_feature():
    df = pd.DataFrame({
        "target": [1, 2, 3, 4],
        "cat_constant": ["A", "A", "A", "A"],
    })

    result = get_features_cat_regression(
        df=df,
        target_col="target",
        pvalue=0.05,
    )

    assert "cat_constant" not in result