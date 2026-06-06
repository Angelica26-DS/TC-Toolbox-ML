import pytest
import pandas as pd
import numpy as np
from toolbox_ml.eda.core import describe_df, tipifica_variables, plot_features_num_regression


# ══════════════════════════════════════════════
# Tests de describe_df
# ══════════════════════════════════════════════

def test_describe_df_devuelve_dataframe():
    """Caso correcto: input válido → retorna un DataFrame."""
    df = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
    resultado = describe_df(df)
    assert isinstance(resultado, pd.DataFrame)


def test_describe_df_columnas_correctas():
    """El DataFrame resultado tiene exactamente las columnas esperadas."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    resultado = describe_df(df)
    assert set(resultado.columns) == {
        "tipo", "porcentaje_nulos", "valores_unicos", "porcentaje_cardinalidad"
    }


def test_describe_df_porcentaje_nulos_correcto():
    """Calcula correctamente el porcentaje de nulos."""
    df = pd.DataFrame({"a": [1, None, None, None]})
    resultado = describe_df(df)
    assert resultado.loc["a", "porcentaje_nulos"] == pytest.approx(75.0, abs=0.01)


def test_describe_df_valores_unicos_correcto():
    """Cuenta bien los valores únicos (sin contar NaN)."""
    df = pd.DataFrame({"col": [1, 2, 2, None]})
    resultado = describe_df(df)
    assert resultado.loc["col", "valores_unicos"] == 2


def test_describe_df_retorna_none_con_input_invalido():
    """Caso de error: input no es DataFrame → retorna None."""
    assert describe_df("esto no es un dataframe") is None
    assert describe_df([1, 2, 3]) is None
    assert describe_df(None) is None


def test_describe_df_indice_son_nombres_de_columnas():
    """El índice del resultado son los nombres de las columnas originales."""
    df = pd.DataFrame({"edad": [25, 30], "ciudad": ["Madrid", "Barcelona"]})
    resultado = describe_df(df)
    assert list(resultado.index) == ["edad", "ciudad"]


# ══════════════════════════════════════════════
# Tests de tipifica_variables
# ══════════════════════════════════════════════

def test_tipifica_variables_devuelve_dataframe():
    """Caso correcto: input válido → retorna un DataFrame."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
    resultado = tipifica_variables(df, umbral_categoria=10, umbral_continua=50.0)
    assert isinstance(resultado, pd.DataFrame)


def test_tipifica_variables_columnas_correctas():
    """El resultado tiene las columnas 'nombre_variable' y 'tipo_sugerido'."""
    df = pd.DataFrame({"a": [1, 2]})
    resultado = tipifica_variables(df, umbral_categoria=10, umbral_continua=50.0)
    assert set(resultado.columns) == {"nombre_variable", "tipo_sugerido"}


def test_tipifica_variables_detecta_binaria():
    """Una columna con 2 valores únicos debe clasificarse como 'Binaria'."""
    df = pd.DataFrame({"sexo": ["hombre", "mujer", "hombre", "mujer"]})
    resultado = tipifica_variables(df, umbral_categoria=10, umbral_continua=50.0)
    fila = resultado[resultado["nombre_variable"] == "sexo"]
    assert fila["tipo_sugerido"].values[0] == "Binaria"


def test_tipifica_variables_detecta_categorica():
    """Una columna con pocos valores únicos debe ser 'Categórica'."""
    df = pd.DataFrame({"color": ["rojo", "azul", "verde", "rojo", "azul", "verde"]})
    resultado = tipifica_variables(df, umbral_categoria=10, umbral_continua=50.0)
    fila = resultado[resultado["nombre_variable"] == "color"]
    assert fila["tipo_sugerido"].values[0] == "Categórica"


def test_tipifica_variables_detecta_numerica_continua():
    """Una columna con alta cardinalidad debe ser 'Numérica Continua'."""
    df = pd.DataFrame({"precio": list(range(100))})
    resultado = tipifica_variables(df, umbral_categoria=10, umbral_continua=50.0)
    fila = resultado[resultado["nombre_variable"] == "precio"]
    assert fila["tipo_sugerido"].values[0] == "Numérica Continua"


def test_tipifica_variables_retorna_none_si_df_invalido():
    """Caso de error: df no es DataFrame → retorna None."""
    assert tipifica_variables("no es un df", 10, 50.0) is None


def test_tipifica_variables_retorna_none_si_umbral_categoria_invalido():
    """Caso de error: umbral_categoria no es entero positivo → retorna None."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert tipifica_variables(df, -5, 50.0) is None
    assert tipifica_variables(df, 0, 50.0) is None


def test_tipifica_variables_retorna_none_si_umbral_continua_invalido():
    """Caso de error: umbral_continua fuera de rango → retorna None."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert tipifica_variables(df, 10, -1.0) is None
    assert tipifica_variables(df, 10, 150.0) is None
