"""
Unit tests for toolbox_ml.eda.core.
"""

# Importamos pandas para crear DataFrames de prueba
import pandas as pd

# Importamos la función que queremos testear desde la estructura final del proyecto
from toolbox_ml.eda.core import get_features_num_regression


def test_get_features_num_regression_detects_correlated_features():
    """
    Comprueba que la función detecta correctamente una variable numérica
    muy correlacionada con la variable objetivo.
    """

    # Creamos un DataFrame de ejemplo controlado
    df = pd.DataFrame({
        # Variable objetivo numérica
        "target": [1, 2, 3, 4, 5],

        # Esta variable está perfectamente correlacionada con target
        "feature_corr": [2, 4, 6, 8, 10],

        # Esta variable no tiene una correlación lineal fuerte con target
        "feature_no_corr": [5, 3, 1, 3, 5],

        # Esta variable es categórica y no debería tenerse en cuenta
        "category": ["a", "b", "a", "b", "a"]
    })

    # Ejecutamos la función con un umbral alto de correlación
    result = get_features_num_regression(
        df=df,
        target_col="target",
        umbral_corr=0.8
    )

    # Comprobamos que la variable correlacionada aparece en el resultado
    assert "feature_corr" in result

    # Comprobamos que la variable numérica poco correlacionada no aparece
    assert "feature_no_corr" not in result

    # Comprobamos que la variable categórica no aparece en el resultado
    assert "category" not in result


def test_get_features_num_regression_returns_none_if_target_missing():
    """
    Comprueba que la función devuelve None si la variable objetivo
    no existe en el DataFrame.
    """

    # Creamos un DataFrame sin columna target
    df = pd.DataFrame({
        "x": [1, 2, 3]
    })

    # Ejecutamos la función con una columna objetivo inexistente
    result = get_features_num_regression(
        df=df,
        target_col="target",
        umbral_corr=0.3
    )

    # La función debe devolver None en vez de lanzar un error
    assert result is None


def test_get_features_num_regression_returns_none_if_target_not_numeric():
    """
    Comprueba que la función devuelve None si la variable objetivo
    no es numérica.
    """

    # Creamos un DataFrame con target categórico
    df = pd.DataFrame({
        "target": ["a", "b", "c"],
        "x": [1, 2, 3]
    })

    # Ejecutamos la función con un target no numérico
    result = get_features_num_regression(
        df=df,
        target_col="target",
        umbral_corr=0.3
    )

    # La función debe devolver None porque target no es numérico
    assert result is None


def test_get_features_num_regression_ignores_constant_features():
    """
    Comprueba que la función ignora variables constantes.
    Las variables constantes no sirven para calcular correlación.
    """

    # Creamos un DataFrame donde x_constant tiene siempre el mismo valor
    df = pd.DataFrame({
        "target": [1, 2, 3, 4, 5],
        "x_constant": [1, 1, 1, 1, 1],
        "x_corr": [10, 20, 30, 40, 50]
    })

    # Ejecutamos la función
    result = get_features_num_regression(
        df=df,
        target_col="target",
        umbral_corr=0.5
    )

    # La variable constante no debe estar en el resultado
    assert "x_constant" not in result

    # La variable correlacionada sí debe estar en el resultado
    assert "x_corr" in result


def test_get_features_num_regression_filters_by_pvalue():
    """
    Comprueba que la función acepta el argumento pvalue
    y lo usa como filtro estadístico.
    """

    # Creamos un DataFrame con una relación lineal clara
    df = pd.DataFrame({
        "target": [1, 2, 3, 4, 5, 6, 7, 8],
        "x_corr": [2, 4, 6, 8, 10, 12, 14, 16],
        "x_random": [8, 1, 6, 3, 7, 2, 5, 4]
    })

    # Ejecutamos la función usando filtro de p-value
    result = get_features_num_regression(
        df=df,
        target_col="target",
        umbral_corr=0.8,
        pvalue=0.05
    )

    # La variable con correlación clara debería aparecer
    assert "x_corr" in result


def test_get_features_num_regression_returns_none_if_df_is_not_dataframe():
    """
    Comprueba que la función devuelve None si df no es un DataFrame.
    """

    # Pasamos una lista en vez de un DataFrame
    df = [1, 2, 3, 4]

    # Ejecutamos la función
    result = get_features_num_regression(
        df=df,
        target_col="target",
        umbral_corr=0.3
    )

    # La función debe devolver None porque df no es un DataFrame
    assert result is None


def test_get_features_num_regression_returns_none_if_umbral_corr_invalid():
    """
    Comprueba que la función devuelve None si umbral_corr
    no está entre 0 y 1.
    """

    # Creamos un DataFrame válido
    df = pd.DataFrame({
        "target": [1, 2, 3],
        "x": [2, 4, 6]
    })

    # Ejecutamos la función con un umbral inválido
    result = get_features_num_regression(
        df=df,
        target_col="target",
        umbral_corr=1.5
    )

    # La función debe devolver None porque el umbral no es válido
    assert result is None


def test_get_features_num_regression_returns_none_if_pvalue_invalid():
    """
    Comprueba que la función devuelve None si pvalue
    no está entre 0 y 1.
    """

    # Creamos un DataFrame válido
    df = pd.DataFrame({
        "target": [1, 2, 3],
        "x": [2, 4, 6]
    })

    # Ejecutamos la función con un pvalue inválido
    result = get_features_num_regression(
        df=df,
        target_col="target",
        umbral_corr=0.3,
        pvalue=1.5
    )

    # La función debe devolver None porque el pvalue no es válido
    assert result is None