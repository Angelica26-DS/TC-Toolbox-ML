# Importamos pandas para crear DataFrames de prueba
import pandas as pd

# Importamos pytest para comprobar que se lanzan errores correctamente
import pytest

# Importamos la función que queremos testear
# OJO: este import puede cambiar según dónde esté guardada la función en el proyecto
from toolbox_ml import get_features_num_regression


def test_get_features_num_regression_detects_correlated_features():
    """
    Comprobamos que la función detecta correctamente una variable numérica
    muy correlacionada con la variable objetivo.
    """

    # Creamos un DataFrame de ejemplo
    df = pd.DataFrame({
        # Variable objetivo numérica
        "target": [1, 2, 3, 4, 5],

        # Esta variable está perfectamente correlacionada con target
        "feature_corr": [2, 4, 6, 8, 10],

        # Esta variable no tiene una correlación lineal clara con target
        "feature_no_corr": [5, 3, 1, 3, 5],

        # Esta variable es categórica y no debería tenerse en cuenta
        "category": ["a", "b", "a", "b", "a"]
    })

    # Ejecutamos la función
    result = get_features_num_regression(
        df=df,
        target_col="target",
        umbral_corr=0.8
    )

    # Comprobamos que la variable correlacionada aparece en el resultado
    assert "feature_corr" in result

    # Comprobamos que la variable categórica no aparece en el resultado
    assert "category" not in result


def test_get_features_num_regression_raises_error_if_target_missing():
    """
    Comprobamos que la función lanza un error si la variable objetivo
    no existe en el DataFrame.
    """

    # Creamos un DataFrame sin columna target
    df = pd.DataFrame({
        "x": [1, 2, 3]
    })

    # Comprobamos que se lanza ValueError al pasar una columna inexistente
    with pytest.raises(ValueError):
        get_features_num_regression(df, "target")


def test_get_features_num_regression_raises_error_if_target_not_numeric():
    """
    Comprobamos que la función lanza un error si la variable objetivo
    no es numérica.
    """

    # Creamos un DataFrame con target categórico
    df = pd.DataFrame({
        "target": ["a", "b", "c"],
        "x": [1, 2, 3]
    })

    # Como target no es numérico, la función debe lanzar ValueError
    with pytest.raises(ValueError):
        get_features_num_regression(df, "target")


def test_get_features_num_regression_ignores_constant_features():
    """
    Comprobamos que la función ignora variables constantes.
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
    Comprobamos que la función acepta el argumento pvalue
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