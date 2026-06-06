import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# ──────────────────────────────────────────────
# Función 1: describe_df
# ──────────────────────────────────────────────

def describe_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un resumen descriptivo de un DataFrame.

    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.

    Retorna:
        pd.DataFrame: DataFrame con una fila por columna del input y las
        siguientes columnas: 'tipo', 'porcentaje_nulos', 'valores_unicos',
        'porcentaje_cardinalidad'.
        Retorna None si el input no es un DataFrame válido.
    """

    # Validación: el input tiene que ser un DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: el argumento 'df' debe ser un pd.DataFrame.")
        return None

    # Número total de filas para calcular porcentajes
    total_filas = len(df)

    # Construimos el resultado fila a fila
    resultado = pd.DataFrame({
        "tipo": df.dtypes,
        "porcentaje_nulos": (df.isnull().sum() / total_filas) * 100,
        "valores_unicos": df.nunique(),
        "porcentaje_cardinalidad": (df.nunique() / total_filas) * 100,
    })

    return resultado
