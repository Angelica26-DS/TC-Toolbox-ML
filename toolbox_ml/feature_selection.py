# Importamos List y Optional para añadir type hints a la función
from typing import List, Optional

# Importamos pandas para trabajar con DataFrames
import pandas as pd

# Importamos pearsonr para calcular correlación de Pearson y p-value
from scipy.stats import pearsonr


def get_features_num_regression(
    df: pd.DataFrame,
    target_col: str,
    umbral_corr: float = 0.3,
    pvalue: Optional[float] = None
) -> List[str]:
    """
    Identifica variables numéricas que tienen una correlación significativa
    con una variable objetivo numérica.

    La función analiza todas las columnas numéricas del DataFrame, excepto
    la columna objetivo, y calcula la correlación de Pearson entre cada una
    de ellas y la variable objetivo.

    Además, permite filtrar por:
    - Umbral mínimo de correlación absoluta.
    - P-value máximo opcional.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        target_col (str): Nombre de la columna objetivo numérica.
        umbral_corr (float): Valor mínimo absoluto de correlación para aceptar
            una variable. Por defecto es 0.3.
        pvalue (Optional[float]): Valor máximo de p-value permitido.
            Si es None, no se aplica filtro por p-value.

    Returns:
        List[str]: Lista con los nombres de las variables numéricas que cumplen
        los criterios de correlación y p-value.

    Raises:
        TypeError: Si df no es un DataFrame.
        TypeError: Si target_col no es un string.
        TypeError: Si umbral_corr no es numérico.
        TypeError: Si pvalue no es numérico ni None.
        ValueError: Si target_col no existe en el DataFrame.
        ValueError: Si target_col no es una variable numérica.
        ValueError: Si umbral_corr no está entre 0 y 1.
        ValueError: Si pvalue no está entre 0 y 1.
    """

    # Validamos que df sea realmente un DataFrame de pandas
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df debe ser un pandas DataFrame.")

    # Validamos que target_col sea un string
    if not isinstance(target_col, str):
        raise TypeError("target_col debe ser un string.")

    # Validamos que la columna objetivo exista dentro del DataFrame
    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no existe en el DataFrame.")

    # Validamos que la columna objetivo sea numérica
    # La correlación de Pearson solo tiene sentido con variables numéricas
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError("La variable objetivo debe ser numérica.")

    # Validamos que umbral_corr sea un número entero o decimal
    if not isinstance(umbral_corr, (int, float)):
        raise TypeError("umbral_corr debe ser un número.")

    # Validamos que umbral_corr esté entre 0 y 1
    # Porque la correlación absoluta va de 0 a 1
    if umbral_corr < 0 or umbral_corr > 1:
        raise ValueError("umbral_corr debe estar entre 0 y 1.")

    # Si el usuario pasa un pvalue, validamos que sea numérico
    if pvalue is not None and not isinstance(pvalue, (int, float)):
        raise TypeError("pvalue debe ser un número o None.")

    # Si el usuario pasa un pvalue, validamos que esté entre 0 y 1
    if pvalue is not None and (pvalue < 0 or pvalue > 1):
        raise ValueError("pvalue debe estar entre 0 y 1.")

    # Creamos una lista vacía donde guardaremos las variables seleccionadas
    selected_features = []

    # Obtenemos todas las columnas numéricas del DataFrame
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Quitamos la variable objetivo de la lista de columnas numéricas
    # No queremos calcular la correlación de target consigo misma
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    # Recorremos cada columna numérica candidata
    for col in numeric_cols:

        # Creamos un DataFrame temporal solo con la columna candidata y el target
        # Eliminamos filas con valores nulos para evitar errores en pearsonr
        temp_df = df[[col, target_col]].dropna()

        # Si después de eliminar nulos hay menos de 2 filas, no se puede calcular correlación
        if len(temp_df) < 2:
            continue

        # Si la variable tiene un único valor, no aporta información
        # Además, pearsonr falla con variables constantes
        if temp_df[col].nunique() <= 1:
            continue

        # Si el target tiene un único valor, tampoco se puede calcular correlación
        if temp_df[target_col].nunique() <= 1:
            continue

        # Calculamos la correlación de Pearson y su p-value
        corr, p_val = pearsonr(temp_df[col], temp_df[target_col])

        # Comprobamos si la correlación absoluta supera el umbral indicado
        cumple_corr = abs(corr) >= umbral_corr

        # Comprobamos si cumple el filtro de p-value
        # Si pvalue es None, significa que no queremos filtrar por p-value
        cumple_pvalue = pvalue is None or p_val <= pvalue

        # Si cumple ambos criterios, añadimos la variable a la lista final
        if cumple_corr and cumple_pvalue:
            selected_features.append(col)

    # Devolvemos la lista con las variables seleccionadas
    return selected_features