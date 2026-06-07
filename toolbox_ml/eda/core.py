"""
Core EDA and Machine Learning helper functions for toolbox_ml.
"""

from scipy.stats import pearsonr
import pandas as pd


def get_features_num_regression(df, target_col, umbral_corr, pvalue=None) -> list:
    """
    Identifica variables numéricas con correlación significativa respecto a
    una variable objetivo numérica usando correlación de Pearson.

    Args:
        df: DataFrame con los datos.
        target_col: Nombre de la variable objetivo numérica.
        umbral_corr: Umbral mínimo de correlación absoluta.
        pvalue: Valor máximo opcional de p-value.

    Returns:
        Lista de variables numéricas que cumplen los criterios.
        Devuelve None si las validaciones no se cumplen.
    """

    # Validamos que df sea un DataFrame
    if not isinstance(df, pd.DataFrame):
        print("df debe ser un pandas DataFrame.")
        return None

    # Validamos que target_col sea un string
    if not isinstance(target_col, str):
        print("target_col debe ser un string.")
        return None

    # Validamos que target_col exista en el DataFrame
    if target_col not in df.columns:
        print(f"La columna objetivo '{target_col}' no existe en el DataFrame.")
        return None

    # Validamos que target_col sea numérica
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("La variable objetivo debe ser numérica.")
        return None

    # Validamos que umbral_corr sea numérico
    if not isinstance(umbral_corr, (int, float)):
        print("umbral_corr debe ser un número.")
        return None

    # Validamos que umbral_corr esté entre 0 y 1
    if umbral_corr < 0 or umbral_corr > 1:
        print("umbral_corr debe estar entre 0 y 1.")
        return None

    # Validamos que pvalue sea numérico si se informa
    if pvalue is not None and not isinstance(pvalue, (int, float)):
        print("pvalue debe ser un número o None.")
        return None

    # Validamos que pvalue esté entre 0 y 1 si se informa
    if pvalue is not None and (pvalue < 0 or pvalue > 1):
        print("pvalue debe estar entre 0 y 1.")
        return None

    # Lista donde guardaremos las variables seleccionadas
    selected_features = []

    # Seleccionamos automáticamente las columnas numéricas
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Quitamos el target para no compararlo consigo mismo
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    # Recorremos cada columna numérica candidata
    for col in numeric_cols:

        # Eliminamos nulos solo en la pareja columna-target
        temp_df = df[[col, target_col]].dropna()

        # Si no hay datos suficientes, saltamos esa columna
        if len(temp_df) < 2:
            continue

        # Si la columna es constante, no se puede calcular correlación
        if temp_df[col].nunique() <= 1:
            continue

        # Si el target es constante, tampoco se puede calcular correlación
        if temp_df[target_col].nunique() <= 1:
            continue

        # Calculamos correlación de Pearson y p-value
        corr, p_val = pearsonr(temp_df[col], temp_df[target_col])

        # Comprobamos si supera el umbral de correlación
        cumple_corr = abs(corr) >= umbral_corr

        # Comprobamos el p-value solo si se ha indicado
        cumple_pvalue = pvalue is None or p_val <= pvalue

        # Si cumple ambos filtros, guardamos la columna
        if cumple_corr and cumple_pvalue:
            selected_features.append(col)

    return selected_features