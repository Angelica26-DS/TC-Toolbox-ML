"""
Core EDA and Machine Learning helper functions for toolbox_ml.
"""

# ──────────────────────────────────────────────
# Función 3: plot_features_num_regression
# ──────────────────────────────────────────────

def plot_features_num_regression(
    df: pd.DataFrame,
    target_col: str = "",
    columns: list = [],
    umbral_corr: float = 0,
    pvalue: float = None
) -> list:
    """
    Pinta pairplots de las columnas numéricas que tengan correlación
    significativa con la variable target.

    Argumentos:
        df (pd.DataFrame): DataFrame con los datos.
        target_col (str): nombre de la columna target (numérica).
        columns (list): lista de columnas candidatas. Si está vacía,
            se usan todas las columnas numéricas del DataFrame.
        umbral_corr (float): umbral mínimo de correlación en valor absoluto (0-1).
        pvalue (float o None): si se indica, solo se incluyen columnas cuyo
            p-valor sea menor que este umbral.

    Retorna:
        list: lista de columnas que cumplen los criterios y se han pintado.
        Retorna None si alguna validación falla.
    """

    # Validación 1: df tiene que ser un DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: el argumento 'df' debe ser un pd.DataFrame.")
        return None

    # Validación 2: target_col tiene que existir en el DataFrame
    if target_col not in df.columns:
        print(f"Error: '{target_col}' no existe en el DataFrame.")
        return None

    # Validación 3: target_col tiene que ser numérica
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: '{target_col}' debe ser una columna numérica.")
        return None

    # Validación 4: umbral_corr tiene que estar entre 0 y 1
    if not isinstance(umbral_corr, (int, float)) or not (0 <= umbral_corr <= 1):
        print("Error: 'umbral_corr' debe ser un número entre 0 y 1.")
        return None

    # Validación 5: pvalue si se indica tiene que estar entre 0 y 1
    if pvalue is not None:
        if not isinstance(pvalue, (int, float)) or not (0 <= pvalue <= 1):
            print("Error: 'pvalue' debe ser un número entre 0 y 1, o None.")
            return None

    # Si no se pasan columnas, usamos todas las numéricas excepto el target
    if not columns:
        columnas_candidatas = [
            col for col in df.select_dtypes(include="number").columns
            if col != target_col
        ]
    else:
        columnas_candidatas = [
            col for col in columns
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and col != target_col
        ]

    # Filtramos por correlación y p-valor
    columnas_seleccionadas = []

    for col in columnas_candidatas:
        datos_validos = df[[col, target_col]].dropna()
        corr, p = stats.pearsonr(datos_validos[col], datos_validos[target_col])

        if abs(corr) >= umbral_corr:
            if pvalue is None or p < pvalue:
                columnas_seleccionadas.append(col)

    # Si no hay columnas que cumplan los criterios avisamos y salimos
    if not columnas_seleccionadas:
        print("No hay columnas que cumplan los criterios de correlación.")
        return columnas_seleccionadas

    # Pintamos pairplots en grupos de 5
    TAMANO_GRUPO = 5

    for i in range(0, len(columnas_seleccionadas), TAMANO_GRUPO):
        grupo = columnas_seleccionadas[i: i + TAMANO_GRUPO]
        columnas_plot = [target_col] + grupo

        sns.pairplot(df[columnas_plot].dropna())
        plt.suptitle(f"Pairplot — grupo {i // TAMANO_GRUPO + 1}", y=1.02)
        plt.show()

    return columnas_seleccionadas
