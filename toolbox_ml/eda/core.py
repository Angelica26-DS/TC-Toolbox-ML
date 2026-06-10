import pandas as pd
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

    # Validación 1: el input tiene que ser un DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: el argumento 'df' debe ser un pd.DataFrame.")
        return None

    # Validación 2: el DataFrame no puede estar vacío
    if df.empty:
        print("Error: el DataFrame está vacío.")
        return None

    # Número total de filas para calcular porcentajes
    total_filas = len(df)

    resultado = pd.DataFrame({
        "tipo": df.dtypes,
        "porcentaje_nulos": (df.isnull().sum() / total_filas) * 100,
        "valores_unicos": df.nunique(),
        "porcentaje_cardinalidad": (df.nunique() / total_filas) * 100,
    })

    return resultado


# ──────────────────────────────────────────────
# Función 2: tipifica_variables
# ──────────────────────────────────────────────

def tipifica_variables(
    df: pd.DataFrame,
    umbral_categoria: int,
    umbral_continua: float
) -> pd.DataFrame:
    """
    Sugiere el tipo estadístico de cada variable de un DataFrame.

    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.
        umbral_categoria (int): número mínimo de valores únicos para que
            una variable no se considere categórica.
        umbral_continua (float): porcentaje mínimo de cardinalidad (0-100)
            para considerar una variable como numérica continua.

    Retorna:
        pd.DataFrame: DataFrame con columnas 'nombre_variable' y 'tipo_sugerido'.
        Retorna None si alguna validación de entrada falla.
    """

    if not isinstance(df, pd.DataFrame):
        print("Error: el argumento 'df' debe ser un pd.DataFrame.")
        return None

    if df.empty:
        print("Error: el DataFrame está vacío.")
        return None

    if not isinstance(umbral_categoria, int) or umbral_categoria <= 0:
        print("Error: 'umbral_categoria' debe ser un entero positivo.")
        return None

    if not isinstance(umbral_continua, float) or not (0 <= umbral_continua <= 100):
        print("Error: 'umbral_continua' debe ser un float entre 0 y 100.")
        return None

    total_filas = len(df)
    tipos_sugeridos = []

    for col in df.columns:
        cardinalidad = df[col].nunique()
        porcentaje_cardinalidad = (cardinalidad / total_filas) * 100

        if cardinalidad == 2:
            tipo = "Binaria"
        elif cardinalidad < umbral_categoria:
            tipo = "Categórica"
        elif porcentaje_cardinalidad >= umbral_continua:
            tipo = "Numérica Continua"
        else:
            tipo = "Numérica Discreta"

        tipos_sugeridos.append({"nombre_variable": col, "tipo_sugerido": tipo})

    return pd.DataFrame(tipos_sugeridos)


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

    if not isinstance(df, pd.DataFrame):
        print("Error: el argumento 'df' debe ser un pd.DataFrame.")
        return None

    if target_col not in df.columns:
        print(f"Error: '{target_col}' no existe en el DataFrame.")
        return None

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: '{target_col}' debe ser una columna numérica.")
        return None

    if not isinstance(umbral_corr, (int, float)) or not (0 <= umbral_corr <= 1):
        print("Error: 'umbral_corr' debe ser un número entre 0 y 1.")
        return None

    if pvalue is not None:
        if not isinstance(pvalue, (int, float)) or not (0 <= pvalue <= 1):
            print("Error: 'pvalue' debe ser un número entre 0 y 1, o None.")
            return None

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

    columnas_seleccionadas = []

    for col in columnas_candidatas:
        datos_validos = df[[col, target_col]].dropna()

        if len(datos_validos) < 2:
            continue

        if datos_validos[col].std() == 0 or datos_validos[target_col].std() == 0:
            continue

        corr, p = stats.pearsonr(datos_validos[col], datos_validos[target_col])

        if abs(corr) >= umbral_corr:
            if pvalue is None or p < pvalue:
                columnas_seleccionadas.append(col)

    if not columnas_seleccionadas:
        print("No hay columnas que cumplan los criterios de correlación.")
        return columnas_seleccionadas

    TAMANO_GRUPO = 4

    for i in range(0, len(columnas_seleccionadas), TAMANO_GRUPO):
        grupo = columnas_seleccionadas[i: i + TAMANO_GRUPO]
        columnas_plot = [target_col] + grupo

        sns.pairplot(df[columnas_plot].dropna())
        plt.suptitle(f"Pairplot — grupo {i // TAMANO_GRUPO + 1}", y=1.02)
        plt.show()

    return columnas_seleccionadas
