"""
Core EDA and Machine Learning helper functions for toolbox_ml.
"""

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
        target_col (str): nombre de la columna target numérica.
        columns (list): lista de columnas candidatas. Si está vacía,
            se usan todas las columnas numéricas del DataFrame.
        umbral_corr (float): umbral mínimo de correlación en valor absoluto.
        pvalue (float o None): si se indica, solo se incluyen columnas cuyo
            p-value sea menor que este umbral.

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
            if col in df.columns
            and pd.api.types.is_numeric_dtype(df[col])
            and col != target_col
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


# ──────────────────────────────────────────────
# Función 4: get_features_num_regression
# ──────────────────────────────────────────────

def get_features_num_regression(
    df: pd.DataFrame,
    target_col: str,
    umbral_corr: float,
    pvalue: float = None
) -> list:
    """
    Identifica variables numéricas con correlación significativa respecto a
    una variable objetivo numérica usando correlación de Pearson.

    Argumentos:
        df (pd.DataFrame): DataFrame con los datos.
        target_col (str): nombre de la variable objetivo numérica.
        umbral_corr (float): umbral mínimo de correlación absoluta.
        pvalue (float o None): valor máximo opcional de p-value.

    Retorna:
        list: lista de variables numéricas que cumplen los criterios.
        Retorna None si las validaciones no se cumplen.
    """

    # Validamos que df sea un DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: el argumento 'df' debe ser un pd.DataFrame.")
        return None

    # Validamos que target_col sea un string
    if not isinstance(target_col, str):
        print("Error: 'target_col' debe ser un string.")
        return None

    # Validamos que target_col exista en el DataFrame
    if target_col not in df.columns:
        print(f"Error: la columna objetivo '{target_col}' no existe en el DataFrame.")
        return None

    # Validamos que target_col sea numérica
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: la variable objetivo '{target_col}' debe ser numérica.")
        return None

    # Validamos que umbral_corr sea numérico
    if not isinstance(umbral_corr, (int, float)):
        print("Error: 'umbral_corr' debe ser un número.")
        return None

    # Validamos que umbral_corr esté entre 0 y 1
    if umbral_corr < 0 or umbral_corr > 1:
        print("Error: 'umbral_corr' debe estar entre 0 y 1.")
        return None

    # Validamos que pvalue sea numérico si se informa
    if pvalue is not None and not isinstance(pvalue, (int, float)):
        print("Error: 'pvalue' debe ser un número o None.")
        return None

    # Validamos que pvalue esté entre 0 y 1 si se informa
    if pvalue is not None and (pvalue < 0 or pvalue > 1):
        print("Error: 'pvalue' debe estar entre 0 y 1.")
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
        corr, p_val = stats.pearsonr(temp_df[col], temp_df[target_col])

        # Comprobamos si supera el umbral de correlación
        cumple_corr = abs(corr) >= umbral_corr

        # Comprobamos el p-value solo si se ha indicado
        cumple_pvalue = pvalue is None or p_val <= pvalue

        # Si cumple ambos filtros, guardamos la columna
        if cumple_corr and cumple_pvalue:
            selected_features.append(col)

    return selected_features



# ──────────────────────────────────────────────
# Función 5: get_features_cat_regression
# ──────────────────────────────────────────────

def get_features_cat_regression(
    df: pd.DataFrame,
    target_col: str,
    pvalue: float = 0.05
) -> list:
    """
    Identifica variables categóricas que tienen una relación estadísticamente
    significativa con una variable objetivo numérica.

    La función selecciona automáticamente el test estadístico en función del
    número de categorías de cada variable categórica:

    - Mann-Whitney U si la variable tiene exactamente 2 categorías.
    - ANOVA de un factor si la variable tiene más de 2 categorías.

    Argumentos:
        df (pd.DataFrame): DataFrame con los datos.
        target_col (str): nombre de la variable objetivo numérica.
        pvalue (float): nivel máximo de p-value para considerar una variable
            como significativa. Por defecto es 0.05.

    Retorna:
        list: lista con los nombres de las variables categóricas significativas.
        Retorna None si alguna validación de entrada falla.
    """

    # Validamos que df sea un DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: el argumento 'df' debe ser un pd.DataFrame.")
        return None

    # Validamos que target_col sea un string
    if not isinstance(target_col, str):
        print("Error: 'target_col' debe ser un string.")
        return None

    # Validamos que target_col exista en el DataFrame
    if target_col not in df.columns:
        print(f"Error: la columna objetivo '{target_col}' no existe en el DataFrame.")
        return None

    # Validamos que target_col sea numérica
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: la variable objetivo '{target_col}' debe ser numérica.")
        return None

    # Validamos que pvalue sea numérico
    if not isinstance(pvalue, (int, float)):
        print("Error: 'pvalue' debe ser un número.")
        return None

    # Validamos que pvalue esté entre 0 y 1
    if pvalue < 0 or pvalue > 1:
        print("Error: 'pvalue' debe estar entre 0 y 1.")
        return None

    # Lista donde guardaremos las variables categóricas significativas
    selected_features = []

    # Seleccionamos automáticamente columnas categóricas
    categorical_cols = df.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()

    # Recorremos cada columna categórica candidata
    for col in categorical_cols:

        # Eliminamos nulos solo en la columna categórica y el target
        temp_df = df[[col, target_col]].dropna()

        # Si no hay datos suficientes, saltamos la columna
        if len(temp_df) < 2:
            continue

        # Obtenemos las categorías únicas de la variable
        categories = temp_df[col].unique()

        # Si tiene menos de 2 categorías, no se puede comparar
        if len(categories) < 2:
            continue

        # Creamos los grupos de valores del target según cada categoría
        groups = [
            temp_df[temp_df[col] == category][target_col]
            for category in categories
        ]

        # Eliminamos grupos con menos de 2 observaciones
        groups = [
            group
            for group in groups
            if len(group) >= 2
        ]

        # Si después de filtrar quedan menos de 2 grupos, no se puede comparar
        if len(groups) < 2:
            continue

        # Si la variable tiene exactamente 2 categorías, usamos Mann-Whitney U
        if len(groups) == 2:
            _, p_val = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")

        # Si la variable tiene más de 2 categorías, usamos ANOVA
        else:
            _, p_val = stats.f_oneway(*groups)

        # Si el p-value es menor que el umbral, guardamos la variable
        if p_val < pvalue:
            selected_features.append(col)

    return selected_features


def plot_features_cat_regression(df, target_col="", columns=[], pvalue=None, with_individual_plot=False):
    import matplotlib.pyplot as plt
    from scipy import stats

    if not isinstance(df, pd.DataFrame):
        print("Error: el argumento 'df' debe ser un pd.DataFrame.")
        return None
    if df.empty:
        print("Error: el DataFrame está vacío.")
        return None
    if target_col not in df.columns:
        print(f"Error: '{target_col}' no existe en el DataFrame.")
        return None
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: '{target_col}' debe ser una columna numérica.")
        return None
    if pvalue is not None:
        if not isinstance(pvalue, (int, float)) or not (0 < pvalue <= 1):
            print("Error: 'pvalue' debe ser un número entre 0 y 1, o None.")
            return None

    if not columns:
        columnas_candidatas = [col for col in df.select_dtypes(include=["object", "category"]).columns if col != target_col]
    else:
        columnas_candidatas = [col for col in columns if col in df.columns and col != target_col]

    columnas_seleccionadas = []
    for col in columnas_candidatas:
        datos_validos = df[[col, target_col]].dropna()
        grupos = [g[target_col].values for _, g in datos_validos.groupby(col)]
        if len(grupos) < 2:
            continue
        if pvalue is not None:
            if len(grupos) == 2:
                _, p = stats.mannwhitneyu(grupos[0], grupos[1], alternative="two-sided")
            else:
                _, p = stats.kruskal(*grupos)
            if p >= pvalue:
                continue
        columnas_seleccionadas.append(col)

    if not columnas_seleccionadas:
        print("No hay columnas que cumplan los criterios.")
        return columnas_seleccionadas

    for col in columnas_seleccionadas:
        datos_validos = df[[col, target_col]].dropna()
        if with_individual_plot:
            categorias = datos_validos[col].unique()
            fig, axes = plt.subplots(1, len(categorias), figsize=(5 * len(categorias), 4))
            if len(categorias) == 1:
                axes = [axes]
            for ax, cat in zip(axes, categorias):
                ax.hist(datos_validos[datos_validos[col] == cat][target_col], bins=20, edgecolor="black")
                ax.set_title(f"{col} = {cat}")
                ax.set_xlabel(target_col)
                ax.set_ylabel("Frecuencia")
            plt.tight_layout()
            plt.show()
        else:
            fig, ax = plt.subplots(figsize=(10, 4))
            for cat, grupo in datos_validos.groupby(col):
                ax.hist(grupo[target_col], bins=20, alpha=0.5, label=str(cat), edgecolor="black")
            ax.set_title(f"Distribución de {target_col} por {col}")
            ax.set_xlabel(target_col)
            ax.set_ylabel("Frecuencia")
            ax.legend()
            plt.tight_layout()
            plt.show()

    return columnas_seleccionadas
