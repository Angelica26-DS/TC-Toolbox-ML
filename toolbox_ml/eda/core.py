import pandas as pd


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

    # Construimos el resultado fila a fila
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

    # Validación 1: df tiene que ser un DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: el argumento 'df' debe ser un pd.DataFrame.")
        return None

    # Validación 2: el DataFrame no puede estar vacío
    if df.empty:
        print("Error: el DataFrame está vacío.")
        return None

    # Validación 3: umbral_categoria tiene que ser un entero positivo
    if not isinstance(umbral_categoria, int) or umbral_categoria <= 0:
        print("Error: 'umbral_categoria' debe ser un entero positivo.")
        return None

    # Validación 4: umbral_continua tiene que ser float entre 0 y 100
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
