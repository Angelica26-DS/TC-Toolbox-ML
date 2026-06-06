"""
Core EDA and Machine Learning helper functions for toolbox_ml.
"""

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

    # Validación 2: umbral_categoria tiene que ser un entero positivo
    if not isinstance(umbral_categoria, int) or umbral_categoria <= 0:
        print("Error: 'umbral_categoria' debe ser un entero positivo.")
        return None

    # Validación 3: umbral_continua tiene que ser un float entre 0 y 100
    if not isinstance(umbral_continua, (int, float)) or not (0 <= umbral_continua <= 100):
        print("Error: 'umbral_continua' debe ser un número entre 0 y 100.")
        return None

    total_filas = len(df)
    tipos_sugeridos = []

    for col in df.columns:
        cardinalidad = df[col].nunique()
        porcentaje_cardinalidad = (cardinalidad / total_filas) * 100

        # Lógica en cascada según la guía
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
