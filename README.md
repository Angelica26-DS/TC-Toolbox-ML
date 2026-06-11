# 🚀 Toolbox ML

## 🔗 Repositorio

https://github.com/Angelica26-DS/TC-Toolbox-ML

## 📖 Descripción

Toolbox ML es una librería desarrollada en Python como parte del Team Challenge del Bootcamp de Data Science.

El proyecto tiene como objetivo proporcionar funciones reutilizables para facilitar tareas de Análisis Exploratorio de Datos (EDA) y selección de variables, permitiendo acelerar las fases iniciales de preparación y análisis de datos en proyectos de Machine Learning.

La librería ha sido desarrollada siguiendo buenas prácticas de ingeniería de software, incluyendo control de versiones con Git, gestión de tareas mediante Issues, revisiones de código a través de Pull Requests y pruebas unitarias automatizadas.

## 🎯 Objetivos

- Automatizar tareas frecuentes de EDA.
- Facilitar la selección de variables para modelos predictivos.
- Crear una librería reutilizable y escalable.
- Aplicar buenas prácticas de desarrollo profesional.
- Trabajar mediante integración continua y revisión de código.

## 📂 Estructura del proyecto

```text

toolbox_ml/
├── __init__.py
└── eda/
    ├── __init__.py
    └── core.py

tests/
├── __init__.py
└── test_core.py

notebooks/
└── demo.ipynb

README.md
requirements.txt
setup.py

```
## 🧰 Tecnologías utilizadas

- Python 3.x
- Pandas
- NumPy
- SciPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Pytest
- Git
- GitHub


## ⚙️ Instalación

### Clonar repositorio

```bash
git clone https://github.com/Angelica26-DS/TC-Toolbox-ML.git
cd TC-Toolbox-ML
```

### Crear entorno virtual

```bash
python -m venv venv
```

### Activar entorno

**Windows**

```bash
venv\Scripts\activate
```

**Linux / Mac**

```bash
source venv/bin/activate
```

### Instalar dependencias

```bash
pip install -r requirements.txt
```

### Instalar el paquete

```bash
pip install -e .
```
## 📦 Importación del paquete

Una vez instalado el paquete, las funciones pueden importarse desde cualquier notebook o script:

```python
from toolbox_ml.eda.core import (
    describe_df,
    tipifica_variables,
    get_features_num_regression,
    plot_features_num_regression,
    get_features_cat_regression,
    plot_features_cat_regression
)
```

## 🛠 Funcionalidades

### *describe_df()*

* Genera un resumen descriptivo de un DataFrame:
    - Tipo de dato.
    - Porcentaje de nulos.
    - Número de valores únicos.
    - Porcentaje de cardinalidad.

### *tipifica_variables()*

* Clasifica automáticamente las variables en:
    - Binaria.
    - Categórica.
    - Numérica Discreta.
    - Numérica Continua.

### *get_features_num_regression()*

* Identifica variables numéricas relacionadas con una variable objetivo numérica mediante correlación de Pearson.

* Permite:
    - Filtrar por umbral de correlación.
    - Filtrar por significancia estadística (p-value).

### *plot_features_num_regression()*

* Genera visualizaciones para variables numéricas relacionadas con una variable objetivo.

* Incluye:
    - Selección automática de variables.
    - Pairplots.
    - Filtrado por correlación.
    - Filtrado opcional por p-value.

### *get_features_cat_regression()*

* Identifica variables categóricas que presentan una relación estadísticamente significativa con una variable objetivo numérica.

* Utiliza:
    - Mann-Whitney U para variables categóricas binarias.
    - ANOVA para variables categóricas con más de dos categorías.

* Permite filtrar resultados mediante un umbral de significancia (*p-value*).

### *plot_features_cat_regression()*

* Genera visualizaciones para variables categóricas relacionadas con una variable objetivo numérica.

* Incluye:
    - Selección automática de variables significativas.
    - Visualización mediante boxplots.
    - Filtrado opcional por significancia estadística (*p-value*).
    - Soporte para múltiples variables categóricas.


### 🚀 Ejemplos de uso

- A continuación se muestran ejemplos básicos de utilización de las funciones implementadas en la librería.


### describe_df()

```python
import pandas as pd
from toolbox_ml.eda.core import describe_df

df = pd.DataFrame({
    "edad": [20, 25, 30, 35],
    "ciudad": ["Madrid", "Sevilla", "Madrid", "Valencia"]
})

resultado = describe_df(df)

print(resultado)
```

### tipifica_variables()

```python
import pandas as pd
from toolbox_ml.eda.core import tipifica_variables

df = pd.DataFrame({
    "sexo": ["M", "F", "M", "F"],
    "edad": [20, 25, 30, 35],
    "precio": [100, 120, 130, 150]
})

resultado = tipifica_variables(
    df,
    umbral_categoria=10,
    umbral_continua=50.0
)

print(resultado)
```

### get_features_num_regression()

```python
import pandas as pd
from toolbox_ml.eda.core import get_features_num_regression

df = pd.DataFrame({
    "target": [1, 2, 3, 4, 5],
    "feature_1": [2, 4, 6, 8, 10],
    "feature_2": [10, 3, 7, 1, 8]
})

variables = get_features_num_regression(
    df=df,
    target_col="target",
    umbral_corr=0.8
)

print(variables)
```

### plot_features_num_regression()

```python
import pandas as pd
from toolbox_ml.eda.core import plot_features_num_regression

df = pd.DataFrame({
    "target": [1, 2, 3, 4, 5],
    "feature_1": [2, 4, 6, 8, 10],
    "feature_2": [10, 3, 7, 1, 8]
})

plot_features_num_regression(
    df=df,
    target_col="target",
    umbral_corr=0.8
)
```

### get_features_cat_regression()

```python
import pandas as pd
from toolbox_ml.eda.core import get_features_cat_regression

df = pd.DataFrame({
    "categoria": ["A", "A", "B", "B", "C", "C"],
    "ventas": [100, 110, 200, 210, 300, 310]
})

variables = get_features_cat_regression(
    df=df,
    target_col="ventas",
    pvalue=0.05
)

print(variables)
```

### plot_features_cat_regression()

```python
import pandas as pd
from toolbox_ml.eda.core import plot_features_cat_regression

df = pd.DataFrame({
    "categoria": ["A", "A", "B", "B", "C", "C"],
    "ventas": [100, 110, 200, 210, 300, 310]
})

plot_features_cat_regression(
    df=df,
    target_col="ventas",
    pvalue=0.05
)
```


## 🧪 Testing

El proyecto incorpora pruebas unitarias desarrolladas con Pytest para validar el correcto funcionamiento de las funciones implementadas.

Para ejecutar todos los tests:

```bash
pytest tests/ -v
```

Resultado esperado:

```text
41 passed
```

Las pruebas cubren:

- Casos de uso válidos.
- Validación de parámetros de entrada.
- Manejo de errores.
- Casos límite.
- Filtros por correlación.
- Filtros por significancia estadística (p-value).

## 🌳 Metodología de trabajo

El desarrollo del proyecto se realizó mediante un flujo de trabajo colaborativo utilizando Git y GitHub.

Se aplicaron las siguientes prácticas:

- Creación de Issues para cada funcionalidad.
- Desarrollo en ramas independientes por funcionalidad.
- Integración mediante Pull Requests.
- Revisión de código entre miembros del equipo.
- Resolución de conflictos de integración.
- Protección de la rama main.
- Uso de pruebas unitarias antes de cada integración.

Esta metodología permitió mantener la calidad del código, la trazabilidad de los cambios y una integración progresiva de todas las funcionalidades desarrolladas.

## 👥 Equipo

| Integrante | Rol | Responsabilidades |
|------------|-----|-------------------|
| Angélica Sánchez | Scrum Master | Coordinación del proyecto, gestión de Issues, revisiones de código, documentación y soporte en integración |
| Hugo | Developer | Desarrollo de funcionalidades EDA y validaciones |
| Carlos | Developer | Desarrollo de funcionalidades de selección de variables y pruebas unitarias |


## 🔀 Flujo de trabajo Git

El proyecto se desarrolló siguiendo un flujo basado en Git Flow simplificado.

### Proceso utilizado

1. Creación de Issues para cada funcionalidad.
2. Creación de ramas feature independientes.
3. Desarrollo y commits en cada rama.
4. Apertura de Pull Request.
5. Revisión por otro integrante.
6. Aprobación del código.
7. Integración mediante Squash & Merge.
8. Actualización de ramas locales.

### Convención de ramas

Durante el desarrollo del proyecto se utilizó una estrategia basada en ramas independientes por funcionalidad para facilitar el trabajo colaborativo y reducir conflictos de integración.

### Estructura utilizada

```text
main
feature/nombre-funcionalidad
```

### Ejemplos

```text
feature/project-setup
feature/describe-df
feature/tipifica-variables
feature/get-features-num-regression
feature/final-docs
```

## 📌 Estado del proyecto

Proyecto desarrollado como parte del Team Challenge del Bootcamp de Data Science.

Estado actual:

- ✅ 6 funcionalidades implementadas.
- ✅ 41 pruebas unitarias automatizadas superadas mediante Pytest.
- ✅ Documentación completa.
- ✅ Ejemplos de uso incluidos.
- ✅ Gestión colaborativa mediante Git y GitHub.
- ✅ Integración mediante Pull Requests y revisiones de código.
- ✅ Notebook de demostración incluido.


