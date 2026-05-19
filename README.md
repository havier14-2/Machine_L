# Proyecto de Machine Learning: Clasificación Binaria de Clientes Bancarios (Monopoly)

Este proyecto implementa un flujo completo de Machine Learning bajo la metodología CRISP-DM para predecir la aceptación de productos financieros por parte de una cartera de clientes bancarios. El código ha sido estructurado y modularizado siguiendo los más altos estándares de desarrollo, abstrayendo la ingeniería de características y el modelamiento en un módulo central para evitar la fuga de datos (*data leakage*) y automatizando todo mediante `Pipeline` de scikit-learn.

## 🚀 Estructura del Proyecto

El repositorio está organizado de la siguiente manera para garantizar escalabilidad y orden:

```text
Machine_L/
│
├── data/
│   └── Base_clientes_Monopoly.xlsx  # Archivo de datos (Ignorado en Git por tamaño/privacidad)
│
├── src/
│   ├── __init__.py                  # Inicializador de paquete Python
│   └── ml_core.py                   # Motor de lógica pesada (Limpieza, Pipelines, SMOTE y Modelos)
│
├── .gitignore                       # Filtro para evitar subir entornos virtuales o datos a GitHub
├── requirements.txt                 # Archivo de dependencias con versiones del proyecto
└── Presentacion_Final.ipynb         # Notebook limpio enfocado en la presentación y analítica de negocio
```

## 🛠️ Requisitos Previos

Asegúrate de tener instalado localmente:

- Python 3.9 o superior.
- Un editor de código compatible con notebooks (se recomienda Visual Studio Code con las extensiones de Python y Jupyter instaladas).

## 💻 Instrucciones de Instalación y Ejecución (Paso a Paso)

Sigue estos comandos desde tu terminal favorita (PowerShell en Windows, Bash en Mac/Linux) dentro de la carpeta del proyecto:

### 1. Clonar el Repositorio

```bash
git clone https://github.com/havier14-2/Machine_L.git
cd Machine_L
```

### 2. Crear el Entorno Virtual Aislado

Para evitar conflictos con otras librerías globales en tu computadora:

```bash
python -m venv venv
```

### 3. Activar el Entorno Virtual

Dependiendo del sistema operativo que uses, ejecuta:

#### En Windows (PowerShell)

```powershell
.\venv\Scripts\Activate.ps1
```

#### En Windows (CMD)

```dos
venv\Scripts\activate.bat
```

#### En Mac / Linux

```bash
source venv/bin/activate
```

> Nota: Sabrás que el entorno está activo porque verás el prefijo `(venv)` al inicio de tu terminal.

### 4. Instalar las Dependencias de Producción

Con el entorno virtual activo, este comando instalará todo lo necesario (incluyendo `xgboost`, `imbalanced-learn` y `jinja2` para el formateo visual):

```bash
pip install -r requirements.txt
```

### 5. Suministrar el Dataset Financiero

Como la carpeta `data/` está listada en el `.gitignore` por buenas prácticas de almacenamiento, debes:

1. Crear una carpeta llamada `data` en la raíz del proyecto (si no existe).
2. Guardar el archivo original `Base_clientes_Monopoly.xlsx` dentro de esa carpeta.

### 6. Configurar el Kernel en Visual Studio Code

1. Abre la carpeta del proyecto en VS Code.
2. Haz doble clic para abrir el notebook `Presentacion_Final.ipynb`.
3. En la esquina superior derecha del notebook, haz clic en **"Select Kernel"**.
4. Elige **"Python Environments..."** y selecciona explícitamente el interpretador que apunta a nuestro entorno virtual local (`venv`).

¡Listo! Ya puedes presionar **"Run All"** o ejecutar celda por celda.

## 🧠 Arquitectura y Flujo Técnico del Core (`src/ml_core.py`)

El archivo `ml_core.py` automatiza las fases críticas evaluadas por la rúbrica mediante funciones parametrizadas:

### `preparar_datos()`

- Descarta variables con más del 40% de nulos para limpieza inicial.
- Divide los datos de forma estratificada (80/20) para blindar el conjunto de test.
- Implementa dos `Pipelines` paralelos:
  - Numérico: imputación por mediana + `StandardScaler`.
  - Categórico: imputación por constante + `OneHotEncoder`.
- Aplica la técnica `SMOTE` balanceando matemáticamente la variable objetivo exclusivamente en el set de entrenamiento para mitigar el sesgo comercial.

### `entrenar_modelos()`

Entrena simultáneamente los 3 algoritmos supervisados distintos exigidos:

- Regresión Logística
- Random Forest
- XGBoost

### `evaluar_modelos()`

Construye la matriz de comparación con las métricas técnicas clave:

- Accuracy
- Precision
- Recall
- F1-Score

### `mostrar_matriz_confusion()`

Renderiza gráficamente los cuadrantes de clasificación para auditar el impacto en el negocio.

### `generar_proyeccion_negocio()`

Extrae y ordena jerárquicamente por probabilidad a los clientes con mayor potencial de conversión para el equipo de ventas.
