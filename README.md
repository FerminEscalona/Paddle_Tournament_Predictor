# Paddle Tournament Predictor

Proyecto de análisis y predicción de partidos y torneos de pádel, con notebooks de exploración, preparación de datos y modelado, y una aplicación Streamlit.

## Estructura del proyecto

La organización sigue las responsabilidades de CRISP-DM. Los notebooks modulares separan el trabajo por fase y mantienen los cálculos del notebook original, conservado como referencia.

```text
Paddle_Tournament_Predictor/
├── data/
│   ├── raw/              # Fuentes originales: videos, jugadores y levantamiento
│   ├── interim/          # Base limpia y features iniciales por jugador/partido
│   └── processed/        # Features completas, features por equipo y resultados
├── notebooks/
│   ├── contexto.ipynb
│   ├── limpieza.ipynb
│   ├── EDA.ipynb
│   ├── modelos.ipynb
│   ├── metricas.ipynb
│   ├── Paddle_Predictor.ipynb
│   └── miAnalisis.ipynb     # Referencia original
├── src/
│   └── app.py            # Dashboard Streamlit
├── docs/                 # Espacio para objetivos, alcance y documentación
├── models/               # Modelos entrenados y estado para calcular métricas
├── reports/
│   └── figures/          # Espacio para métricas, conclusiones y gráficos
├── requirements.txt
├── .gitignore
└── README.md
```

El entorno local `venv/` se mantiene en su ubicación original.

## Relación con CRISP-DM

| Fase | Ubicación |
| --- | --- |
| 1. Comprensión del negocio | `docs/`: objetivos, criterios de éxito y alcance |
| 2. Comprensión de los datos | `data/raw/`, `notebooks/contexto.ipynb` y `notebooks/EDA.ipynb` |
| 3. Preparación de los datos | `notebooks/limpieza.ipynb`, `data/interim/` y `data/processed/` |
| 4. Modelado | `notebooks/modelos.ipynb` y `models/` |
| 5. Evaluación | `notebooks/metricas.ipynb` y `reports/` |
| 6. Despliegue | `src/app.py` |

## Ejecución de los notebooks

Los cinco notebooks modulares cargan sus entradas desde archivos y pueden ejecutarse con kernels separados. Sus rutas admiten iniciar Jupyter desde la raíz del proyecto o desde `notebooks/`.

1. `contexto.ipynb`: carga los videos originales para revisar su estructura y valores nulos.
2. `limpieza.ipynb`: carga los videos originales y las fuentes de jugadores y partidos; guarda los datasets en `data/interim/` y `data/processed/`.
3. `EDA.ipynb`: carga los datasets procesados de jugadores y equipos para explorarlos.
4. `modelos.ipynb`: carga los datasets procesados, entrena los modelos y guarda `models/estado_modelos.joblib` con los modelos, escaladores, encoders y resultados necesarios para la evaluación.
5. `metricas.ipynb`: carga ese artefacto y los datos de jugadores para comparar modelos, generar rankings y ejecutar las simulaciones existentes.

Después de cambiar los datos, vuelve a ejecutar `limpieza.ipynb` y `modelos.ipynb` antes de calcular métricas. Se mantienen los cálculos originales; se añadieron las importaciones, cargas y guardados necesarios para conectar los notebooks.

`miAnalisis.ipynb` y `Paddle_Predictor.ipynb` se conservan como referencia y no forman parte de este flujo modular.

La aplicación conserva las referencias a `data_export/` para cargar `tabla_posiciones.csv`, `stats_equipos.csv`, `ranking_jugadores.csv`, `modelo_entrenado.pkl` y `features_modelo.json`. El artefacto de los notebooks modulares no genera esos archivos del dashboard.

## Dependencias

Python >= 3.10; el flujo modular se verifica con Python 3.11. Para instalar las dependencias desde la raíz:

```bash
pip install -r requirements.txt
```

Una vez adaptadas las rutas y generados los artefactos requeridos, el comando del dashboard desde la raíz será:

```bash
streamlit run src/app.py
```
