# 🎾 Paddle Tournament Predictor

Este proyecto es una solución integral de **Machine Learning** diseñada para predecir el desempeño de jugadores de pádel y simular resultados de torneos. Utiliza datos de tracking de video (Computer Vision) para analizar métricas físicas y técnicas, y emplea modelos avanzados para estimar probabilidades de victoria.

## 🎯 Objetivo del Proyecto

El objetivo principal es responder a la pregunta: **¿Quién tiene más probabilidad de ganar el torneo basándose en su rendimiento dentro y fuera de la cancha?**

A diferencia de los rankings tradicionales basados solo en puntos pasados, este sistema analiza la "calidad de juego" objetiva extraída de video:
*   **Velocidad y Aceleración:** ¿Qué tan rápido se mueve el jugador?
*   **Control de Red:** ¿Qué porcentaje del tiempo domina la posición de ataque?
*   **Desgaste Físico:** Distancia total recorrida y potencia de golpeo.

---

## 🛠️ Tecnologías Utilizadas

El proyecto combina varias tecnologías modernas de Data Science y Desarrollo Web:

*   **Python:** Lenguaje núcleo del proyecto.
*   **Pandas & NumPy:** Manipulación y análisis de datos estructurados.
*   **XGBoost (Extreme Gradient Boosting):**
    *   *¿Para qué sirve aquí?* Es el cerebro del sistema. Un modelo de clasificación entrenado para predecir la probabilidad de que un equipo gane un punto/partido basándose en las métricas físicas de los jugadores (velocidad, distancia, etc.).
*   **Simulación de Monte Carlo:**
    *   *¿Para qué sirve aquí?* Como el pádel tiene un componente de suerte y variabilidad, no basta con predecir un solo partido. Esta técnica simula el torneo miles de veces con pequeñas variaciones aleatorias para calcular una probabilidad robusta de campeonato (ej: "El Jugador X ganó el torneo en el 15% de las 1000 simulaciones").
*   **Streamlit:** Framework para crear la interfaz web interactiva (Frontend).
*   **Plotly:** Librería de visualización para gráficos interactivos (Radar Charts, Barras).

---

## 🚀 Guía de Instalación y Ejecución

Sigue estos pasos para poner en marcha el proyecto en tu máquina local.

### 1. Prerrequisitos
Asegúrate de tener Python instalado (versión 3.9 o superior recomendada).

### 2. Instalación de Dependencias
El proyecto cuenta con un archivo `requirements.txt` que lista todas las librerías necesarias. Ejecuta el siguiente comando en tu terminal:

```bash
pip install -r requirements.txt
```

### 3. Ejecución del Pipeline de Datos (Notebook)
El corazón del análisis reside en el Jupyter Notebook. Aquí se procesan los datos crudos, se entrena el modelo y se generan las simulaciones.

1.  Abre el notebook:
    ```bash
    jupyter notebook Paddle_Predictor.ipynb
    ```
2.  Ejecuta todas las celdas en orden. Esto realizará:
    *   Limpieza de datos de video.
    *   Ingeniería de características (Feature Engineering).
    *   Entrenamiento del modelo XGBoost.
    *   Simulación de Monte Carlo del torneo.
    *   **Exportación de Artefactos:** Al finalizar, el notebook generará dos archivos críticos para la app:
        *   `Ranking_Tournament_Prediction.csv`: El ranking final probabilístico.
        *   `xgboost_paddle_model.json`: El modelo entrenado (debes asegurarte de ejecutar la celda de exportación o usar el script `export_model.py` si tienes problemas).

### 4. Ejecución de la Interfaz Web (App)
Una vez generados los datos, levanta la interfaz gráfica para interactuar con los resultados.

```bash
streamlit run app.py
```

Esto abrirá una pestaña en tu navegador (usualmente en `http://localhost:8501`) con tres secciones:
1.  **Dashboard:** Vista macro del torneo con el ranking de favoritos.
2.  **Comparador:** Análisis "cara a cara" de jugadores usando gráficos de radar.
3.  **Simulador en Vivo:** Herramienta para predecir el ganador de un partido hipotético entre dos jugadores seleccionados.

---

## 📂 Estructura del Proyecto

*   `Paddle_Predictor.ipynb`: Notebook principal con toda la lógica de Data Science.
*   `app.py`: Código de la aplicación web (Frontend).
*   `export_model.py`: Script auxiliar para entrenar y exportar el modelo XGBoost independientemente.
*   `Ranking_Tournament_Prediction.csv`: Resultados de la simulación (Input para el Dashboard).
*   `Dataset_Maestro_Real_Target.csv`: Base de datos histórica de estadísticas (Input para el Comparador).
*   `xgboost_paddle_model.json`: Archivo del modelo entrenado.
