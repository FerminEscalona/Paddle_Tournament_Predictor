# Paddle Tournament Predictor

Dashboard y modelo de machine learning para predecir el ganador de partidos/torneos de pádel a partir de métricas de juego. Incluye notebooks de análisis, pipelines de features y un front-end interactivo en Streamlit para explorar rankings, comparativas face-to-face y simulaciones de partidos.

## Contenido del repo
- `app.py`: dashboard Streamlit que consume los datasets procesados y el modelo entrenado.
- `miAnalisis.ipynb`: notebook principal que limpia datos, crea features y guarda los artefactos listos para el dashboard.
- `data_export/`: salidas generadas por el notebook (`tabla_posiciones.csv`, `stats_equipos.csv`, `ranking_jugadores.csv`, `modelo_entrenado.pkl`, `features_modelo.json`).
- `Paddle_Predictor.ipynb`, `Base_Videos_Final*.csv`, `Resultados_Partidos.csv`, etc.: fuentes originales y experimentos de modelado.
- `requirements.txt`: dependencias necesarias para reproducir el pipeline y correr el dashboard.

## Requisitos previos
- Python >= 3.10
- pip (o pipx) para instalar dependencias
- (Opcional) Entorno virtual recomendado para aislar las libs

## Instrucciones de uso
1) Preparar entorno
```bash
python -m venv .venv
source .venv/bin/activate  # en Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) Generar artefactos de datos y modelo (si no existen)
- Ejecuta todas las celdas de `miAnalisis.ipynb`. El notebook crea la carpeta `data_export/` con los CSV procesados y `modelo_entrenado.pkl` requerido por el dashboard.
- Verifica que existan estos archivos: `data_export/tabla_posiciones.csv`, `data_export/stats_equipos.csv`, `data_export/ranking_jugadores.csv`, `data_export/modelo_entrenado.pkl`, `data_export/features_modelo.json`.

3) Correr el dashboard
```bash
streamlit run app.py
```
La app se abrirá en tu navegador en `http://localhost:8501`. Si aparece un error de carga de datos, revisa el paso 2.

## Flujo resumido del proyecto
1. Recoleccion y limpieza: CSV crudos de partidos y jugadores.
2. Feature engineering: cálculo de métricas de velocidad, golpes, distancia, tiempo ofensivo, etc. en `miAnalisis.ipynb`.
3. Entrenamiento: modelo de regresión logística (ver métricas en el sidebar del dashboard).
4. Serving: `app.py` lee los artefactos de `data_export/` y expone cuatro tabs (ranking, radar face-to-face, simulador de predicción y análisis de factores de éxito).

## Notas
- Si cambias datos o reentrenas, vuelve a ejecutar el notebook para actualizar `data_export/`.
- Para inspeccionar la configuración del modelo o las features utilizadas, consulta `data_export/features_modelo.json`.
