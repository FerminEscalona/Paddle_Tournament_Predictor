"""
Dashboard de Predicción de Torneos de Pádel
============================================
Dashboard interactivo para visualizar rankings de equipos, 
comparaciones face-to-face y simulaciones de partidos.

Author: Lead Data Scientist
Date: 2025-11-25
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================
st.set_page_config(
    page_title="Paddle Tournament Predictor",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# FUNCIONES DE CARGA DE DATOS (CON CACHE)
# ============================================================================

@st.cache_data
def load_tabla_posiciones():
    """Carga la tabla de posiciones de equipos"""
    try:
        df = pd.read_csv('data_export/tabla_posiciones.csv')
        return df
    except FileNotFoundError:
        st.error("⚠️ Archivo 'tabla_posiciones.csv' no encontrado. Por favor ejecuta el notebook completo.")
        return None

@st.cache_data
def load_stats_equipos():
    """Carga las estadísticas detalladas de equipos para radar"""
    try:
        df = pd.read_csv('data_export/stats_equipos.csv')
        return df
    except FileNotFoundError:
        st.error("⚠️ Archivo 'stats_equipos.csv' no encontrado. Por favor ejecuta el notebook completo.")
        return None

@st.cache_data
def load_ranking_jugadores():
    """Carga el ranking de jugadores individuales"""
    try:
        df = pd.read_csv('data_export/ranking_jugadores.csv')
        return df
    except FileNotFoundError:
        st.warning("⚠️ Archivo 'ranking_jugadores.csv' no encontrado.")
        return None

@st.cache_resource
def load_modelo():
    """Carga el modelo entrenado y sus componentes"""
    try:
        modelo_data = joblib.load('data_export/modelo_entrenado.pkl')
        return modelo_data
    except FileNotFoundError:
        st.error("⚠️ Archivo 'modelo_entrenado.pkl' no encontrado. Por favor ejecuta el notebook completo.")
        return None

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def crear_grafico_radar(equipo1_data, equipo2_data, equipo1_nombre, equipo2_nombre):
    """
    Crea un gráfico de radar comparativo entre dos equipos
    
    Args:
        equipo1_data: Series con stats del equipo 1
        equipo2_data: Series con stats del equipo 2
        equipo1_nombre: Nombre del equipo 1
        equipo2_nombre: Nombre del equipo 2
    """
    # Métricas para el radar (normalizadas 0-100)
    metricas = [
        'Velocidad Media (m/s)',
        'Golpes Totales',
        'Tiempo Cerca Pelota (%)',
        'Tiempo Zona Ofensiva (%)',
        'Aceleración Media (m/s²)',
        'Distancia Total (m)'
    ]
    
    # Función para normalizar valores a escala 0-100
    def normalizar(valor, min_val, max_val):
        if max_val == min_val:
            return 50
        return ((valor - min_val) / (max_val - min_val)) * 100
    
    # Obtener rangos de todas las estadísticas para normalización
    stats_df = load_stats_equipos()
    
    valores_eq1 = []
    valores_eq2 = []
    
    for metrica in metricas:
        min_val = stats_df[metrica].min()
        max_val = stats_df[metrica].max()
        
        val1 = equipo1_data[metrica]
        val2 = equipo2_data[metrica]
        
        # Para "Distancia a Pelota", menor es mejor (invertir)
        if metrica == 'Distancia a Pelota (m)':
            valores_eq1.append(normalizar(max_val - val1, 0, max_val - min_val))
            valores_eq2.append(normalizar(max_val - val2, 0, max_val - min_val))
        else:
            valores_eq1.append(normalizar(val1, min_val, max_val))
            valores_eq2.append(normalizar(val2, min_val, max_val))
    
    # Crear figura de radar
    fig = go.Figure()
    
    # Equipo 1
    fig.add_trace(go.Scatterpolar(
        r=valores_eq1,
        theta=metricas,
        fill='toself',
        name=equipo1_nombre[:40],
        line=dict(color='rgba(0, 123, 255, 0.8)', width=3),
        fillcolor='rgba(0, 123, 255, 0.2)'
    ))
    
    # Equipo 2
    fig.add_trace(go.Scatterpolar(
        r=valores_eq2,
        theta=metricas,
        fill='toself',
        name=equipo2_nombre[:40],
        line=dict(color='rgba(255, 99, 71, 0.8)', width=3),
        fillcolor='rgba(255, 99, 71, 0.2)'
    ))
    
    # Configuración del layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10),
                showticklabels=True
            ),
            angularaxis=dict(
                tickfont=dict(size=11)
            )
        ),
        showlegend=True,
        title={
            'text': "Comparación de Métricas (Normalizado 0-100)",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def predecir_partido(equipo1_data, equipo2_data, modelo_data):
    """
    Predice el resultado de un partido entre dos equipos
    
    Args:
        equipo1_data: Series con stats del equipo 1
        equipo2_data: Series con stats del equipo 2
        modelo_data: Diccionario con modelo, scaler y features
        
    Returns:
        prob_eq1: Probabilidad de victoria del equipo 1
        prob_eq2: Probabilidad de victoria del equipo 2
    """
    modelo = modelo_data['model']
    scaler = modelo_data['scaler']
    features = modelo_data['features']
    
    # Mapear nombres de columnas con alias amigables (UI) a nombres originales del modelo
    map_cols = {
        'Velocidad Media (m/s)': 'velocidad_media_equipo_mps',
        'Golpes Totales': 'golpes_totales_equipo',
        'Distancia a Pelota (m)': 'distancia_media_pelota_equipo_m',
        'Tiempo Cerca Pelota (%)': 'pct_frames_cerca_pelota_equipo',
        'Distancia Total (m)': 'distancia_total_equipo_m',
        'Aceleración Media (m/s²)': 'aceleracion_media_mps2',
        'Tiempo Zona Ofensiva (%)': 'pct_tiempo_zona_ofensiva'
    }
    
    def get_feature(row, feature):
        """Obtiene un feature usando nombre original o alias amigable"""
        if feature in row.index:
            return row[feature]
        # Buscar alias si viene con nombres del dashboard
        for alias, orig in map_cols.items():
            if orig == feature and alias in row.index:
                return row[alias]
        return None
    
    def safe(value):
        """Convierte None/NaN en 0 para evitar sesgos extremos por faltantes."""
        if value is None:
            return 0
        try:
            if np.isnan(value):
                return 0
        except Exception:
            pass
        return value
    
    # Extraer valores disponibles para cada equipo
    eq1_vals = {feat: get_feature(equipo1_data, feat) for feat in features}
    eq2_vals = {feat: get_feature(equipo2_data, feat) for feat in features}
    
    # Construir features para la predicción alineados con el modelo entrenado
    features_dict = {}
    for feat in features:
        if feat == 'diff_golpes_vs_rival':
            features_dict[feat] = safe(eq1_vals.get('golpes_totales_equipo')) - safe(eq2_vals.get('golpes_totales_equipo'))
        elif feat == 'diff_distancia_pelota_vs_rival':
            features_dict[feat] = safe(eq2_vals.get('distancia_media_pelota_equipo_m')) - safe(eq1_vals.get('distancia_media_pelota_equipo_m'))
        elif feat == 'diff_pct_cerca_vs_rival':
            features_dict[feat] = safe(eq1_vals.get('pct_frames_cerca_pelota_equipo')) - safe(eq2_vals.get('pct_frames_cerca_pelota_equipo'))
        elif feat == 'diff_velocidad_vs_rival':
            features_dict[feat] = safe(eq1_vals.get('velocidad_media_equipo_mps')) - safe(eq2_vals.get('velocidad_media_equipo_mps'))
        else:
            features_dict[feat] = safe(eq1_vals.get(feat))
    
    # Crear DataFrame con las features
    X = pd.DataFrame([features_dict])
    
    # Asegurar orden correcto de columnas
    X = X[features]
    
    # Escalar
    X_scaled = scaler.transform(X)
    
    # Predecir
    prob_eq1_raw = modelo.predict_proba(X_scaled)[0, 1]
    
    # Aplicar suavizado (mismo que en entrenamiento)
    if prob_eq1_raw > 0.85:
        prob_eq1 = 0.70 + (prob_eq1_raw - 0.85) * 0.67
    elif prob_eq1_raw < 0.15:
        prob_eq1 = 0.30 - (0.15 - prob_eq1_raw) * 0.67
    else:
        prob_eq1 = 0.50 + (prob_eq1_raw - 0.50) * 0.85
    
    prob_eq2 = 1 - prob_eq1
    
    return prob_eq1, prob_eq2

# ============================================================================
# HEADER PRINCIPAL
# ============================================================================

st.title("🎾 Paddle Tournament Predictor")
st.markdown("""
Sistema de predicción de resultados para torneos de pádel basado en Machine Learning.
Analiza estadísticas de jugadores y equipos para predecir probabilidades de victoria.
""")

st.divider()

# ============================================================================
# CARGAR DATOS
# ============================================================================

with st.spinner("Cargando datos..."):
    tabla_posiciones = load_tabla_posiciones()
    stats_equipos = load_stats_equipos()
    ranking_jugadores = load_ranking_jugadores()
    modelo_data = load_modelo()

# Verificar que los datos se cargaron correctamente
if tabla_posiciones is None or stats_equipos is None or modelo_data is None:
    st.error("❌ Error al cargar los datos. Asegúrate de haber ejecutado el notebook completo.")
    st.info("📝 Instrucciones:\n1. Ejecuta todas las celdas del notebook `miAnalisis.ipynb`\n2. Asegúrate de que se haya creado la carpeta `data_export/` con los archivos necesarios")
    st.stop()

# Mostrar métricas del modelo en el sidebar
with st.sidebar:
    st.header("📊 Info del Modelo")
    st.metric("Tipo de Modelo", "Logistic Regression")
    st.metric("ROC-AUC", f"{modelo_data['metrics']['roc_auc']:.3f}")
    st.metric("Accuracy", f"{modelo_data['metrics']['accuracy']:.3f}")
    st.metric("F1-Score", f"{modelo_data['metrics']['f1_score']:.3f}")
    
    st.divider()
    
    st.header("📈 Estadísticas")
    st.metric("Total Equipos", len(tabla_posiciones))
    if ranking_jugadores is not None:
        st.metric("Total Jugadores", len(ranking_jugadores))
    st.metric("Features del Modelo", len(modelo_data['features']))

# ============================================================================
# TABS PRINCIPALES
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Ranking Global", 
    "⚡ Face-to-Face (Radar)", 
    "🎯 Simulador de Predicción",
    "🔍 Análisis de Factores de Éxito"
])

# ============================================================================
# TAB 1: RANKING GLOBAL (MACRO)
# ============================================================================

with tab1:
    st.header("🏆 Ranking de Equipos")
    st.markdown("Vista general del ranking de parejas basado en probabilidad promedio de victoria")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Tabla de Posiciones")
        
        # Filtro de búsqueda
        search = st.text_input("🔍 Buscar equipo o jugador:", "")
        
        # Filtrar tabla
        if search:
            tabla_filtrada = tabla_posiciones[
                tabla_posiciones['Equipo'].str.contains(search, case=False, na=False) |
                tabla_posiciones['Jugador 1'].str.contains(search, case=False, na=False) |
                tabla_posiciones['Jugador 2'].str.contains(search, case=False, na=False)
            ]
        else:
            tabla_filtrada = tabla_posiciones.head(50)
        
        # Formatear probabilidades como porcentajes
        tabla_display = tabla_filtrada.copy()
        tabla_display['Prob. Victoria'] = tabla_display['Prob. Victoria'].apply(lambda x: f"{x*100:.2f}%")
        
        st.dataframe(
            tabla_display,
            use_container_width=True,
            height=600,
            hide_index=True
        )
    
    with col2:
        st.subheader("📊 Top 20 Equipos")
        
        # Gráfico de barras horizontales
        top_20 = tabla_posiciones.head(20)
        
        fig = px.bar(
            top_20,
            y='Equipo',
            x='Prob. Victoria',
            orientation='h',
            title='Probabilidad de Victoria (Top 20)',
            labels={'Prob. Victoria': 'Probabilidad', 'Equipo': ''},
            color='Prob. Victoria',
            color_continuous_scale='RdYlGn',
            text='Prob. Victoria'
        )
        
        fig.update_traces(
            texttemplate='%{text:.1%}',
            textposition='outside'
        )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'},
            xaxis={'tickformat': '.0%'},
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Ranking de jugadores individuales (si está disponible)
    if ranking_jugadores is not None:
        st.divider()
        st.subheader("👤 Ranking de Jugadores Individuales")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            ranking_display = ranking_jugadores.head(30).copy()
            ranking_display['Score Promedio'] = ranking_display['Score Promedio'].apply(lambda x: f"{x*100:.2f}%")
            
            st.dataframe(
                ranking_display,
                use_container_width=True,
                hide_index=True,
                height=400
            )
        
        with col2:
            # Top 10 jugadores
            top_10_jug = ranking_jugadores.head(10)
            
            fig_jug = px.bar(
                top_10_jug,
                y='Jugador',
                x='Score Promedio',
                orientation='h',
                title='Top 10 Mejores Jugadores',
                color='Score Promedio',
                color_continuous_scale='Viridis',
                text='Score Promedio'
            )
            
            fig_jug.update_traces(
                texttemplate='%{text:.1%}',
                textposition='outside'
            )
            
            fig_jug.update_layout(
                height=400,
                showlegend=False,
                yaxis={'categoryorder': 'total ascending'},
                xaxis={'tickformat': '.0%'},
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig_jug, use_container_width=True)

# ============================================================================
# TAB 2: FACE-TO-FACE (DIVERGENTE)
# ============================================================================

with tab2:
    st.header("⚡ Comparación Face-to-Face")
    st.markdown("Compara las métricas de dos equipos mediante un gráfico de radar")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔵 Equipo A")
        equipo_a = st.selectbox(
            "Selecciona el Equipo A:",
            options=stats_equipos['Equipo'].tolist(),
            key='equipo_a'
        )
        
        if equipo_a:
            datos_a = stats_equipos[stats_equipos['Equipo'] == equipo_a].iloc[0]
            
            # Mostrar info del equipo
            st.info(f"**Ranking:** {datos_a['Ranking']}")
            st.info(f"**Jugadores:** {datos_a['Jugador 1']} & {datos_a['Jugador 2']}")
            st.metric("Prob. Victoria", f"{datos_a['Prob. Victoria']*100:.2f}%")
            
            # Mostrar métricas clave
            with st.expander("📊 Ver todas las métricas"):
                st.write(f"**Velocidad Media:** {datos_a['Velocidad Media (m/s)']:.2f} m/s")
                st.write(f"**Golpes Totales:** {datos_a['Golpes Totales']:.0f}")
                st.write(f"**Distancia a Pelota:** {datos_a['Distancia a Pelota (m)']:.2f} m")
                st.write(f"**Tiempo Cerca Pelota:** {datos_a['Tiempo Cerca Pelota (%)']*100:.1f}%")
                st.write(f"**Distancia Total:** {datos_a['Distancia Total (m)']:.0f} m")
                st.write(f"**Aceleración Media:** {datos_a['Aceleración Media (m/s²)']:.3f} m/s²")
                st.write(f"**Tiempo Zona Ofensiva:** {datos_a['Tiempo Zona Ofensiva (%)']*100:.1f}%")
    
    with col2:
        st.subheader("🔴 Equipo B")
        equipo_b = st.selectbox(
            "Selecciona el Equipo B:",
            options=stats_equipos['Equipo'].tolist(),
            key='equipo_b'
        )
        
        if equipo_b:
            datos_b = stats_equipos[stats_equipos['Equipo'] == equipo_b].iloc[0]
            
            # Mostrar info del equipo
            st.info(f"**Ranking:** {datos_b['Ranking']}")
            st.info(f"**Jugadores:** {datos_b['Jugador 1']} & {datos_b['Jugador 2']}")
            st.metric("Prob. Victoria", f"{datos_b['Prob. Victoria']*100:.2f}%")
            
            # Mostrar métricas clave
            with st.expander("📊 Ver todas las métricas"):
                st.write(f"**Velocidad Media:** {datos_b['Velocidad Media (m/s)']:.2f} m/s")
                st.write(f"**Golpes Totales:** {datos_b['Golpes Totales']:.0f}")
                st.write(f"**Distancia a Pelota:** {datos_b['Distancia a Pelota (m)']:.2f} m")
                st.write(f"**Tiempo Cerca Pelota:** {datos_b['Tiempo Cerca Pelota (%)']*100:.1f}%")
                st.write(f"**Distancia Total:** {datos_b['Distancia Total (m)']:.0f} m")
                st.write(f"**Aceleración Media:** {datos_b['Aceleración Media (m/s²)']:.3f} m/s²")
                st.write(f"**Tiempo Zona Ofensiva:** {datos_b['Tiempo Zona Ofensiva (%)']*100:.1f}%")
    
    st.divider()
    
    # Gráfico de radar
    if equipo_a and equipo_b:
        if equipo_a == equipo_b:
            st.warning("⚠️ Por favor selecciona dos equipos diferentes para compararlos.")
        else:
            st.subheader("📊 Gráfico de Radar Comparativo")
            
            try:
                fig_radar = crear_grafico_radar(datos_a, datos_b, equipo_a, equipo_b)
                st.plotly_chart(fig_radar, use_container_width=True)
                
                # Análisis comparativo
                st.subheader("🔍 Análisis Comparativo")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    dif_vel = datos_a['Velocidad Media (m/s)'] - datos_b['Velocidad Media (m/s)']
                    st.metric(
                        "Diferencia en Velocidad",
                        f"{abs(dif_vel):.3f} m/s",
                        delta=f"{dif_vel:.3f}",
                        delta_color="normal" if dif_vel > 0 else "inverse"
                    )
                
                with col2:
                    dif_golpes = datos_a['Golpes Totales'] - datos_b['Golpes Totales']
                    st.metric(
                        "Diferencia en Golpes",
                        f"{abs(dif_golpes):.0f}",
                        delta=f"{dif_golpes:.0f}",
                        delta_color="normal" if dif_golpes > 0 else "inverse"
                    )
                
                with col3:
                    dif_tiempo = datos_a['Tiempo Cerca Pelota (%)'] - datos_b['Tiempo Cerca Pelota (%)']
                    st.metric(
                        "Diferencia en Tiempo Cerca",
                        f"{abs(dif_tiempo)*100:.1f}%",
                        delta=f"{dif_tiempo*100:.1f}%",
                        delta_color="normal" if dif_tiempo > 0 else "inverse"
                    )
                
            except Exception as e:
                st.error(f"Error al crear el gráfico de radar: {str(e)}")

# ============================================================================
# TAB 3: SIMULADOR DE PREDICCIÓN (CONVERGENTE)
# ============================================================================

with tab3:
    st.header("🎯 Simulador de Predicción de Partidos")
    st.markdown("Predice el resultado de un partido entre dos equipos usando el modelo de Machine Learning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 Equipo Local")
        equipo_local = st.selectbox(
            "Selecciona el Equipo Local:",
            options=stats_equipos['Equipo'].tolist(),
            key='equipo_local'
        )
    
    with col2:
        st.subheader("✈️ Equipo Visitante")
        equipo_visitante = st.selectbox(
            "Selecciona el Equipo Visitante:",
            options=stats_equipos['Equipo'].tolist(),
            key='equipo_visitante'
        )
    
    st.divider()
    
    # Botón de predicción
    if st.button("🔮 PREDECIR RESULTADO", type="primary", use_container_width=True):
        if equipo_local == equipo_visitante:
            st.error("⚠️ Por favor selecciona dos equipos diferentes.")
        else:
            with st.spinner("Calculando predicción..."):
                # Obtener datos de los equipos
                datos_local = stats_equipos[stats_equipos['Equipo'] == equipo_local].iloc[0]
                datos_visitante = stats_equipos[stats_equipos['Equipo'] == equipo_visitante].iloc[0]
                
                # Realizar predicción
                try:
                    prob_local, prob_visitante = predecir_partido(
                        datos_local, 
                        datos_visitante, 
                        modelo_data
                    )
                    
                    st.success("✅ Predicción completada!")
                    
                    # Mostrar resultados
                    st.subheader("📊 Resultados de la Predicción")
                    
                    # Métricas principales
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col1:
                        st.metric(
                            f"🏠 {equipo_local[:30]}...",
                            f"{prob_local*100:.1f}%",
                            delta=f"{(prob_local-0.5)*100:+.1f}%",
                            delta_color="normal"
                        )
                    
                    with col2:
                        # Determinar favorito
                        if abs(prob_local - prob_visitante) < 0.05:
                            favorito = "Partido Parejo"
                            emoji = "⚖️"
                        elif prob_local > prob_visitante:
                            favorito = "Local Favorito"
                            emoji = "🏠"
                        else:
                            favorito = "Visitante Favorito"
                            emoji = "✈️"
                        
                        st.metric(
                            "Pronóstico",
                            favorito,
                            emoji
                        )
                    
                    with col3:
                        st.metric(
                            f"✈️ {equipo_visitante[:30]}...",
                            f"{prob_visitante*100:.1f}%",
                            delta=f"{(prob_visitante-0.5)*100:+.1f}%",
                            delta_color="normal"
                        )
                    
                    st.divider()
                    
                    # Visualización de probabilidades
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Gráfico de barras comparativo
                        fig_prob = go.Figure()
                        
                        fig_prob.add_trace(go.Bar(
                            x=[equipo_local[:40], equipo_visitante[:40]],
                            y=[prob_local*100, prob_visitante*100],
                            text=[f"{prob_local*100:.1f}%", f"{prob_visitante*100:.1f}%"],
                            textposition='outside',
                            marker=dict(
                                color=['rgba(0, 123, 255, 0.7)', 'rgba(255, 99, 71, 0.7)'],
                                line=dict(color='black', width=2)
                            )
                        ))
                        
                        fig_prob.update_layout(
                            title="Probabilidades de Victoria",
                            yaxis_title="Probabilidad (%)",
                            showlegend=False,
                            height=400,
                            yaxis=dict(range=[0, 100])
                        )
                        
                        st.plotly_chart(fig_prob, use_container_width=True)
                    
                    with col2:
                        # Gauge chart para confianza
                        confianza = abs(prob_local - prob_visitante)
                        
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=confianza*100,
                            title={'text': "Nivel de Confianza<br>(Diferencia de Prob.)"},
                            delta={'reference': 50},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 20], 'color': "lightgray"},
                                    {'range': [20, 40], 'color': "gray"},
                                    {'range': [40, 100], 'color': "darkgray"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        
                        fig_gauge.update_layout(height=400)
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Interpretación
                    st.subheader("💡 Interpretación")
                    
                    if confianza < 0.1:
                        nivel = "MUY BAJO"
                        interpretacion = "El partido está muy parejo. Cualquiera de los dos equipos tiene posibilidades reales de ganar."
                        color = "blue"
                    elif confianza < 0.2:
                        nivel = "BAJO"
                        interpretacion = "Hay una ligera ventaja para el favorito, pero el resultado es bastante incierto."
                        color = "green"
                    elif confianza < 0.3:
                        nivel = "MODERADO"
                        interpretacion = "El favorito tiene una ventaja clara, pero no es un resultado garantizado."
                        color = "orange"
                    else:
                        nivel = "ALTO"
                        interpretacion = "El favorito tiene una ventaja significativa y es probable que gane el partido."
                        color = "red"
                    
                    st.markdown(f"""
                    **Nivel de Confianza:** :{color}[{nivel}] ({confianza*100:.1f}%)
                    
                    {interpretacion}
                    """)
                    
                    # Información adicional
                    with st.expander("ℹ️ Más información sobre la predicción"):
                        st.markdown(f"""
                        **Sobre el Modelo:**
                        - Tipo: Logistic Regression
                        - ROC-AUC: {modelo_data['metrics']['roc_auc']:.3f}
                        - Accuracy: {modelo_data['metrics']['accuracy']:.3f}
                        - Features utilizadas: {len(modelo_data['features'])}
                        
                        **Cómo interpretar los resultados:**
                        - Las probabilidades representan la estimación del modelo sobre las chances de cada equipo.
                        - Una probabilidad del 50% indica un partido completamente parejo.
                        - Las probabilidades extremas (>80% o <20%) se suavizan para reflejar la incertidumbre real del deporte.
                        - El modelo considera múltiples métricas de rendimiento histórico de los jugadores.
                        """)
                        
                except Exception as e:
                    st.error(f"❌ Error al realizar la predicción: {str(e)}")
                    st.info("Asegúrate de que los archivos de datos estén correctamente formateados.")

# ============================================================================
# TAB 4: ANÁLISIS DE FACTORES DE ÉXITO
# ============================================================================

with tab4:
    st.header("🔍 Análisis: ¿Por qué unos equipos ganan más que otros?")
    
    st.markdown("""
    Esta sección analiza las **características que diferencian a los mejores equipos** de los equipos con menor rendimiento,
    ayudándote a entender qué factores son más importantes para la victoria.
    """)
    
    # Cargar stats_equipos para análisis
    if stats_equipos is not None and len(stats_equipos) > 0:
        
        # Dividir en top y bottom performers
        top_percentile = 20  # Top 20%
        bottom_percentile = 20  # Bottom 20%
        
        top_n = max(1, int(len(stats_equipos) * top_percentile / 100))
        bottom_n = max(1, int(len(stats_equipos) * bottom_percentile / 100))
        
        top_equipos = stats_equipos.head(top_n)
        bottom_equipos = stats_equipos.tail(bottom_n)
        
        # Métricas clave para comparar
        metricas_comparacion = {
            'Velocidad Media (m/s)': 'Velocidad de desplazamiento promedio',
            'Golpes Totales': 'Cantidad total de golpes',
            'Tiempo Cerca Pelota (%)': 'Porcentaje del tiempo cerca de la pelota',
            'Distancia Total (m)': 'Distancia total recorrida',
            'Aceleración Media (m/s²)': 'Capacidad de aceleración',
            'Tiempo Zona Ofensiva (%)': 'Tiempo en posición ofensiva'
        }
        
        # ====================================================================
        # 1. COMPARACIÓN DE PROMEDIOS
        # ====================================================================
        st.subheader("📊 Comparación de Métricas: Top 20% vs Bottom 20%")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### 🥇 Top {top_percentile}% Mejores Equipos")
            st.metric("Equipos", top_n)
            st.metric("Probabilidad Promedio", f"{top_equipos['Prob. Victoria'].mean():.1%}")
        
        with col2:
            st.markdown(f"### 📉 Bottom {bottom_percentile}% Equipos")
            st.metric("Equipos", bottom_n)
            st.metric("Probabilidad Promedio", f"{bottom_equipos['Prob. Victoria'].mean():.1%}")
        
        st.divider()
        
        # ====================================================================
        # 2. ANÁLISIS MÉTRICA POR MÉTRICA
        # ====================================================================
        st.subheader("🎯 Diferencias Clave en el Rendimiento")
        
        # Calcular diferencias
        diferencias = []
        for metrica, descripcion in metricas_comparacion.items():
            if metrica in top_equipos.columns and metrica in bottom_equipos.columns:
                top_mean = top_equipos[metrica].mean()
                bottom_mean = bottom_equipos[metrica].mean()
                diff_absoluta = top_mean - bottom_mean
                diff_porcentual = ((top_mean - bottom_mean) / bottom_mean * 100) if bottom_mean != 0 else 0
                
                diferencias.append({
                    'Métrica': metrica,
                    'Descripción': descripcion,
                    'Top 20%': top_mean,
                    'Bottom 20%': bottom_mean,
                    'Diferencia': diff_absoluta,
                    'Diferencia %': diff_porcentual
                })
        
        df_diferencias = pd.DataFrame(diferencias)
        df_diferencias = df_diferencias.sort_values('Diferencia %', key=lambda x: abs(x), ascending=False)
        
        # Mostrar las 3 métricas más importantes
        st.markdown("### 🔥 Top 3 Factores Más Diferenciales")
        
        for idx, row in df_diferencias.head(3).iterrows():
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**{row['Métrica']}**")
                    st.caption(row['Descripción'])
                
                with col2:
                    st.metric(
                        "Top 20%", 
                        f"{row['Top 20%']:.2f}",
                        delta=f"{row['Diferencia %']:+.1f}%"
                    )
                
                with col3:
                    st.metric(
                        "Bottom 20%", 
                        f"{row['Bottom 20%']:.2f}"
                    )
                
                # Interpretación
                if row['Diferencia %'] > 0:
                    st.success(f"✅ Los mejores equipos tienen un **{abs(row['Diferencia %']):.1f}% MÁS** en esta métrica")
                else:
                    st.info(f"ℹ️ Los mejores equipos tienen un **{abs(row['Diferencia %']):.1f}% MENOS** en esta métrica")
                
                st.divider()
        
        # ====================================================================
        # 3. GRÁFICO COMPARATIVO
        # ====================================================================
        st.subheader("📈 Visualización Comparativa")
        
        # Crear gráfico de barras comparativo
        fig_comparacion = go.Figure()
        
        metricas_plot = [row['Métrica'] for _, row in df_diferencias.iterrows()]
        
        # Normalizar valores para mejor visualización (0-100)
        top_values_norm = []
        bottom_values_norm = []
        
        for metrica in metricas_plot:
            row = df_diferencias[df_diferencias['Métrica'] == metrica].iloc[0]
            max_val = max(row['Top 20%'], row['Bottom 20%'])
            min_val = min(row['Top 20%'], row['Bottom 20%'])
            rango = max_val - min_val if max_val != min_val else 1
            
            top_norm = ((row['Top 20%'] - min_val) / rango) * 100
            bottom_norm = ((row['Bottom 20%'] - min_val) / rango) * 100
            
            top_values_norm.append(top_norm)
            bottom_values_norm.append(bottom_norm)
        
        fig_comparacion.add_trace(go.Bar(
            name=f'Top {top_percentile}%',
            y=metricas_plot,
            x=top_values_norm,
            orientation='h',
            marker=dict(color='#28a745'),
            text=[f"{v:.1f}" for v in top_values_norm],
            textposition='outside'
        ))
        
        fig_comparacion.add_trace(go.Bar(
            name=f'Bottom {bottom_percentile}%',
            y=metricas_plot,
            x=bottom_values_norm,
            orientation='h',
            marker=dict(color='#dc3545'),
            text=[f"{v:.1f}" for v in bottom_values_norm],
            textposition='outside'
        ))
        
        fig_comparacion.update_layout(
            title='Comparación Normalizada de Métricas (0-100)',
            xaxis_title='Puntuación Normalizada',
            yaxis_title='',
            barmode='group',
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_comparacion, use_container_width=True)
        
        # ====================================================================
        # 4. CONCLUSIONES Y RECOMENDACIONES
        # ====================================================================
        st.subheader("💡 Conclusiones y Recomendaciones")
        
        # Identificar la métrica más importante
        metrica_mas_importante = df_diferencias.iloc[0]
        
        st.markdown(f"""
        ### Factores Clave del Éxito
        
        Basado en el análisis de **{len(stats_equipos)} equipos**, hemos identificado los siguientes patrones:
        
        **🎯 Factor #1: {metrica_mas_importante['Métrica']}**
        - Los equipos del top 20% tienen un **{abs(metrica_mas_importante['Diferencia %']):.1f}%** 
          {"más" if metrica_mas_importante['Diferencia %'] > 0 else "menos"} que los equipos del bottom 20%
        - Promedio top: **{metrica_mas_importante['Top 20%']:.2f}**
        - Promedio bottom: **{metrica_mas_importante['Bottom 20%']:.2f}**
        
        ### 📋 Recomendaciones para Mejorar
        """)
        
        # Generar recomendaciones basadas en las métricas
        recomendaciones = []
        
        for _, row in df_diferencias.head(3).iterrows():
            if 'Velocidad' in row['Métrica'] and row['Diferencia %'] > 0:
                recomendaciones.append("🏃 **Trabaja en la velocidad de desplazamiento**: Los equipos ganadores se mueven más rápido en la cancha")
            elif 'Golpes' in row['Métrica'] and row['Diferencia %'] > 0:
                recomendaciones.append("🎾 **Aumenta la agresividad ofensiva**: Más golpes generalmente indican mayor control del juego")
            elif 'Cerca Pelota' in row['Métrica'] and row['Diferencia %'] > 0:
                recomendaciones.append("🎯 **Mejora el posicionamiento**: Estar cerca de la pelota es clave para anticipar jugadas")
            elif 'Distancia Total' in row['Métrica'] and row['Diferencia %'] > 0:
                recomendaciones.append("💪 **Incrementa la resistencia física**: Los mejores equipos recorren más distancia")
            elif 'Aceleración' in row['Métrica'] and abs(row['Diferencia %']) > 5:
                recomendaciones.append("⚡ **Desarrolla explosividad**: La capacidad de acelerar rápidamente marca diferencias")
            elif 'Zona Ofensiva' in row['Métrica'] and row['Diferencia %'] > 0:
                recomendaciones.append("⚔️ **Mantén presión ofensiva**: Pasar más tiempo en zona ofensiva aumenta probabilidad de victoria")
        
        for i, rec in enumerate(recomendaciones[:4], 1):
            st.markdown(f"{i}. {rec}")
        
        # Mostrar tabla completa
        with st.expander("📊 Ver Tabla Completa de Comparación"):
            st.dataframe(
                df_diferencias.style.format({
                    'Top 20%': '{:.2f}',
                    'Bottom 20%': '{:.2f}',
                    'Diferencia': '{:.2f}',
                    'Diferencia %': '{:+.1f}%'
                }),
                use_container_width=True,
                height=400
            )
    
    else:
        st.warning("⚠️ No hay datos suficientes para realizar el análisis comparativo.")
        st.info("Asegúrate de que el archivo 'stats_equipos.csv' esté disponible y contenga datos.")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📊 Datos:**")
    st.markdown(f"- {len(tabla_posiciones)} equipos analizados")
    if ranking_jugadores is not None:
        st.markdown(f"- {len(ranking_jugadores)} jugadores únicos")

with col2:
    st.markdown("**🤖 Modelo:**")
    st.markdown("- Logistic Regression")
    st.markdown(f"- ROC-AUC: {modelo_data['metrics']['roc_auc']:.3f}")

with col3:
    st.markdown("**ℹ️ Info:**")
    st.markdown("- Dashboard v1.0")
    st.markdown("- © 2025 Paddle Analytics")

st.caption("Desarrollado con ❤️ usando Streamlit y Scikit-learn")
