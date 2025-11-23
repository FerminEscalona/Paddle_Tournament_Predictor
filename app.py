import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
import os
import json

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Paddle Tournament Predictor",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONSTANTS ---
DATA_PATH = "Ranking_Tournament_Prediction.csv"
STATS_PATH = "Dataset_Maestro_Real_Target.csv"

# Features used in the model (must match training order)
FEATURES = [
    'video_distancia_total_m',
    'video_velocidad_max',
    'video_velocidad_media',
    'video_potencia_media',
    'video_pct_red',
    'video_golpes_totales'
]

# --- DATA LOADING ---
@st.cache_data
def load_ranking_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_player_stats():
    if not os.path.exists(STATS_PATH):
        return None
    df = pd.read_csv(STATS_PATH)
    # Calculate average stats per player
    # Filter only feature columns + player name
    cols = ['player_name_clean'] + FEATURES
    # Check if columns exist
    available_cols = [c for c in cols if c in df.columns]
    if 'player_name_clean' not in available_cols:
        return None
        
    df_stats = df[available_cols].groupby('player_name_clean').mean().reset_index()
    return df_stats


# --- UI COMPONENTS ---

def render_dashboard(df_ranking):
    st.header("🏆 Dashboard del Torneo")
    
    if df_ranking is None:
        st.error(f"No se encontró el archivo {DATA_PATH}")
        return

    # KPI Cards
    best_player = df_ranking.loc[df_ranking['Probabilidad_Campeonato_%'].idxmax()]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Favorito del Torneo", best_player['Jugador'].title())
    with col2:
        st.metric("Probabilidad de Victoria", f"{best_player['Probabilidad_Campeonato_%']}%")
    with col3:
        st.metric("Victorias Simuladas", int(best_player['Victorias_Simuladas']))

    st.markdown("---")

    # Chart and Table
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Probabilidades de Campeonato")
        fig = px.bar(
            df_ranking.sort_values('Probabilidad_Campeonato_%', ascending=True),
            x='Probabilidad_Campeonato_%',
            y='Jugador',
            orientation='h',
            text='Probabilidad_Campeonato_%',
            color='Probabilidad_Campeonato_%',
            color_continuous_scale='Viridis',
            title="Probabilidad de Ganar el Torneo (%)"
        )
        fig.update_layout(showlegend=False, height=600)
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("Tabla de Posiciones")
        st.dataframe(
            df_ranking.style.format({"Probabilidad_Campeonato_%": "{:.1f}%"}),
            use_container_width=True,
            height=600
        )

def render_comparator(df_stats):
    st.header("🆚 Comparador de Jugadores")
    
    if df_stats is None:
        st.error(f"No se encontró el archivo de estadísticas {STATS_PATH}")
        return

    players = df_stats['player_name_clean'].unique().tolist()
    players.sort()
    
    col1, col2 = st.columns(2)
    with col1:
        p1 = st.selectbox("Seleccionar Jugador 1", players, index=0)
    with col2:
        p2 = st.selectbox("Seleccionar Jugador 2", players, index=1 if len(players) > 1 else 0)

    if p1 and p2:
        stats_p1 = df_stats[df_stats['player_name_clean'] == p1].iloc[0]
        stats_p2 = df_stats[df_stats['player_name_clean'] == p2].iloc[0]
        
        # Radar Chart Features
        radar_features = ['video_velocidad_media', 'video_potencia_media', 'video_pct_red', 'video_distancia_total_m']
        # Normalize for visualization (simple min-max scaling based on current max in dataset could be better, but raw is requested)
        # For better radar, let's normalize against the max of the whole dataset
        
        categories = ['Velocidad Media', 'Potencia Media', '% en Red', 'Distancia Total']
        
        values_p1 = [stats_p1[f] for f in radar_features]
        values_p2 = [stats_p2[f] for f in radar_features]
        
        # Normalize values to 0-1 range for the chart to look good, or just plot raw if scales are similar.
        # Scales are very different (pct is 0-100, speed is small). We should normalize.
        max_values = df_stats[radar_features].max()
        
        values_p1_norm = [v / m if m > 0 else 0 for v, m in zip(values_p1, max_values)]
        values_p2_norm = [v / m if m > 0 else 0 for v, m in zip(values_p2, max_values)]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values_p1_norm,
            theta=categories,
            fill='toself',
            name=p1.title()
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=values_p2_norm,
            theta=categories,
            fill='toself',
            name=p2.title()
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Comparación de Atributos Físicos (Normalizado)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Raw Stats Table
        st.subheader("Estadísticas Crudas (Promedio)")
        comp_df = pd.DataFrame({
            'Métrica': categories,
            p1.title(): values_p1,
            p2.title(): values_p2
        })
        st.table(comp_df)



# --- MAIN APP ---

def main():
    st.sidebar.title("🎾 Paddle Predictor")
    
    # Navigation
    page = st.sidebar.radio("Navegación", ["Dashboard", "Comparador"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("Proyecto de Machine Learning para predicción de torneos de Pádel.")
    
    # Load Data
    df_ranking = load_ranking_data()
    df_stats = load_player_stats()
    
    if page == "Dashboard":
        render_dashboard(df_ranking)
    elif page == "Comparador":
        render_comparator(df_stats)

if __name__ == "__main__":
    main()
