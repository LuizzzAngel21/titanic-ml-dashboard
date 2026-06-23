# =============================================================================
# PC2 — Dashboard Interactivo Titanic | USIL 2026-1
# Streamlit app — Panel A: EDA Interactivo | Panel B: Predicción
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic ML Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fuente y fondo general */
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
    .main { background-color: #F8FAFC; }

    /* Encabezado principal */
    .hero {
        background: linear-gradient(135deg, #1B3A6B 0%, #2E5CA8 100%);
        border-radius: 12px;
        padding: 28px 36px;
        margin-bottom: 24px;
        color: white;
    }
    .hero h1 { font-size: 2rem; font-weight: 700; margin: 0 0 6px 0; color: white; }
    .hero p  { font-size: 1rem; margin: 0; opacity: 0.85; color: #E8F0FF; }

    /* Tarjetas KPI */
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 18px 20px;
        border-left: 4px solid #2E5CA8;
        box-shadow: 0 1px 6px rgba(0,0,0,0.07);
        text-align: center;
    }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #1B3A6B; }
    .kpi-label { font-size: 0.82rem; color: #64748B; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.05em; }

    /* Sección panel */
    .panel-title {
        font-size: 1.25rem; font-weight: 700; color: #1B3A6B;
        border-bottom: 2px solid #2E5CA8;
        padding-bottom: 6px; margin-bottom: 16px;
    }

    /* Resultado predicción */
    .pred-survived {
        background: #DCFCE7; border: 2px solid #16A34A;
        border-radius: 12px; padding: 20px 24px; text-align: center;
    }
    .pred-died {
        background: #FEE2E2; border: 2px solid #DC2626;
        border-radius: 12px; padding: 20px 24px; text-align: center;
    }
    .pred-label { font-size: 1.6rem; font-weight: 700; margin-bottom: 4px; }
    .pred-sub   { font-size: 0.95rem; color: #374151; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #EFF4FF; border-radius: 8px 8px 0 0;
        padding: 10px 24px; font-weight: 600; color: #1B3A6B;
    }
    .stTabs [aria-selected="true"] {
        background: #2E5CA8 !important; color: white !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #F0F4FB; }
    .sidebar-section { font-weight: 700; color: #1B3A6B; font-size: 0.9rem;
                       text-transform: uppercase; letter-spacing: 0.06em;
                       margin-top: 16px; margin-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS Y MODELO
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def cargar_datos():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)

@st.cache_resource
def cargar_modelo():
    """Entrena el KNN si no existe el .pkl, o lo carga desde disco."""
    if os.path.exists("mejor_modelo_knn.pkl") and os.path.exists("scaler.pkl"):
        modelo  = joblib.load("mejor_modelo_knn.pkl")
        scaler  = joblib.load("scaler.pkl")
        return modelo, scaler

    # Entrenar en tiempo real si no hay .pkl
    df = cargar_datos()
    df = df.drop(columns=['PassengerId','Name','Ticket','Cabin'])
    df['Age']      = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare']     = df['Fare'].fillna(df['Fare'].median())
    df['Sex']      = LabelEncoder().fit_transform(df['Sex'])
    df = pd.get_dummies(df, columns=['Embarked','Pclass'], drop_first=False)
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    X = df.drop(columns=['Survived'])
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    modelo = KNeighborsClassifier(n_neighbors=7, metric='minkowski', weights='distance')
    modelo.fit(X_train_sc, y_train)
    return modelo, scaler

df_raw  = cargar_datos()
modelo, scaler = cargar_modelo()

# ═══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <h1>🚢 Titanic ML Dashboard</h1>
    <p>Herramienta interactiva de Machine Learning — Clasificación de supervivencia · PC2 USIL 2026-1</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — FILTROS GLOBALES
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/320px-RMS_Titanic_3.jpg",
             use_container_width=True)
    st.markdown("### 🎛️ Filtros del Dataset")

    st.markdown('<div class="sidebar-section">Clase del pasajero</div>', unsafe_allow_html=True)
    clases = st.multiselect("Pclass", [1, 2, 3], default=[1, 2, 3],
                             format_func=lambda x: f"{'1ª' if x==1 else '2ª' if x==2 else '3ª'} Clase")

    st.markdown('<div class="sidebar-section">Sexo</div>', unsafe_allow_html=True)
    sexo = st.multiselect("Sex", ["male", "female"], default=["male", "female"],
                           format_func=lambda x: "Hombre" if x=="male" else "Mujer")

    st.markdown('<div class="sidebar-section">Rango de edad</div>', unsafe_allow_html=True)
    edad_min, edad_max = st.slider("Edad", 0, 80, (0, 80))

    st.markdown('<div class="sidebar-section">Puerto de embarque</div>', unsafe_allow_html=True)
    puertos = st.multiselect("Embarked", ["S", "C", "Q"], default=["S", "C", "Q"],
                              format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x])

    st.markdown("---")
    st.caption("PC2 · Agentes Inteligentes · USIL 2026-1")

# ── Aplicar filtros ───────────────────────────────────────────────────────────
df = df_raw.copy()
df_f = df[
    (df['Pclass'].isin(clases)) &
    (df['Sex'].isin(sexo)) &
    (df['Age'].fillna(df['Age'].median()).between(edad_min, edad_max)) &
    (df['Embarked'].isin(puertos if puertos else ["S","C","Q"]))
]

# ═══════════════════════════════════════════════════════════════════════════════
# TABS — PANEL A y PANEL B
# ═══════════════════════════════════════════════════════════════════════════════
tab_a, tab_b = st.tabs(["📊  Panel A — Análisis de Datos", "🤖  Panel B — Predicción"])

# ───────────────────────────────────────────────────────────────────────────────
# PANEL A — ANÁLISIS DE DATOS
# ───────────────────────────────────────────────────────────────────────────────
with tab_a:
    st.markdown('<div class="panel-title">📊 Análisis Exploratorio Interactivo</div>',
                unsafe_allow_html=True)

    # ── KPIs ─────────────────────────────────────────────────────────────────
    total      = len(df_f)
    n_surv     = df_f['Survived'].sum()
    n_no_surv  = total - n_surv
    pct_surv   = (n_surv / total * 100) if total > 0 else 0
    edad_media = df_f['Age'].mean()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{total}</div><div class="kpi-label">Pasajeros activos</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{n_surv}</div><div class="kpi-label">Sobrevivieron</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{pct_surv:.1f}%</div><div class="kpi-label">Tasa de supervivencia</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{edad_media:.1f}</div><div class="kpi-label">Edad promedio (años)</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Fila 1: Gráfico de barras + Pie ──────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Supervivencia por Clase y Sexo**")
        if len(df_f) > 0:
            fig, ax = plt.subplots(figsize=(7, 4))
            surv_cs = df_f.groupby(['Pclass','Sex'])['Survived'].mean().unstack()
            surv_cs.plot(kind='bar', ax=ax,
                         color=['#E74C3C','#3498DB'], edgecolor='white',
                         width=0.65)
            ax.set_title("Tasa de Supervivencia por Clase y Sexo", fontsize=11, fontweight='bold')
            ax.set_xlabel("Clase del Pasajero")
            ax.set_ylabel("Tasa de Supervivencia")
            ax.set_xticklabels(["1ª Clase","2ª Clase","3ª Clase"], rotation=0)
            ax.legend(["Hombre","Mujer"], title="Sexo")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0%}'))
            ax.set_ylim(0, 1.1)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Sin datos con los filtros seleccionados.")

    with col2:
        st.markdown("**Distribución de Supervivencia**")
        if len(df_f) > 0:
            fig, ax = plt.subplots(figsize=(5, 4))
            vals  = [n_no_surv, n_surv]
            etqs  = [f'No sobrevivió\n({n_no_surv})', f'Sobrevivió\n({n_surv})']
            colrs = ['#E74C3C','#2ECC71']
            wedges, texts, autotexts = ax.pie(
                vals, labels=etqs, colors=colrs,
                autopct='%1.1f%%', startangle=90,
                wedgeprops={'edgecolor':'white','linewidth':2})
            for at in autotexts:
                at.set_fontsize(11); at.set_fontweight('bold')
            ax.set_title("Distribución Survived", fontsize=11, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Sin datos con los filtros seleccionados.")

    # ── Fila 2: Histograma de edad + Mapa de calor ────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Distribución de Edad por Supervivencia**")
        if len(df_f) > 0:
            fig, ax = plt.subplots(figsize=(7, 4))
            ages_no = df_f[df_f['Survived']==0]['Age'].dropna()
            ages_si = df_f[df_f['Survived']==1]['Age'].dropna()
            ax.hist(ages_no, bins=20, alpha=0.7, color='#E74C3C',
                    label='No sobrevivió', edgecolor='white')
            ax.hist(ages_si, bins=20, alpha=0.7, color='#2ECC71',
                    label='Sobrevivió', edgecolor='white')
            ax.set_xlabel("Edad (años)")
            ax.set_ylabel("Número de Pasajeros")
            ax.set_title("Distribución de Edad según Supervivencia", fontsize=11, fontweight='bold')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Sin datos con los filtros seleccionados.")

    with col4:
        st.markdown("**Matriz de Correlación**")
        if len(df_f) > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            num_cols = ['Survived','Pclass','Age','SibSp','Parch','Fare']
            corr = df_f[num_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                        center=0, ax=ax, linewidths=0.5,
                        annot_kws={'size': 9}, mask=mask,
                        cbar_kws={'shrink': 0.8})
            ax.set_title("Correlación de Pearson — Variables Numéricas",
                         fontsize=11, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Sin datos con los filtros seleccionados.")

    # ── Tabla de datos filtrados ──────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 Ver tabla de datos filtrados"):
        cols_show = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
        st.dataframe(
            df_f[cols_show].reset_index(drop=True),
            use_container_width=True, height=260
        )
        st.caption(f"Mostrando {len(df_f)} de {len(df_raw)} registros")

    # ── Estadísticos descriptivos ─────────────────────────────────────────────
    with st.expander("📈 Estadísticos descriptivos del subconjunto"):
        desc_cols = ['Age','Fare','SibSp','Parch']
        st.dataframe(df_f[desc_cols].describe().round(2), use_container_width=True)

# ───────────────────────────────────────────────────────────────────────────────
# PANEL B — PREDICCIÓN
# ───────────────────────────────────────────────────────────────────────────────
with tab_b:
    st.markdown('<div class="panel-title">🤖 Predicción de Supervivencia — Modelo KNN (k=7)</div>',
                unsafe_allow_html=True)

    st.info("📌 **Ingresa los datos del pasajero** y el modelo KNN predecirá si habría sobrevivido al naufragio del Titanic (1912).", icon="ℹ️")

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("#### 👤 Datos del Pasajero")

        pclass_input = st.selectbox(
            "Clase del pasajero (Pclass)",
            options=[1, 2, 3],
            format_func=lambda x: f"{'1ª Clase (Alta)' if x==1 else '2ª Clase (Media)' if x==2 else '3ª Clase (Económica)'}",
        )
        sex_input = st.radio(
            "Sexo", ["male", "female"],
            format_func=lambda x: "👨 Hombre" if x=="male" else "👩 Mujer",
            horizontal=True
        )
        age_input = st.slider("Edad (años)", 0, 80, 28)
        fare_input = st.slider("Tarifa pagada (£)", 0.0, 520.0, 32.0, step=0.5)

        st.markdown("#### 👨‍👩‍👧 Familia a bordo")
        sibsp_input = st.number_input("Hermanos / Cónyuge (SibSp)", 0, 8, 0)
        parch_input = st.number_input("Padres / Hijos (Parch)", 0, 6, 0)

        embarked_input = st.selectbox(
            "Puerto de embarque",
            options=["S", "C", "Q"],
            format_func=lambda x: {"S": "🇬🇧 Southampton (S)", "C": "🇫🇷 Cherbourg (C)", "Q": "🇮🇪 Queenstown (Q)"}[x]
        )

        predecir = st.button("🔍 Predecir supervivencia", use_container_width=True, type="primary")

    with col_result:
        st.markdown("#### 📊 Resultado de la Predicción")

        if predecir:
            # ── Preparar input ─────────────────────────────────────────────
            sex_enc       = 0 if sex_input == "female" else 1
            embarked_C    = 1 if embarked_input == "C" else 0
            embarked_Q    = 1 if embarked_input == "Q" else 0
            embarked_S    = 1 if embarked_input == "S" else 0
            pclass_1      = 1 if pclass_input == 1 else 0
            pclass_2      = 1 if pclass_input == 2 else 0
            pclass_3      = 1 if pclass_input == 3 else 0

            X_input = np.array([[sex_enc, age_input, sibsp_input, parch_input,
                                  fare_input,
                                  embarked_C, embarked_Q, embarked_S,
                                  pclass_1, pclass_2, pclass_3]])

            X_input_sc = scaler.transform(X_input)
            pred       = modelo.predict(X_input_sc)[0]
            proba      = modelo.predict_proba(X_input_sc)[0]
            conf       = proba[pred] * 100

            if pred == 1:
                st.markdown(f"""
                <div class="pred-survived">
                    <div class="pred-label">✅ SOBREVIVIÓ</div>
                    <div class="pred-sub">Confianza del modelo: <strong>{conf:.1f}%</strong></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-died">
                    <div class="pred-label">❌ NO SOBREVIVIÓ</div>
                    <div class="pred-sub">Confianza del modelo: <strong>{conf:.1f}%</strong></div>
                </div>
                """, unsafe_allow_html=True)

            # ── Probabilidades ─────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Probabilidad por clase:**")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.metric("❌ No sobrevive", f"{proba[0]*100:.1f}%")
            with p_col2:
                st.metric("✅ Sobrevive", f"{proba[1]*100:.1f}%")

            # Barra de probabilidad
            fig_p, ax_p = plt.subplots(figsize=(5, 1.2))
            ax_p.barh([""], [proba[0]], color='#E74C3C', label='No sobrevive')
            ax_p.barh([""], [proba[1]], left=[proba[0]], color='#2ECC71', label='Sobrevive')
            ax_p.set_xlim(0, 1)
            ax_p.set_xlabel("Probabilidad")
            ax_p.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0%}'))
            ax_p.legend(loc='upper right', fontsize=8)
            ax_p.set_title("Distribución de probabilidad", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_p)
            plt.close()

            # ── Explicación contextualizada ────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            clase_str = "1ª clase (alta)" if pclass_input==1 else "2ª clase (media)" if pclass_input==2 else "3ª clase (económica)"
            sexo_str  = "mujer" if sex_input=="female" else "hombre"

            if pred == 1:
                st.success(f"""
**¿Qué significa este resultado?**

El modelo predice que este pasajero ({sexo_str}, {age_input} años, {clase_str}) 
**habría sobrevivido** al naufragio del Titanic con una confianza del {conf:.1f}%.

Factores clave: el sexo femenino y la clase alta aumentan significativamente 
la probabilidad de supervivencia, ya que la política de evacuación priorizó 
a "mujeres y niños primero" y los botes salvavidas estaban más accesibles 
para los pasajeros de primera clase.
                """)
            else:
                st.error(f"""
**¿Qué significa este resultado?**

El modelo predice que este pasajero ({sexo_str}, {age_input} años, {clase_str}) 
**no habría sobrevivido** al naufragio con una confianza del {conf:.1f}%.

Factores clave: los hombres de tercera clase tuvieron la tasa de supervivencia 
más baja (13%). La ubicación de los camarotes en cubierta inferior y la 
distancia a los botes salvavidas fueron factores determinantes.
                """)
        else:
            st.markdown("""
            <div style="background:#F0F4FB; border-radius:10px; padding:40px;
                        text-align:center; color:#64748B; margin-top:20px;">
                <div style="font-size:3rem;">🚢</div>
                <div style="font-size:1.1rem; font-weight:600; margin-top:12px;">
                    Completa el formulario y presiona<br><strong>«Predecir supervivencia»</strong>
                </div>
                <div style="font-size:0.85rem; margin-top:8px;">
                    Modelo: KNN (k=7) · Accuracy: 82.7% · F1-Score: 0.77
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Info del modelo
        with st.expander("ℹ️ Sobre el modelo utilizado"):
            st.markdown("""
| Parámetro | Valor |
|---|---|
| Algoritmo | K-Nearest Neighbors (KNN) |
| k vecinos | 7 |
| Ponderación | Por distancia (weights='distance') |
| Accuracy (test) | **82.68%** |
| F1-Score (test) | **0.7704** |
| Precision | 0.7879 |
| Recall | 0.7536 |
| Scaler | StandardScaler |
| Dataset | Titanic — Kaggle (891 registros) |
| Split | 80% train / 20% test (random_state=42) |
            """)
