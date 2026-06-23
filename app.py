# =============================================================================
# PC2 — Dashboard Interactivo Titanic | USIL 2026-1
# Streamlit app — Panel A: EDA Interactivo | Panel B: Predicción
# Design system: light, clean, tipo "data product" profesional
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib, os, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic · Supervivencia ML",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS + CSS GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
# Paleta: Navy #0F2B5B  |  Blue #1E54B7  |  Sky #4F86F7  |  Mist #EDF2FF
# Accent vivo: Coral #E8533A (no sobrevivió)  |  Emerald #1FAB6E (sobrevivió)
# Neutral: #F7F9FC fondo  |  #E4EAF4 borde  |  #6B7A99 texto secundario

st.markdown("""
<style>
/* ── Reset & base ─────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    color: #1A2340;
}
.main .block-container {
    background: #F7F9FC;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1280px;
}

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E4EAF4;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.25rem;
}
.sb-logo {
    display: flex; align-items: center; gap: 10px;
    padding: 0 0 16px 0;
    border-bottom: 1px solid #E4EAF4;
    margin-bottom: 18px;
}
.sb-logo-icon { font-size: 2rem; line-height: 1; }
.sb-logo-title { font-size: 1rem; font-weight: 700; color: #0F2B5B; line-height: 1.2; }
.sb-logo-sub   { font-size: 0.7rem; color: #6B7A99; }

.sb-section {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #6B7A99;
    margin: 18px 0 6px 0;
}
.sb-divider { border: none; border-top: 1px solid #E4EAF4; margin: 14px 0; }

/* ── Hero banner ──────────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(118deg, #0F2B5B 0%, #1E54B7 60%, #4F86F7 100%);
    border-radius: 14px;
    padding: 26px 32px;
    margin-bottom: 20px;
    display: flex; align-items: center; gap: 18px;
}
.hero-icon { font-size: 3rem; }
.hero-text h1 {
    font-size: 1.55rem; font-weight: 800; color: #FFFFFF;
    margin: 0 0 4px 0; letter-spacing: -0.02em;
}
.hero-text p {
    font-size: 0.88rem; color: #B8CCEE; margin: 0;
}
.hero-badge {
    margin-left: auto;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 8px; padding: 8px 14px; text-align: center;
    color: white; white-space: nowrap;
}
.hero-badge .hb-value { font-size: 1.3rem; font-weight: 700; }
.hero-badge .hb-label { font-size: 0.68rem; opacity: 0.8; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── KPI cards ────────────────────────────────────────────────────────────── */
.kpi-row { display: flex; gap: 14px; margin-bottom: 22px; }
.kpi {
    flex: 1; background: #FFFFFF;
    border: 1px solid #E4EAF4;
    border-radius: 12px; padding: 18px 20px;
    box-shadow: 0 1px 4px rgba(30,84,183,0.06);
    min-width: 0;
}
.kpi-val  { font-size: 2rem; font-weight: 800; color: #0F2B5B; line-height: 1.1; }
.kpi-lbl  { font-size: 0.72rem; color: #6B7A99; text-transform: uppercase;
             letter-spacing: 0.07em; margin-top: 3px; }
.kpi-delta { font-size: 0.78rem; font-weight: 600; margin-top: 6px; }
.kpi-pos  { color: #1FAB6E; }
.kpi-neg  { color: #E8533A; }

/* ── Section header ───────────────────────────────────────────────────────── */
.sec-header {
    display: flex; align-items: center; gap: 8px;
    margin: 0 0 14px 0;
}
.sec-header-icon { font-size: 1.1rem; }
.sec-header-text { font-size: 1rem; font-weight: 700; color: #0F2B5B; }
.sec-header-line {
    flex: 1; height: 1px; background: #E4EAF4; margin-left: 6px;
}

/* ── Chart cards ──────────────────────────────────────────────────────────── */
.chart-card {
    background: #FFFFFF; border: 1px solid #E4EAF4;
    border-radius: 12px; padding: 18px 20px;
    box-shadow: 0 1px 4px rgba(30,84,183,0.05);
    margin-bottom: 14px;
}
.chart-title {
    font-size: 0.82rem; font-weight: 700; color: #0F2B5B;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin-bottom: 12px;
}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0; background: #EDF2FF;
    border-radius: 10px; padding: 4px;
    margin-bottom: 20px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px; padding: 9px 22px;
    font-weight: 600; font-size: 0.9rem;
    color: #6B7A99; background: transparent;
    border: none; transition: all 0.18s;
}
.stTabs [aria-selected="true"] {
    background: #FFFFFF !important;
    color: #0F2B5B !important;
    box-shadow: 0 1px 6px rgba(15,43,91,0.12) !important;
}

/* ── Pred result boxes ────────────────────────────────────────────────────── */
.pred-box {
    border-radius: 14px; padding: 24px 28px;
    text-align: center; margin: 16px 0;
}
.pred-survived {
    background: linear-gradient(135deg,#E8F9F1 0%,#D1F2E4 100%);
    border: 2px solid #1FAB6E;
}
.pred-died {
    background: linear-gradient(135deg,#FEF0ED 0%,#FDDDD6 100%);
    border: 2px solid #E8533A;
}
.pred-icon  { font-size: 2.8rem; margin-bottom: 6px; }
.pred-label { font-size: 1.5rem; font-weight: 800; margin-bottom: 4px; }
.pred-survived .pred-label { color: #0D7A47; }
.pred-died     .pred-label { color: #B83424; }
.pred-conf  { font-size: 0.9rem; color: #4A5568; }

/* ── Prob bar ─────────────────────────────────────────────────────────────── */
.prob-row { display: flex; gap: 10px; margin-top: 14px; }
.prob-cell {
    flex: 1; border-radius: 10px; padding: 12px 16px; text-align: center;
}
.prob-cell.died    { background: #FEF0ED; border: 1px solid #F5B8A8; }
.prob-cell.survived{ background: #E8F9F1; border: 1px solid #8FDBB6; }
.prob-pct { font-size: 1.4rem; font-weight: 800; }
.prob-died .prob-pct    { color: #B83424; }
.prob-survived .prob-pct{ color: #0D7A47; }
.prob-lbl { font-size: 0.72rem; color: #6B7A99; text-transform: uppercase;
             letter-spacing: 0.06em; margin-top: 2px; }

/* ── Form inputs ──────────────────────────────────────────────────────────── */
.form-section {
    background: #FFFFFF; border: 1px solid #E4EAF4;
    border-radius: 12px; padding: 20px 22px; margin-bottom: 14px;
}
.form-section-title {
    font-size: 0.75rem; font-weight: 700; color: #0F2B5B;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 14px; padding-bottom: 8px;
    border-bottom: 1px solid #E4EAF4;
}

/* ── Model badge ──────────────────────────────────────────────────────────── */
.model-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #EDF2FF; border: 1px solid #C4D4F5;
    border-radius: 20px; padding: 4px 12px;
    font-size: 0.78rem; font-weight: 600; color: #1E54B7;
    margin-bottom: 16px;
}

/* ── Info callout ─────────────────────────────────────────────────────────── */
.callout {
    background: #EDF2FF; border-left: 4px solid #1E54B7;
    border-radius: 0 10px 10px 0; padding: 12px 16px;
    font-size: 0.88rem; color: #0F2B5B; margin-bottom: 18px;
}

/* ── Context explanation ──────────────────────────────────────────────────── */
.context-box {
    background: #F7F9FC; border: 1px solid #E4EAF4;
    border-radius: 10px; padding: 16px 18px; margin-top: 14px;
    font-size: 0.87rem; color: #374151; line-height: 1.65;
}
.context-box strong { color: #0F2B5B; }

/* ── Empty state ──────────────────────────────────────────────────────────── */
.empty-state {
    background: #FFFFFF; border: 2px dashed #D0DCF5;
    border-radius: 14px; padding: 52px 32px; text-align: center; color: #6B7A99;
}
.empty-state .es-icon { font-size: 3.2rem; margin-bottom: 10px; }
.empty-state .es-title { font-size: 1rem; font-weight: 700; color: #0F2B5B; margin-bottom: 6px; }
.empty-state .es-sub   { font-size: 0.84rem; }

/* ── Matplotlib override ──────────────────────────────────────────────────── */
div[data-testid="stImage"] img { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB THEME
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#FFFFFF',
    'axes.facecolor':    '#FFFFFF',
    'axes.edgecolor':    '#E4EAF4',
    'axes.grid':          True,
    'grid.color':        '#EDF2FF',
    'grid.linewidth':     0.7,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.spines.left':   False,
    'axes.spines.bottom': True,
    'font.family':       'DejaVu Sans',
    'axes.labelcolor':   '#6B7A99',
    'xtick.color':       '#6B7A99',
    'ytick.color':       '#6B7A99',
    'axes.labelsize':     9,
    'xtick.labelsize':    8.5,
    'ytick.labelsize':    8.5,
    'axes.titlesize':    11,
    'axes.titleweight': 'bold',
    'axes.titlecolor':  '#0F2B5B',
    'axes.titlepad':    10,
})

C_RED   = '#E8533A'
C_GREEN = '#1FAB6E'
C_BLUE  = '#1E54B7'
C_SKY   = '#4F86F7'
C_NAVY  = '#0F2B5B'
C_MIST  = '#EDF2FF'

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS Y MODELO
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cargar_datos():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)

@st.cache_resource(show_spinner=False)
def cargar_modelo():
    if os.path.exists("mejor_modelo_knn.pkl") and os.path.exists("scaler.pkl"):
        return joblib.load("mejor_modelo_knn.pkl"), joblib.load("scaler.pkl")
    df = cargar_datos().copy()
    df.drop(columns=['PassengerId','Name','Ticket','Cabin'], inplace=True)
    df['Age']      = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare']     = df['Fare'].fillna(df['Fare'].median())
    df['Sex']      = LabelEncoder().fit_transform(df['Sex'])
    df = pd.get_dummies(df, columns=['Embarked','Pclass'], drop_first=False)
    df[df.select_dtypes('bool').columns] = df.select_dtypes('bool').astype(int)
    X, y = df.drop(columns=['Survived']), df['Survived']
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    sc = StandardScaler()
    X_sc = sc.fit_transform(X_tr)
    knn = KNeighborsClassifier(n_neighbors=7, metric='minkowski', weights='distance')
    knn.fit(X_sc, y_tr)
    return knn, sc

with st.spinner("Cargando datos y modelo…"):
    df_raw = cargar_datos()
    modelo, scaler = cargar_modelo()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-icon">🚢</div>
        <div>
            <div class="sb-logo-title">Titanic ML</div>
            <div class="sb-logo-sub">PC2 · USIL 2026-1</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Clase del pasajero</div>', unsafe_allow_html=True)
    clases = st.multiselect(
        label="clase", options=[1,2,3], default=[1,2,3], label_visibility="collapsed",
        format_func=lambda x: f"{'1ª' if x==1 else '2ª' if x==2 else '3ª'} Clase"
    )

    st.markdown('<div class="sb-section">Sexo</div>', unsafe_allow_html=True)
    sexo = st.multiselect(
        label="sexo", options=["male","female"], default=["male","female"],
        label_visibility="collapsed",
        format_func=lambda x: "Hombre" if x=="male" else "Mujer"
    )

    st.markdown('<div class="sb-section">Rango de edad</div>', unsafe_allow_html=True)
    edad_min, edad_max = st.slider("edad", 0, 80, (0,80), label_visibility="collapsed")

    st.markdown('<div class="sb-section">Puerto de embarque</div>', unsafe_allow_html=True)
    puerto_map = {"S":"Southampton","C":"Cherbourg","Q":"Queenstown"}
    puertos = st.multiselect(
        label="puerto", options=["S","C","Q"], default=["S","C","Q"],
        label_visibility="collapsed",
        format_func=lambda x: puerto_map[x]
    )

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    total_raw = len(df_raw)
    st.caption(f"Dataset: **{total_raw} pasajeros** · Kaggle Titanic")
    st.caption("Modelo: KNN k=7 · Acc 82.7% · F1 0.77")

# ─────────────────────────────────────────────────────────────────────────────
# FILTRO
# ─────────────────────────────────────────────────────────────────────────────
_clases  = clases  if clases  else [1,2,3]
_sexo    = sexo    if sexo    else ["male","female"]
_puertos = puertos if puertos else ["S","C","Q"]

df = df_raw.copy()
df['Age'] = df['Age'].fillna(df['Age'].median())
df_f = df[
    df['Pclass'].isin(_clases) &
    df['Sex'].isin(_sexo) &
    df['Age'].between(edad_min, edad_max) &
    df['Embarked'].isin(_puertos)
]

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
n_total = len(df_f)
n_surv  = int(df_f['Survived'].sum())
pct     = n_surv/n_total*100 if n_total else 0

st.markdown(f"""
<div class="hero">
    <div class="hero-icon">🚢</div>
    <div class="hero-text">
        <h1>Titanic · Predictor de Supervivencia</h1>
        <p>Herramienta de Machine Learning — Clasificación binaria · PC2 Agentes Inteligentes USIL 2026-1</p>
    </div>
    <div class="hero-badge">
        <div class="hb-value">{n_total}</div>
        <div class="hb-label">pasajeros activos</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_a, tab_b = st.tabs(["📊  Análisis de Datos", "🤖  Predicción"])

# ══════════════════════════════════════════════════════════════════════════════
# PANEL A
# ══════════════════════════════════════════════════════════════════════════════
with tab_a:

    # ── KPIs ─────────────────────────────────────────────────────────────────
    edad_med = df_f['Age'].mean()
    fare_med = df_f['Fare'].mean()
    n_no     = n_total - n_surv

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi">
            <div class="kpi-val">{n_total}</div>
            <div class="kpi-lbl">Pasajeros</div>
        </div>
        <div class="kpi">
            <div class="kpi-val">{n_surv}</div>
            <div class="kpi-lbl">Sobrevivieron</div>
            <div class="kpi-delta kpi-pos">↑ {pct:.1f}% del total</div>
        </div>
        <div class="kpi">
            <div class="kpi-val">{n_no}</div>
            <div class="kpi-lbl">No sobrevivieron</div>
            <div class="kpi-delta kpi-neg">↓ {100-pct:.1f}% del total</div>
        </div>
        <div class="kpi">
            <div class="kpi-val">{edad_med:.0f}</div>
            <div class="kpi-lbl">Edad promedio</div>
        </div>
        <div class="kpi">
            <div class="kpi-val">£{fare_med:.0f}</div>
            <div class="kpi-lbl">Tarifa promedio</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Fila 1 ───────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        st.markdown('<div class="chart-card"><div class="chart-title">Tasa de supervivencia por clase y sexo</div>', unsafe_allow_html=True)
        if n_total > 0:
            surv_cs = df_f.groupby(['Pclass','Sex'])['Survived'].mean().unstack()
            fig, ax = plt.subplots(figsize=(6.5, 3.5))
            colors = [C_RED, C_SKY]
            surv_cs.plot(kind='bar', ax=ax, color=colors, edgecolor='white',
                         linewidth=1.2, width=0.6)
            ax.set_title("Supervivencia por Clase y Sexo")
            ax.set_xlabel("Clase del Pasajero")
            ax.set_ylabel("Tasa de Supervivencia")
            ax.set_xticklabels(["1ª Clase","2ª Clase","3ª Clase"], rotation=0)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.0%}'))
            ax.set_ylim(0, 1.12)
            ax.legend(["Hombre","Mujer"], title="Sexo", framealpha=0.9,
                      fontsize=8, title_fontsize=8)
            for container in ax.containers:
                ax.bar_label(container, fmt=lambda v: f'{v:.0%}',
                             fontsize=7.5, padding=3, color='#374151')
            plt.tight_layout(pad=1.2)
            st.pyplot(fig, use_container_width=True)
            plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card"><div class="chart-title">Distribución de supervivencia</div>', unsafe_allow_html=True)
        if n_total > 0:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            vals  = [n_no, n_surv]
            cols  = [C_RED, C_GREEN]
            lbls  = [f'No sobrevivió\n{n_no} ({100-pct:.1f}%)',
                     f'Sobrevivió\n{n_surv} ({pct:.1f}%)']
            wedges, _, autos = ax.pie(
                vals, labels=None, colors=cols, autopct='%1.1f%%',
                startangle=90, pctdistance=0.72,
                wedgeprops={'edgecolor':'white','linewidth':2.5, 'width':0.6}
            )
            for a in autos:
                a.set_fontsize(10); a.set_fontweight('bold'); a.set_color('white')
            ax.legend(wedges, lbls, loc="lower center", bbox_to_anchor=(0.5,-0.12),
                      fontsize=8, framealpha=0.9, ncol=2)
            ax.set_title("Distribución Survived")
            plt.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True)
            plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Fila 2 ───────────────────────────────────────────────────────────────
    c3, c4 = st.columns(2, gap="medium")

    with c3:
        st.markdown('<div class="chart-card"><div class="chart-title">Distribución de edad por supervivencia</div>', unsafe_allow_html=True)
        if n_total > 0:
            fig, ax = plt.subplots(figsize=(6.5, 3.5))
            a0 = df_f[df_f['Survived']==0]['Age'].dropna()
            a1 = df_f[df_f['Survived']==1]['Age'].dropna()
            ax.hist(a0, bins=22, alpha=0.75, color=C_RED,
                    label='No sobrevivió', edgecolor='white', linewidth=0.6)
            ax.hist(a1, bins=22, alpha=0.75, color=C_GREEN,
                    label='Sobrevivió', edgecolor='white', linewidth=0.6)
            ax.axvline(df_f['Age'].median(), color=C_NAVY, linestyle='--',
                       linewidth=1.3, alpha=0.7,
                       label=f'Mediana {df_f["Age"].median():.0f} a.')
            ax.set_title("Distribución de Edad según Supervivencia")
            ax.set_xlabel("Edad (años)")
            ax.set_ylabel("N° de pasajeros")
            ax.legend(fontsize=8, framealpha=0.9)
            plt.tight_layout(pad=1.2)
            st.pyplot(fig, use_container_width=True)
            plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="chart-card"><div class="chart-title">Matriz de correlación — variables numéricas</div>', unsafe_allow_html=True)
        if n_total > 0:
            num = ['Survived','Pclass','Age','SibSp','Parch','Fare']
            corr = df_f[num].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            fig, ax = plt.subplots(figsize=(6, 3.8))
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                        cmap=cmap, center=0, vmin=-1, vmax=1,
                        ax=ax, linewidths=0.6, linecolor='#F7F9FC',
                        annot_kws={'size':8.5, 'weight':'600'},
                        cbar_kws={'shrink':0.75, 'pad':0.02})
            ax.set_title("Correlación de Pearson")
            plt.tight_layout(pad=1.2)
            st.pyplot(fig, use_container_width=True)
            plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Tabla ─────────────────────────────────────────────────────────────────
    with st.expander(f"📋  Ver los {n_total} registros filtrados"):
        cols_show = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
        st.dataframe(
            df_f[cols_show].reset_index(drop=True)
                .rename(columns={'Survived':'Sobrevivió','Pclass':'Clase',
                                 'Sex':'Sexo','Age':'Edad','Fare':'Tarifa',
                                 'Embarked':'Puerto'}),
            use_container_width=True, height=240
        )
    with st.expander("📈  Estadísticos descriptivos"):
        st.dataframe(df_f[['Age','Fare','SibSp','Parch']].describe().round(2),
                     use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — PREDICCIÓN
# ══════════════════════════════════════════════════════════════════════════════
with tab_b:

    st.markdown("""
    <div class="model-badge">
        🤖 KNN · k=7 · Accuracy 82.7% · F1-Score 0.77
    </div>
    <div class="callout">
        Ingresa los datos del pasajero y el modelo predice si habría sobrevivido
        al naufragio del RMS Titanic el 15 de abril de 1912.
    </div>
    """, unsafe_allow_html=True)

    col_form, col_res = st.columns([1, 1], gap="large")

    # ── Formulario ────────────────────────────────────────────────────────────
    with col_form:
        st.markdown('<div class="form-section"><div class="form-section-title">👤 Perfil del pasajero</div>', unsafe_allow_html=True)

        pclass_in = st.selectbox(
            "Clase del pasajero", [1,2,3],
            format_func=lambda x: {1:"🥇 1ª Clase — Alta",2:"🥈 2ª Clase — Media",3:"🥉 3ª Clase — Económica"}[x]
        )
        sex_in = st.radio(
            "Sexo", ["female","male"],
            format_func=lambda x: "👩 Mujer" if x=="female" else "👨 Hombre",
            horizontal=True
        )
        age_in = st.slider("Edad", 0, 80, 28,
                            help="Arrastra para ajustar la edad del pasajero")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="form-section"><div class="form-section-title">💰 Tarifa y embarque</div>', unsafe_allow_html=True)
        fare_in = st.slider("Tarifa pagada (£)", 0.0, 520.0, 32.0, step=1.0)
        emb_in  = st.selectbox(
            "Puerto de embarque", ["S","C","Q"],
            format_func=lambda x: {"S":"🇬🇧 Southampton","C":"🇫🇷 Cherbourg","Q":"🇮🇪 Queenstown"}[x]
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="form-section"><div class="form-section-title">👨‍👩‍👧 Familia a bordo</div>', unsafe_allow_html=True)
        fc1, fc2 = st.columns(2)
        with fc1:
            sibsp_in = st.number_input("Hermanos / Cónyuge", 0, 8, 0)
        with fc2:
            parch_in = st.number_input("Padres / Hijos", 0, 6, 0)
        st.markdown('</div>', unsafe_allow_html=True)

        predecir = st.button("🔍  Predecir supervivencia",
                              use_container_width=True, type="primary")

    # ── Resultado ─────────────────────────────────────────────────────────────
    with col_res:
        if predecir:
            sex_enc = 0 if sex_in=="female" else 1
            X_in = np.array([[
                sex_enc, age_in, sibsp_in, parch_in, fare_in,
                1 if emb_in=="C" else 0,
                1 if emb_in=="Q" else 0,
                1 if emb_in=="S" else 0,
                1 if pclass_in==1 else 0,
                1 if pclass_in==2 else 0,
                1 if pclass_in==3 else 0,
            ]])
            X_sc   = scaler.transform(X_in)
            pred   = modelo.predict(X_sc)[0]
            proba  = modelo.predict_proba(X_sc)[0]
            conf   = proba[pred]*100
            p_no   = proba[0]*100
            p_si   = proba[1]*100

            # ── Caja resultado ────────────────────────────────────────────────
            clase_str = {1:"1ª clase",2:"2ª clase",3:"3ª clase"}[pclass_in]
            sexo_str  = "mujer" if sex_in=="female" else "hombre"

            if pred==1:
                st.markdown(f"""
                <div class="pred-box pred-survived">
                    <div class="pred-icon">✅</div>
                    <div class="pred-label">SOBREVIVIÓ</div>
                    <div class="pred-conf">Confianza del modelo: <strong>{conf:.1f}%</strong></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-box pred-died">
                    <div class="pred-icon">❌</div>
                    <div class="pred-label">NO SOBREVIVIÓ</div>
                    <div class="pred-conf">Confianza del modelo: <strong>{conf:.1f}%</strong></div>
                </div>""", unsafe_allow_html=True)

            # ── Barras de probabilidad ────────────────────────────────────────
            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-cell died">
                    <div class="prob-pct" style="color:#B83424">{p_no:.1f}%</div>
                    <div class="prob-lbl">No sobrevive</div>
                </div>
                <div class="prob-cell survived">
                    <div class="prob-pct" style="color:#0D7A47">{p_si:.1f}%</div>
                    <div class="prob-lbl">Sobrevive</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Barra visual ──────────────────────────────────────────────────
            fig_b, ax_b = plt.subplots(figsize=(5.5, 0.7))
            ax_b.barh([""], [p_no/100],  color=C_RED,   height=0.55)
            ax_b.barh([""], [p_si/100],  color=C_GREEN, height=0.55, left=[p_no/100])
            ax_b.set_xlim(0,1)
            ax_b.set_xlabel("Probabilidad")
            ax_b.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.0%}'))
            ax_b.grid(False)
            ax_b.spines['bottom'].set_visible(False)
            plt.tight_layout(pad=0.4)
            st.pyplot(fig_b, use_container_width=True)
            plt.close()

            # ── Explicación contextualizada ───────────────────────────────────
            if pred==1:
                ctx = (f"El modelo predice que este pasajero — <strong>{sexo_str}, "
                       f"{age_in} años, {clase_str}</strong> — <strong>habría sobrevivido</strong> "
                       f"al naufragio con una confianza del {conf:.1f}%. "
                       f"Los factores más influyentes son el sexo femenino y la clase alta: "
                       f"la política de evacuación 'mujeres y niños primero' y "
                       f"la cercanía de los camarotes de 1ª clase a los botes salvavidas "
                       f"aumentaron significativamente las probabilidades de supervivencia.")
            else:
                ctx = (f"El modelo predice que este pasajero — <strong>{sexo_str}, "
                       f"{age_in} años, {clase_str}</strong> — <strong>no habría sobrevivido</strong> "
                       f"al naufragio con una confianza del {conf:.1f}%. "
                       f"Los hombres de 3ª clase registraron la tasa de supervivencia más baja "
                       f"(~13%). La ubicación de sus camarotes en cubiertas inferiores y "
                       f"la prioridad dada a mujeres y pasajeros de primera clase "
                       f"en los botes salvavidas fueron factores determinantes.")

            st.markdown(f'<div class="context-box">{ctx}</div>', unsafe_allow_html=True)

            # ── Mini modelo info ──────────────────────────────────────────────
            with st.expander("ℹ️ Detalles del modelo"):
                st.markdown("""
| Parámetro | Valor |
|---|---|
| Algoritmo | K-Nearest Neighbors |
| k vecinos | 7 |
| Ponderación | Por distancia |
| Accuracy test | **82.68%** |
| F1-Score test | **0.7704** |
| Precision | 0.7879 |
| Recall | 0.7536 |
| Normalización | StandardScaler |
| Split | 80/20 · seed 42 |
                """)
        else:
            # ── Empty state ───────────────────────────────────────────────────
            st.markdown("""
            <div class="empty-state">
                <div class="es-icon">🔮</div>
                <div class="es-title">El modelo está listo</div>
                <div class="es-sub">
                    Completa el formulario de la izquierda<br>
                    y presiona <strong>«Predecir supervivencia»</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Mini resumen comparativo de modelos
            st.markdown("""
            <div class="chart-card">
            <div class="chart-title">Comparativa de los 5 modelos entrenados</div>
            """, unsafe_allow_html=True)

            comp_df = pd.DataFrame({
                'Modelo': ['Reg. Logística','Árbol Decisión','Random Forest','SVM (RBF)','KNN ★'],
                'Accuracy': [0.8045, 0.7821, 0.8045, 0.7989, 0.8268],
                'F1-Score': [0.7552, 0.6880, 0.7368, 0.7353, 0.7704],
            })
            fig_c, ax_c = plt.subplots(figsize=(5.5, 3))
            x = np.arange(len(comp_df))
            w = 0.35
            b1 = ax_c.bar(x-w/2, comp_df['Accuracy'], w,
                          color=C_SKY, label='Accuracy', edgecolor='white')
            b2 = ax_c.bar(x+w/2, comp_df['F1-Score'], w,
                          color=C_BLUE, label='F1-Score', edgecolor='white')
            ax_c.set_xticks(x)
            ax_c.set_xticklabels(comp_df['Modelo'], fontsize=7.5, rotation=18, ha='right')
            ax_c.set_ylim(0.6, 0.9)
            ax_c.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.0%}'))
            ax_c.set_title("Accuracy vs F1-Score por modelo")
            ax_c.legend(fontsize=8)
            ax_c.bar_label(b1, fmt='%.2f', fontsize=7, padding=2)
            ax_c.bar_label(b2, fmt='%.2f', fontsize=7, padding=2)
            plt.tight_layout(pad=1.2)
            st.pyplot(fig_c, use_container_width=True)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)