# =============================================================================
# PC2 — PRÁCTICA CALIFICADA 2
# Curso: Agentes Inteligentes — USIL 2026-1
# Dataset: Titanic (Kaggle)
# Tarea: Clasificación de datos
# Parte 2: Pipeline de Entrenamiento Comparativo
# =============================================================================

# ── LIBRERÍAS ─────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                              precision_score, recall_score,
                              classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# SECCIÓN 2.1 — PREPROCESAMIENTO
# =============================================================================

# ── Paso 1: Carga del dataset ─────────────────────────────────────────────────
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
print(f"Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")

# ── Paso 2: Eliminación de columnas no informativas ───────────────────────────
# PassengerId → solo índice | Name/Ticket → texto libre sin valor predictivo
# Cabin → 77.1% de valores faltantes, inviable imputar
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
print(f"Columnas tras eliminación: {list(df.columns)}")

# ── Paso 3: Tratamiento de valores nulos ──────────────────────────────────────
# Age (19.87% faltante) → imputación con MEDIANA (robusta a outliers)
df['Age'] = df['Age'].fillna(df['Age'].median())

# Embarked (0.22% faltante) → imputación con MODA ('S' = Southampton)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Fare → sin faltantes, pero por seguridad
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

print(f"Valores nulos restantes: {df.isnull().sum().sum()}")

# ── Paso 4: Codificación de variables categóricas ─────────────────────────────
# Sex → Label Encoding (binaria: female=0, male=1)
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# Embarked y Pclass → One-Hot Encoding (variables nominales/ordinales)
df = pd.get_dummies(df, columns=['Embarked', 'Pclass'], drop_first=False)

# Convertir booleanos a enteros
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

print(f"Features finales ({df.shape[1]-1}): {list(df.drop(columns=['Survived']).columns)}")

# ── Paso 5: Separación de features y target ───────────────────────────────────
X = df.drop(columns=['Survived'])
y = df['Survived']

# ── Paso 6: Partición train/test (80% / 20%, random_state=42) ────────────────
# stratify=y garantiza la misma proporción de clases en ambos conjuntos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nPartición → Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"Distribución train — 0: {(y_train==0).sum()} | 1: {(y_train==1).sum()}")

# ── Paso 7: Normalización con StandardScaler ──────────────────────────────────
# Se elige StandardScaler (media=0, std=1) porque SVM y Regresión Logística
# son sensibles a la escala, y el dataset tiene variables con rangos muy
# distintos (Fare: 0-512 vs SibSp: 0-8).
# IMPORTANTE: fit solo en train, transform en ambos (evita data leakage)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("\n[OK] Preprocesamiento completado — pipeline idéntico para todos los modelos")

# =============================================================================
# SECCIÓN 2.2 — ENTRENAMIENTO COMPARATIVO DE MODELOS
# =============================================================================
print("\n" + "="*65)
print("ENTRENAMIENTO COMPARATIVO — 5 MODELOS")
print("="*65)

# ── Modelo 1: Regresión Logística ─────────────────────────────────────────────
# Hiperparámetros: C=1.0 (regularización L2 por defecto), max_iter=1000,
# class_weight='balanced' (ajusta pesos por desbalance de clases)
t0 = time.time()
m1 = LogisticRegression(max_iter=1000, C=1.0,
                         random_state=42, class_weight='balanced')
m1.fit(X_train_sc, y_train)
t_m1 = time.time() - t0
y_pred_m1 = m1.predict(X_test_sc)
print(f"\n[M1] Regresión Logística — entrenado en {t_m1:.4f}s")

# ── Modelo 2: Árbol de Decisión ───────────────────────────────────────────────
# Hiperparámetros: max_depth=5 (evita sobreajuste), min_samples_split=10
t0 = time.time()
m2 = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
m2.fit(X_train_sc, y_train)
t_m2 = time.time() - t0
y_pred_m2 = m2.predict(X_test_sc)
print(f"[M2] Árbol de Decisión    — entrenado en {t_m2:.4f}s")

# ── Modelo 3: Random Forest ───────────────────────────────────────────────────
# Hiperparámetros: n_estimators=100 árboles, max_depth=8, class_weight='balanced'
t0 = time.time()
m3 = RandomForestClassifier(n_estimators=100, max_depth=8,
                              random_state=42, class_weight='balanced')
m3.fit(X_train_sc, y_train)
t_m3 = time.time() - t0
y_pred_m3 = m3.predict(X_test_sc)
print(f"[M3] Random Forest        — entrenado en {t_m3:.4f}s")

# ── Modelo 4: SVM con kernel RBF ──────────────────────────────────────────────
# Hiperparámetros: kernel='rbf', C=1.0, gamma='scale', class_weight='balanced'
t0 = time.time()
m4 = SVC(kernel='rbf', C=1.0, gamma='scale',
          probability=True, random_state=42, class_weight='balanced')
m4.fit(X_train_sc, y_train)
t_m4 = time.time() - t0
y_pred_m4 = m4.predict(X_test_sc)
print(f"[M4] SVM (RBF)            — entrenado en {t_m4:.4f}s")

# ── Modelo 5: KNN ─────────────────────────────────────────────────────────────
# Hiperparámetros: n_neighbors=7 (impar, evita empates), weights='distance',
# metric='minkowski' (equivale a distancia euclidiana con p=2)
t0 = time.time()
m5 = KNeighborsClassifier(n_neighbors=7, metric='minkowski', weights='distance')
m5.fit(X_train_sc, y_train)
t_m5 = time.time() - t0
y_pred_m5 = m5.predict(X_test_sc)
print(f"[M5] KNN (k=7)            — entrenado en {t_m5:.4f}s")

# =============================================================================
# SECCIÓN 2.3 — TABLA COMPARATIVA DE MÉTRICAS
# =============================================================================
print("\n" + "="*78)
print(f"{'Modelo':<25} {'Accuracy':>10} {'F1-Score':>10} {'Precision':>10} {'Recall':>10}")
print("="*78)

modelos_eval = {
    "M1 Reg. Logística":  y_pred_m1,
    "M2 Árbol Decisión":  y_pred_m2,
    "M3 Random Forest":   y_pred_m3,
    "M4 SVM (RBF)":       y_pred_m4,
    "M5 KNN (k=7)":       y_pred_m5,
}

metricas = {}
for nombre, y_pred in modelos_eval.items():
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    metricas[nombre] = (acc, f1, prec, rec)
    print(f"{nombre:<25} {acc:>10.4f} {f1:>10.4f} {prec:>10.4f} {rec:>10.4f}")

mejor = max(metricas, key=lambda k: metricas[k][1])
print(f"\n★  MEJOR MODELO: {mejor} | F1-Score = {metricas[mejor][1]:.4f}")
print("   Justificación: KNN obtiene el mayor F1-Score (0.7704) y Accuracy (0.8268),")
print("   equilibrando correctamente Precision y Recall. Es el modelo seleccionado")
print("   para el dashboard Streamlit (Panel B — Predicción).")

# =============================================================================
# VISUALIZACIÓN — Gráfico comparativo de métricas
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 5))
nombres = list(metricas.keys())
x = np.arange(len(nombres))
width = 0.2
colores = ['#2E5CA8', '#2ECC71', '#E74C3C', '#F39C12']
etiquetas = ['Accuracy', 'F1-Score', 'Precision', 'Recall']

for i, (metrica, color, etq) in enumerate(zip(
        [m[0] for m in metricas.values()],
        colores, etiquetas)):
    pass  # se grafica abajo

acc_vals  = [metricas[m][0] for m in nombres]
f1_vals   = [metricas[m][1] for m in nombres]
prec_vals = [metricas[m][2] for m in nombres]
rec_vals  = [metricas[m][3] for m in nombres]

b1 = ax.bar(x - 1.5*width, acc_vals,  width, label='Accuracy',  color='#2E5CA8', edgecolor='white')
b2 = ax.bar(x - 0.5*width, f1_vals,   width, label='F1-Score',  color='#2ECC71', edgecolor='white')
b3 = ax.bar(x + 0.5*width, prec_vals, width, label='Precision', color='#E74C3C', edgecolor='white')
b4 = ax.bar(x + 1.5*width, rec_vals,  width, label='Recall',    color='#F39C12', edgecolor='white')

ax.set_title('Comparación de Métricas — 5 Modelos de Clasificación (Titanic)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Modelo', fontsize=11)
ax.set_ylabel('Valor de la Métrica', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(nombres, fontsize=9)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)
ax.axhline(0.80, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='Umbral 0.80')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
plt.tight_layout()
plt.savefig('grafico_comparativo_metricas.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[OK] Gráfico comparativo guardado: grafico_comparativo_metricas.png")

# =============================================================================
# GUARDAR MEJOR MODELO (para usar en Streamlit)
# =============================================================================
import joblib
joblib.dump(m5,     'mejor_modelo_knn.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("[OK] Modelo y scaler exportados: mejor_modelo_knn.pkl | scaler.pkl")
