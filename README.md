# Titanic ML Dashboard — PC2 USIL 2026-1

Dashboard interactivo de Machine Learning para predicción de supervivencia en el Titanic.

## Estructura del repositorio

```
├── app.py                  # Dashboard Streamlit (Panel A + Panel B)
├── titanic_pc2.py          # Pipeline de entrenamiento comparativo (5 modelos)
├── requirements.txt        # Dependencias Python
└── README.md
```

##  Cómo ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

##  Dashboard en línea
[Ver dashboard en Streamlit Cloud](URL_AQUI)

##  Modelos comparados
| Modelo | Accuracy | F1-Score |
|---|---|---|
| Regresión Logística | 0.8045 | 0.7552 |
| Árbol de Decisión | 0.7821 | 0.6880 |
| Random Forest | 0.8045 | 0.7368 |
| SVM (RBF) | 0.7989 | 0.7353 |
| **KNN ★** | **0.8268** | **0.7704** |

##  Dataset
- **Fuente:** Kaggle — Titanic: Machine Learning from Disaster
- **Registros:** 891 pasajeros
- **Tarea:** Clasificación binaria (Survived: 0/1)
