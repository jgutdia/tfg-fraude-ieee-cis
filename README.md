# tfg-fraude-ieee-cis
# Detección de fraude transaccional — TFG

Código Python del Trabajo de Fin de Grado *«Aplicación de técnicas de clasificación supervisada para la detección de fraude transaccional»*.

## Conjunto de datos

El estudio emplea el dataset [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) publicado en Kaggle en 2019. Para reproducir los experimentos es necesario descargar los archivos `train_transaction.csv` y `train_identity.csv` de dicha competición y situarlos en el mismo directorio en el que se encuentren los códigos a ejecutar.

## Estructura del repositorio

El código se organiza en dos carpetas, cada una de las cuales implementa una estrategia distinta frente al desbalanceo de clases tal y como se describe en la sección 4.1 de la memoria:

```
├── sin_undersampling/            # Estrategia A: entrenamiento con la distribución original
│   ├── preprocesado.py           # Depuración del dataset original (desarrollado en la sección 2.3.3 de la memoria)
│   ├── preparacion.py            # Partición train/test y validación cruzada
│   ├── regresion_logistica.py    # Regresión logística: backward elimination + Elastic Net
│   ├── random_forest.py          # Bosque aleatorio (RandomForestClassifier)
│   └── gradient_boosting.py      # Gradient boosting (HistGradientBoostingClassifier)
│
├── con_undersampling/            # Estrategia B: undersampling de la clase mayoritaria (45/55)
│   ├── preprocesado.py           # Idéntico al de la estrategia A
│   ├── preparacion.py            # Partición + generación del train balanceado
│   ├── regresion_logistica.py
│   ├── random_forest.py
│   └── gradient_boosting.py
│
└── README.md
```

## Orden de ejecución

Dentro de cada carpeta, los scripts deben ejecutarse en el siguiente orden:

1. **`preprocesado.py`** — Lee los CSV originales de Kaggle, aplica las transformaciones descritas en la sección 2.3.3 de la memoria (eliminación de variables, imputación, winsorizing, filtrado por correlación) y genera `df_preprocesado.csv`.

2. **`preparacion.py` / `preparacion_und.py`** — A partir de `df_preprocesado.csv`, genera la codificación one-hot, la partición estratificada 80/20 y (solo en la estrategia B) la muestra de entrenamiento balanceada. Produce los archivos CSV de train y test y el objeto de validación cruzada (`cv_folds.joblib`).

3. **Scripts de los modelos** — Pueden ejecutarse en cualquier orden. Cada uno realiza la búsqueda de hiperparámetros, el ajuste del umbral de decisión, la evaluación en test, la generación de gráficas y el cálculo de la importancia de variables. Los resultados se guardan en carpetas `resultados_*/`.

## Requisitos

- Python ≥ 3.9
- Bibliotecas: `scikit-learn`, `pandas`, `numpy`, `statsmodels`, `joblib`, `matplotlib`

Instalación rápida:
```bash
pip install scikit-learn pandas numpy statsmodels joblib matplotlib
```

## Autor

Jesús Gutiérrez Díaz — Grado en Ingeniería Matemática, UCM (2025-2026).
