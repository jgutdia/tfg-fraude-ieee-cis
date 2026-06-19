# ================================================================================================
# PREPARACIÓN DE CONJUNTOS
# ================================================================================================

import os
import time
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")  # silencia avisos no críticos de las librerías.


# ------------------------------------------------------------------------------------------------
#  CONFIGURACIÓN
# ------------------------------------------------------------------------------------------------

# Al estar el CSV del preprocesado en el mismo directorio que el código pondremos "./".
# Si no, habría que especificar el nombre de la carpeta en la que se encuentran los datos.
CARPETA_DATOS   = "./"
ARCHIVO_ENTRADA = os.path.join(CARPETA_DATOS, "df_preprocesado.csv")
CARPETA_SALIDA  = "datos_preparados"

# --- Parámetros ---
SEMILLA          = 1111
COLUMNA_OBJETIVO = "isFraud"
PROP_TEST        = 0.20      # partición 80/20 estratificada.
N_FOLDS_CV       = 5         # 5-fold StratifiedKFold para los modelos.
PROP_MAYORITARIA = 0.55      # proporción máxima de desbalance (resp. a la clase mayoritaria)


# ================================================================================================
#  FUNCIONES AUXILIARES
# ================================================================================================
def encabezado(texto):
    print("\n" + "=" * 72)
    print(f"  {texto}")
    print("=" * 72)


def info(texto):
    print(f"   - {texto}")


# ------------------------------------------------------------------------------------------------
#  PASO 1 - CARGA
# ------------------------------------------------------------------------------------------------
def paso_1_carga():
    encabezado("PASO 1 - Carga del dataset preprocesado")
    df = pd.read_csv(ARCHIVO_ENTRADA, index_col=0)
    info(f"Forma cargada: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    info(f"Variable objetivo presente: {COLUMNA_OBJETIVO in df.columns}")
    info(f"Proporción de fraude: {df[COLUMNA_OBJETIVO].mean():.4f}")
    return df


# ------------------------------------------------------------------------------------------------
#  PASO 2 - SEPARACIÓN X / y
# ------------------------------------------------------------------------------------------------
def paso_2_separar_X_y(df):
    encabezado("PASO 2 - Separación de la variable objetivo y los predictores")
    y = df[COLUMNA_OBJETIVO].astype(int)
    X = df.drop(columns=[COLUMNA_OBJETIVO])
    info(f"X: {X.shape[0]:,} filas x {X.shape[1]} predictores")
    info(f"y: {y.shape[0]:,} valores ({y.sum():,} positivos)")
    return X, y


# ------------------------------------------------------------------------------------------------
#  PASO 3 - CREACIÓN DE VARIABLES DUMMIES
# ------------------------------------------------------------------------------------------------
def paso_3_dummies(X):
    encabezado("PASO 3 - Creación de variables dummies (one-hot-encoding)")

    # Identificamos las variables categóricas (las que no son numéricas).
    cols_cat = X.select_dtypes(exclude=[np.number]).columns.tolist()
    cols_num = X.select_dtypes(include=[np.number]).columns.tolist()
    info(f"Categóricas detectadas ({len(cols_cat)}): {cols_cat}")
    info(f"Numéricas: {len(cols_num)}")

    if not cols_cat:
        info("No hay categóricas: nada que codificar")
        return X

    n_cols_antes = X.shape[1]
    # drop_first=True para prescindir de una categoría (la de "referencia") de cada variable
    # categórica y así evitar problemas de multicolinealidad.
    X = pd.get_dummies(X, columns=cols_cat, drop_first=True, dtype=int)

    info(f"Columnas antes: {n_cols_antes}  -->  después: {X.shape[1]}")
    info(f"Nuevas dummies creadas: {X.shape[1] - len(cols_num)}")
    return X


# ------------------------------------------------------------------------------------------------
#  PASO 4 - PARTICIÓN EN DATOS TRAIN / TEST ESTRATIFICADA
# ------------------------------------------------------------------------------------------------
def paso_4_split(X, y):
    encabezado(f"PASO 4 - Partición en datos train/test estratificada "
            f"({int((1-PROP_TEST)*100)}/{int(PROP_TEST*100)})")

    # stratify=y.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=PROP_TEST,
        random_state=SEMILLA,
        stratify=y,  # asegura que la proporción de fraude se conserva en ambos conjuntos
                     # (esencial con clases muy desbalanceadas).
    )

    info(f"Train: {X_train.shape[0]:,} filas | fraude: {y_train.mean():.4f}")
    info(f"Test:  {X_test.shape[0]:,} filas | fraude: {y_test.mean():.4f}")
    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------------------------------------------
#  PASO 5 - BALANCEO DEL TRAIN POR UNDERSAMPLING DE LA CLASE MAYORITARIA
# ------------------------------------------------------------------------------------------------
def paso_5_balanceo(X_train, y_train):
    encabezado(f"PASO 5 - Balanceo del train (proporción objetivo "
            f"{int(PROP_MAYORITARIA*100)}/{int((1-PROP_MAYORITARIA)*100)})")

    # Validar el parámetro.
    assert 0.50 <= PROP_MAYORITARIA <= 0.55, \
        "PROP_MAYORITARIA debe estar en [0.50, 0.55]"

    # Todos los positivos (minoritaria) se conservan.
    idx_pos = y_train[y_train == 1].index
    idx_neg = y_train[y_train == 0].index
    n_pos = len(idx_pos)

    # Calculamos cuántos negativos coger para alcanzar la proporción objetivo.
    # Si la prop. mayoritaria objetivo es p, entonces n_neg / (n_pos + n_neg) = p
    # -> n_neg = n_pos * p / (1 - p).
    n_neg_target = int(round(n_pos * PROP_MAYORITARIA / (1 - PROP_MAYORITARIA)))
    n_neg_disponibles = len(idx_neg)
    assert n_neg_target <= n_neg_disponibles, \
        f"No hay suficientes negativos ({n_neg_disponibles}) para alcanzar el objetivo"

    info(f"Positivos en train: {n_pos:,}")
    info(f"Negativos en train: {n_neg_disponibles:,}")
    info(f"Negativos a muestrear: {n_neg_target:,}")

    # Muestreo aleatorio reproducible de negativos.
    rng = np.random.RandomState(SEMILLA)
    idx_neg_muestreados = rng.choice(idx_neg, size=n_neg_target, replace=False)

    # Construimos el train balanceado uniendo positivos + negativos muestreados, y mezclamos
    # las filas para que las clases no queden agrupadas en bloques (importante para CV).
    idx_balanceado = np.concatenate([idx_pos.values, idx_neg_muestreados])
    rng.shuffle(idx_balanceado)

    X_train_bal = X_train.loc[idx_balanceado].copy()
    y_train_bal = y_train.loc[idx_balanceado].copy()

    info(f"Train balanceado: {X_train_bal.shape[0]:,} filas "
        f"({X_train_bal.shape[0] / X_train.shape[0]:.1%} del train original)")
    info(f"Proporción de fraude en train balanceado: {y_train_bal.mean():.4f}")
    return X_train_bal, y_train_bal


# ------------------------------------------------------------------------------------------------
#  PASO 6 - OBJETO DE VALIDACIÓN CRUZADA
# ------------------------------------------------------------------------------------------------
def paso_6_cv():
    encabezado(f"PASO 6 - Creación del objeto StratifiedKFold ({N_FOLDS_CV} folds)")
    # Usamos shuffle=True y random_state para obtener particiones aleatorias pero reproducibles.
    cv = StratifiedKFold(n_splits=N_FOLDS_CV, shuffle=True, random_state=SEMILLA)
    info(f"Objeto cv creado. Se reutilizará en todos los scripts de modelos.")
    return cv


# ------------------------------------------------------------------------------------------------
#  PASO 7 - GUARDADO
# ------------------------------------------------------------------------------------------------
def paso_7_guardar(X_train, X_test, y_train, y_test,
                    X_train_bal, y_train_bal, cv):
    encabezado("PASO 7 - Guardado de los conjuntos preparados")
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    # Conjuntos originales (se conservan por si se quieren reutilizar).
    X_train.to_csv(os.path.join(CARPETA_SALIDA, "X_train.csv"))
    X_test.to_csv (os.path.join(CARPETA_SALIDA, "X_test.csv"))
    y_train.to_csv(os.path.join(CARPETA_SALIDA, "y_train.csv"))
    y_test.to_csv (os.path.join(CARPETA_SALIDA, "y_test.csv"))

    # Conjunto de train balanceado (es el que usarán los scripts de modelo).
    X_train_bal.to_csv(os.path.join(CARPETA_SALIDA, "X_train_balanced.csv"))
    y_train_bal.to_csv(os.path.join(CARPETA_SALIDA, "y_train_balanced.csv"))

    # Objeto CV (común para ambos trains).
    joblib.dump(cv, os.path.join(CARPETA_SALIDA, "cv_folds.joblib"))

    info(f"X_train.csv, X_test.csv, y_train.csv, y_test.csv -> {CARPETA_SALIDA}/")
    info(f"X_train_balanced.csv, y_train_balanced.csv -> {CARPETA_SALIDA}/")
    info(f"cv_folds.joblib (objeto StratifiedKFold) -> {CARPETA_SALIDA}/")


# ================================================================================================
#  PROGRAMA PRINCIPAL
# ================================================================================================
def main():
    t0 = time.time()
    encabezado("PREPARACIÓN DE LOS CONJUNTOS DE DATOS")

    df = paso_1_carga()
    X, y = paso_2_separar_X_y(df)
    X = paso_3_dummies(X)
    X_train, X_test, y_train, y_test = paso_4_split(X, y)
    X_train_bal, y_train_bal = paso_5_balanceo(X_train, y_train)
    cv = paso_6_cv()
    paso_7_guardar(X_train, X_test, y_train, y_test,
                    X_train_bal, y_train_bal, cv)

    encabezado(f"PREPARACIÓN COMPLETADA en {time.time() - t0:.1f} s")
    print(f"\nResumen final:")
    print(f"   Train original:   {X_train.shape[0]:,} filas x {X_train.shape[1]} predictores"
        f"  | fraude: {y_train.mean():.4f}")
    print(f"   Train balanceado: {X_train_bal.shape[0]:,} filas x {X_train_bal.shape[1]} predictores"
        f"  | fraude: {y_train_bal.mean():.4f}")
    print(f"   Test:             {X_test.shape[0]:,} filas x {X_test.shape[1]} predictores"
        f"  | fraude: {y_test.mean():.4f}")


if __name__ == "__main__":
    main()