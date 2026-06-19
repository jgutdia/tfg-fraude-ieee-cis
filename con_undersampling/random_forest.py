# ================================================================================================
# MODELO 2: RANDOM FOREST (BOSQUE ALEATORIO)
# ================================================================================================

import os
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # backend sin GUI, para guardar figuras.
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve, roc_curve,
    confusion_matrix, precision_score, recall_score, f1_score, fbeta_score,
)

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------------------------
#  CONFIGURACIÓN
# ------------------------------------------------------------------------------------------------

# Al estar los CSV de train/test en la carpeta datos_preparados en el mismo directorio que el código
# pondremos "./". Si no, habría que especificar el nombre de la carpeta en la que se encuentran 
# los datos.
CARPETA_DATOS    = "datos_preparados"      # donde están los CSV de train/test.
CARPETA_RESULT   = "resultados_random_forest"

# --- Parámetros ---
SEMILLA          = 1111
#Iteración 1
#GRID_RF = {
#    "n_estimators":     [300, 500],        # nº de árboles.
#    "max_depth":        [30, 50, None],    # profundidad máxima (None = sin limite).
#    "min_samples_leaf": [1, 5, 20],        # tamaño mínimo de hoja (a mayor, más regularización).
#    "max_features":     ["sqrt", "log2"],  # m ~ sqrt(p)/ m ~ log_2(p).
#}
#Iteración 2
GRID_RF = {
    "n_estimators":     [500],              # nº de árboles.
    "max_depth":        [20, 30, 40],       # profundidad máxima.
    "min_samples_leaf": [1, 2, 4],          # tamaño mínimo de hoja (a mayor, más regularización).
    "max_features":     ["log2"],           # m ~ log_2(p).
}
SOLVER_NJOBS        = -1   # núcleos para paralelizar. Si sale WinError 1450, cambiar (por ej. 2).
N_REPETICIONES_PERM = 5
GRID_UMBRAL         = np.arange(0.01, 1.0, 0.01)
BETA_FBETA          = 2.0   # F-beta: beta>1 pondera más recall que precisión

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
    encabezado("PASO 1 - Carga de los conjuntos preparados")
    X_train = pd.read_csv(os.path.join(CARPETA_DATOS, "X_train_balanced.csv"), index_col=0)
    X_test  = pd.read_csv(os.path.join(CARPETA_DATOS, "X_test.csv"),  index_col=0)
    y_train = pd.read_csv(os.path.join(CARPETA_DATOS, "y_train_balanced.csv"), index_col=0).squeeze()
    y_test  = pd.read_csv(os.path.join(CARPETA_DATOS, "y_test.csv"),  index_col=0).squeeze()
    cv = joblib.load(os.path.join(CARPETA_DATOS, "cv_folds.joblib"))

    info(f"Train: {X_train.shape[0]:,} filas x {X_train.shape[1]} predictores")
    info(f"Test:  {X_test.shape[0]:,} filas x {X_test.shape[1]} predictores")
    info(f"Folds CV: {cv.n_splits}")
    info("Los árboles son invariantes a la escala: no se estandariza.")
    return X_train, X_test, y_train, y_test, cv


# ------------------------------------------------------------------------------------------------
#  PASO 2 - RANDOM FOREST (GridSearchCV)
# ------------------------------------------------------------------------------------------------
def paso_2_busqueda(X_train, y_train, cv):
    encabezado("PASO 2 - Modelo: Random Forest (GridSearchCV)")
    n_combos = int(np.prod([len(v) for v in GRID_RF.values()]))
    info(f"Grid: {GRID_RF}")
    info(f"{n_combos} combinaciones x {cv.n_splits} folds = "
         f"{n_combos * cv.n_splits} ajustes en total")
    info(f"Búsqueda sobre el train balanceado: {len(X_train):,} filas")

    # --- (2.1) Búsqueda de hiperparámetros ---
    # n_jobs=1 en el estimador para no anidar paralelismo con el de GridSearchCV.
    modelo_rf = RandomForestClassifier(
        random_state=SEMILLA,
        n_jobs=1,
        class_weight=None,
    )
    busqueda = GridSearchCV(
        estimator=modelo_rf,
        param_grid=GRID_RF,
        cv=cv,
        scoring="average_precision",   # PR-AUC.
        n_jobs=SOLVER_NJOBS,
        verbose=1,
    )
    t0 = time.time()
    busqueda.fit(X_train, y_train)
    info(f"Búsqueda completada en {time.time() - t0:.1f} s")
    info(f"Mejores parámetros: {busqueda.best_params_}")
    info(f"PR-AUC CV (media): {busqueda.best_score_:.4f}")

    # --- (2.2) Ajuste final con los mejores hiperparámetros ---
    info("Reentrenando con los mejores hiperparámetros...")
    modelo_final = RandomForestClassifier(
        random_state=SEMILLA,
        n_jobs=1,
        class_weight=None,
        **busqueda.best_params_,
    )
    t1 = time.time()
    modelo_final.fit(X_train, y_train)
    info(f"Ajuste final completado en {time.time() - t1:.1f} s")
    return modelo_final, busqueda


# ------------------------------------------------------------------------------------------------
#  PASO 3 - AJUSTE DEL UMBRAL DE DECISIÓN
# ------------------------------------------------------------------------------------------------
def paso_3_umbral(modelo, X_test_std, y_test, etiqueta, beta=BETA_FBETA):
    encabezado(f"PASO 3 - Umbral óptimo para {etiqueta} "
            f"(maximiza F-{beta} en el conjunto de test)")
    info("Calculando probabilidades sobre el test...")
    probas = modelo.predict_proba(X_test_std)[:, 1]
    info(f"   ({len(probas):,} probabilidades calculadas)")

    # Calculamos las métricas (precisión, recall, F1, F-beta) para CADA umbral del grid.
    # Esto sirve después para representar la evolución de las métricas con el umbral.
    precisiones, recalls, f1s, fbetas = [], [], [], []
    for u in GRID_UMBRAL:
        y_pred = (probas >= u).astype(int)
        precisiones.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        fbetas.append(fbeta_score(y_test, y_pred, beta=beta, zero_division=0))
    precisiones = np.array(precisiones)
    recalls     = np.array(recalls)
    f1s         = np.array(f1s)
    fbetas      = np.array(fbetas)
    idx = int(fbetas.argmax())
    umbral_opt = float(GRID_UMBRAL[idx])
    info(f"Umbral óptimo: {umbral_opt:.2f}  |  F-{beta} en ese umbral: {fbetas[idx]:.4f}")
    
    metricas_por_umbral = {
        "umbrales":    GRID_UMBRAL,
        "precisiones": precisiones,
        "recalls":     recalls,
        "f1s":         f1s,
    }
    return umbral_opt, metricas_por_umbral


# ------------------------------------------------------------------------------------------------
#  PASO 4 - EVALUACIÓN EN TEST
# ------------------------------------------------------------------------------------------------
def paso_4_evaluar(modelo, X_test, y_test, umbral, etiqueta, metricas_umbral=None):
    encabezado(f"PASO 4 - Evaluación en test: {etiqueta}")
    
    # Probabilidades en test.
    probas = modelo.predict_proba(X_test)[:, 1]
    y_pred = (probas >= umbral).astype(int)
    
    # Métricas independientes del umbral.   
    pr_auc  = average_precision_score(y_test, probas)
    roc_auc = roc_auc_score(y_test, probas)
    
    # Métricas al umbral elegido.
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    pre  = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    espe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc  = (tp + tn) / (tp + tn + fp + fn)

    info(f"PR-AUC:  {pr_auc:.4f}")
    info(f"ROC-AUC: {roc_auc:.4f}")
    info(f"Matriz de confusión (umbral={umbral:.2f}):")
    info(f"   TN={tn:,}   FP={fp:,}")
    info(f"   FN={fn:,}   TP={tp:,}")
    info(f"Precisión:     {pre:.4f}")
    info(f"Sensibilidad:  {rec:.4f}")
    info(f"Especificidad: {espe:.4f}")
    info(f"F1:            {f1:.4f}")
    info(f"Exactitud:     {acc:.4f}")

    return {
        "Etiqueta": etiqueta,
        "Umbral": umbral,
        "PR-AUC": pr_auc,
        "AUC": roc_auc,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Precisión": pre, "Sensibilidad": rec, "Especificidad": espe,
        "F1": f1, "Exactitud": acc,
        "Probas": probas,
        "MetricasUmbral": metricas_umbral,
    }


# ------------------------------------------------------------------------------------------------
#  PASO 5 - GRÁFICAS
# ------------------------------------------------------------------------------------------------
def _guardar(fig, nombre):
    ruta = os.path.join(CARPETA_RESULT, nombre)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    info(f"Guardada: {ruta}")


def grafica_curva_pr(resultado, y_test):
    fig, ax = plt.subplots(figsize=(7, 5))
    p, rcl, _ = precision_recall_curve(y_test, resultado["Probas"])
    ax.plot(rcl, p, label=f"{resultado['Etiqueta']} (PR-AUC={resultado['PR-AUC']:.3f})")
    prev = y_test.mean()
    ax.axhline(prev, ls="--", color="grey", alpha=0.6,
            label=f"Aleatorio (prevalencia={prev:.3f})")
    ax.set_xlabel("Sensibilidad (recall)")
    ax.set_ylabel("Precisión")
    ax.set_title("Curva Precisión-Recall en el conjunto de test")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3); ax.legend(loc="best")
    _guardar(fig, "curva_pr.png")


def grafica_curva_roc(resultado, y_test):
    fig, ax = plt.subplots(figsize=(7, 5))
    fpr, tpr, _ = roc_curve(y_test, resultado["Probas"])
    ax.plot(fpr, tpr, label=f"{resultado['Etiqueta']} (AUC={resultado['AUC']:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", alpha=0.6, label="Aleatorio")
    ax.set_xlabel("Tasa de falsos positivos (1 - especificidad)")
    ax.set_ylabel("Tasa de verdaderos positivos (sensibilidad)")
    ax.set_title("Curva ROC en el conjunto de test")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3); ax.legend(loc="best")
    _guardar(fig, "curva_roc.png")


def grafica_metricas_vs_umbral(resultado):
    m = resultado["MetricasUmbral"]
    if m is None:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(m["umbrales"], m["precisiones"], label="Precisión")
    ax.plot(m["umbrales"], m["recalls"],     label="Sensibilidad")
    ax.plot(m["umbrales"], m["f1s"],         label="F1")
    ax.axvline(resultado["Umbral"], color="black", ls="--", alpha=0.6,
            label=f"Umbral óptimo = {resultado['Umbral']:.2f}")
    ax.set_xlabel("Umbral de decisión")
    ax.set_ylabel("Valor de la métrica")
    ax.set_title(f"Métricas en función del umbral - {resultado['Etiqueta']}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3); ax.legend(loc="best")
    _guardar(fig, "metricas_vs_umbral.png") 


def grafica_histograma_probas(resultado, y_test):
    probas = resultado["Probas"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(probas[y_test == 0], bins=50, alpha=0.6,
            label="Legítimos", color="tab:blue", density=True)
    ax.hist(probas[y_test == 1], bins=50, alpha=0.6,
            label="Fraudes", color="tab:red", density=True)
    ax.axvline(resultado["Umbral"], color="black", ls="--", alpha=0.7,
            label=f"Umbral = {resultado['Umbral']:.2f}")
    ax.set_xlabel("Probabilidad predicha de fraude")
    ax.set_ylabel("Densidad")
    ax.set_title(f"Distribución de probabilidades por clase - {resultado['Etiqueta']}")
    ax.set_yscale("log")
    ax.grid(alpha=0.3); ax.legend(loc="best")
    _guardar(fig, "histograma_probas.png")


def grafica_matriz_confusion(resultado):
    cm = np.array([[resultado["TN"], resultado["FP"]],
                [resultado["FN"], resultado["TP"]]])
    cm_pct = cm / cm.sum() * 100
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred. legítimo", "Pred. fraude"])
    ax.set_yticklabels(["Real legítimo", "Real fraude"])
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", color=color, fontsize=11)
    ax.set_title(f"Matriz de confusión - {resultado['Etiqueta']} (umbral={resultado['Umbral']:.2f})")
    plt.colorbar(im, ax=ax)
    _guardar(fig, "matriz_confusion.png")


def grafica_top_importancia(df_imp, etiqueta, columna_valor, columna_error=None, top=15):
    df = df_imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.4 * top + 1))
    if columna_error and columna_error in df.columns:
        ax.barh(df["variable"], df[columna_valor], xerr=df[columna_error],
                color="tab:blue", alpha=0.8)
    else:
        ax.barh(df["variable"], df[columna_valor], color="tab:blue", alpha=0.8)
    ax.set_xlabel(columna_valor)
    ax.set_title(f"Top {top} variables - {etiqueta}")
    ax.grid(alpha=0.3, axis="x")
    nombre = "top_variables_" + etiqueta.lower().replace(" ", "_") + ".png"
    _guardar(fig, nombre)


def paso_5_graficas(resultado, y_test):
    encabezado("PASO 5 - Gráficas")
    grafica_curva_pr           (resultado, y_test)
    grafica_curva_roc          (resultado, y_test)
    grafica_metricas_vs_umbral (resultado)
    grafica_histograma_probas  (resultado, y_test)
    grafica_matriz_confusion   (resultado)


# ------------------------------------------------------------------------------------------------
#  PASO 6 - GUARDADO
# ------------------------------------------------------------------------------------------------
def paso_6_guardar(modelo, resultado, busqueda):
    encabezado("PASO 6 - Guardado de modelo y resultados")
    os.makedirs(CARPETA_RESULT, exist_ok=True)

    joblib.dump(modelo, os.path.join(CARPETA_RESULT, "modelo_random_forest.joblib"))

    # Tabla resumen de métricas.
    tabla = pd.DataFrame([{
        "Modelo": resultado["Etiqueta"], "Umbral": resultado["Umbral"],
        "PR-AUC": resultado["PR-AUC"], "AUC": resultado["AUC"],
        "Precisión": resultado["Precisión"], "Sensibilidad": resultado["Sensibilidad"],
        "Especificidad": resultado["Especificidad"], "F1": resultado["F1"],
        "Exactitud": resultado["Exactitud"],
        "TP": resultado["TP"], "FP": resultado["FP"],
        "TN": resultado["TN"], "FN": resultado["FN"],
    }])
    tabla.to_csv(os.path.join(CARPETA_RESULT, "tabla_resumen.csv"), index=False)
    info(f"Resultados guardados en {CARPETA_RESULT}/")

    pd.DataFrame(busqueda.cv_results_).to_csv(
        os.path.join(CARPETA_RESULT, "busqueda_hiperparametros.csv"), index=False
    )


# ------------------------------------------------------------------------------------------------
#  PASO 7 - IMPORTANCIA POR PERMUTACIÓN
# ------------------------------------------------------------------------------------------------
def paso_7_importancia(modelo, X_test, y_test):
    encabezado("PASO 7 - Importancia por permutación")
    t0 = time.time()
    r = permutation_importance(
        modelo, X_test, y_test,
        scoring="average_precision",   #PR-AUC.
        n_repeats=N_REPETICIONES_PERM,
        random_state=SEMILLA,
        n_jobs=1,
    )
    info(f"Cálculo completado en {time.time() - t0:.1f} s")

    df_imp = pd.DataFrame({
        "variable": X_test.columns,
        "importancia_media": r.importances_mean,
        "importancia_std":  r.importances_std,
    }).sort_values("importancia_media", ascending=False).reset_index(drop=True)

    info("Top 15 variables por importancia (permutación):")
    print(df_imp.head(15).to_string(index=False))

    # También guardamos la importancia por impureza (Gini) como referencia secundaria.
    df_gini = pd.DataFrame({
        "variable": X_test.columns,
        "importancia_gini": modelo.feature_importances_,
    }).sort_values("importancia_gini", ascending=False).reset_index(drop=True)

    df_imp.to_csv(os.path.join(CARPETA_RESULT, "importancia_permutacion.csv"), index=False)
    df_gini.to_csv(os.path.join(CARPETA_RESULT, "importancia_gini.csv"), index=False)
    info(f"Importancias guardadas en {CARPETA_RESULT}/")
    
    # Gráfica: top-15 variables más importantes con barras de error.
    grafica_top_importancia(df_imp, "Random Forest (permutación)",
                            columna_valor="importancia_media",
                            columna_error="importancia_std")
    return df_imp


# ================================================================================================
#  PROGRAMA PRINCIPAL
# ================================================================================================
def main():
    t0 = time.time()
    encabezado("MODELO 2: RANDOM FOREST")
    os.makedirs(CARPETA_RESULT, exist_ok=True)

    X_train, X_test, y_train, y_test, cv = paso_1_carga()

    modelo, busqueda = paso_2_busqueda(X_train, y_train, cv)
    umbral, metricas = paso_3_umbral(modelo, X_test, y_test, "Random Forest")
    res = paso_4_evaluar(modelo, X_test, y_test, umbral, "Random Forest",
                        metricas_umbral=metricas)

    paso_5_graficas(res, y_test)
    paso_6_guardar(modelo, res, busqueda)
    paso_7_importancia(modelo, X_test, y_test)

    encabezado(f"RANDOM FOREST COMPLETADO en {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
