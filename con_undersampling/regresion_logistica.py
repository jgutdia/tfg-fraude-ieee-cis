# ================================================================================================
# MODELO DE REGRESIÓN LOGÍSTICA
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

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve, roc_curve,
    confusion_matrix, precision_score, recall_score, f1_score, fbeta_score,
)

import statsmodels.api as sm

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------------------------
#  CONFIGURACIÓN
# ------------------------------------------------------------------------------------------------

# Al estar los CSV de train/test en la carpeta datos_preparados en el mismo directorio que el código
# pondremos "./". Si no, habría que especificar el nombre de la carpeta en la que se encuentran
# los datos.
CARPETA_DATOS    = "datos_preparados"      # donde están los CSV de train/test.
CARPETA_RESULT   = "resultados_logistica"

#--- Parámetros ---
SEMILLA          = 1111
#Iteración 1
#GRID_C           = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
#GRID_L1_RATIO    = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
#Iteración 2
GRID_C           = [0.05, 0.1, 0.15, 0.25, 0.4]
GRID_L1_RATIO    = [0.65, 0.7, 0.75, 0.8, 0.85]
SOLVER_EN        = "saga"  # único solver que admite Elastic Net en sklearn.
MAX_ITER_EN      = 5000
TOL_EN           = 1e-4
GRID_UMBRAL      = np.arange(0.01, 1.0, 0.01)
P_VALOR_MAX      = 0.05  # umbral de significación para backward
BETA_FBETA       = 2.0   # F-beta: beta>1 pondera más recall que precisión


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
    return X_train, X_test, y_train, y_test, cv


# ------------------------------------------------------------------------------------------------
#  PASO 2 - ESTANDARIZACIÓN
# ------------------------------------------------------------------------------------------------
def paso_2_estandarizar(X_train, X_test):
    encabezado("PASO 2 - Estandarización de los predictores")
    scaler = StandardScaler()
    X_train_std = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns, index=X_train.index,
    )
    X_test_std = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns, index=X_test.index,
    )
    info("StandardScaler ajustado en train y aplicado a ambos conjuntos")
    return X_train_std, X_test_std, scaler


# ------------------------------------------------------------------------------------------------
#  PASO 3 - MODELO A: REGRESIÓN LOGÍSTICA SIN REGULARIZAR (statsmodel, selección, backward)
# ------------------------------------------------------------------------------------------------
def _ajustar_glm(y, X_iter):
    """Ajusta un GLM binomial. Intenta primero con IRLS (rápido y preciso); si falla por problemas
    numéricos (matriz casi singular, multicolinealidad), reintenta con BFGS, que no resuelve
    sistemas lineales y es más robusto.
    Devuelve el modelo ajustado o lanza la excepción si ambos fallan."""
    try:
        return sm.GLM(y, X_iter, family=sm.families.Binomial()).fit(
            maxiter=200, tol=1e-6, disp=0
        )
    except Exception:
        return sm.GLM(y, X_iter, family=sm.families.Binomial()).fit(
            method="bfgs", maxiter=300, disp=0
        )


def paso_3_logistica_sin_reg(X_train_std, y_train):
    encabezado("PASO 3 - Modelo A: Regresión logística sin regularizar "
            "(backward elimination por p-valor)")

    import gc  # para forzar liberación de memoria entre iteraciones

    variables_actuales = list(X_train_std.columns)
    historial = []
    iteracion = 0
    t0 = time.time()

    info(f"Inicio: {len(variables_actuales)} predictores, umbral p-valor = {P_VALOR_MAX}")

    # Conversión a float32 para reducir a la mitad el consumo de memoria.
    X_train_f32 = X_train_std.astype(np.float32)
    y_train_f32 = y_train.astype(np.float32)
    info("Datos convertidos a float32 para reducir el consumo de memoria")

    modelo = None
    while True:
        iteracion += 1
        X_iter = sm.add_constant(X_train_f32[variables_actuales], has_constant="add")

        try:
            modelo = _ajustar_glm(y_train_f32, X_iter)
        except Exception as e:
            info(f"Iter {iteracion:>3} | Ajuste fallido ({type(e).__name__}): {e}")
            info("Se devuelve el último modelo ajustado correctamente.")
            break

        # p-valores de los predictores.
        p_valores = modelo.pvalues.drop("const", errors="ignore")
        p_max_actual = p_valores.max()
        var_peor = p_valores.idxmax()

        if p_max_actual <= P_VALOR_MAX:
            historial.append({
                "iteracion": iteracion,
                "n_variables": len(variables_actuales),
                "variable_eliminada": None,
                "p_valor_eliminada": None,
            })
            info(f"Iter {iteracion:>3} | nº variables: {len(variables_actuales):>3} | "
                f"max p-valor: {p_max_actual:.4g} -> CONVERGENCIA "
                f"(todas las variables con p-valor <= {P_VALOR_MAX})")
            break

        # Registrar e imprimir la eliminación.
        historial.append({
            "iteracion": iteracion,
            "n_variables": len(variables_actuales),
            "variable_eliminada": var_peor,
            "p_valor_eliminada": p_max_actual,
        })
        info(f"Iter {iteracion:>3} | nº variables: {len(variables_actuales)-1:>3} | "
            f"eliminada: {var_peor:<30s} | p-valor: {p_max_actual:.4g}")

        variables_actuales.remove(var_peor)

        # Liberar memoria explícitamente antes de la siguiente iteración.
        del X_iter
        gc.collect()

        if len(variables_actuales) == 0:
            info("Se han eliminado todas las variables.")
            break

    info(f"Backward elimination completada en {time.time() - t0:.1f} s")
    info(f"Variables iniciales: {X_train_std.shape[1]} | "
        f"Variables finales: {len(variables_actuales)} | "
        f"Eliminadas: {X_train_std.shape[1] - len(variables_actuales)}")

    # Guardar historial en CSV.
    os.makedirs(CARPETA_RESULT, exist_ok=True)
    ruta_hist = os.path.join(CARPETA_RESULT, "historial_backward_elimination.csv")
    pd.DataFrame(historial).to_csv(ruta_hist, index=False)
    info(f"Historial guardado en {ruta_hist}")

    return modelo, variables_actuales


# ------------------------------------------------------------------------------------------------
#  PASO 4 - MODELO B: REGRESIÓN LOGÍSTICA CON ELASTIC NET (sklearn, CV)
# ------------------------------------------------------------------------------------------------
def paso_4_elastic_net(X_train_std, y_train, cv):
    encabezado("PASO 4 - Modelo B: Regresión logística con Elastic Net (sklearn + CV)")
    n_combos = len(GRID_C) * len(GRID_L1_RATIO)
    info(f"Grid: C={GRID_C}, l1_ratio={GRID_L1_RATIO}")
    info(f"{n_combos} combinaciones x {cv.n_splits} folds = "
         f"{n_combos * cv.n_splits} ajustes en total")
    info(f"Búsqueda sobre el train balanceado: {len(X_train_std):,} filas")

    # --- (4.1) Búsqueda de hiperparámetros ---
    modelo_en = LogisticRegression(
        penalty="elasticnet",
        solver=SOLVER_EN,
        max_iter=MAX_ITER_EN,
        tol=TOL_EN,
        random_state=SEMILLA,
    )
    busqueda = GridSearchCV(
        estimator=modelo_en,
        param_grid={"C": GRID_C, "l1_ratio": GRID_L1_RATIO},
        cv=cv,
        scoring="average_precision",   # PR-AUC.
        n_jobs=1,
        verbose=1,
    )
    t0 = time.time()
    busqueda.fit(X_train_std, y_train)
    info(f"Búsqueda completada en {time.time() - t0:.1f} s")
    info(f"Mejor C: {busqueda.best_params_['C']} | "
        f"Mejor l1_ratio: {busqueda.best_params_['l1_ratio']}")
    info(f"PR-AUC CV (media): {busqueda.best_score_:.4f}")

    # --- (4.2) Ajuste final con los mejores hiperparámetros ---
    info("Reentrenando con los mejores hiperparámetros...")
    modelo_final = LogisticRegression(
        penalty="elasticnet",
        solver=SOLVER_EN,
        max_iter=MAX_ITER_EN,
        tol=TOL_EN,
        random_state=SEMILLA,
        C=busqueda.best_params_["C"],
        l1_ratio=busqueda.best_params_["l1_ratio"],
    )
    t1 = time.time()
    modelo_final.fit(X_train_std, y_train)
    info(f"Ajuste final completado en {time.time() - t1:.1f} s")

    return modelo_final, busqueda


# ------------------------------------------------------------------------------------------------
#  PASO 5 - AJUSTE DEL UMBRAL DE DECISIÓN
# ------------------------------------------------------------------------------------------------
def paso_5_umbral(modelo, X_test_std, y_test, etiqueta, beta=BETA_FBETA):
    encabezado(f"PASO 5 - Umbral óptimo para {etiqueta} "
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
#  PASO 6 - EVALUACIÓN EN TEST
# ------------------------------------------------------------------------------------------------
def paso_6_evaluar(modelo, X_test_std, y_test, umbral, etiqueta, metricas_umbral=None):
    encabezado(f"PASO 6 - Evaluación en test: {etiqueta}")

    # Probabilidades en test.
    probas = modelo.predict_proba(X_test_std)[:, 1]
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
#  PASO 7 - GRÁFICAS
# ------------------------------------------------------------------------------------------------
def _guardar(fig, nombre):
    ruta = os.path.join(CARPETA_RESULT, nombre)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    info(f"Guardada: {ruta}")


def grafica_curva_pr(resultados, y_test):
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in resultados:
        p, rcl, _ = precision_recall_curve(y_test, r["Probas"])
        ax.plot(rcl, p, label=f"{r['Etiqueta']} (PR-AUC={r['PR-AUC']:.3f})")
    # Línea horizontal de referencia: clasificador aleatorio (prevalencia).
    prev = y_test.mean()
    ax.axhline(prev, ls="--", color="grey", alpha=0.6,
            label=f"Aleatorio (prevalencia={prev:.3f})")
    ax.set_xlabel("Sensibilidad (recall)")
    ax.set_ylabel("Precisión")
    ax.set_title("Curvas Precisión-Recall en el conjunto de test")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3); ax.legend(loc="best")
    _guardar(fig, "curva_pr.png")


def grafica_curva_roc(resultados, y_test):
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in resultados:
        fpr, tpr, _ = roc_curve(y_test, r["Probas"])
        ax.plot(fpr, tpr, label=f"{r['Etiqueta']} (AUC={r['AUC']:.3f})")
    # Diagonal de azar.
    ax.plot([0, 1], [0, 1], ls="--", color="grey", alpha=0.6, label="Aleatorio")
    ax.set_xlabel("Tasa de falsos positivos (1 - especificidad)")
    ax.set_ylabel("Tasa de verdaderos positivos (sensibilidad)")
    ax.set_title("Curvas ROC en el conjunto de test")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3); ax.legend(loc="best")
    _guardar(fig, "curva_roc.png")


def grafica_metricas_vs_umbral(resultados):
    for r in resultados:
        m = r["MetricasUmbral"]
        if m is None:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(m["umbrales"], m["precisiones"], label="Precisión")
        ax.plot(m["umbrales"], m["recalls"],     label="Sensibilidad")
        ax.plot(m["umbrales"], m["f1s"],         label="F1")
        ax.axvline(r["Umbral"], color="black", ls="--", alpha=0.6,
                label=f"Umbral óptimo = {r['Umbral']:.2f}")
        ax.set_xlabel("Umbral de decisión")
        ax.set_ylabel("Valor de la métrica")
        ax.set_title(f"Métricas en función del umbral - {r['Etiqueta']}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3); ax.legend(loc="best")
        nombre = "metricas_vs_umbral_" + r["Etiqueta"].lower().replace(" ", "_") + ".png"
        _guardar(fig, nombre)


def grafica_histograma_probas(resultados, y_test):
    for r in resultados:
        probas = r["Probas"]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(probas[y_test == 0], bins=50, alpha=0.6,
                label="Legítimos", color="tab:blue", density=True)
        ax.hist(probas[y_test == 1], bins=50, alpha=0.6,
                label="Fraudes", color="tab:red", density=True)
        ax.axvline(r["Umbral"], color="black", ls="--", alpha=0.7,
                label=f"Umbral = {r['Umbral']:.2f}")
        ax.set_xlabel("Probabilidad predicha de fraude")
        ax.set_ylabel("Densidad")
        ax.set_title(f"Distribución de probabilidades por clase - {r['Etiqueta']}")
        ax.set_yscale("log")  # escala log porque hay muchísimos más legítimos.
        ax.grid(alpha=0.3); ax.legend(loc="best")
        nombre = "histograma_probas_" + r["Etiqueta"].lower().replace(" ", "_") + ".png"
        _guardar(fig, nombre)


def grafica_matriz_confusion(resultados):
    for r in resultados:
        cm = np.array([[r["TN"], r["FP"]],
                    [r["FN"], r["TP"]]])
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
        ax.set_title(f"Matriz de confusión - {r['Etiqueta']} (umbral={r['Umbral']:.2f})")
        plt.colorbar(im, ax=ax)
        nombre = "matriz_confusion_" + r["Etiqueta"].lower().replace(" ", "_") + ".png"
        _guardar(fig, nombre)


def grafica_top_importancia(df_imp, etiqueta, columna_valor, columna_error=None, top=15):
    df = df_imp.head(top).iloc[::-1]   # invertimos para que la más importante quede arriba.
    fig, ax = plt.subplots(figsize=(7, 0.4 * top + 1))
    if columna_error and columna_error in df.columns:
        ax.barh(df["variable"], df[columna_valor], xerr=df[columna_error],
                color="tab:blue", alpha=0.8)
    else:
        # Colorear positivo/negativo si son coeficientes.
        colores = ["tab:red" if v < 0 else "tab:blue" for v in df[columna_valor]]
        ax.barh(df["variable"], df[columna_valor], color=colores, alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel(columna_valor)
    ax.set_title(f"Top {top} variables - {etiqueta}")
    ax.grid(alpha=0.3, axis="x")
    nombre = "top_variables_" + etiqueta.lower().replace(" ", "_") + ".png"
    _guardar(fig, nombre)


def paso_7_graficas(resultados, y_test):
    encabezado("PASO 7 - Gráficas")
    grafica_curva_pr        (resultados, y_test)
    grafica_curva_roc       (resultados, y_test)
    grafica_metricas_vs_umbral(resultados)
    grafica_histograma_probas (resultados, y_test)
    grafica_matriz_confusion  (resultados)


# ------------------------------------------------------------------------------------------------
#  PASO 8 - GUARDADO
# ------------------------------------------------------------------------------------------------
def paso_8_guardar(modelo_sm_sklearn, modelo_sm_stats, modelo_en, resultados, busqueda, scaler):
    encabezado("PASO 8 - Guardado de modelos y resultados")
    os.makedirs(CARPETA_RESULT, exist_ok=True)

    joblib.dump(modelo_sm_sklearn, os.path.join(CARPETA_RESULT, "modelo_sin_reg.joblib"))
    joblib.dump(modelo_sm_stats, os.path.join(CARPETA_RESULT, "modelo_sin_reg_statsmodels.joblib"))
    joblib.dump(modelo_en, os.path.join(CARPETA_RESULT, "modelo_elastic_net.joblib"))
    joblib.dump(scaler,    os.path.join(CARPETA_RESULT, "scaler.joblib"))

    # Tabla resumen de métricas.
    tabla = pd.DataFrame([{
        "Modelo": r["Etiqueta"], "Umbral": r["Umbral"],
        "PR-AUC": r["PR-AUC"], "AUC": r["AUC"],
        "Precisión": r["Precisión"], "Sensibilidad": r["Sensibilidad"],
        "Especificidad": r["Especificidad"], "F1": r["F1"],
        "Exactitud": r["Exactitud"],
        "TP": r["TP"], "FP": r["FP"], "TN": r["TN"], "FN": r["FN"],
    } for r in resultados])
    tabla.to_csv(os.path.join(CARPETA_RESULT, "tabla_resumen.csv"), index=False)
    info(f"Resultados guardados en {CARPETA_RESULT}/")

    pd.DataFrame(busqueda.cv_results_).to_csv(
        os.path.join(CARPETA_RESULT, "busqueda_hiperparametros.csv"), index=False
    )


# ------------------------------------------------------------------------------------------------
#  PASO 9 - COEFICIENTES ESTANDARIZADOS
# ------------------------------------------------------------------------------------------------
def _coefs_de_modelo(modelo, nombres):
    """Devuelve un DataFrame con los coeficientes y, si los hay, errores
    estándar y p-valores."""
    if hasattr(modelo, "params") and hasattr(modelo, "pvalues"):
        # statsmodels: tiene intercepto como 'const'.
        df = pd.DataFrame({
            "variable": modelo.params.index,
            "coef": modelo.params.values,
            "std_err": modelo.bse.values,
            "p_valor": modelo.pvalues.values,
        })
        df = df[df["variable"] != "const"].copy()
    else:
        coefs = modelo.coef_.ravel()
        df = pd.DataFrame({
            "variable": nombres,
            "coef": coefs,
            "std_err": np.nan,
            "p_valor": np.nan,
        })
    df["coef_abs"] = df["coef"].abs()
    df = df.sort_values("coef_abs", ascending=False).reset_index(drop=True)
    return df


def paso_9_coeficientes(modelo_sm, vars_glm, modelo_en, X_train_std):
    encabezado("PASO 9 - Coeficientes estandarizados (importancia)")
    nombres = X_train_std.columns.tolist()

    df_sm = _coefs_de_modelo(modelo_sm, vars_glm)
    df_en = _coefs_de_modelo(modelo_en, nombres)

    info(f"Top 15 variables - Modelo sin regularizar ({len(vars_glm)} variables finales):")
    print(df_sm.head(15).to_string(index=False))
    info("Top 15 variables - Modelo Elastic Net:")
    print(df_en.head(15).to_string(index=False))

    n_anuladas = int((df_en["coef"] == 0).sum())
    info(f"Variables anuladas por Elastic Net (coef = 0): {n_anuladas} "
        f"de {len(df_en)}")

    df_sm.to_csv(os.path.join(CARPETA_RESULT, "coefs_sin_reg.csv"), index=False)
    df_en.to_csv(os.path.join(CARPETA_RESULT, "coefs_elastic_net.csv"), index=False)
    info(f"Coeficientes guardados en {CARPETA_RESULT}/")
    
    # Gráficas de top-15 variables más importantes (por valor absoluto del coeficiente).
    grafica_top_importancia(df_sm, "Sin regularizar", "coef")
    grafica_top_importancia(df_en, "Elastic Net",     "coef")

# ================================================================================================
#  PROGRAMA PRINCIPAL
# ================================================================================================
def main():
    t0 = time.time()
    encabezado("MODELO 1: REGRESIÓN LOGÍSTICA")
    os.makedirs(CARPETA_RESULT, exist_ok=True)

    X_train, X_test, y_train, y_test, cv = paso_1_carga()
    X_train_std, X_test_std, scaler = paso_2_estandarizar(X_train, X_test)

    # Modelo A: backward elimination sobre el train completo.
    modelo_sm, vars_glm = paso_3_logistica_sin_reg(X_train_std, y_train)

    # Modelo B: Elastic Net.
    modelo_en, busqueda = paso_4_elastic_net(X_train_std, y_train, cv)

    info("Ajustando modelo sklearn sin penalización sobre las variables seleccionadas...")
    modelo_sm_sklearn = LogisticRegression(
        penalty=None, solver="lbfgs", max_iter=2000, random_state=SEMILLA
    )
    modelo_sm_sklearn.fit(X_train_std[vars_glm], y_train)

    # Umbrales óptimos (uno por modelo).
    umbral_sm, metricas_sm = paso_5_umbral(modelo_sm_sklearn, X_test_std[vars_glm], y_test,
                            "regresión logística sin regularización (backward)")
    umbral_en, metricas_en = paso_5_umbral(modelo_en, X_test_std, y_test,
                            "regresión logística con Elastic Net")

    # Evaluación en test.
    res_sm = paso_6_evaluar(modelo_sm_sklearn, X_test_std[vars_glm], y_test, umbral_sm,
                            "Sin regularizar", metricas_umbral=metricas_sm)
    res_en = paso_6_evaluar(modelo_en, X_test_std, y_test, umbral_en,
                            "Elastic Net", metricas_umbral=metricas_en)

    # Gráficas.
    paso_7_graficas([res_sm, res_en], y_test)

    # Guardado.
    paso_8_guardar(modelo_sm_sklearn, modelo_sm, modelo_en, 
                [res_sm, res_en], busqueda, scaler)

    # Importancia.
    paso_9_coeficientes(modelo_sm, vars_glm, modelo_en, X_train_std)

    encabezado(f"REGRESIÓN LOGÍSTICA COMPLETADA en {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
