# ================================================================================================
# PREPROCESADO
# ================================================================================================

import os
import time
import warnings

import numpy as np
import pandas as pd

from statsmodels.stats.diagnostic import lilliefors  # test de normalidad (paso 7).

from scipy.stats import skew  # asimetría, para decidir media vs mediana (paso 5).

warnings.filterwarnings("ignore")  # silencia avisos no críticos de las librerías.


# ------------------------------------------------------------------------------------------------
#  CONFIGURACIÓN
# ------------------------------------------------------------------------------------------------

# Al estar los CSV descargados de Kaggle en el mismo directorio que el código pondremos "./".
# Si no, habría que especificar el nombre de la carpeta en la que se encuentran los datos.
CARPETA_DATOS = "./"

ARCHIVO_TRANSACTION = os.path.join(CARPETA_DATOS, "train_transaction.csv")
ARCHIVO_IDENTITY    = os.path.join(CARPETA_DATOS, "train_identity.csv")
ARCHIVO_SALIDA      = os.path.join(CARPETA_DATOS, "df_preprocesado.csv")

# --- Parámetros ---
SEMILLA                = 1111      
COLUMNA_OBJETIVO       = "isFraud"
COLUMNA_ID             = "TransactionID"

TAM_MUESTRA_LILLIEFORS = 5000    # tamaño de muestra para el test de normalidad.
ALPHA_NORMALIDAD       = 0.05    # nivel de significación del test de normalidad.
UMBRAL_CORRELACION     = 0.70    
FACTOR_IQR             = 3       # anchura de los bigotes en el winsorizing (IQR x 3).

# Tramos de faltantes (en tanto por uno).
TRAMO_BORRAR_FILA      = 0.05    # <5%    -> borrar filas afectadas.
TRAMO_CENTRAL          = 0.35    # 5-35%  -> tendencia central (mediana/moda).
                                 # >35%   -> borrar variable.

np.random.seed(SEMILLA)


# ================================================================================================
#  FUNCIONES AUXILIARES
# ================================================================================================
def encabezado(texto):
    print("\n" + "=" * 72)
    print(f"  {texto}")
    print("=" * 72)


def info(texto):
    print(f"   - {texto}")


def es_categorica(serie):
    """Devuelve True si la variable no es numerica (es categórica, text, etc.)."""
    return not pd.api.types.is_numeric_dtype(serie)


# ------------------------------------------------------------------------------------------------
#  PASO 1 - CARGA Y UNIÓN DE LAS DOS TABLAS
# ------------------------------------------------------------------------------------------------
def paso_1_cargar_y_unir():
    encabezado("PASO 1 - Carga de las dos tablas y merge por TransactionID")

    if not os.path.exists(ARCHIVO_TRANSACTION):
        raise FileNotFoundError(
            f"No encuentro '{ARCHIVO_TRANSACTION}'. Revisa CARPETA_DATOS "
            f"y que los CSV de Kaggle esten ahí."
        )

    info("Cargando train_transaction.csv (tabla grande, puede tardar un poco)...")
    t0 = time.time()
    transaction = pd.read_csv(ARCHIVO_TRANSACTION)
    info(f"transaction: {transaction.shape[0]:,} filas x {transaction.shape[1]} cols "
        f"({time.time() - t0:.1f} s)")

    info("Cargando train_identity.csv...")
    identity = pd.read_csv(ARCHIVO_IDENTITY)
    info(f"identity: {identity.shape[0]:,} filas x {identity.shape[1]} cols")

    # left join: conservamos todas las transacciones aunque no tengan identity.
    info("Uniendo ambas tablas a partir de TransactionID...")
    df = transaction.merge(identity, how="left", on=COLUMNA_ID)
    info(f"Dataset unido: {df.shape[0]:,} filas x {df.shape[1]} columnas")

    # Liberamos memoria (tablas originales).
    del transaction, identity

    return df


# ------------------------------------------------------------------------------------------------
#  PASO 2 - ELIMINAR VARIABLES IRRELEVANTES
# ------------------------------------------------------------------------------------------------
def paso_2_eliminar_irrelevantes(df):
    encabezado("PASO 2 - Eliminación de variables irrelevantes")

    # TransactionID es solo una clave: no aporta poder predictivo.
    # No lo borramos todavía: lo usamos como índice para alinear la variable objetivo al final.
    df = df.set_index(COLUMNA_ID)
    info(f"'{COLUMNA_ID}' pasa a ser un índice (no se usa como predictor)")

    return df


# ------------------------------------------------------------------------------------------------
#  PASO 3 - CAMBIOS EN VARIABLES
# ------------------------------------------------------------------------------------------------
def paso_3_transformar_variables(df):
    encabezado("PASO 3 - Transformación de variables")

    # ----- 3.1  Transformar marcadores de ausencia ("NULL","Undefined",...) en NA ---------------
    marcadores_na = ["", " ", "NULL", "null", "Null",
                    "Undefined", "undefined",
                    "Missing", "missing", "NaN", "nan", "None", "none"]
    info("Reemplazando marcadores de ausencia (NULL/Undefined/vacíos...) por NA")
    df = df.replace(marcadores_na, np.nan)
    # Tras quitar los marcadores de texto, algunas columnas numéricas pueden haber quedado con dtype
    # 'object'. Las reconvertimos a numérico para que los pasos posteriores las traten correctamente.
    reconvertidas = 0
    for col in df.columns:
        if es_categorica(df[col]):
            convertida = pd.to_numeric(df[col], errors="coerce")
            # Solo aceptamos la conversión si no se generan NA nuevos.
            if convertida.notna().sum() == df[col].notna().sum():
                df[col] = convertida
                reconvertidas += 1
    if reconvertidas:
        info(f"Reconvertidas a numérico {reconvertidas} columnas que estaban como texto")

    # ----- 3.2  Agrupar variables por categoricas para reducir uan alta cardinalidad ---------------
    # (a) Dominios de email (P_emaildomain y R_emaildomain) -> agrupamos por proveedor principal.
    def proveedor_email(dominio):
        if pd.isna(dominio):
            return np.nan
        d = str(dominio).lower()
        if "gmail" in d:                                      return "google"
        if "yahoo" in d or "ymail" in d or "rocketmail" in d: return "yahoo"
        if ("hotmail" in d or "outlook" in d or "live" in d
                or "msn" in d or "passport" in d):            return "microsoft"
        if "aol" in d:                                        return "aol"
        if "icloud" in d or "mac.com" in d or "me.com" in d:  return "apple"
        if "anonymous" in d:                                  return "anonymous"
        return "otros"

    for col in ["P_emaildomain", "R_emaildomain"]:
        if col in df.columns:
            n_antes = df[col].nunique(dropna=True)
            df[col] = df[col].apply(proveedor_email)
            info(f"{col}: {n_antes} dominios -> {df[col].nunique(dropna=True)} proveedores")

    # (b) DeviceInfo -> agrupamos por familia de dispositivo.
    def familia_dispositivo(valor):
        if pd.isna(valor):
            return np.nan
        v = str(valor).lower()
        if "windows" in v:                                       return "windows"
        if "ios" in v or "iphone" in v or "ipad" in v:           return "ios"
        if "mac" in v:                                           return "macos"
        if ("sm-" in v or "samsung" in v or "moto" in v 
                or "huawei" in v or "lg-" in v or "redmi" in v 
                or "android" in v or "build" in v):              return "android"
        if "trident" in v or "rv:" in v:                         return "windows"
        return "otros"

    if "DeviceInfo" in df.columns:
        n_antes = df["DeviceInfo"].nunique(dropna=True)
        df["DeviceInfo"] = df["DeviceInfo"].apply(familia_dispositivo)
        info(f"DeviceInfo: {n_antes} valores -> {df['DeviceInfo'].nunique(dropna=True)} familias")

    # (c) id_31 (navegador) -> versión agrupada.
    if "id_31" in df.columns:
        n_antes = df["id_31"].nunique(dropna=True)
        df["id_31"] = (df["id_31"].astype(str).str.lower()
                    .str.replace(r"[\d\.]+", "", regex=True).str.strip())
        df["id_31"] = df["id_31"].replace("nan", np.nan)
        info(f"id_31 (navegador): {n_antes} -> {df['id_31'].nunique(dropna=True)} versiones agrupadas")
    return df


# ------------------------------------------------------------------------------------------------
#  PASO 4 - REGLAS DE INCONSISTENCIA Y TRANSFORMACIÓN A NA
# ------------------------------------------------------------------------------------------------
def paso_4_reglas_inconsistencia(df):
    encabezado("PASO 4 - Reglas de inconsistencia (valores imposibles -> NA)")

    total_corregidos = 0

    # (a) Importe <= 0 es imposible para una transacción válida.
    if "TransactionAmt" in df.columns:
        mask = df["TransactionAmt"] <= 0
        n = int(mask.sum())
        df.loc[mask, "TransactionAmt"] = np.nan
        total_corregidos += n
        info(f"TransactionAmt <= 0 -> NA: {n} valores")

    # (b) Distancias negativas (dist1, dist2) son imposibles.
    for col in ["dist1", "dist2"]:
        if col in df.columns:
            mask = df[col] < 0
            n = int(mask.sum())
            df.loc[mask, col] = np.nan
            total_corregidos += n
            info(f"{col} < 0 -> NA: {n} valores")

    # (c) Contadores Cxx negativos son imposibles (son conteos).
    cols_C = [c for c in df.columns if c.startswith("C") and c[1:].isdigit()]
    for col in cols_C:
        mask = df[col] < 0
        n = int(mask.sum())
        if n:
            df.loc[mask, col] = np.nan
            total_corregidos += n
    if cols_C:
        info(f"Contadores C negativos -> NA (revisadas {len(cols_C)} columnas C)")

    # (d) Deltas temporales Dxx negativos son imposibles (son días transcurridos).
    cols_D = [c for c in df.columns if c.startswith("D") and c[1:].isdigit()]
    for col in cols_D:
        mask = df[col] < 0
        n = int(mask.sum())
        if n:
            df.loc[mask, col] = np.nan
            total_corregidos += n
    if cols_D:
        info(f"Deltas temporales D negativos -> NA (revisadas {len(cols_D)} columnas D)")

    info(f"Total de valores marcados como NA por inconsistencia: {total_corregidos}")
    return df


# ------------------------------------------------------------------------------------------------
#  PASO 5 - TRATAMIENTO DE VALORES FALTANTES 
# ------------------------------------------------------------------------------------------------
def _imputar_central(df, col):
    """Imputa la variable por tendencia central:
    - categórica  -> moda
    - numérica     -> media si es simétrica, mediana si es asimétrica
                        (criterio: |skewness| > 1 se considera asimétrica)."""
    if es_categorica(df[col]):
        valor = df[col].mode(dropna=True)
        valor = valor.iloc[0] if len(valor) else "desconocido"
        df[col] = df[col].fillna(valor)
    else:
        serie = df[col].dropna()
        asimetria = skew(serie) if len(serie) > 2 else 0
        if abs(asimetria) > 1:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mean())
    return df


def paso_5_faltantes(df):
    encabezado("PASO 5 - Tratamiento de valores faltantes")

    contador = {"borrar_var": 0, "borrar_fila_eventos": 0, "filas_borradas": 0,
                "central": 0}
    t0 = time.time()

    # Para evitar bucles infinitos: si una variable quedara con NA residual tras imputarla,
    # se registra y la próxima vez se elimina.
    intentos_fallidos = set()

    while True:
        porc_na = df.isna().mean()
        porc_na = porc_na[porc_na > 0]  # solo variables que aún tienen faltantes.
        if porc_na.empty:
            break

        # Nos quedamos la variable con menor porcentaje de faltantes (>0).
        col = porc_na.idxmin()
        p = porc_na[col]

        # Control: si la variable falló antes, la eliminamos.
        if col in intentos_fallidos:
            df = df.drop(columns=[col])
            contador["borrar_var"] += 1
            continue

        # --- Tratamiento según el tramo ---
        # Primero miramos el tramo <5% porque borrar filas reduce los faltantes del resto y una
        # variable puede cambiar de tramo de una iteración a otra.
        if p <= TRAMO_BORRAR_FILA:
            # <5% -> borrar las filas con NA en esta variable.
            antes = len(df)
            df = df.dropna(subset=[col])
            contador["borrar_fila_eventos"] += 1
            contador["filas_borradas"] += antes - len(df)

        elif p > TRAMO_CENTRAL:
            # >35% -> eliminar la variable entera.
            df = df.drop(columns=[col])
            contador["borrar_var"] += 1

        else:
            # 5-35% -> tendencia central.
            df = _imputar_central(df, col)
            contador["central"] += 1
            # Comprobamos que no quede NA residual en esa variable.
            if df[col].isna().any():
                intentos_fallidos.add(col)

        # Tras tratar esta variable, el while vuelve a empezar, recalcula los porcentajes de
        # todas las demás con faltantes y pasa a la siguiente de menor porcentaje.
        # Continúa hasta que no queden variables con faltantes.

    # Resumen
    info(f"Eventos de borrado de filas (<5%):                  {contador['borrar_fila_eventos']} "
        f"({contador['filas_borradas']:,} filas en total)")
    info(f"Variables imputadas por tendencia central (5-35%):  {contador['central']}")
    info(f"Variables eliminadas (>35% NA):                     {contador['borrar_var']}")
    info(f"Faltantes restantes: {int(df.isna().sum().sum())}")
    info(f"Paso 5 completado en {time.time() - t0:.1f} s")

    return df


# ------------------------------------------------------------------------------------------------
#  PASO 6 - TRATAMIENTO DE ATÍPICOS: WINSORIZING POR IQR x 3
# ------------------------------------------------------------------------------------------------
def paso_6_outliers(df):
    encabezado("PASO 6 - Outliers: winsorizing por IQR x 3")
    
    # Exclusivo para variables numéricas.
    cols_num = df.select_dtypes(include=[np.number]).columns.tolist()

    n_recortadas = 0
    for col in cols_num:
        # Si la variable tiene <=2 valores distintos, es practicamente binaria: la saltamos.
        if df[col].nunique(dropna=True) <= 2:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue  # no hay dispersion: nada que winsorizar.
        lim_inf = q1 - FACTOR_IQR * iqr
        lim_sup = q3 + FACTOR_IQR * iqr
        # clip: recorta los valores extremos a los limites.
        antes = ((df[col] < lim_inf) | (df[col] > lim_sup)).sum()
        if antes:
            df[col] = df[col].clip(lower=lim_inf, upper=lim_sup)
            n_recortadas += 1

    info(f"Winsorizing aplicado (bigotes a IQR x {FACTOR_IQR:g})")
    info(f"Variables con algun valor recortado: {n_recortadas} de {len(cols_num)}")
    return df


# ------------------------------------------------------------------------------------------------
#  PASO 7 - CORRELACIÓN
# ------------------------------------------------------------------------------------------------
def es_normal(serie):
    """Test de Lilliefors sobre una muestra de hasta TAM_MUESTRA_LILLIEFORS.
    Devuelve True si no se rechaza la normalidad (p-valor > ALPHA_NORMALIDAD)."""
    serie = serie.dropna()
    if serie.nunique() < 5:
        return False
    if len(serie) > TAM_MUESTRA_LILLIEFORS:
        serie = serie.sample(TAM_MUESTRA_LILLIEFORS, random_state=SEMILLA)
    try:
        _, p_valor = lilliefors(serie, dist="norm")
    except Exception:
        return False
    return p_valor > ALPHA_NORMALIDAD


def paso_7_correlacion(df):
    encabezado("PASO 7 - Correlación y eliminación de variables altamente correladas")
    # Solo tiene sentido estudiar la normalidad y medir la correlación en variables numéricas.
    cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(cols_num) < 2:
        info("Menos de 2 variables numéricas: no hay correlaciones que evaluar")
        return df

    # Si todas las variables implicadas son normales -> Coeficiente de correlación de Pearson;
    # En otro caso -> Coeficiente de correlación de Spearman (mas robusto y sin supuesto de normalidad).
    info(f"Comprobando normalidad (Lilliefors, muestra<= {TAM_MUESTRA_LILLIEFORS}) de {len(cols_num)} variables...")
    n_normales = sum(es_normal(df[c]) for c in cols_num)
    proporcion_normales = n_normales / len(cols_num)
    info(f"Variables que pasan el test de normalidad: {n_normales}/{len(cols_num)} "
        f"({proporcion_normales:.0%})")

    metodo = "pearson" if proporcion_normales > 0.5 else "spearman"
    info(f"Método de correlación elegido: {metodo.capitalize()}")

    info("Calculando matriz de correlación (puede tardar con muchas variables)...")
    t0 = time.time()
    corr = df[cols_num].corr(method=metodo).abs()
    info(f"Matriz calculada en {time.time() - t0:.1f} s")
    
    a_eliminar = []
    # Trabajamos sobre una copia de la matriz de correlación.
    M = corr.copy()
    for c in M.columns:
        M.loc[c, c] = 0.0 # anulamos la diagonal

    while True:
        # Si quedan menos de 2 variables, no hay pares posibles.
        if M.shape[0] < 2:
            break
        valores = M.to_numpy()
        # Máximo valor de correlación que queda en la matriz.
        max_corr = valores.max()
        if max_corr <= UMBRAL_CORRELACION:
            break  # no hay ningún par por encima del umbral.

        # Localizamos el par (i, j) responsable de ese máximo.
        i, j = np.unravel_index(np.argmax(valores), valores.shape)
        var_i, var_j = M.columns[i], M.columns[j]

        # Correlación media de cada una con todas las demás variables.
        # La que presente la tasa más alta es la que se eliminará.
        media_i = M[var_i].mean()
        media_j = M[var_j].mean()
        descartada = var_i if media_i >= media_j else var_j

        a_eliminar.append(descartada)
        # Eliminamos la variable y recalculamos la matriz de correlaciones con las variables restantes.
        M = M.drop(index=descartada, columns=descartada)
    # Resumen.
    if a_eliminar:
        df = df.drop(columns=a_eliminar)
        info(f"Eliminadas {len(a_eliminar)} variables por correlación > {UMBRAL_CORRELACION}")
    else:
        info("Ninguna pareja supera el umbral: no se elimina ninguna variable")

    return df


# ------------------------------------------------------------------------------------------------
#  PASO 8 - GUARDADO
# ------------------------------------------------------------------------------------------------
def paso_8_guardar(df, objetivo):
    encabezado("PASO 8 - Guardado del dataset preprocesado")

    # Reincorporamos la variable objetivo (alineada por el índice TransactionID).
    df = df.join(objetivo, how="left")
    info(f"Reincorporada la variable objetivo '{COLUMNA_OBJETIVO}'")

    # Si el archivo de salida ya existe, lo sobrescribimos.
    if os.path.exists(ARCHIVO_SALIDA):
        os.remove(ARCHIVO_SALIDA)

    df.to_csv(ARCHIVO_SALIDA, index=True)  # index=True conserva TransactionID.
    info(f"Dataset final: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    info(f"Guardado en: {os.path.abspath(ARCHIVO_SALIDA)}")
    return df


# ================================================================================================
#  FUNCIÓN PRINCIPAL
# ================================================================================================
def main():
    t_inicio = time.time()
    encabezado("INICIO DEL PREPROCESADO")

    df = paso_1_cargar_y_unir()
    df = paso_2_eliminar_irrelevantes(df)

    # Apartamos la variable objetivo para evitar pérdida de información.
    if COLUMNA_OBJETIVO not in df.columns:
        raise KeyError(f"No encuentro la variable objetivo '{COLUMNA_OBJETIVO}'")
    objetivo = df[COLUMNA_OBJETIVO].copy()
    df = df.drop(columns=[COLUMNA_OBJETIVO])
    info(f"Variable objetivo '{COLUMNA_OBJETIVO}' apartada ({int(objetivo.sum()):,} fraudes, "
        f"{objetivo.mean():.2%} del total)")

    df = paso_3_transformar_variables(df)
    df = paso_4_reglas_inconsistencia(df)
    df = paso_5_faltantes(df)
    df = paso_6_outliers(df)
    df = paso_7_correlacion(df)
    df = paso_8_guardar(df, objetivo)

    encabezado(f"PREPROCESADO COMPLETADO en {time.time() - t_inicio:.1f} s")
    print("\nResumen final:")
    print(f"   Filas:    {df.shape[0]:,}")
    print(f"   Columnas: {df.shape[1]} (incluida la objetivo)")
    print(f"   Fraude:   {df[COLUMNA_OBJETIVO].mean():.2%}")
    numericas = df.select_dtypes(include="number").columns.tolist()
    categoricas = df.select_dtypes(exclude="number").columns.tolist()
    print(f" --- Variables numéricas: ({len(numericas)}) ---")
    for c in numericas:
        print(c)
    print(f"\n --- Variables categóricas: ({len(categoricas)}) ---")
    for c in categoricas:
        print(c)


if __name__ == "__main__":
    main()
