import time 
import pandas as pd
from pathlib import Path 

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parents[1]

X_TRAIN_PATH = BASE_DIR / "data" / "processed" / "X_train.csv"
X_TEST_PATH = BASE_DIR / "data" / "processed" / "X_test.csv"
Y_TRAIN_PATH = BASE_DIR / "data" / "processed" / "y_train.csv"
Y_TEST_PATH = BASE_DIR / "data" / "processed" / "y_test.csv"

RESULTS_DIR = BASE_DIR / "results"

def cargar_datos_procesados():
    X_train = pd.read_csv(X_TRAIN_PATH).values
    X_test = pd.read_csv(X_TEST_PATH).values

    y_train = pd.read_csv(Y_TRAIN_PATH).values.ravel()
    y_test = pd.read_csv(Y_TEST_PATH).values.ravel()

    return X_train, X_test, y_train, y_test

def entrenar_random_forest_secuencial():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("espera...cargando..datos..procesados")
    X_train, X_test, y_train, y_test = cargar_datos_procesados()

    print("Tamaño de X_train:", X_train.shape)
    print("Tamaño de X_test:", X_test.shape)
    print("Tamaño de y_train:", y_train.shape)
    print("Tamaño de y_test:", y_test.shape)

    print("\n Entrenando Bosque Aleatorio secuencial")

    inicio_entrenamiento=time.time()

    modelo=RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=1
    )

    modelo.fit(X_train, y_train)
    fin_entrenamiento=time.time()

    print("realizando..predicciones...")

    inicio_prediccion=time.time()
    y_pred=modelo.predict(X_test)
    fin_prediccion=time.time()

    tiempo_entrenamiento = fin_entrenamiento - inicio_entrenamiento
    tiempo_prediccion = fin_prediccion - inicio_prediccion
    tiempo_total = tiempo_entrenamiento + tiempo_prediccion

    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    matriz = confusion_matrix(y_test, y_pred)
    reporte = classification_report(y_test, y_pred, zero_division=0)

    resultados =pd.DataFrame([{
        "modo": "secuencial",
        "modelo": "Random Forest",
        "procesos": 1,
        "n_estimators": 100,
        "n_jobs": 1,
        "tiempo_entrenamiento": tiempo_entrenamiento,
        "tiempo_prediccion": tiempo_prediccion,
        "tiempo_total": tiempo_total,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted
    }])

    resultados.to_csv(RESULTS_DIR / "resultados_secuencial.csv", index=False)
    pd.DataFrame(matriz).to_csv(RESULTS_DIR / "matriz_confusion_secuencial.csv", index=False)

    with open(RESULTS_DIR / "reporte_clasificacion_secuencial.txt", "w", encoding="utf-8") as archivo:
        archivo.write(reporte)

    print("\n Resultados del Bosque Aleatorio secuencial:")
    print(resultados) 

    print("\nMatriz de confusión:") 
    print(matriz) 

    print("\nReporte de clasificación:") 
    print(reporte) 

    print("\nArchivos generados:") 
    print("results/resultados_secuencial.csv") 
    print("results/matriz_confusion_secuencial.csv") 
    print("results/reporte_clasificacion_secuencial.txt") 

if __name__=="__main__":
    entrenar_random_forest_secuencial()
