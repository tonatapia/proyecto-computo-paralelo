#entrenamiento secuencial con random forest
#se ocupa n_jobs=1 un solo proceso sin paralelismo
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

#-------Rutas de archivos-------
BASE_DIR = Path(__file__).resolve().parents[1]


#-------Rutas de los datos preprocesados--------
X_TRAIN_PATH = BASE_DIR / "data" / "processed" / "X_train.csv"
X_TEST_PATH = BASE_DIR / "data" / "processed" / "X_test.csv"
Y_TRAIN_PATH = BASE_DIR / "data" / "processed" / "y_train.csv"
Y_TEST_PATH = BASE_DIR / "data" / "processed" / "y_test.csv"


#-----carpeta donde se guardarán los resultados generados----
RESULTS_DIR = BASE_DIR / "results"


#-----Funciones------
def cargar_datos_procesados():
    #Lee los archivos  CSV de train y test ya preprocesados y los convierte 
    # a arrays de Numpy y listos para sklearn. 
    X_train = pd.read_csv(X_TRAIN_PATH).values
    X_test = pd.read_csv(X_TEST_PATH).values

    #.ravel() convierte el vector columna (n,1) a vector plano (n,)
    y_train = pd.read_csv(Y_TRAIN_PATH).values.ravel()
    y_test = pd.read_csv(Y_TEST_PATH).values.ravel()

    return X_train, X_test, y_train, y_test


def entrenar_random_forest_secuencial():
    #Entrena un random forest en modo secuencial (n_jobs=1)
    #evalua su desempeño y guarda los resultados en archivos CSV y TXT.
    RESULTS_DIR.mkdir(exist_ok=True)

    #1.carga de datos
    print("espera...cargando..datos..procesados")
    X_train, X_test, y_train, y_test = cargar_datos_procesados()

    print("Tamaño de X_train:", X_train.shape)
    print("Tamaño de X_test:", X_test.shape)
    print("Tamaño de y_train:", y_train.shape)
    print("Tamaño de y_test:", y_test.shape)

    #2.Entrenamiento
    print("\n Entrenando Bosque Aleatorio secuencial")

    inicio_entrenamiento=time.time()

    modelo=RandomForestClassifier(
        n_estimators=100, # número de árboles en el bosque
        random_state=42, #semilla para reproducibilidad
        n_jobs=1  #1=ejecución secuencial
    )

    modelo.fit(X_train, y_train)
    fin_entrenamiento=time.time()

    #3.predicción
    print("realizando..predicciones...")

    inicio_prediccion=time.time()
    y_pred=modelo.predict(X_test)
    fin_prediccion=time.time()

    #4. Tiempos
    tiempo_entrenamiento = fin_entrenamiento - inicio_entrenamiento
    tiempo_prediccion = fin_prediccion - inicio_prediccion
    tiempo_total = tiempo_entrenamiento + tiempo_prediccion

    #---5. Métricas de evaluación--
    #Exactitud general del modelo
    accuracy = accuracy_score(y_test, y_pred)
    # Métricas promediadas de forma macro (todas las clases con el mismo peso)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    #Métricas promediadas de forma ponderada (peso proporcional al soporte de cada clase)
    precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    #Matriz de confusión y reporte completo por clase
    matriz = confusion_matrix(y_test, y_pred)
    reporte = classification_report(y_test, y_pred, zero_division=0)

#---6. Guardado de resultados---
 
    #DataFrame con el resumen de métricas y tiempos.
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


    #Guardar CSV de métricas
    resultados.to_csv(RESULTS_DIR / "resultados_secuencial.csv", index=False)
    
    #guardar CSV de la matriz de confusión
    pd.DataFrame(matriz).to_csv(RESULTS_DIR / "matriz_confusion_secuencial.csv", index=False)

    #Guarda el reporte de clasificación
    with open(RESULTS_DIR / "reporte_clasificacion_secuencial.txt", "w", encoding="utf-8") as archivo:
        archivo.write(reporte)

    
    #Impresión de resumen en consola
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


#punto de entrada
if __name__=="__main__":
    entrenar_random_forest_secuencial()
