import pandas as pd
import numpy as np
import time
from mpi4py import MPI
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from itertools import product
import warnings

# Sirve para ignorar la advertencia de scikit-learn y no ensuciar la consola
warnings.filterwarnings('ignore')

#Inicializacion de MPI para prueba 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# rutas del dataset
TRAIN_PATH = r"D:\archive\mitbih_train.csv"
TEST_PATH = r"D:\archive\mitbih_test.csv"

if rank == 0:
    print(f"\n[{time.strftime('%H:%M:%S')}] Iniciando nodo Maestro")#comprobar que ya inicio la prueba y el tiempo en que inicio
    print(f"Simulando Clúster con {size} nucleo(s)")

# cargamos y submuestreamos los datos para hacer una prueba rapida y poder justificar el cluster
train_df = pd.read_csv(TRAIN_PATH, header=None)
test_df = pd.read_csv(TEST_PATH, header=None)

train_sample = train_df.sample(frac=0.02, random_state=42)
test_sample = test_df.sample(frac=0.02, random_state=42)

# las primeras 188 columnas son caracteristicas y la 189 es la etiqueta
X_train = train_sample.iloc[:, :-1].values
y_train = train_sample.iloc[:, -1].values
X_test = test_sample.iloc[:, :-1].values
y_test = test_sample.iloc[:, -1].values

# El maestro genera la busqueda de hiperparametros 
if rank == 0:
    # construimos 180 convinaciones de busqueda
    n_estimators = [50, 100, 150, 200, 250]      # 5 opciones
    max_depth = [10, 20, None]                   # 3 opciones
    min_samples_split = [2, 5, 10]               # 3 opciones
    criterion = ['gini', 'entropy']              # 2 opciones
    class_weight = [None, 'balanced']            # 2 opciones
    
    all_combinations = list(product(n_estimators, max_depth, min_samples_split, criterion, class_weight))
    print(f"[{time.strftime('%H:%M:%S')}] Maestro: Matriz de {len(all_combinations)} combinaciones generada.")
    
    # Dividir el trabajo entre los nucleos equitativamente
    chunks = np.array_split(all_combinations, size)
else:
    chunks = None

# se reparte el trabajo entre todos los trabajadores
my_chunk = comm.scatter(chunks, root=0)

# se entrena localmente simulando los maestros y trabajadores
my_results = []
start_time = time.time()#iniciamos toma de tiempo

for combo in my_chunk:
    n_est, m_depth, min_split, crit, c_weight = combo
    
    # se usa n_jobs=1 para asegura que cada RF use solo 1 hilo 
    # MPI ya se encarga del paralelismo a nivel de proceso.
    clf = RandomForestClassifier(
        n_estimators=int(n_est),
        max_depth=m_depth,
        min_samples_split=int(min_split),
        criterion=crit,
        class_weight=c_weight,
        n_jobs=1, 
        random_state=42
    )
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # metricas para el reporte
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    
    my_results.append({
        'params': combo,
        'accuracy': acc,
        'f1_score': f1,
        'recall': rec
    })

local_time = time.time() - start_time

# se recuperan los diccionarios de metricas
all_results_gathered = comm.gather(my_results, root=0)
all_times = comm.gather(local_time, root=0)

# trabajo de solo el maestro
if rank == 0:
    # se aplana la lista
    flat_results = [res for sublist in all_results_gathered for res in sublist]
    
    # el cluster que mas tardo es el que define el tiempo 
    max_time = max(all_times) 
    
    # se busca al mejor usando el F1-Score
    best_model = max(flat_results, key=lambda x: x['f1_score'])
    
    print("           RESULTADOS")
    print(f"Núcleos utilizados : {size}")
    print(f"Tiempo de ejecución: {max_time:.2f} segundos")
    print("Mejor Configuración")
    print(f" Árboles (n_estimators)   : {best_model['params'][0]}")
    print(f" Profundidad (max_depth)  : {best_model['params'][1]}")
    print(f" Min Samples Split        : {best_model['params'][2]}")
    print(f" Criterio                 : {best_model['params'][3]}")
    print(f" Pesos de Clase           : {best_model['params'][4]}")
    print("Métricas")
    print(f" Exactitud (Accuracy)     : {best_model['accuracy']:.4f}")
    print(f" F1-Score (Ponderado)     : {best_model['f1_score']:.4f}")
    print(f" Recall                   : {best_model['recall']:.4f}")
