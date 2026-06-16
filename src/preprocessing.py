import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Carga el dataset MIT-BIH desde un archivo CSV
def cargar_datos(ruta_csv):
    df = pd.read_csv(ruta_csv, header=None)

    # Separa las señales ECG de la etiqueta de clase
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    return X, y

# Divide los datos y aplica normalización
def preprocesar_datos(X, y):
    # 80% entrenamiento y 20% prueba, manteniendo la proporción de clases
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Normaliza las características para mejorar el entrenamiento del modelo
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# Punto de entrada del script
if __name__ == "__main__":
    # Crea la carpeta donde se guardarán los datos procesados
    os.makedirs("data/processed", exist_ok=True)

    # Carga el dataset original
    X, y = cargar_datos("data/raw/mitbih_test.csv")

    # Ejecuta el preprocesamiento
    X_train, X_test, y_train, y_test = preprocesar_datos(X, y)

    # Guarda los conjuntos procesados en archivos CSV
    pd.DataFrame(X_train).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv("data/processed/X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv("data/processed/y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv("data/processed/y_test.csv", index=False)

    # Muestra confirmación y dimensiones de los datos generados
    print("Datos preprocesados correctamente")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
