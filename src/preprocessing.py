import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def cargar_datos(ruta_csv):
    df = pd.read_csv(ruta_csv, header=None)

    # Características
    X = df.iloc[:, :-1].values

    # Etiquetas
    y = df.iloc[:, -1].values

    return X, y

def preprocesar_datos(X, y):
    # División entrenamiento/prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Normalización
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    X, y = cargar_datos("data/raw/mitbih_test.csv")

    X_train, X_test, y_train, y_test = preprocesar_datos(X, y)

    # Guardar datos procesados
    pd.DataFrame(X_train).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv("data/processed/X_test.csv", index=False)

    pd.DataFrame(y_train).to_csv("data/processed/y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv("data/processed/y_test.csv", index=False)

    print("Datos preprocesados correctamente")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
