import pandas as pd

def calcularMetricas():
    print("Calculo de speedup y eficiencia con la muestra de 2%\n")

    # datos obtenidos del cluster
    datos = {
        "nucleos": [1, 6, 11],
        "tiempo": [228.55, 80.77, 67.61]
    }

    df = pd.DataFrame(datos)

    # tiempo base de 1 nucleo
    t1 = df[df["nucleos"] == 1]["tiempo"].values[0]

    # calculo de speedup
    df["speedup"] = t1 / df["tiempo"]

    # calculo de eficiencia
    df["eficiencia"] = (df["speedup"] / df["nucleos"]) * 100

    # redondeo para mejorar la lectura
    df["speedup"] = df["speedup"].round(2)
    df["eficiencia"] = df["eficiencia"].round(1)

    print(df.to_string(index=False))

    # guardar resultados
    df.to_csv("tiempos finales.csv", index=False)
    print("\nGuardado")

if __name__ == "__main__":
    calcularMetricas()
