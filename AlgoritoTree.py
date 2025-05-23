import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def cargar_datos():
    try:
        datos = pd.read_csv('datos.csv')
        print("Datos cargados exitosamente!")
        print(f"Registros cargados: {len(datos)}")
        return datos
    except FileNotFoundError:
        print("\nError: No se encontró el archivo 'datos.csv'")
        print("Asegúrate de que:")
        print("1. El archivo existe en la misma carpeta que este script")
        print("2. Se llama exactamente 'datos.csv'")
        return None
    except Exception as e:
        print(f"\nError al leer el archivo: {str(e)}")
        return None

def preparar_modelo(datos):

    columnas_requeridas = {'Edad', 'Ingresos', 'Deuda', 'Empleo_Anios', 'Aprobado'}
    if not columnas_requeridas.issubset(datos.columns):
        print("\nError: El CSV debe contener estas columnas exactas:")
        print("Edad, Ingresos, Deuda, Empleo_Anios, Aprobado")
        print("\nColumnas encontradas:")
        print(datos.columns.tolist())
        return None, None, None
 
    datos_limpios = datos.dropna()
    if len(datos_limpios) == 0:
        print("\nError: No hay datos válidos después de limpiar valores faltantes")
        return None, None, None

    le = LabelEncoder()
    datos_limpios['Aprobado'] = le.fit_transform(datos_limpios['Aprobado'])

    X = datos_limpios[['Edad', 'Ingresos', 'Deuda', 'Empleo_Anios']]
    y = datos_limpios['Aprobado']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = DecisionTreeClassifier(random_state=42, max_depth=3)
    modelo.fit(X_scaled, y)
    
    return modelo, scaler, le

def predecir(modelo, scaler):
    print("\nIngrese los datos del cliente:")
    try:
        edad = int(input("Edad: "))
        ingresos = float(input("Ingresos Anuales: "))
        deuda = float(input("Deuda actual: "))
        empleo = int(input("Años en el empleo actual: "))
        
        nuevo = pd.DataFrame([[edad, ingresos, deuda, empleo]], 
                           columns=['Edad', 'Ingresos', 'Deuda', 'Empleo_Anios'])
        
        nuevo_scaled = scaler.transform(nuevo)
        aprobado = modelo.predict(nuevo_scaled)[0]
        
        print("\nResultado de la precalificacion:")
        print("--------------------------")
        print("APROBADO: Sí" if aprobado == 1 else "APROBADO: No")
        print("--------------------------")
    except ValueError:
        print("Error: Ingrese valores numéricos válidos")

def main():
    print("\nSISTEMA DE PRECALIFICACION DE CLIENTE")
    print("------------------------------------")

    datos = cargar_datos()
    if datos is None:
        return

    modelo, scaler, _ = preparar_modelo(datos)
    if modelo is None:
        return

    while True:
        print("\nOpciones:")
        print("1. Precalificar Cliente")
        print("2. Salir")
        
        opcion = input("Seleccione una opción (1-2): ").strip()
        
        if opcion == '1':
            predecir(modelo, scaler)
        elif opcion == '2':
            print("\nSaliendo del sistema...")
            break
        else:
            print("Opción no válida")
        
        if opcion == '1':
            if input("\n¿Otra predicción? (s/n): ").lower() != 's':
                print("\nSaliendo del sistema...")
                break

if __name__ == "__main__":
    main()