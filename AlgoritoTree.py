import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# 1. Crear datos de ejemplo (simulando historial de solicitudes)
data = {
    'Nombre': ['Juan', 'María', 'Pedro', 'Ana', 'Luis', 'Laura', 'Carlos', 'Sofía', 'Diego', 'Elena'],
    'Edad': [25, 45, 30, 50, 35, 28, 42, 55, 33, 40],
    'Ingresos': [30000, 70000, 45000, 80000, 60000, 35000, 75000, 90000, 55000, 65000],
    'Puntuacion_Credito': [650, 720, 680, 780, 710, 630, 750, 800, 690, 730],
    'Deuda': [5000, 20000, 15000, 10000, 8000, 12000, 18000, 5000, 10000, 15000],
    'Empleo_Anios': [2, 10, 5, 15, 8, 3, 12, 20, 6, 9],
    'Aprobado': ['No', 'Sí', 'No', 'Sí', 'Sí', 'No', 'Sí', 'Sí', 'Sí', 'Sí']
}

datos = pd.DataFrame(data)


le = LabelEncoder()
datos['Aprobado'] = le.fit_transform(datos['Aprobado'])

# Separar características y variable objetivo
X = datos.drop(['Nombre', 'Aprobado'], axis=1)
y = datos['Aprobado']

# Escalar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Entrenar modelo
modelo = DecisionTreeClassifier(random_state=42, max_depth=3)
modelo.fit(X_scaled, y)

# 4. Función para predecir en tiempo real
def predecir_aprobacion():
    print("\nIngresa los datos del solicitante:")
    nombre = input("Nombre: ")
    edad = int(input("Edad: "))
    ingresos = float(input("Ingresos anuales: "))
    puntuacion = int(input("Puntuación de crédito (300-850): "))
    deuda = float(input("Deuda actual: "))
    empleo_anios = int(input("Años en el empleo actual: "))
    
    # Crear dataframe con los nuevos datos
    nuevo_solicitante = pd.DataFrame({
        'Edad': [edad],
        'Ingresos': [ingresos],
        'Puntuacion_Credito': [puntuacion],
        'Deuda': [deuda],
        'Empleo_Anios': [empleo_anios]
    })
    
    # Escalar los datos igual que el conjunto de entrenamiento
    nuevo_solicitante_scaled = scaler.transform(nuevo_solicitante)
    
    # Predecir
    prediccion = modelo.predict(nuevo_solicitante_scaled)
    
    # Convertir predicción numérica a texto
    resultado = "SÍ" if prediccion[0] == 1 else "NO"
    
    print(f"\nResultado para {nombre}:")
    print("----------------------------")
    print(f"¿La solicitud será aprobada? {resultado}")
    print("----------------------------")

# 5. Interfaz de consola
while True:
    print("\nSistema de Predicción de Aprobación de Crédito")
    print("1. Predecir aprobación")
    print("2. Salir")
    
    opcion = input("Selecciona una opción (1/2): ")
    
    if opcion == '1':
        predecir_aprobacion()
    elif opcion == '2':
        print("Saliendo del sistema...")
        break
    else:
        print("Opción no válida. Por favor ingresa 1 o 2.")

    continuar = input("\n¿Deseas realizar otra predicción? (s/n): ")
    if continuar.lower() != 's':
        print("Saliendo del sistema...")
        break