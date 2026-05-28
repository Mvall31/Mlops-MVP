import pandas as pd
import requests
import json

# Cargar dataset
df = pd.read_csv("diabetes.csv")

# Separar features
X = df.drop("Outcome", axis=1)

# Tomar algunas filas de ejemplo
sample = X.iloc[:5].values.tolist()

# Endpoint del modelo servido
url = "http://localhost:1235/invocations"

# Payload correcto para MLflow
payload = {
    "dataframe_split": {
        "columns": list(X.columns),
        "data": sample
    }
}

headers = {"Content-Type": "application/json"}

# Enviar request
response = requests.post(url, data=json.dumps(payload), headers=headers)

# Mostrar respuesta completa (útil para debug)
print("\nRespuesta API:")
print(response.json())

# Obtener predicciones
preds = response.json()["predictions"]

print("\n=== PREDICCIONES ===\n")

for i, p in enumerate(preds):
    print(f"Fila {i+1} -> Predicción: {p}")