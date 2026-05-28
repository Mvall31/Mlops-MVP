import pandas as pd
import requests
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Cargar dataset
df = pd.read_csv("diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Reproducir split 70/20/10
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.33, random_state=42, stratify=y_temp
)

# Endpoint de la API
url = "http://localhost:1235/invocations"

predictions = []

for _, row in X_test.iterrows():

    payload = {
        "dataframe_split": {
            "columns": list(X.columns),
            "data": [row.tolist()]
        }
    }

    headers = {"Content-Type": "application/json"}

    response = requests.post(url, data=json.dumps(payload), headers=headers)

    # imprimir respuesta si hay error
    resp_json = response.json()

    if "predictions" not in resp_json:
        print("Error respuesta API:", resp_json)
        continue

    pred = resp_json["predictions"][0]

    predictions.append(pred)

# Calcular métricas
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("\n=== RESULTADOS TEST (API) ===\n")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)