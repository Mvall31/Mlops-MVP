import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_NAME = "Diabetes Model"

with open("last_model_uri.txt", "r") as f:
    model_uri = f.read().strip()

client = MlflowClient()

result = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

print("\nModelo registrado:", MODEL_NAME)
print("Versión:", result.version)