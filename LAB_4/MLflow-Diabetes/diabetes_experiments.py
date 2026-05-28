import pandas as pd
import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# MLflow config
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("diabetes_experiments")

MODEL_NAME = "Diabetes Model"
client = MlflowClient()

# Load dataset
df = pd.read_csv("diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 70 / 20 / 10 split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.33,
    random_state=42,
    stratify=y_temp
)

C_values = [0.01, 0.1, 1, 10]
solvers = ["liblinear", "lbfgs"]
class_weights = [None, "balanced"]

results = []

# Grid search
for C in C_values:
    for solver in solvers:
        for cw in class_weights:

            model = LogisticRegression(
                C=C,
                solver=solver,
                class_weight=cw,
                max_iter=1000
            )

            model.fit(X_train, y_train)

            preds = model.predict(X_val)

            accuracy = accuracy_score(y_val, preds)
            precision = precision_score(y_val, preds)
            recall = recall_score(y_val, preds)
            f1 = f1_score(y_val, preds)

            # SALIDA FORMATEADA EXACTAMENTE COMO QUIERES
            print(f"C={C}, solver={solver}, class_weight={cw}")
            print(f"accuracy={accuracy:.3f}, recall={recall:.3f}, f1={f1:.3f}\n")

            results.append({
                "C": C,
                "solver": solver,
                "class_weight": cw,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "model": model
            })

# Ordenar modelos por F1
results = sorted(results, key=lambda x: x["f1"], reverse=True)

print("\n==============================")
print("TOP 3 MODELOS SELECCIONADOS")
print("==============================\n")

top_models = results[:3]

best_version = None

# Registrar los 3 mejores
for i, r in enumerate(top_models):

    print(f"MODELO {i+1}")
    print(f"C={r['C']}, solver={r['solver']}, class_weight={r['class_weight']}")
    print(f"accuracy={r['accuracy']:.3f}, recall={r['recall']:.3f}, f1={r['f1']:.3f}\n")

    with mlflow.start_run():

        mlflow.log_param("C", r["C"])
        mlflow.log_param("solver", r["solver"])
        mlflow.log_param("class_weight", r["class_weight"])

        mlflow.log_metric("accuracy", r["accuracy"])
        mlflow.log_metric("precision", r["precision"])
        mlflow.log_metric("recall", r["recall"])
        mlflow.log_metric("f1_score", r["f1"])

        mlflow.sklearn.log_model(
            sk_model=r["model"],
            name="model"
        )

        run_id = mlflow.active_run().info.run_id

        model_uri = f"runs:/{run_id}/model"

        result = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME
        )

        print("Modelo registrado versión:", result.version, "\n")

        if i == 0:
            best_version = result.version

# Asignar champion
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=best_version
)

print("================================")
print("CHAMPION ASIGNADO")
print("================================")
print("Versión:", best_version)
print("Modelo:", MODEL_NAME)