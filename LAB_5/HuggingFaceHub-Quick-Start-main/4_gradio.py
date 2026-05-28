# Import the necessary libraries
import gradio as gr
import joblib
import numpy as np
from huggingface_hub import hf_hub_download

# Download both models from HF Hub at startup
# Both models live in the same repository: brjapon/iris-dt
dt_path = hf_hub_download(repo_id="brjapon/iris-dt", filename="iris_dt.joblib", repo_type="model")
lr_path = hf_hub_download(repo_id="brjapon/iris-dt", filename="iris_logreg.joblib", repo_type="model")

models = {
    "Decision Tree": joblib.load(dt_path),
    "Logistic Regression": joblib.load(lr_path),
}

LABELS = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

# The function now accepts a model_choice parameter (from a Gradio dropdown)
def predict_iris(model_choice, sepal_length, sepal_width, petal_length, petal_width):
    pipeline = models[model_choice]
    input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = pipeline.predict(input)
    return LABELS.get(int(prediction[0]), "Invalid prediction")

interface = gr.Interface(
    fn=predict_iris,
    inputs=[
        gr.Dropdown(choices=["Decision Tree", "Logistic Regression"], label="Model", value="Decision Tree"),
        gr.Number(label="Sepal Length (cm)"),
        gr.Number(label="Sepal Width (cm)"),
        gr.Number(label="Petal Length (cm)"),
        gr.Number(label="Petal Width (cm)"),
    ],
    outputs="text",
    live=True,
    title="Iris Species Identifier",
    description="Choose a model and enter the four measurements to predict the Iris species.",
    flagging_mode="manual",
    flagging_dir="flagged"
)

if __name__ == "__main__":
    interface.launch()
    
'''
# The Flag button allows users (or testers) to mark or “flag”
# a particular input-output interaction for later review.
# When someone clicks Flag, Gradio saves the input values (and often the output) to a log.csv file
# letting you keep track of interesting or potentially problematic cases for debugging or analysis later on
'''
