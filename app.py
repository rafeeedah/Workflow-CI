from fastapi import FastAPI
import mlflow.pyfunc

app = FastAPI()

# load model when container starts
model = mlflow.pyfunc.load_model("/mlflow/model")

@app.post("/predict")
def predict(data: dict):
    preds = model.predict(data["inputs"])
    return {"predictions": preds.tolist()}
