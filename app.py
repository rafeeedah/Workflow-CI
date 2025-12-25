from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import mlflow.pyfunc

app = FastAPI()

model = mlflow.pyfunc.load_model("/opt/ml/model")

class PredictRequest(BaseModel):
    inputs: List[List[float]]

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    preds = model.predict(req.inputs)
    return {"predictions": preds.tolist()}
