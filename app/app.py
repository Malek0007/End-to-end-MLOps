import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import warnings

warnings.filterwarnings("ignore")

app = FastAPI()

# Load model at startup
print("Loading model from models/us_accident_model.pkl...")
model = joblib.load("models/us_accident_model.pkl")
print("Model loaded:", type(model))


class PredictionInput(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(input_data: PredictionInput):
    if len(input_data.features) != 8:
        return {"error": "Expected 8 features"}

    arr = np.array(input_data.features).reshape(1, -1)
    prediction = model.predict(arr)

    return {
        "prediction": int(prediction[0])
    }