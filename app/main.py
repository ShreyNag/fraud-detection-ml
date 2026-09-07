import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Load trained model
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "fraud_model.pkl")
model = joblib.load(MODEL_PATH)

INDEX_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html")


class Transaction(BaseModel):
    features: list


@app.get("/", response_class=HTMLResponse)
def home():
    with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    return {"message": "Fraud Detection API is running"}


@app.post("/predict")
def predict(transaction: Transaction):
    data = np.array(transaction.features).reshape(1, -1)
    
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[:, 1][0]

    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": float(probability)
    }
