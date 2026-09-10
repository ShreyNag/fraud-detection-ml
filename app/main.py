import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
from pydantic import BaseModel, conlist

app = FastAPI()

# Load trained model + threshold
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "fraud_model.pkl")

model_bundle = None
if os.path.exists(MODEL_PATH):
    model_bundle = joblib.load(MODEL_PATH)

INDEX_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html")


class Transaction(BaseModel):
    features: conlist(float, min_length=30, max_length=30)


@app.get("/", response_class=HTMLResponse)
def home():
    with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    return {"message": "Fraud Detection API is running"}


@app.post("/predict")
def predict(transaction: Transaction):
    if model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="fraud_model.pkl not found. Run src/train.py first to train and save the model.",
        )

    pipeline = model_bundle["pipeline"]
    threshold = model_bundle["threshold"]

    data = np.array(transaction.features).reshape(1, -1)
    probability = pipeline.predict_proba(data)[:, 1][0]
    prediction = int(probability >= threshold)

    return {
        "fraud_prediction": prediction,
        "fraud_probability": float(probability),
        "threshold_used": threshold,
    }
