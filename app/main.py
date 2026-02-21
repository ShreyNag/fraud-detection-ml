from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Load trained model
model = joblib.load("fraud_model.pkl")


class Transaction(BaseModel):
    features: list


@app.get("/")
def home():
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
