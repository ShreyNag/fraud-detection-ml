import joblib
import numpy as np

# Load model
model = joblib.load("fraud_model.pkl")

# Example input (30 features)
sample = np.random.rand(1, 30)

prediction = model.predict(sample)
probability = model.predict_proba(sample)[:, 1]

print("Prediction:", prediction[0])
print("Fraud Probability:", probability[0])
