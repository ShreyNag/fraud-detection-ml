import os
import joblib
import numpy as np

# Load model
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(ROOT_DIR, "fraud_model.pkl"))

# Example input (30 features)
sample = np.random.rand(1, 30)

prediction = model.predict(sample)
probability = model.predict_proba(sample)[:, 1]

print("Prediction:", prediction[0])
print("Fraud Probability:", probability[0])
