import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report, roc_auc_score

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load dataset
df = pd.read_csv(os.path.join(ROOT_DIR, "data", "creditcard.csv"))

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build Pipeline (Scaling + SMOTE + Model). imblearn's Pipeline resamples
# during fit() and skips resampling during predict()/predict_proba(), so
# SMOTE only ever sees the training fold and never touches the test data.
# Scaling happens before SMOTE so neighbour distances aren't dominated by
# Time and Amount, which are orders of magnitude larger than the V1-V28
# PCA features.
pipeline = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(max_iter=1000)),
])

# Train model
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

# Evaluation
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# Save model
model_path = os.path.join(ROOT_DIR, "fraud_model.pkl")
joblib.dump(pipeline, model_path)

print(f"Model saved as {model_path}")
