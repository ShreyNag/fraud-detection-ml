📌 Credit Card Fraud Detection System

A Machine Learning project to detect fraudulent credit card transactions, with imbalance handling, cost-based threshold tuning, and API deployment.

🚀 Project Overview

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent a very small percentage of total transactions.

In this project, I:

Performed Exploratory Data Analysis (EDA)

Analyzed class imbalance

Applied SMOTE for oversampling, correctly ordered after feature scaling

Trained a Logistic Regression baseline

Tuned the decision threshold using a cost function

Evaluated using PR-AUC (primary), ROC-AUC, Precision, Recall, F1-score

Deployed the model using FastAPI, with a Dockerfile for containerised serving

📊 Dataset

Source: Kaggle – Credit Card Fraud Detection Dataset

Contains anonymized transaction features (V1–V28), Time, Amount

Target variable:

0 → Legitimate transaction

1 → Fraudulent transaction

⚠ Due to licensing and size limits, the dataset is not included in this repository.

You can download it from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the file inside a data/ folder before training.

🧠 Machine Learning Pipeline

1️⃣ Data Preprocessing

Checked for missing values

Verified class imbalance (~0.17% fraud cases)

Feature scaling (StandardScaler)

2️⃣ Handling Imbalanced Data

Used SMOTE (Synthetic Minority Oversampling Technique), applied via an imblearn Pipeline so scaling happens before resampling. This matters: SMOTE picks neighbours by Euclidean distance, and Time/Amount are orders of magnitude larger than the V1–V28 PCA features, so scaling first keeps SMOTE from being dominated by those two columns. The imblearn Pipeline also ensures resampling only happens during fit, never at prediction time.

**SMOTE vs. class weighting.** Rather than assuming SMOTE is the right call, `src/train.py` trains both approaches on the same split and compares them:

| Approach | PR-AUC | ROC-AUC | Threshold | Precision | Recall |
|---|---|---|---|---|---|
| SMOTE | 0.7245 | 0.9708 | 0.69 | 0.1002 | 0.9082 |
| Class weighting | 0.7190 | 0.9721 | 0.74 | 0.1369 | 0.9082 |

(Threshold, precision, and recall above are at `FP_COST = 1`; PR-AUC and ROC-AUC are threshold-independent and unaffected by that constant.)

SMOTE has the higher PR-AUC, so it's the one saved to `fraud_model.pkl`. The margin is small, and the code keeps both pipelines implemented so the comparison stays reproducible on re-run.

Class weighting reweights the loss function rather than fabricating minority samples. SMOTE, by contrast, assumes the minority class is locally convex — questionable for fraud, where distinct attack types occupy different regions of feature space, so interpolating between two unrelated frauds can produce a point that corresponds to no real fraud.

3️⃣ Model

Logistic Regression is the implemented baseline. It's interpretable and a strong baseline on PCA-transformed features, which is why it's the only model in this repo.

**Future work:** tree-based and gradient-boosted models (Random Forest, XGBoost/LightGBM) are the natural next comparison, but they are not implemented here.

4️⃣ Evaluation Metrics

Because of heavy class imbalance, accuracy is not reliable, and ROC-AUC can be misleading too: its false-positive-rate denominator is the (large) negative class, so hundreds of false alarms barely move the curve. **PR-AUC (average precision) is reported as the headline metric** because it's driven by precision and recall on the minority (fraud) class directly, which is what actually matters at ~0.17% positive rate. ROC-AUC is reported as a secondary metric.

Also reported:

Precision, Recall, F1-score

Confusion Matrix at the chosen threshold

5️⃣ Threshold Tuning

Rather than using the default 0.5 cutoff, the training script sweeps thresholds from 0.01 to 0.99 and picks the one that minimises:

    cost = (total dollar amount of missed frauds) + (false positives * FP_COST)

False negatives are weighted by the actual transaction `Amount`, so missing a large fraud costs more than missing a small one. `FP_COST` is a placeholder constant representing the cost of wrongly declining a legitimate transaction; the real value would come from the business.

📈 Results

Fraud Rate in dataset: ~0.17%

Run `src/train.py` to reproduce; PR-AUC, ROC-AUC, the threshold sweep table, and the chosen threshold are printed to stdout for both the SMOTE and class-weighting approaches (see the comparison above), and the winning pipeline is saved to `fraud_model.pkl`.

⚙️ Running

**`fraud_model.pkl` is not committed to this repository.** You must run training before starting the API or building the Docker image:

```
python src/train.py
```

This trains the pipeline, tunes the threshold, and writes `fraud_model.pkl` to the project root.

Then run the API locally:

```
uvicorn app.main:app --reload
```

Or build the Docker image (requires `fraud_model.pkl` to already exist, since it's copied in at build time):

```
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
```

⚠ Known limitations

- **V1–V28 are anonymised PCA components.** No feature engineering or interpretation of individual features is possible.
- **The train/test split is random, not temporal.** This is optimistic compared to a real deployment, since fraud patterns evolve over time and a random split lets the model "see the future" relative to any given transaction.
- **`Time` is seconds since the dataset's first transaction**, not a real timestamp. It has no meaningful value for a live transaction and the model would need to be retrained without it (or with a properly engineered time feature) for real-world deployment.
