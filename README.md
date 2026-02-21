📌 Credit Card Fraud Detection System

A Machine Learning project to detect fraudulent credit card transactions using classification models, imbalance handling techniques, and API deployment.

🚀 Project Overview

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent a very small percentage of total transactions.

In this project, I:

Performed Exploratory Data Analysis (EDA)

Analyzed class imbalance

Applied SMOTE for oversampling

Trained multiple ML models

Optimized decision threshold for recall

Evaluated using Precision, Recall, F1-score, ROC-AUC

Deployed the model using FastAPI

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

Used SMOTE (Synthetic Minority Oversampling Technique)

Balanced fraud vs non-fraud samples

3️⃣ Models Trained

Logistic Regression

Random Forest

MLP Classifier (Neural Network)

4️⃣ Evaluation Metrics

Because of heavy class imbalance, accuracy is not reliable.

Used:

Precision

Recall

F1-score

ROC-AUC score

Confusion Matrix

Threshold tuning to maximize recall

5️⃣ Model Selection

Final model selected based on:

High Recall (to catch fraud cases)

Balanced Precision

Strong ROC-AUC

📈 Results

Fraud Rate in dataset: ~0.17%

ROC-AUC: High discriminatory power

Threshold tuned to improve recall

Successfully detects majority of fraudulent transactions