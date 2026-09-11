import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cost of wrongly declining a legitimate transaction (a false positive).
# This is a placeholder — the real figure would come from the business
# (e.g. customer friction, support load, lost transaction revenue).
FP_COST = 5.0

# Load dataset
df = pd.read_csv(os.path.join(ROOT_DIR, "data", "creditcard.csv"))

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

amounts_test = X_test["Amount"].to_numpy()
thresholds = np.arange(0.01, 1.00, 0.01)


def train_and_evaluate(name, pipeline, X_train, y_train, X_test, y_test):
    """Fit `pipeline` and evaluate it on the held-out test set.

    Returns a dict with PR-AUC, ROC-AUC, and the cost-minimising threshold
    together with the precision/recall achieved at that threshold, so
    different imbalance-handling strategies can be compared on equal terms.
    """
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\n=== {name} ===")
    print("PR-AUC (average precision):", pr_auc)
    print("ROC-AUC:", roc_auc)
    print(classification_report(y_test, y_pred))

    # Threshold tuning: minimise cost = (dollar amount of missed frauds) +
    # (false positives * FP_COST), rather than using the arbitrary 0.5 default.
    results = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="binary", zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        fn_amount = amounts_test[(preds == 0) & (y_test.to_numpy() == 1)].sum()
        cost = fn_amount + fp * FP_COST
        results.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fp": fp,
            "fn": fn,
            "fn_amount": fn_amount,
            "cost": cost,
        })

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["cost"].idxmin()]
    best_threshold = float(best_row["threshold"])

    print("\nThreshold sweep (sample):")
    sample = results_df.iloc[::10]  # every 10th threshold, ~10 rows
    print(sample[["threshold", "precision", "recall", "f1", "fp", "fn", "cost"]]
          .to_string(index=False))

    print(f"\nCost-minimising threshold: {best_threshold:.2f}")
    print(f"  Precision: {best_row['precision']:.4f}")
    print(f"  Recall:    {best_row['recall']:.4f}")
    print(f"  F1:        {best_row['f1']:.4f}")
    print(f"  Cost:      {best_row['cost']:.2f}")

    final_preds = (y_prob >= best_threshold).astype(int)
    print("\nConfusion matrix at chosen threshold:")
    print(confusion_matrix(y_test, final_preds))

    return {
        "name": name,
        "pipeline": pipeline,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "threshold": best_threshold,
        "precision": float(best_row["precision"]),
        "recall": float(best_row["recall"]),
    }


# Approach A: SMOTE. Scaling happens before SMOTE so neighbour distances
# aren't dominated by Time and Amount, which are orders of magnitude larger
# than the V1-V28 PCA features. imblearn's Pipeline resamples during fit()
# and skips resampling during predict()/predict_proba(), so SMOTE only ever
# sees the training fold and never touches the test data.
smote_pipeline = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(max_iter=1000)),
])

# Approach B: class weighting. No resampling at all — the loss function is
# reweighted so misclassifying a minority-class sample costs more, instead
# of synthesizing new minority-class rows.
weighted_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

smote_result = train_and_evaluate(
    "SMOTE", smote_pipeline, X_train, y_train, X_test, y_test
)
weighted_result = train_and_evaluate(
    "Class weighting", weighted_pipeline, X_train, y_train, X_test, y_test
)

# Comparison table
print("\n=== Comparison ===")
comparison_df = pd.DataFrame([
    {
        "approach": r["name"],
        "pr_auc": r["pr_auc"],
        "roc_auc": r["roc_auc"],
        "threshold": r["threshold"],
        "precision": r["precision"],
        "recall": r["recall"],
    }
    for r in (smote_result, weighted_result)
])
print(comparison_df.to_string(index=False))

# Select whichever approach has the higher PR-AUC, since PR-AUC is the
# primary metric at this class imbalance (see README).
best_result = max(smote_result, weighted_result, key=lambda r: r["pr_auc"])
other_result = weighted_result if best_result is smote_result else smote_result

print(
    f"\nSelected: {best_result['name']} "
    f"(PR-AUC {best_result['pr_auc']:.4f} vs {other_result['name']} "
    f"{other_result['pr_auc']:.4f})"
)

# Save the winning pipeline + its threshold together
model_path = os.path.join(ROOT_DIR, "fraud_model.pkl")
joblib.dump(
    {"pipeline": best_result["pipeline"], "threshold": best_result["threshold"]},
    model_path,
)

print(f"\nModel saved as {model_path}")
