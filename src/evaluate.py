import pandas as pd
import joblib
import json
import numpy as np
from sklearn.metrics import classification_report, average_precision_score, confusion_matrix
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def evaluate_model_on_test(test_csv_path="../data/creditcard.csv"):
    """
    Evaluate final model on the test split using locked threshold, pipeline, and model.
    Assumes the original creditcard.csv and same splitting logic.

    Parameters:
        test_csv_path (str): Path to full dataset CSV (we'll split out test set)
    """

    SEED = 42

    # --- Load full data
    df = pd.read_csv(test_csv_path)
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # --- Recreate train/test split (must match train.py logic)
    from sklearn.model_selection import train_test_split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    # --- Load pipeline, model, threshold
    pipeline = joblib.load("../models/pipeline.pkl")
    model = joblib.load("../models/rf_model.pkl")
    with open("../models/threshold.json") as f:
        threshold = json.load(f)["threshold"]

    # --- Preprocess test data
    X_test_processed = pipeline.transform(X_test)

    # --- Predict
    y_proba = model.predict_proba(X_test_processed)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # --- Metrics
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    pr_auc = average_precision_score(y_test, y_proba)
    print(f"PR AUC: {pr_auc:.4f}")

    return {
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "pr_auc": pr_auc
    }


if __name__ == "__main__":
    evaluate_model_on_test()
