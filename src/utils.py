import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os, sys

def tune_thresholds(y_true, y_proba, model_name='Model'):
    thresholds = np.arange(0.0, 1.01, 0.01)
    precisions = []
    recalls = []
    f1s = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred))
        f1s.append(f1_score(y_true, y_pred))

    best_f1_idx = np.argmax(f1s)
    best_threshold = thresholds[best_f1_idx]

    print(f"\n{model_name} — Best F1 at threshold {best_threshold:.2f}")
    print(f"Precision: {precisions[best_f1_idx]:.4f}, Recall: {recalls[best_f1_idx]:.4f}, F1: {f1s[best_f1_idx]:.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f1s, label='F1 Score')
    plt.axvline(x=best_threshold, color='gray', linestyle='--', label=f'Best F1 Threshold = {best_threshold:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'{model_name} - Threshold Tuning')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return best_threshold

def run_logreg(X, y, scale=False, desc="", SEED=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=SEED
    )

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=SEED)
    model.fit(X_train, y_train)
    
    precision, recall, pr_auc = evaluate_model(model, X_test, y_test, model_name=desc)
    return precision, recall, pr_auc


def evaluate_model(model, X_test, y_test, model_name='Model'):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Scores
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"\nEvaluation: {model_name}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc:.4f} (reference)")
    print(f"PR AUC Score: {pr_auc:.4f}")

    # For plotting PR curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    return precision, recall, pr_auc

import pandas as pd
import numpy as np
import json

def generate_dummy_sample(
    data_path=None,
    feature_path=None,
    noise_scale=0.1,
    fraud_only=False,
    seed=None
):
    """
    Generate a dummy sample from the credit card dataset with slight noise, ready for inference.

    Parameters:
        data_path (str): Full path to creditcard.csv (default: relative to project root)
        feature_path (str): Full path to feature_names.json (default: relative to project root)
        noise_scale (float): Standard deviation of Gaussian noise to apply
        fraud_only (bool): Whether to sample only from fraud rows
        seed (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: A single dummy sample with correct column order
    """
    # Get base path of the project (two levels up from this file)
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Default paths (if none provided)
    if data_path is None:
        data_path = os.path.join(base_path, "data", "creditcard.csv")
    if feature_path is None:
        feature_path = os.path.join(base_path, "models", "feature_names.json")

    if seed is not None:
        np.random.seed(seed)

    df = pd.read_csv(data_path)

    # Filter if fraud_only is True
    if fraud_only:
        df = df[df["Class"] == 1]

    # Drop target column
    df = df.drop(columns="Class")

    # Sample a row
    base = df.sample(1)
    dummy = base.copy()

    # Add small noise to numerical features
    for col in base.columns:
        if col.startswith("V") or col in ["Amount"]:
            dummy[col] += np.random.normal(scale=noise_scale)

    # Load feature order
    with open(feature_path, "r") as f:
        feature_names = json.load(f)

    # Reorder columns
    dummy = dummy[feature_names]

    return dummy