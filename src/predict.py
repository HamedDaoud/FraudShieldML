import joblib
import json
import numpy as np
import pandas as pd
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.utils import generate_dummy_sample


def predict_new_data(
    df,
    pipeline_path=None,
    model_path=None,
    threshold_path=None
):
    """
    Predict fraud labels and probabilities on new raw input data.

    Parameters:
        df (pd.DataFrame): New input data (includes Time, Amount, V1–V28)
        pipeline_path (str): Path to saved preprocessing pipeline
        model_path (str): Path to saved model
        threshold_path (str): Path to threshold JSON

    Returns:
        preds (np.ndarray): Predicted labels (0 or 1)
        proba (np.ndarray): Fraud probability scores
    """
    # Resolve project root based on this file location
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Default to correct relative paths
    if pipeline_path is None:
        pipeline_path = os.path.join(base_path, "models", "pipeline.pkl")
    if model_path is None:
        model_path = os.path.join(base_path, "models", "rf_model.pkl")
    if threshold_path is None:
        threshold_path = os.path.join(base_path, "models", "threshold.json")

    # Load pipeline, model, threshold
    pipeline = joblib.load(pipeline_path)
    model = joblib.load(model_path)
    with open(threshold_path, "r") as f:
        threshold = json.load(f)["threshold"]

    # Transform input and predict
    X = pipeline.transform(df)
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    return preds, proba


if __name__ == "__main__":
    dummy = generate_dummy_sample(fraud_only=True, noise_scale=0.1, seed=None)
    preds, probas = predict_new_data(dummy)
    print("Prediction:", preds[0])
    print("Probability:", probas[0])