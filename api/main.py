# got to http://localhost:8000/docs to see the API documentation

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from src.utils import generate_dummy_sample

# ----------- Startup: Load Model, Pipeline, Threshold -----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))

PIPELINE_PATH = os.path.join(MODELS_DIR, "pipeline.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "rf_model.pkl")
THRESHOLD_PATH = os.path.join(MODELS_DIR, "threshold.json")
FEATURE_PATH = os.path.join(MODELS_DIR, "feature_names.json")

# Load on startup
pipeline = joblib.load(PIPELINE_PATH)
model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH) as f:
    threshold = json.load(f)["threshold"]

with open(FEATURE_PATH) as f:
    feature_names = json.load(f)


# ----------- FastAPI App Setup -----------
app = FastAPI(
    title="Fraud Detection API",
    description="API for predicting fraudulent credit card transactions.",
    version="1.0.0"
)

# ----------- Pydantic Input Schema -----------
class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


# ----------- Routes -----------

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(transaction: Transaction):
    # Convert to DataFrame
    input_dict = transaction.dict()
    df = pd.DataFrame([input_dict])

    # Ensure column order
    df = df[feature_names]

    # Transform + Predict
    X = pipeline.transform(df)
    proba = model.predict_proba(X)[:, 1][0]
    pred = int(proba >= threshold)

    return {
        "prediction": pred,
        "probability": round(proba, 6)
    }


@app.get("/generate_dummy")
def generate(fraud_only: bool = False, seed: int = 42):
    dummy = generate_dummy_sample(fraud_only=fraud_only, seed=seed)
    return dummy.to_dict(orient="records")[0]
