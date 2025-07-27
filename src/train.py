import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.preprocessing import build_preprocessing_pipeline
from src.utils import tune_thresholds

SEED = 42

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv("../data/creditcard.csv")
    X_raw = df.drop("Class", axis=1)
    y = df["Class"]

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(X_raw, y, test_size=0.2, stratify=y, random_state=SEED)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=SEED)

    # Build and fit pipeline
    pipeline = build_preprocessing_pipeline()
    X_train = pipeline.fit_transform(X_train_raw)
    X_val = pipeline.transform(X_val_raw)
    X_test = pipeline.transform(X_test)

    # Save pipeline and features
    joblib.dump(pipeline, "../models/pipeline.pkl")
    with open("../models/feature_names.json", "w") as f:
        json.dump(X_train_raw.columns.tolist(), f)

    # Apply SMOTE
    print("Applying SMOTE to training data...")
    smote = SMOTE(random_state=SEED)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Train model
    print("Training Random Forest model...")
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
    rf.fit(X_train_smote, y_train_smote)
    joblib.dump(rf, "../models/rf_model.pkl")

    # Tune threshold
    y_val_proba = rf.predict_proba(X_val)[:, 1]
    best_threshold = tune_thresholds(y_val, y_val_proba, model_name="RF Final")
    with open("../models/threshold.json", "w") as f:
        json.dump({"threshold": best_threshold}, f)

    print("\nTraining complete. Artifacts saved to /models.")

if __name__ == "__main__":
    main()