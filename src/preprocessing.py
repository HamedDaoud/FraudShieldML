import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
import joblib
import os, sys


# For V1 to V28
v_features = [f'V{i}' for i in range(1, 29)]

# --- Custom Transformer: Time → Cyclic Hour ---
def time_to_hour_cyclic_array(X):
    hour = (X[:, 0] // 3600) % 24
    return np.column_stack([
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24)
    ])

# --- Preprocessing Pipeline ---
def build_preprocessing_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ('log_amount', FunctionTransformer(np.log1p, validate=True), ['Amount']),
            ('cyclic_time', FunctionTransformer(time_to_hour_cyclic_array, validate=True), ['Time']),
            ('v_features', 'passthrough', v_features)
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler())
    ])
    
    return pipeline