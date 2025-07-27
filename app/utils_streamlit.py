# app/utils_streamlit.py

import json
import requests
import numpy as np
from pathlib import Path
import streamlit as st

API_URL = "http://localhost:8000"

@st.cache_data
def load_feature_names():
    """Load feature names from saved JSON."""
    project_root = Path(__file__).resolve().parent.parent
    path = project_root / "models" / "feature_names.json"
    with open(path) as f:
        return json.load(f)

def fetch_dummy_sample(fraud_only=False):
    """Call the API to generate a dummy sample."""
    try:
        params = {
            "fraud_only": fraud_only,
            "seed": np.random.randint(10000)
        }
        r = requests.get(f"{API_URL}/generate_dummy", params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"❌ Failed to fetch dummy sample: {e}")
        return None

def predict_transaction(input_data):
    """Call the API to predict a transaction."""
    try:
        r = requests.post(f"{API_URL}/predict", json=input_data)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        return None