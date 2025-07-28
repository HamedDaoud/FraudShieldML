import streamlit as st
import pandas as pd
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from app.utils_streamlit import load_feature_names, fetch_dummy_sample, predict_transaction

# --- Config
st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("💳 FraudShieldML")

# --- Load features
FEATURE_NAMES = load_feature_names()

# --- Sidebar: Sample controls
st.sidebar.header("⚙️ Generate Sample")
sample_type = st.sidebar.radio("Choose sample type:", ["Random", "Fraud-like"])
generate = st.sidebar.button("🔁 Generate Example")

# --- Session state
if "input_values" not in st.session_state:
    st.session_state.input_values = {f: 0.0 for f in FEATURE_NAMES}

if generate:
    with st.spinner("Generating sample..."):
        sample = fetch_dummy_sample(fraud_only=(sample_type == "Fraud-like"))
        if sample:
            st.session_state.input_values = sample

# --- Input form
st.markdown("### ✏️ Input Features")

input_data = {}
cols = st.columns(4) 
for idx, feature in enumerate(FEATURE_NAMES):
    with cols[idx % 4]:
        input_data[feature] = st.number_input(
            label=feature,
            value=float(st.session_state.input_values.get(feature, 0.0)),
            format="%.4f"
        )

# --- Prediction
if st.button("🚀 Predict"):
    with st.spinner("Predicting..."):
        result = predict_transaction(input_data)
        if result:
            pred_label = "⚠️ Fraud" if result["prediction"] == 1 else "✅ Not Fraud"
            probability = result["probability"] * 100

            st.markdown("---")
            st.subheader("🔍 Prediction Result")
            st.markdown(f"**Prediction:** {pred_label}")
            st.markdown(f"**Fraud Probability:** `{probability:.2f}%`")