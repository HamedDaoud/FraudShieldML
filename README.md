# Credit Card Fraud Detection System

A robust, end-to-end machine learning solution for detecting fraudulent credit card transactions, featuring a modular data pipeline, a high-performing Random Forest model, an interactive Streamlit web interface, and a FastAPI backend for real-time predictions.

![Demo](./assets/demo.gif)

## Overview

This project delivers a production-ready system for identifying credit card fraud with a focus on scalability, modularity, and ease of use. Built on the Kaggle Credit Card Fraud Detection Dataset, it combines advanced preprocessing, a tuned machine learning model, and a seamless integration of a web UI and REST API.

- **Data Pipeline**: Handles data cleaning, feature engineering, and SMOTE resampling for class imbalance.
- **Model**: RandomForestClassifier with tuned threshold for optimal precision-recall tradeoff.
- **Web UI**: Streamlit interface for user-friendly input, predictions, and dummy data generation.
- **API**: FastAPI backend for health checks, predictions, and sample generation.
- **Utilities**: Modular scripts for preprocessing, training, and evaluation, with path-agnostic file handling.

## Key Features

- **Effective Fraud Detection**: Achieves ~100% accuracy, 0.93 precision, 0.82 recall, and 0.86 PR AUC on the test set.
- **Custom Feature Engineering**: Cyclic encoding of time and log-transformed transaction amounts for improved model performance.
- **Interactive UI**: Streamlit app allows manual input, random/fraud-like sample generation, and displays prediction probabilities.
- **Scalable API**: FastAPI endpoints for predictions and sample generation, with Pydantic for schema validation.
- **Modular Design**: Organized codebase with separate modules for preprocessing, training, and utilities.

## Tech Stack

| Component          | Tools/Frameworks                     |
|--------------------|--------------------------------------|
| **Data Processing**| Pandas, Scikit-learn, Imbalanced-learn|
| **Model**          | RandomForestClassifier              |
| **Frontend**       | Streamlit                           |
| **Backend**        | FastAPI, Uvicorn, Pydantic          |
| **Utilities**      | Joblib, Pathlib, JSON               |
| Python Version     | Python 3.10                         |

## Dataset

- **Source**: Kaggle Credit Card Fraud Detection Dataset
- **Size**: 284,807 transactions, with 492 fraud cases (0.17%)
- **Features**: 30 features (Time, Amount, V1–V28 PCA-transformed features)

## Machine Learning Workflow

### Preprocessing
- **Amount**: Log-transformed using `np.log1p` for better distribution.
- **Time**: Converted to hour of day (0–23) and encoded as cyclic features (sin/cos).
- **V1–V28**: PCA-transformed features left untouched.
- **Scaling**: Applied `StandardScaler` for consistent feature ranges.
- **SMOTE**: Used to address class imbalance before training.

### Model Training
- **Algorithm**: RandomForestClassifier (100 trees).
- **Threshold Tuning**: Custom utility optimizes precision-recall tradeoff (best threshold: 0.50).
- **Artifacts**: Saved pipeline (`pipeline.pkl`), model (`rf_model.pkl`), threshold (`threshold.json`), and feature names (`feature_names.json`).

### Evaluation
- **Metrics**:
  - Precision: 0.93
  - Recall: 0.82
  - F1 Score: 0.87
  - PR AUC: 0.86
  - Accuracy: ~100%

## Project Structure

```bash
credit-fraud-detection/
├── assets/
│   └── demo.gif
├── api/                  # FastAPI server
│   └── main.py
├── app/                  # Streamlit frontend
│   ├── streamlit_app.py
│   └── utils_streamlit.py
├── data/                 # Raw and processed data -> Not included in repository
│   ├── creditcard.csv
│   └── test_predictions.csv
├── models/               # Trained model and pipeline artifacts
│   ├── feature_names.json
│   ├── pipeline.pkl
│   ├── rf_model.pkl
│   └── threshold.json
├── notebooks/            # Exploratory data analysis and experiments
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_class_imbalances.ipynb
│   └── 04_final_model.ipynb
├── src/                  # Core logic for preprocessing and training
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── requirements.txt      # Python dependencies
└── .streamlit/           # Streamlit theme configuration
```

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/HamedDaoud/credit-fraud-detection.git
   cd credit-fraud-detection
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the FastAPI Backend**:
   ```bash
   uvicorn api.main:app --reload
   ```
4. **Run the Streamlit Frontend**:
   ```bash
   streamlit run app/streamlit_app.py
   ```
5. **Access the App**:
   - Streamlit UI: `http://localhost:8501`
   - FastAPI: `http://localhost:8000` (endpoints: `/health`, `/predict`, `/generate_dummy`)

## API Endpoints

| Endpoint           | Method | Description                          |
|--------------------|--------|--------------------------------------|
| `/health`          | GET    | Checks API and model availability   |
| `/predict`         | POST   | Predicts fraud from 30-feature input |
| `/generate_dummy`  | GET    | Generates random/fraud-like samples  |

## Future Enhancements

- Add Docker support for containerized deployment.
- Implement unit tests with pytest.
- Explore model explainability with SHAP.
- Deploy Streamlit app to the cloud for public access.

## License

This project is licensed under the [MIT License](LICENSE).