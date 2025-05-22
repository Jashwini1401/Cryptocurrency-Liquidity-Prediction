import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import pathlib
from sklearn.metrics import mean_squared_error, r2_score

# Set page config
st.set_page_config(page_title="Crypto Liquidity Predictor", layout="centered")

st.title("📊 Cryptocurrency Liquidity Prediction App")

# --- Load model and features ---
@st.cache_resource
def load_pipeline_and_features():
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
    model_path = BASE_DIR / 'model' / 'full_pipeline.pkl'
    features_path = BASE_DIR / 'model' / 'features.json'

    pipeline = joblib.load(model_path)
    with open(features_path, 'r') as f:
        features = json.load(f)
    return pipeline, features

pipeline, features = load_pipeline_and_features()

# --- Upload CSV file ---
uploaded_file = st.file_uploader("Upload a CSV file with the required features:", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Check required features
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        st.error(f"Missing required features in uploaded file: {missing_features}")
    else:
        # Clean the data
        df[features] = df[features].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=features)
        df = df[np.abs(df[features]) < 1e10].dropna()  # Clip out extreme values

        if df.empty:
            st.warning("No valid rows left after cleaning.")
        else:
            # Make predictions
            predictions = pipeline.predict(df[features])
            df['predicted_liquidity_ratio'] = predictions

            st.success("✅ Prediction complete.")
            st.dataframe(df[['predicted_liquidity_ratio'] + features].head(10))

            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download predictions as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
else:
    st.info("Please upload a CSV file with the correct input format to begin.")
