import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import os

def evaluate(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f'MSE: {mse:.4f}')
    print(f'R²: {r2:.4f}')

def main():
    # Paths
    input_path = os.path.join("data", "cleaned", "feature_engineered_data.csv")
    model_path = os.path.join("model", "random_forest_model.pkl")

    # Load data
    df = pd.read_csv(input_path)
    print(f"Loaded data shape: {df.shape}")

    # Load model and features
    model, features = joblib.load(model_path)
    print(f"Loaded model with features: {features}")

    target = 'liquidity_ratio_7d'

    # Clean data
    df = df[features + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Shape after cleaning: {df.shape}")

    X = df[features]
    y = df[target]

    evaluate(model, X, y)

if __name__ == "__main__":
    main()

