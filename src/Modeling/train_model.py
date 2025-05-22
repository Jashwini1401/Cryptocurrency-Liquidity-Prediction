import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Paths
input_path = os.path.join("data", "cleaned", "feature_engineered_data.csv")
model_dir = os.path.join("model")
model_output_path = os.path.join(model_dir, "random_forest_model.pkl")

# Make sure model directory exists
os.makedirs(model_dir, exist_ok=True)

# Load data
df = pd.read_csv(input_path)
print("📥 Loaded data shape:", df.shape)

# Define features and target
target = 'liquidity_ratio_7d'
features = [
    'price', 'price_ma7', 'price_ma30', 'volatility_7d',
    '24h_volume', 'volume_change_pct', 'price_change_pct', 'mkt_cap'
]

X = df[features]
y = df[target]

# ⚠️ Replace infinite values and handle NaNs
if np.isinf(X).values.any():
    print("⚠️ Found infinite values in X, replacing with NaN...")
    X = X.replace([np.inf, -np.inf], np.nan)

print("🧹 Dropping rows with NaNs...")
initial_shape = X.shape
X = X.dropna()
y = y.loc[X.index]  # Align y with cleaned X
print(f"✅ Cleaned shape: {X.shape} (before: {initial_shape})")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("🚀 Training RandomForestRegressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 Model Evaluation:\n   MSE: {mse:.4f}\n   R²: {r2:.4f}")

# Save model
joblib.dump((model, features), model_output_path)
print(f"✅ Model and features saved to: {model_output_path}")


