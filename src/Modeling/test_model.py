import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score

# 📍 Define paths
data_path = os.path.join("data", "cleaned", "feature_engineered_data.csv")
pipeline_path = os.path.join("model", "full_pipeline.pkl") 

# 🎯 Define features and target 
features = ['price', '1h', '24h', '7d', '24h_volume', 'mkt_cap',
            'price_ma7', 'price_ma30', 'volatility_7d',
            'volume_change_pct', 'price_change_pct']
target = 'liquidity_ratio'  # or 'liquidity_ratio_7d' if that's what your model used

# 📄 Load feature-engineered data
df = pd.read_csv(data_path)
print(f"✅ Loaded data with shape: {df.shape}")
print(f"📊 Columns in data: {list(df.columns)}")

# 🧼 Clean data: drop rows with missing or infinite target or features
df[target] = df[target].replace([np.inf, -np.inf], np.nan)
df[features] = df[features].replace([np.inf, -np.inf], np.nan)

df.dropna(subset=features + [target], inplace=True)

# 🔍 Extract X and y
X = df[features]
y_true = df[target]
print(f"✅ Cleaned dataset shape: {X.shape}")


# 📦 Load the full pipeline
if os.path.exists(pipeline_path):
    pipeline = joblib.load(pipeline_path)
    print("✅ Full pipeline loaded successfully.")
else:
    raise FileNotFoundError(f"❌ Pipeline not found at: {pipeline_path}")

# 🔮 Predict using the pipeline
y_pred = pipeline.predict(X)

# 📊 Evaluation
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\n📈 Model Evaluation Results:")
print(f"🔹 Mean Squared Error (MSE): {mse:.4f}")
print(f"🔹 R² Score: {r2:.4f}")

# 💾 Save predictions (optional)
results = df.copy()
results['predicted_liquidity_ratio'] = y_pred
results.to_csv(os.path.join("data", "results", "predictions.csv"), index=False)
print("✅ Predictions saved to data/results/predictions.csv")
