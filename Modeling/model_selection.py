import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# Load feature engineered data
data_path = os.path.join("data", "cleaned", "feature_engineered_data.csv")
df = pd.read_csv(data_path)

print(f"Loaded data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Define features and target
features = [
    'price', 'price_ma7', 'price_ma30', 'volatility_7d',
    '24h_volume', 'volume_change_pct', 'price_change_pct', 'mkt_cap'
]
target = 'liquidity_ratio_7d'

# Drop rows with missing values in features or target
df_model = df[features + [target]].dropna()
print(f"Shape after dropping missing values: {df_model.shape}")

# Split features and target
X = df_model[features].copy()
y = df_model[target].copy()

# Replace infinite values with NaN and drop rows containing NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)
y.replace([np.inf, -np.inf], np.nan, inplace=True)

mask = X.notnull().all(axis=1) & y.notnull()
X = X.loc[mask]
y = y.loc[mask]

print(f"Shape after cleaning infinite values: {X.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"  MSE: {mse:.4f}")
print(f"  R²: {r2:.4f}")



