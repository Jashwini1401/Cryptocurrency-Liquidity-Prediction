import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load feature engineered data
data_path = "data/cleaned/feature_engineered_data.csv"
df = pd.read_csv(data_path)

print(f"Loaded data shape: {df.shape}")
print(f"Columns in dataset: {df.columns.tolist()}")

# Define features and target 
features = ['price', 'price_ma7', 'price_ma30', 'volatility_7d',
            '24h_volume', 'volume_change_pct', 'price_change_pct', 'mkt_cap']
target = 'liquidity_ratio_7d'

# Replace inf with NaN for all relevant columns (features + target)
df[features] = df[features].replace([np.inf, -np.inf], np.nan)
df[target] = df[target].replace([np.inf, -np.inf], np.nan)

# Drop rows with NaNs in features or target
df.dropna(subset=features + [target], inplace=True)

print(f"Shape after cleaning: {df.shape}")

# Prepare feature matrix and target vector
X = df[features].copy()
y = df[target].copy()

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize model
rf = RandomForestRegressor(random_state=42)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2', None]
}

# Setup GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1  # to see progress
)

# Run hyperparameter tuning
grid_search.fit(X_train, y_train)

print(f"Best hyperparameters: {grid_search.best_params_}")

# Evaluate best model on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MSE: {mse:.4f}")
print(f"Test R²: {r2:.4f}")




