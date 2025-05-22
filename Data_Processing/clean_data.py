import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# Paths
input_path = os.path.join('data', 'merged', 'merged_raw_data.csv')
output_path = os.path.join('data', 'cleaned', 'cleaned_data.csv')

# Load data
df = pd.read_csv(input_path)

# --- Data Cleaning Properly ---

# Identify numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Safely fill missing values: no inplace, no chained assignment
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Normalize only numeric columns
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Save the cleaned data
df.to_csv(output_path, index=False)

print(f"Cleaned data saved to: {output_path}")


