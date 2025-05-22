import pandas as pd
import os

def main():
    print("Starting feature engineering...", flush=True)

    data_path = os.path.join('data', 'cleaned', 'cleaned_data.csv')
    print(f"Loading data from: {data_path}", flush=True)
    df = pd.read_csv(data_path)
    print(f"Data loaded. Shape: {df.shape}", flush=True)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        print("Converted 'date' column to datetime.", flush=True)

    print("Creating features...", flush=True)
    df['price_ma7'] = df['price'].rolling(window=7, min_periods=1).mean()
    df['price_ma30'] = df['price'].rolling(window=30, min_periods=1).mean()
    df['volatility_7d'] = df['price'].rolling(window=7, min_periods=1).std()
    df['volume_change_pct'] = df['24h_volume'].pct_change()
    df['price_change_pct'] = df['price'].pct_change()
    df['liquidity_ratio'] = df['24h_volume'] / df['mkt_cap']
    df['liquidity_ratio_7d'] = df['liquidity_ratio'].rolling(window=7, min_periods=1).mean()

    df.fillna(0, inplace=True)

    output_path = os.path.join('data', 'cleaned', 'feature_engineered_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved feature engineered data at: {output_path}", flush=True)

if __name__ == "__main__":
    main()



