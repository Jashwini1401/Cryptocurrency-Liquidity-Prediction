import pandas as pd
import os


data_dir = "Data"


csv_files = sorted([f for f in os.listdir(data_dir) if f.startswith("coin_gecko_") and f.endswith(".csv")])

if not csv_files:
    print("❌ No coin_gecko_*.csv files found in the 'Data' folder.")
else:
    print(f"✅ Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print("   -", f)


merged_df = pd.DataFrame()

for file in csv_files:
    file_path = os.path.join(data_dir, file)
    print(f"📄 Reading {file_path}...")
    df = pd.read_csv(file_path)
    df['SourceFile'] = file
    merged_df = pd.concat([merged_df, df], ignore_index=True)


if 'Date' in merged_df.columns:
    print("📅 Converting 'Date' column to datetime...")
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    merged_df = merged_df.sort_values(by='Date').reset_index(drop=True)


output_path = os.path.join(data_dir, "merged_raw_data.csv")
merged_df.to_csv(output_path, index=False)
print("✅ Merged dataset saved at:", output_path)
