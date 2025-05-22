import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os

sns.set(style="whitegrid")

# 📍 Print current working directory
print("Current working directory:", os.getcwd())

# 📁 Absolute path to cleaned data 
data_path = r"C:\Users\jashw\Cryptocurrency Liquidity Prediction for Market Stability\data\cleaned\cleaned_data.csv"

# 🧪 Optional: Show files in the cleaned data folder for debug
cleaned_dir = r"C:\Users\jashw\Cryptocurrency Liquidity Prediction for Market Stability\data\cleaned"
print("📂 Files in 'data/cleaned':", os.listdir(cleaned_dir))

# ❗ Check if file exists
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Data file not found at: {data_path}")

# ✅ Load data
df = pd.read_csv(data_path)
print("✅ Data loaded successfully.")

# 🎯 Select numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

# 📄 Start generating PDF
with PdfPages('EDA_report.pdf') as pdf:

    # 📊 Page 1: Summary Statistics
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    summary_text = df.describe().to_string()
    ax.text(0, 1, "Summary Statistics\n\n" + summary_text, fontsize=10, verticalalignment='top', family='monospace')
    pdf.savefig(fig)
    plt.close()

    # 📈 Pages 2+: Histograms for numeric features
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, bins=30, ax=ax)
        ax.set_title(f'Distribution of {col}')
        pdf.savefig(fig)
        plt.close()

    # 🔗 Correlation Matrix
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title("Correlation Matrix of Numerical Features")
    pdf.savefig(fig)
    plt.close()

print("✅ EDA_report.pdf generated successfully!")

