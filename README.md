
# Cryptocurrency Liquidity Prediction for Market Stability

## Author
**J. Ashwini**  
📧 jashwini5410@gmail.com  

---

## 📌 Project Overview

This project focuses on predicting cryptocurrency liquidity to help maintain market stability. It involves:
- Data cleaning
- Feature engineering
- Exploratory Data Analysis (EDA)
- Model training and evaluation
- Model deployment using Flask or Streamlit

---

## 📁 Folder Structure

```
Cryptocurrency Liquidity Prediction for Market Stability/
│
├── App/
│   └── app.py
│
├── Data/
│   ├── Cleaned/
│   │   ├── cleaned_data.csv
│   │   ├── deployment_data.csv
│   │   └── feature_engineered_data.csv
│   ├── Merged/
│   │   └── merged_raw_data.csv
│   ├── Raw/
│   │   ├── coin_gecko_2022-03-16.csv
│   │   └── coin_gecko_2022-03-17.csv
│   ├── results/
│   │   └── predictions.csv
│   └── model/
│       ├── best_random_forest_model.pkl
│       ├── evaluation_metrics.json
│       ├── features.json
│       ├── full_pipeline.pkl
│       ├── imputer.pkl
│       ├── random_forest_model.pkl
│       └── scaler.pkl
│
├── Notebook/
│   ├── Eda.ipynb
│   ├── feature_engineering.ipynb
│   ├── Hyperparameter_tuning.ipynb
│   ├── model_selection_and_tuning.ipynb
│   ├── model_training_and_evaluation.ipynb
│   └── prediction_and_deployment_test.ipynb
│
├── Reports/
│   ├── Final_Report.pdf
│   ├── generate_eda_report.py
│   ├── HLD_Document.pdf
│   ├── LLD_Document.pdf
│   └── Pipeline_Architecture.pdf
│
├── Src/
│   ├── Data_Processing/
│   │   ├── clean_data.py
│   │   ├── feature_engineering.py
│   │   └── merge_daily_csvs.py
│   └── Modeling/
│       ├── evaluate_model.py
│       ├── hyperparameter_tuning.py
│       ├── model_selection.py
│       ├── test_model.py
│       └── train_model.py
│
├── venv/
│
├── EDA_Report.pdf
├── README.md
└── requirements.txt
```

---

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/Jashwini1401/Cryptocurrency-Liquidity-Prediction.git
cd Cryptocurrency-Liquidity-Prediction
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate    # For Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the App

To run the deployment application:
```bash
python App/app.py
```

---

## 📊 Model Performance

Final evaluation metrics (from `evaluation_metrics.json`):
- **Test MSE**: 5.31  
- **Test R² Score**: 0.635  

---

## 📂 Reports

Detailed documents included:
- 📄 `Final_Report.pdf`
- 📄 `EDA_Report.pdf`
- 📄 `LLD_Document.pdf`
- 📄 `HLD_Document.pdf`
- 📄 `Pipeline_Architecture.pdf`

---

## ✅ Features

- Random Forest model with hyperparameter tuning
- Data processing pipelines
- Modular codebase for easy updates
- Exported model and pipeline objects for deployment

---

## 📬 Contact

For queries or collaborations, please reach out to:  
📧 **jashwini5410@gmail.com**
