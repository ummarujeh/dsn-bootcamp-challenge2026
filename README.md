# DSN Bootcamp Challenge: Sales Forecasting

A predictive model for DSN Mart to forecast total product-store sales for the DSN AI Bootcamp qualification hackathon.

## 📋 Project Overview

**Objective**: Build a machine learning model that predicts total sales for a given product at a given store, based on product and outlet characteristics.

**Dataset**: Historical product-store sales records from DSN Mart (a retail chain across Nigeria)

**Evaluation Metric**: Root Mean Squared Error (RMSE) — lower is better

**Challenge Type**: Regression (supervised learning)

---

## ✅ Final Project Status

This project successfully built and validated a sales forecasting pipeline for DSN Mart using a time-aware machine learning workflow. The work includes exploratory analysis, feature engineering, a baseline LightGBM model, and a final improvement stage with engineered features, categorical target encoding, lag/rolling features, and Optuna-based hyperparameter tuning.

The validated notebook in [notebooks/04_Model_Improvements.ipynb](notebooks/04_Model_Improvements.ipynb) produced strong results under time-series cross-validation:

- Improved-feature LightGBM CV RMSE: 1116.49
- Optuna-tuned LightGBM best CV RMSE: 1025.61
- XGBoost comparison mean CV RMSE: 1115.81

The project is now considered complete and reproducible, with the best current prediction artifact saved in [submissions/submission_v3.csv](submissions/submission_v3.csv). The repo reflects the finalized workflow, validated model results, and milestone tracking in [PROGRESS.md](PROGRESS.md).

---

## 📁 Project Structure

```
dsn-bootcamp-challenge/
├── data/                          # Training and test data
│   ├── train.csv                  # Historical sales data with target
│   ├── test.csv                   # Test set (no target)
│   └── README.md                  # Data dictionary & schema
├── notebooks/                     # Jupyter notebooks for analysis & modeling
│   ├── 01_EDA.ipynb               # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Baseline_Model.ipynb
├── src/                           # Reusable Python modules
│   ├── features.py                # Feature engineering functions
│   ├── models.py                  # Model training & evaluation
│   └── utils.py                   # Utility functions
├── submissions/                   # Prediction files for competition
│   └── submission_v1.csv
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
├── PROGRESS.md                    # Milestone tracking
└── README.md                      # This file
```

---

## 🎯 Milestones & Tasks

Track your progress in [PROGRESS.md](PROGRESS.md).

1. **Setup** ✓ (this tracking file)
2. **Data Exploration & Cleaning**
3. **Feature Engineering** (date, lag, rolling, store/product aggregates)
4. **Baseline Model** (LightGBM) with time-aware CV
5. **Model Improvements** (target encoding, tuning, ensembles)
6. **Final Training & Submission**

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/oumar-code/dsn-bootcamp-challenge.git
cd dsn-bootcamp-challenge
```

### 2. Create a Python Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Training Data
Download the dataset and place files in the `data/` directory:
- `data/train.csv` — training data with target variable
- `data/test.csv` — test set for predictions

---

## 📊 Key Concepts & Best Practices

### Time-Based Cross-Validation
- **Why**: Prevent data leakage by ensuring training dates predate validation dates
- **Implementation**: Use `TimeSeriesSplit` or custom time-aware splits
- **Key Rule**: Never train on data from the future

### Evaluation Metric: RMSE
```
RMSE = sqrt(mean((y_pred - y_true)^2))
```
- Lower values are better
- Penalizes large errors more heavily than MAE

### Reproducibility Checklist
- [ ] Record random seeds (Python, NumPy, sklearn, etc.)
- [ ] Document package versions in `requirements.txt`
- [ ] Save notebook outputs (metrics, feature importance, plots)
- [ ] Log hyperparameters and model configurations

---

## 🔧 Tech Stack

- **Data Processing**: pandas, NumPy
- **EDA**: matplotlib, seaborn
- **Modeling**: scikit-learn, LightGBM, XGBoost
- **Metrics**: scikit-learn's evaluation functions
- **Notebooks**: Jupyter Lab

---

## 📈 Expected Workflow

1. **Understand** — Explore the dataset structure, distributions, relationships
2. **Analyse** — Identify trends, seasonality, and factors influencing sales
3. **Model** — Engineer features, train baselines, and optimize
4. **Predict** — Generate final predictions on test set
5. **Communicate** — Document findings and reasoning

---

## 📝 Competition Details

- **Track**: Machine Learning (ML)
- **Goal**: Predict total sales at product-store level
- **Leaderboard**: Rank influences DSN AI Bootcamp selection
- **Note**: Participation alone does NOT guarantee bootcamp selection

---

## 🚀 How to Submit

1. Train your final model
2. Generate predictions for `test.csv`
3. Save as `submissions/submission_v{version}.csv` with format: `[ID, predicted_sales]`
4. Upload to the competition platform

---

## 📚 Resources

- [Kaggle Competition](https://kaggle.com) — Platform for submissions
- [Time Series Cross-Validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Feature Engineering Guide](https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/)

---

## 📧 Contact & Support

For competition questions, refer to the official hackathon guidelines.

---

**Last Updated**: 2026-09-17  
**Status**: Completed ✅
