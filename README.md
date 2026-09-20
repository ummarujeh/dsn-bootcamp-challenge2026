# DSN Bootcamp Challenge: Sales Forecasting

A predictive model for DSN Mart to forecast total product-store sales for the DSN AI Bootcamp qualification hackathon.

## 📋 Project Overview

**Objective**: Build a machine learning model that predicts total sales for a given product at a given store, based on product and outlet characteristics.

**Dataset**: Historical product-store sales records from DSN Mart (a retail chain across Nigeria)

**Evaluation Metric**: Root Mean Squared Error (RMSE) — lower is better

**Challenge Type**: Regression (supervised learning)

## 🚀 How to Submit

1. Train your final model.
2. Generate predictions for `data/test.csv`.
3. Save the submission with exactly these columns, in this order:

```csv
id,total_sales
row_00009,3151.322925
row_00015,6152.628648
```

4. Ensure the CSV does not contain a pandas index column.
5. Upload the resulting CSV to Kaggle.

The prediction column must be named exactly `total_sales`. Do not use `predicted_sales`; Kaggle rejects it as an unexpected column.

To validate a submission before uploading:

```python
import pandas as pd

submission = pd.read_csv("submissions/submission_final.csv")
assert list(submission.columns) == ["id", "total_sales"]
assert submission["id"].notna().all()
assert submission["total_sales"].notna().all()
```

## 📁 Project Structure

```
dsn-bootcamp-challenge2026/
├── data/                          # Training and test data
├── notebooks/                     # Analysis and modeling notebooks
├── scripts/                       # Reproducibility and submission utilities
├── src/                           # Reusable Python modules
├── submissions/                   # Prediction files for competition
├── requirements.txt
├── PROGRESS.md
└── README.md
```

## ⚙️ Setup Instructions

```bash
git clone https://github.com/ummarujeh/dsn-bootcamp-challenge2026.git
cd dsn-bootcamp-challenge2026
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

Place `train.csv` and `test.csv` in the `data/` directory, run the validated modeling notebook, and check the generated submission header before uploading.

## 📊 Modeling Notes

- Use time-based cross-validation to avoid data leakage.
- The evaluation metric is RMSE.
- Train using `total_sales` as the target column.
- The test set must not contain the target column.

---

**Last Updated**: 2026-09-20
