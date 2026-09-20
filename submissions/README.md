# Submissions Directory

This directory contains prediction files generated during the DSN Bootcamp challenge workflow.

## Required Kaggle Format

Every submission must contain exactly two columns, in this order:

```csv
id,total_sales
row_00009,3151.322925
row_00015,6152.628648
```

### Requirements

- Header row: `id,total_sales`
- `id`: The unique identifier copied from `data/test.csv`
- `total_sales`: Numeric predicted sales values
- No pandas index column
- No extra columns
- No missing IDs or predictions

The name `predicted_sales` is not accepted by Kaggle for this competition.

## Validate Before Uploading

```python
import pandas as pd

submission = pd.read_csv("submissions/submission_final.csv")

assert list(submission.columns) == ["id", "total_sales"]
assert len(submission) > 0
assert submission["id"].notna().all()
assert submission["total_sales"].notna().all()
```

If an older file still uses `predicted_sales`, convert it before uploading:

```python
submission = submission.rename(columns={"predicted_sales": "total_sales"})
submission.to_csv("submissions/submission_kaggle.csv", index=False)
```

## Current Files

- `submission_v3.csv` — Improved LightGBM model with engineered features and tuned hyperparameters
- `submission_final.csv` — Final competition-ready submission

Both files should use the required `id,total_sales` schema before they are uploaded.

**Last Updated**: 2026-09-20
