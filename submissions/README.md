# Submissions Directory

This directory contains the prediction files generated during the DSN Bootcamp challenge workflow. The project uses time-aware validation and keeps a versioned record of model iterations and final output files.

## Submission Format

Each CSV file follows the required format:

```
id,predicted_sales
row_00009,3151.322925
row_00015,6152.628648
...
```

### Format Requirements:
- **Header Row**: `id,predicted_sales`
- **ID Column**: Unique identifier from the test set
- **Predictions Column**: Float values representing forecasted total sales
- **No Index Column**: CSV should not include row indices

## Current Files

- `submission_v3.csv` — Improved LightGBM model with engineered features and tuned hyperparameters
- `submission_final.csv` — Final competition-ready submission based on the validated tuned model

## Model Status

The validated pipeline in the project notebook achieved the following time-series CV results:

- Improved-feature LightGBM CV RMSE: 1116.49
- Optuna-tuned LightGBM best CV RMSE: 1025.61
- XGBoost comparison mean CV RMSE: 1115.81

The tuned LightGBM configuration is the best-performing model in the project workflow and is the one used to generate the final submission file.

## Before Submission

Checklist before uploading to the competition platform:

- [x] Predictions are for all rows in test.csv
- [x] No missing values (NaN) in predictions
- [x] Predictions are reasonable and within expected sales ranges
- [x] CSV format is correct (`id,predicted_sales`)
- [x] File is not corrupted and can be read
- [x] Model was trained using time-aware validation (no leakage)

## Leaderboard Tracking

| Version | Date | CV RMSE | Public RMSE | Private RMSE | Notes |
|---------|------|---------|------------|-------------|-------|
| v1 | 2026-09-10 | ~1234.57 | — | — | Baseline LightGBM |
| v2 | 2026-09-17 | — | — | — | Feature engineering iteration |
| v3 | 2026-09-17 | 1025.61 | — | — | Tuned LightGBM |
| final | 2026-09-17 | — | — | — | Final competition submission |

---

**Last Updated**: 2026-09-17
