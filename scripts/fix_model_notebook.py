from pathlib import Path
import ast
import json
import traceback

import nbformat as nbf

repo_root = Path(__file__).resolve().parents[1]
notebook_path = repo_root / 'notebooks' / '04_Model_Improvements.ipynb'

cells = [
    nbf.v4.new_markdown_cell(
        '# 04. Model Improvements\n## DSN Bootcamp Challenge: Sales Forecasting\n\n**Milestone**: 5 - Model Improvements\n\n**Objective**: Improve the baseline LightGBM model with feature engineering, target encoding, and model ensemble tuning.\n\nThis notebook is designed to produce a Kaggle-compatible submission with the exact required schema: `id,total_sales`.\n'
    ),
    nbf.v4.new_markdown_cell('## 1. Setup & Imports'),
    nbf.v4.new_code_cell(
        """import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (14, 6)

import sys
repo_root = Path.cwd()
if not (repo_root / 'data').exists():
    repo_root = repo_root.parent
sys.path.insert(0, str(repo_root))
from src.utils import set_seed, print_seed_info, rmse, mae

set_seed(42)
print_seed_info(42)
"""
    ),
    nbf.v4.new_markdown_cell('## 2. Data Loading & Validation'),
    nbf.v4.new_code_cell(
        """train_df = pd.read_csv(repo_root / 'data' / 'train.csv')
test_df = pd.read_csv(repo_root / 'data' / 'test.csv')

print(f'Train shape: {train_df.shape}')
print(f'Test shape: {test_df.shape}')
print(f'Train columns: {list(train_df.columns)}')
print(f'Test columns: {list(test_df.columns)}')

assert 'total_sales' in train_df.columns, 'Target missing from train'
assert 'total_sales' not in test_df.columns, 'Target should not be in test'
assert set(train_df.columns) - {'total_sales'} == set(test_df.columns), 'Feature mismatch'
print('✓ Data validation passed')
"""
    ),
    nbf.v4.new_markdown_cell('## 3. Missing Value Handling'),
    nbf.v4.new_code_cell(
        """for col in ['product_weight_kg']:
    med = train_df[col].median()
    train_df[col] = train_df[col].fillna(med)
    test_df[col] = test_df[col].fillna(med)

for col in ['fat_content', 'product_category', 'store_size', 'store_location_tier', 'store_format']:
    mode = train_df[col].mode(dropna=True)
    if not mode.empty:
        train_df[col] = train_df[col].fillna(mode.iloc[0])
        test_df[col] = test_df[col].fillna(mode.iloc[0])

print('Missing values in train:\n', train_df.isna().sum()[train_df.isna().sum() > 0])
print('Missing values in test:\n', test_df.isna().sum()[test_df.isna().sum() > 0])
"""
    ),
    nbf.v4.new_markdown_cell('## 4. Aggregate and Interaction Features'),
    nbf.v4.new_code_cell(
        """train_df = train_df.sort_values('id').reset_index(drop=True)
test_df = test_df.sort_values('id').reset_index(drop=True)

train_df['time_index'] = np.arange(len(train_df))
test_df['time_index'] = np.arange(len(test_df)) + len(train_df)

store_stats = train_df.groupby('store_code')['total_sales'].agg(['mean', 'std', 'median'])
store_stats.columns = ['store_mean_sales', 'store_std_sales', 'store_median_sales']
train_df = train_df.merge(store_stats, on='store_code', how='left')
test_df = test_df.merge(store_stats, on='store_code', how='left')

product_stats = train_df.groupby('product_code')['total_sales'].agg(['mean', 'std', 'median'])
product_stats.columns = ['product_mean_sales', 'product_std_sales', 'product_median_sales']
train_df = train_df.merge(product_stats, on='product_code', how='left')
test_df = test_df.merge(product_stats, on='product_code', how='left')

category_stats = train_df.groupby('product_category')['total_sales'].agg(['mean', 'std', 'median'])
category_stats.columns = ['category_mean_sales', 'category_std_sales', 'category_median_sales']
train_df = train_df.merge(category_stats, on='product_category', how='left')
test_df = test_df.merge(category_stats, on='product_category', how='left')

train_df['price_to_category_mean'] = train_df['product_price'] / (train_df['category_mean_sales'] + 1e-6)
test_df['price_to_category_mean'] = test_df['product_price'] / (test_df['category_mean_sales'] + 1e-6)
train_df['price_to_store_mean'] = train_df['product_price'] / (train_df['store_mean_sales'] + 1e-6)
test_df['price_to_store_mean'] = test_df['product_price'] / (test_df['store_mean_sales'] + 1e-6)
train_df['weight_price_ratio'] = train_df['product_weight_kg'] / (train_df['product_price'] + 1e-6)
test_df['weight_price_ratio'] = test_df['product_weight_kg'] / (test_df['product_price'] + 1e-6)

size_map = {'Small': 1, 'Medium': 2, 'Large': 3}
tier_map = {'Tier_1': 1, 'Tier_2': 2, 'Tier_3': 3}
train_df['store_size_rank'] = train_df['store_size'].map(size_map)
test_df['store_size_rank'] = test_df['store_size'].map(size_map)
train_df['tier_rank'] = train_df['store_location_tier'].map(tier_map)
test_df['tier_rank'] = test_df['store_location_tier'].map(tier_map)

for col in ['store_mean_sales', 'store_std_sales', 'store_median_sales', 'product_mean_sales', 'product_std_sales', 'product_median_sales', 'category_mean_sales', 'category_std_sales', 'category_median_sales']:
    train_df[col] = train_df[col].fillna(train_df[col].median())
    test_df[col] = test_df[col].fillna(test_df[col].median())

print('Engineered features created for train and test')
"""
    ),
    nbf.v4.new_markdown_cell('## 5. Lag Features & Rolling Statistics'),
    nbf.v4.new_code_cell(
        """train_df = train_df.sort_values(['store_code', 'product_code', 'time_index']).reset_index(drop=True)
for lag in [1, 7, 30]:
    train_df[f'sales_lag_{lag}'] = train_df.groupby(['store_code', 'product_code'])['total_sales'].transform(lambda s: s.shift(lag))

for window in [7, 30]:
    train_df[f'rolling_mean_{window}'] = train_df.groupby(['store_code', 'product_code'])['total_sales'].transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
    train_df[f'rolling_std_{window}'] = train_df.groupby(['store_code', 'product_code'])['total_sales'].transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).std())

for col in ['sales_lag_1', 'sales_lag_7', 'sales_lag_30', 'rolling_mean_7', 'rolling_std_7', 'rolling_mean_30', 'rolling_std_30']:
    train_df[col] = train_df[col].fillna(train_df[col].median())
    test_df[col] = train_df[col].median()

print('Lag and rolling features created')
"""
    ),
    nbf.v4.new_markdown_cell('## 6. Target Encoding for Categorical Variables'),
    nbf.v4.new_code_cell(
        """cat_cols = ['fat_content', 'product_category', 'store_size', 'store_location_tier', 'store_format']
for col in cat_cols:
    overall = train_df['total_sales'].mean()
    means = train_df.groupby(col)['total_sales'].mean()
    counts = train_df.groupby(col)['total_sales'].count()
    encoded = (counts * means + 10 * overall) / (counts + 10)
    train_df[f'{col}_target_mean'] = train_df[col].map(encoded)
    test_df[f'{col}_target_mean'] = test_df[col].map(encoded).fillna(overall)

print('Target encoding completed')
"""
    ),
    nbf.v4.new_markdown_cell('## 7. Prepare Modeling Data'),
    nbf.v4.new_code_cell(
        """exclude_cols = ['id', 'product_code', 'store_code', 'total_sales', 'time_index']
feature_cols = [c for c in train_df.columns if c not in exclude_cols]

X_train = train_df[feature_cols].copy()
y_train = train_df['total_sales'].copy()
X_test = test_df[feature_cols].copy()

for col in X_train.columns:
    if X_train[col].isna().any():
        val = X_train[col].median()
        X_train[col] = X_train[col].fillna(val)
    if X_test[col].isna().any():
        val = X_test[col].median()
        X_test[col] = X_test[col].fillna(val)

for col in X_train.columns:
    if pd.api.types.is_object_dtype(X_train[col]) or pd.api.types.is_string_dtype(X_train[col]) or str(X_train[col].dtype) == 'str':
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

cat_cols = [c for c in feature_cols if str(X_train[c].dtype).startswith('category')]
print(f'Feature columns used: {len(feature_cols)}')
print(f'Categorical columns: {cat_cols}')
"""
    ),
    nbf.v4.new_markdown_cell('## 8. TimeSeriesSplit Validation'),
    nbf.v4.new_code_cell(
        """n_splits = 5
ts_cv = TimeSeriesSplit(n_splits=n_splits)
fold_results = []
for fold_idx, (train_idx, val_idx) in enumerate(ts_cv.split(X_train), start=1):
    fold_results.append({'Fold': fold_idx, 'Train_Size': len(train_idx), 'Val_Size': len(val_idx)})
print(pd.DataFrame(fold_results).to_string(index=False))
"""
    ),
    nbf.v4.new_markdown_cell('## 9. LightGBM Baseline with Improved Features'),
    nbf.v4.new_code_cell(
        """lgb_scores = []
oof_preds = np.zeros(len(y_train), dtype=float)
for fold_idx, (train_idx, val_idx) in enumerate(ts_cv.split(X_train), start=1):
    X_tr = X_train.iloc[train_idx].copy()
    y_tr = y_train.iloc[train_idx].copy()
    X_val = X_train.iloc[val_idx].copy()
    y_val = y_train.iloc[val_idx].copy()

    for col in cat_cols:
        X_tr[col] = X_tr[col].astype('category')
        X_val[col] = X_val[col].astype('category')

    model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        boosting_type='gbdt',
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        random_state=42,
        n_estimators=500,
        verbose=-1
    )
    model.fit(X_tr, y_tr, categorical_feature=cat_cols)
    pred = model.predict(X_val)
    oof_preds[val_idx] = pred
    fold_rmse = rmse(y_val, pred)
    lgb_scores.append(fold_rmse)
    print(f'Fold {fold_idx}: RMSE = {fold_rmse:.4f}')

print('Mean CV RMSE:', round(np.mean(lgb_scores), 4))
print('Std CV RMSE:', round(np.std(lgb_scores), 4))
print('OOF RMSE:', round(rmse(y_train, oof_preds), 4))
"""
    ),
    nbf.v4.new_markdown_cell('## 10. Optuna Tuning'),
    nbf.v4.new_code_cell(
        """def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 80),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': 5,
        'random_state': 42,
        'n_estimators': 400,
        'verbose': -1
    }

    fold_scores = []
    for train_idx, val_idx in ts_cv.split(X_train):
        X_tr = X_train.iloc[train_idx].copy()
        y_tr = y_train.iloc[train_idx].copy()
        X_val = X_train.iloc[val_idx].copy()
        y_val = y_train.iloc[val_idx].copy()

        for col in cat_cols:
            X_tr[col] = X_tr[col].astype('category')
            X_val[col] = X_val[col].astype('category')

        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, categorical_feature=cat_cols)
        pred = model.predict(X_val)
        fold_scores.append(rmse(y_val, pred))

    return float(np.mean(fold_scores))

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=3)
print('Best trial:', study.best_trial.number)
print('Best params:', study.best_trial.params)
print('Best CV RMSE:', study.best_value)
"""
    ),
    nbf.v4.new_markdown_cell('## 11. XGBoost Comparison'),
    nbf.v4.new_code_cell(
        """xgb_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=False)
xgb_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=False)

common_cols = xgb_train.columns.intersection(xgb_test.columns)
xgb_train = xgb_train[common_cols]
xgb_test = xgb_test[common_cols]

xgb_scores = []
for train_idx, val_idx in ts_cv.split(xgb_train):
    X_tr = xgb_train.iloc[train_idx].to_numpy()
    y_tr = y_train.iloc[train_idx].to_numpy()
    X_val = xgb_train.iloc[val_idx].to_numpy()
    y_val = y_train.iloc[val_idx].to_numpy()

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    pred = model.predict(X_val)
    xgb_scores.append(rmse(y_val, pred))

print('XGBoost CV RMSE per fold:', np.round(xgb_scores, 4))
print('Mean CV RMSE:', round(np.mean(xgb_scores), 4))
"""
    ),
    nbf.v4.new_markdown_cell('## 12. Weighted Ensemble'),
    nbf.v4.new_code_cell(
        """final_lgb = lgb.LGBMRegressor(
    objective='regression',
    metric='rmse',
    boosting_type='gbdt',
    num_leaves=31,
    learning_rate=0.05,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42,
    n_estimators=500,
    verbose=-1
)
final_lgb.fit(X_train, y_train, categorical_feature=cat_cols)

final_xgb = xgb.XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
final_xgb.fit(pd.get_dummies(X_train, columns=cat_cols, drop_first=False).to_numpy(), y_train.to_numpy())

lgb_pred = final_lgb.predict(X_test)
xgb_pred = final_xgb.predict(pd.get_dummies(X_test, columns=cat_cols, drop_first=False).to_numpy())
ensemble_pred = 0.6 * lgb_pred + 0.4 * xgb_pred

print('LightGBM pred min/max:', round(lgb_pred.min(), 2), round(lgb_pred.max(), 2))
print('XGBoost pred min/max:', round(xgb_pred.min(), 2), round(xgb_pred.max(), 2))
print('Ensemble pred min/max:', round(ensemble_pred.min(), 2), round(ensemble_pred.max(), 2))
"""
    ),
    nbf.v4.new_markdown_cell('## 13. Final Submission'),
    nbf.v4.new_code_cell(
        """submission_df = pd.DataFrame({
    'id': test_df['id'].values,
    'total_sales': ensemble_pred
})

os.makedirs(repo_root / 'submissions', exist_ok=True)
submission_path = repo_root / 'submissions' / 'submission_v3.csv'
submission_df.to_csv(submission_path, index=False)
print('Saved submission:', submission_path)
print(submission_df.head().to_string(index=False))

assert list(submission_df.columns) == ['id', 'total_sales']
"""
    ),
    nbf.v4.new_markdown_cell(
        '## 14. Summary\n\nThis notebook follows the project plan in PROGRESS.md by improving the baseline with engineered features, strict time-series validation, Optuna tuning, and a simple ensemble. It saves a Kaggle-compatible CSV with the required `id,total_sales` schema.\n'
    ),
]

nb = nbf.v4.new_notebook(cells=cells)
notebook_path.write_text(nbf.writes(nb), encoding='utf-8')

# Validate syntax and run code cells in order
nb_json = json.loads(notebook_path.read_text(encoding='utf-8'))
ns = {'__name__': '__main__'}
for idx, cell in enumerate(nb_json['cells']):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    ast.parse(src)
    exec(compile(src, f'<cell {idx}>', 'exec'), ns, ns)
    print(f'Executed cell {idx}')

print(f'Notebook rebuilt successfully: {notebook_path}')
