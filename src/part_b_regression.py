# part_b_regression.py  (run from repo root: python part_b_regression.py)
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.shipment_prep import FEATURES   # the 17 pinned columns, in order

DATA = Path("data/modelling_table.csv")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
TARGET, SEED = "delivery_hours", 42


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, y_pred)}


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


df = pd.read_csv(DATA)
assert len(FEATURES) == 17
X, y = df[FEATURES], df[TARGET]

# 1. split 840 / 180 / 180
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=SEED)
assert (len(X_train), len(X_val), len(X_test)) == (840, 180, 180)

# 2. scaling without leakage: fit on train only, transform test
scaler = StandardScaler().fit(X_train)
train_scaled = scaler.transform(X_train)
test_scaled = scaler.transform(X_test)
pd.DataFrame({
    "metric": ["train_rows", "val_rows", "test_rows",
               "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"],
    "value": [len(X_train), len(X_val), len(X_test),
              float(np.abs(train_scaled.mean(axis=0)).max()),
              float(np.abs(test_scaled.mean(axis=0)).max())],
}).to_csv(OUT / "part_b_checks.csv", index=False)

# 3. baseline: mean of y_train for every validation row
metric_rows = [{"model": "mean_baseline", "split": "val",
                **regression_metrics(y_val, np.full(len(y_val), y_train.mean()))}]

# 4. tree-depth diagnostic
depth_rows = []
for d in [1, 2, 3, 4, 6, 8, 12, None]:
    t = DecisionTreeRegressor(max_depth=d, random_state=SEED).fit(X_train, y_train)
    depth_rows.append({"max_depth": "unlimited" if d is None else d,
