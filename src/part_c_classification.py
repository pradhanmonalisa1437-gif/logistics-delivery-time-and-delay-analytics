"""Part C - Delay classifier.

Reads data/modelling_table.csv, writes the CSVs in outputs/.
Run from the repo root:  python part_c_classifier.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Pinned in Part A / src: the 17 feature columns (in order) and the 5 noise columns.
from src.shipment_prep import FEATURES, NOISE_COLUMNS

DATA = Path("data/modelling_table.csv")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
TARGET = "is_delayed"   # <-- change if your modelling table names the 0/1 label differently
SEED = 42
K = 7


# ------------------------------------------------------------- helpers
def cells_to_metrics(tn, fp, fn, tp):
    """All metrics come from the four confusion-matrix cells."""
    n = tn + fp + fn + tp
    acc = (tn + tp) / n
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return cells_to_metrics(tn, fp, fn, tp)


# ---------------------------------------------------------------- load
df = pd.read_csv(DATA)
assert len(FEATURES) == 17 and len(NOISE_COLUMNS) == 5
X, y = df[FEATURES], df[TARGET].astype(int)

# ------------------------------------------- 1. stratified 840/180/180
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp)
assert (len(X_train), len(X_val), len(X_test)) == (840, 180, 180)
assert (int(y_train.sum()), int(y_val.sum()), int(y_test.sum())) == (195, 42, 41), (
    int(y_train.sum()), int(y_val.sum()), int(y_test.sum()))

# ------------------------------- 2. blind baseline + three classifiers
rows = []

blind_pred = np.zeros(len(y_val), dtype=int)
rows.append({"model": "blind_all_on_time", **score(y_val, blind_pred)})  # precision = 0

logit = Pipeline([("scaler", StandardScaler()),
                  ("model", LogisticRegression(max_iter=1000))]).fit(X_train, y_train)
rows.append({"model": "logistic_regression", **score(y_val, logit.predict(X_val))})

best_d, best_f1 = None, -1.0
for d in [2, 3, 4, 5, 6, 8, 10]:          # ascending, strict > keeps smaller d on ties
    t = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=SEED)
    t.fit(X_train, y_train)
    f1 = score(y_val, t.predict(X_val))["f1"]
    if f1 > best_f1:
        best_d, best_f1 = d, f1
tree = DecisionTreeClassifier(criterion="gini", max_depth=best_d, random_state=SEED)
tree.fit(X_train, y_train)
rows.append({"model": f"decision_tree_depth_{best_d}", **score(y_val, tree.predict(X_val))})

rf_default = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(X_train, y_train)
rows.append({"model": "random_forest_default", **score(y_val, rf_default.predict(X_val))})

cols = ["model", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"]
pd.DataFrame(rows, columns=cols).to_csv(OUT / "part_c_models.csv", index=False)

# ------------------------------------------ 3. which columns carry signal
rf_imp = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(X_train, y_train)
sel_all = SelectKBest(f_classif, k="all").fit(X_train, y_train)
fs = pd.DataFrame({
    "feature": FEATURES,
    "f_score": sel_all.scores_,
    "rf_importance": rf_imp.feature_importances_,
})
fs["f_rank"] = fs["f_score"].rank(ascending=False, method="first").astype(int)
fs["is_noise"] = fs["feature"].isin(NOISE_COLUMNS)
fs[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]].to_csv(
    OUT / "part_c_feature_scores.csv", index=False)

selk_rows = []
for k in [5, 7, 10, 12, 17]:
    pipe = Pipeline([("selector", SelectKBest(f_classif, k=k)),
                     ("model", RandomForestClassifier(n_estimators=100, random_state=SEED))])
    pipe.fit(X_train, y_train)
    selk_rows.append({"k": k, "val_f1": score(y_val, pipe.predict(X_val))["f1"]})
pd.DataFrame(selk_rows).to_csv(OUT / "part_c_selectk.csv", index=False)

# --------------------------------------------------------- 4. tuning
selector = SelectKBest(f_classif, k=K).fit(X_train, y_train)     # training rows only
Xtr7, Xva7 = selector.transform(X_train), selector.transform(X_val)

param_grid = {"n_estimators": [50, 100, 150],
              "max_depth": [4, 6, 8],
              "min_samples_leaf": [1, 5, 10]}


def search_row(name, search, n_cand, folds):
    search.fit(Xtr7, y_train)
    return {
        "search": name,
        "candidates": n_cand,
        "cv_folds": folds,
        "total_fits": n_cand * folds,
        "best_params": dict(search.best_params_),
        "best_cv_accuracy": float(search.best_score_),
        "val_f1": score(y_val, search.best_estimator_.predict(Xva7))["f1"],
    }


grid = GridSearchCV(RandomForestClassifier(random_state=SEED), param_grid, cv=5)
rand = RandomizedSearchCV(RandomForestClassifier(random_state=SEED), param_grid,
                          n_iter=9, cv=5, random_state=SEED)
tuning = [search_row("grid_search", grid, 27, 5),
          search_row("randomized_search", rand, 9, 5)]

default_params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
rf7 = RandomForestClassifier(random_state=SEED, **default_params).fit(Xtr7, y_train)
tuning.append({
    "search": "default_forest_same_7_features",
    "candidates": 1, "cv_folds": 0, "total_fits": 1,
    "best_params": default_params,
    "best_cv_accuracy": np.nan,                     # left blank
    "val_f1": score(y_val, rf7.predict(Xva7))["f1"],
})
tuning_df = pd.DataFrame(tuning)
tuning_df.to_csv(OUT / "part_c_tuning.csv", index=False)

# ------------------------------------- 5. choose the final configuration
priority = {"default_forest_same_7_features": 0, "grid_search": 1, "randomized_search": 2}
tuning_df["_f1r"] = tuning_df["val_f1"].round(12)
tuning_df["_p"] = tuning_df["search"].map(priority)
winner = tuning_df.sort_values(["_f1r", "_p"], ascending=[False, True]).iloc[0]
params = dict(winner["best_params"])

final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=K)),
    ("model", RandomForestClassifier(random_state=SEED, **params)),
]).fit(X_train, y_train)

# ------------------------------------------------ 6. alert threshold
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
p_val = final_model.predict_proba(X_val)[:, 1]
th_rows = []
for t in THRESHOLDS:
    th_rows.append({"threshold": t, **score(y_val, (p_val >= t).astype(int))})
th_df = pd.DataFrame(th_rows, columns=["threshold", "TN", "FP", "FN", "TP", "accuracy",
                                       "precision", "recall", "f1"])
th_df.to_csv(OUT / "part_c_thresholds.csv", index=False)

# recall >= 0.80, compared in integers to avoid float boundary issues
ok = th_df[(th_df["TP"] * 5) >= (4 * (th_df["TP"] + th_df["FN"]))]
threshold = float(ok["threshold"].max()) if len(ok) else 0.1

# ----------------------------------------- 7. one test evaluation
p_test = final_model.predict_proba(X_test)[:, 1]
test_metrics = score(y_test, (p_test >= threshold).astype(int))
pd.DataFrame([{"threshold": threshold, **test_metrics}],
             columns=["threshold", "TN", "FP", "FN", "TP", "accuracy",
                      "precision", "recall", "f1"]).to_csv(
    OUT / "part_c_final_test.csv", index=False)

pd.DataFrame([{
    "selector_k": K,
    "n_estimators": params["n_estimators"],
    "max_depth": "unlimited" if params["max_depth"] is None else params["max_depth"],
    "min_samples_leaf": params["min_samples_leaf"],
    "threshold": threshold,
    "chosen_from": winner["search"],
}]).to_csv(OUT / "part_c_final_config.csv", index=False)

# ------------------------------------------------ printout for README
print(pd.read_csv(OUT / "part_c_models.csv").round(3).to_string(index=False))
print("\nFeature scores:\n", fs.sort_values("f_rank").round(3).to_string(index=False))
print("\nSelectK:\n", pd.DataFrame(selk_rows).round(3).to_string(index=False))
print("\nTuning:\n", tuning_df.drop(columns=["_f1r", "_p"]).to_string(index=False))
print("\nThresholds:\n", th_df.round(3).to_string(index=False))
print("\nChosen threshold:", threshold, "| from:", winner["search"], "|", params)
print("Test:", {k: round(v, 3) if isinstance(v, float) else v for k, v in test_metrics.items()})
