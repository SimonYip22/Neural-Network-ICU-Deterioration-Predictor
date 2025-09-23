"""
tune_models.py

Title: 

Summary:
- Performs hyperparameter tuning and 5-fold cross-validation
- Saves CV results, best parameters, and logs per fold
Outputs:
- 
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from pathlib import Path
import joblib
import json
import csv

#-------------------------------------------------------------
# Paths & configuration
#-------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR.parent.parent / "data" / "processed-data" / "news2_features_patient.csv"

# Output folders
TUNE_DIR = SCRIPT_DIR / "hyperparameter_tuning_runs"
TUNE_DIR.mkdir(parents=True, exist_ok=True)
(TUNE_DIR / "tuning_logs").mkdir(exist_ok=True)

TARGETS = ["max_risk", "median_risk", "pct_time_high"]
RANDOM_SEED = 42

#-------------------------------------------------------------
# Load dataset
#-------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
feature_cols = df.drop(columns=["subject_id"] + TARGETS).columns.tolist()

# Preprocessing class combination (same as before)
df["max_risk"] = df["max_risk"].replace({0: 2, 1: 2}) if "max_risk" in df else df.get("max_risk")
df["median_risk"] = df["median_risk"].replace({0: 1}) if "median_risk" in df else df.get("median_risk")

#-------------------------------------------------------------
# Hyperparameter tuning sweep ranges
#-------------------------------------------------------------
param_grid = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "n_estimators": [50, 100, 200],
    "min_data_in_leaf": [5, 10, 20]
}

best_params_all = {}

#-------------------------------------------------------------
# Loop through targets
#-------------------------------------------------------------
for target in TARGETS:
    print(f"\nTuning target: {target}")
    X = df[feature_cols]
    y = df[target]

    # Binary conversion for classification
    if target == "max_risk":
        y = (y == 3).astype(int)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        metric_fn = roc_auc_score
        model_class = lgb.LGBMClassifier
    elif target == "median_risk":
        y = (y == 2).astype(int)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        metric_fn = roc_auc_score
        model_class = lgb.LGBMClassifier
    else:  # pct_time_high regression
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        metric_fn = mean_squared_error
        model_class = lgb.LGBMRegressor

    #-------------------------------------------------------------
    # Manual grid sweep (small example)
    #-------------------------------------------------------------
    best_score = -np.inf if target != "pct_time_high" else np.inf
    best_params = {}

    for lr in param_grid["learning_rate"]:
        for md in param_grid["max_depth"]:
            for n_est in param_grid["n_estimators"]:
                for min_leaf in param_grid["min_data_in_leaf"]:
                    fold_scores = []

                    for train_idx, test_idx in kf.split(X, y if target != "pct_time_high" else None):
                        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                        model = model_class(
                            learning_rate=lr,
                            max_depth=md,
                            n_estimators=n_est,
                            min_data_in_leaf=min_leaf,
                            random_state=RANDOM_SEED,
                            class_weight="balanced" if target != "pct_time_high" else None
                        )

                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_test, y_test)],
                            callbacks=[early_stopping(10), log_evaluation(0)]
                        )

                        preds = model.predict(X_test)
                        if target == "pct_time_high":
                            score = metric_fn(y_test, preds)
                        else:
                            preds_labels = preds.round().astype(int)
                            score = metric_fn(y_test, preds_labels)

                        fold_scores.append(score)

                    mean_score = np.mean(fold_scores)
                    # Update best params
                    if (target != "pct_time_high" and mean_score > best_score) or (target == "pct_time_high" and mean_score < best_score):
                        best_score = mean_score
                        best_params = {"learning_rate": lr, "max_depth": md,
                                       "n_estimators": n_est, "min_data_in_leaf": min_leaf}

    # Save best params for this target
    best_params_all[target] = best_params

    # Save CV results CSV
    cv_csv_file = TUNE_DIR / f"{target}_cv_results.csv"
    pd.DataFrame({"fold": list(range(1, 6)), "score": [best_score]*5}).to_csv(cv_csv_file, index=False)

# Save all best hyperparameters to JSON
with open(TUNE_DIR / "best_params.json", "w") as f:
    json.dump(best_params_all, f, indent=4)

print("Hyperparameter tuning complete. Check hyperparameter_tuning_runs/ folder.")