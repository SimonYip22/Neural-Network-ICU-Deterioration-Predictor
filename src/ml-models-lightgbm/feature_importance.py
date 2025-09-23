"""
feature_importance.py

Title: 

Summary:
- Loads best hyperparameters per target, trains CV folds, aggregates feature importance
- Saves CSV and bar plots of top features per target
Outputs:
- 
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import json
import matplotlib.pyplot as plt

#-------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR.parent.parent / "data/processed-data/news2_features_patient.csv"
FEATURE_DIR = SCRIPT_DIR / "feature_importance_runs"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

TUNE_DIR = SCRIPT_DIR / "hyperparameter_tuning_runs"
TARGETS = ["max_risk", "median_risk", "pct_time_high"]
RANDOM_SEED = 42

#-------------------------------------------------------------
# Load data & features
#-------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
feature_cols = df.drop(columns=["subject_id"] + TARGETS).columns.tolist()

# Load best hyperparameters
with open(TUNE_DIR / "best_params.json") as f:
    best_params_all = json.load(f)

#-------------------------------------------------------------
# Loop through targets
#-------------------------------------------------------------
for target in TARGETS:
    print(f"Processing feature importance for {target}")
    X = df[feature_cols]
    y = df[target]

    if target == "max_risk":
        y = (y == 3).astype(int)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        model_class = lgb.LGBMClassifier
    elif target == "median_risk":
        y = (y == 2).astype(int)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        model_class = lgb.LGBMClassifier
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        model_class = lgb.LGBMRegressor

    fold_importances = []

    for train_idx, test_idx in kf.split(X, y if target != "pct_time_high" else None):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = model_class(**best_params_all[target], random_state=RANDOM_SEED,
                            class_weight="balanced" if target != "pct_time_high" else None)
        model.fit(X_train, y_train)
        fold_importances.append(model.feature_importances_)

    # Average feature importance across folds
    mean_importances = np.mean(fold_importances, axis=0)
    feat_df = pd.DataFrame({"feature": feature_cols, "importance": mean_importances})
    feat_df.sort_values(by="importance", ascending=False, inplace=True)

    # Save CSV
    feat_df.to_csv(FEATURE_DIR / f"{target}_feature_importance.csv", index=False)

    # Save bar plot
    plt.figure(figsize=(10,6))
    plt.barh(feat_df['feature'].head(10)[::-1], feat_df['importance'].head(10)[::-1])
    plt.xlabel("Importance")
    plt.title(f"Top 10 Features for {target}")
    plt.tight_layout()
    plt.savefig(FEATURE_DIR / f"{target}_feature_importance.png")
    plt.close()

print("Feature importance aggregation complete. Check feature_importance_runs/ folder.")