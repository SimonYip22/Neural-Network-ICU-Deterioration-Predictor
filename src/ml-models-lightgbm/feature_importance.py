"""
feature_importance.py

Title: Feature Importance Analysis and Visualisation Plots

Summary:
- Loads best hyperparameters per target from the tuning stage (best_params.json), trains CV folds, aggregates feature importance by finding the average per target
- Calculates which features (clinical variables, NEWS2 features, etc.) were most important in predicting the outcomes.
- Saves CSV and bar plots of top 10 features per target
- Tells us which clinical features the model relied on the most across targets to drive predictions, and differences across targets show which predictors matter most for different risk outcomes.
Outputs:
- 3x `{target}_feature_importance.csv` (CSV file per target with importance scores for all features e.g. resp_rate ranked by importance)
- 3x `{target}_feature_importance.png` (bar plots per target of the Top 10 features for quick visualisation)
"""
#-------------------------------------------------------------
# Imports
#-------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import lightgbm as lgb
from pathlib import Path
import json
import matplotlib.pyplot as plt

#-------------------------------------------------------------
# Directories
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

# Load best hyperparameters for each target from tuning
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

    # Train models fold by fold
    for train_idx, test_idx in kf.split(X, y if target != "pct_time_high" else None):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Trains model with the best params for this target
        model = model_class(**best_params_all[target], random_state=RANDOM_SEED,
                            class_weight="balanced" if target != "pct_time_high" else None)
        model.fit(X_train, y_train)
        # Append to list for each fold an array of importance values for each feature
        fold_importances.append(model.feature_importances_)

    # Average feature importance across folds (stability, avoids one fold bias)
    # Makes a DataFrame of features ranked by importance score, then sorts by descending
    # axis=0 means: average column-wise (feature 1’s score is averaged across folds, same for feature 2, etc.)
    mean_importances = np.mean(fold_importances, axis=0)
    feat_df = pd.DataFrame({"feature": feature_cols, "importance": mean_importances})
    feat_df.sort_values(by="importance", ascending=False, inplace=True)

    # Save CSV with all features + their average importance.
    feat_df.to_csv(FEATURE_DIR / f"{target}_feature_importance.csv", index=False)

    # Save bar plot
    # head(10) → take the first 10 rows (top features).
    # [::-1] → reverse the order so the highest feature appears at the top of the horizontal bar chart.
    plt.figure(figsize=(10,6))
    plt.barh(feat_df['feature'].head(10)[::-1], feat_df['importance'].head(10)[::-1])
    plt.xlabel("Relative importance (LightGBM split counts)") # x-axis label
    plt.title(f"Top 10 Features for {target}") # title with target name 
    plt.tight_layout() # avoids cutoff text
    plt.savefig(FEATURE_DIR / f"{target}_feature_importance.png")# saves PNG file
    plt.close() # closes the figure (no overlap with next loop)

print("Feature importance aggregation complete. Check feature_importance_runs/ folder.")