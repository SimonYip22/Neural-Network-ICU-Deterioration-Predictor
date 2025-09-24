"""
tune_models.py

Title: Hyperparameter Tuning for Classification and Regression Models and Evaluate performance using 5-fold cross-validation

Summary:
- Performs hyperparameter tuning and 5-fold cross-validation
	- `learning_rate` → controls step size; balances speed vs overfitting.
	- `max_depth` / `num_leaves` → limits tree complexity; prevents overfitting small dataset.
	- `n_estimators` → total number of trees.
	- `min_data_in_leaf` → ensures each leaf has enough samples, stabilising predictions.
- Inside each loop, For each combo of hyperparameters:
	1. Split the dataset into 5 folds (train/test).
	2. Train a model on 4 folds, predict on the remaining 1.
	3. Record the performance (AUROC or MSE).
	4. Average the 5 scores → mean_score.
	5. Compare mean_score to the best so far: if better → save this param set as the new best_params.
- Saves CV results, best parameters, and logs per fold
- Optimises baseline performance without overfitting, especially critical with only 100 patients.
- Ensures that the model is robust, reproducible, and interpretable.
Outputs:
- 3x tuning_logs/{target}_tuning_log.csv (3 files, one per target, containing every parameter sets values and the computed mean score)
- 3x {target}_cv_results.csv (3 files, one per target, containing all 5 fold-level scores for the winning parameter set. You can compute mean score for each target by averaging the folds scores)
- 1x best_params.json (one file with a dictionary of the best parameter set for every target)
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

# Preprocessing class combination (to match training script)
df["max_risk"] = df["max_risk"].replace({0: 2, 1: 2})
df["median_risk"] = df["median_risk"].replace({0: 1})

#-------------------------------------------------------------
# Hyperparameter tuning sweep ranges
#-------------------------------------------------------------
# Small, conservative grid tuned for a small dataset (100 patients). 
# These params control model capacity and overfitting (avoid exhaustive search on small data).
param_grid = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "n_estimators": [50, 100, 200],
    "min_data_in_leaf": [5, 10, 20]
}

# Dictionary to store the chosen best parameter combination per target.
best_params_all = {}

#-------------------------------------------------------------
# Loop through targets
#-------------------------------------------------------------
# Loop through targets and  choose CV and metric appropriate to task type.
for target in TARGETS:
    print(f"\nTuning target: {target}")
    X = df[feature_cols] # X and y extracted for the current target
    y = df[target]

    # Binary conversion for classification
    # Choose CV + metric
    if target == "max_risk":
        y = (y == 3).astype(int) # label = 1 if y==3 else 0.
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        metric_fn = roc_auc_score
        model_class = lgb.LGBMClassifier
    elif target == "median_risk":
        y = (y == 2).astype(int) # label = 1 if y==2 else 0.
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        metric_fn = roc_auc_score
        model_class = lgb.LGBMClassifier
    else:  # pct_time_high regression
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED) # No stratification
        metric_fn = mean_squared_error # MSE for regression instead of ROC-AUC
        model_class = lgb.LGBMRegressor

    #-------------------------------------------------------------
    # Manual grid sweep (small example)
    #-------------------------------------------------------------
    # Initialise best_score to -inf for classification (we want to maximise AUROC) and to inf for regression (we want to minimise MSE).
    # We want to track best mean CV score across grid combos.
    if target != "pct_time_high":   # classification task
        best_score = -np.inf        # baseline is -∞ so the first mean AUROC is bigger and updates.
    else:                           # regression task
        best_score = np.inf         # baseline is +∞ so the first mean MSE (e.g. 0.45) is smaller and updates.

    best_params = {}
    best_fold_scores = []

    # collect all tuning log rows for this target
    tuning_rows = []

    # Grid sweep (4-level nested loops)
    # This is a Cartesian product of all parameter choices.
    # Loop order: iterates over all combinations of lr, max_depth, n_estimators, min_data_in_leaf. This is a manual grid search.
    # Each loop nests inside the previous one, so all possible parameter combinations are tested.
    # # 3 learning rates × 3 depths × 3 n_estimators × 3 min_leaf = 81 total combos.
    # Each one is cross-validated (5 folds), so we train 81 × 5 = 405 models per target.
    for lr in param_grid["learning_rate"]:
        for md in param_grid["max_depth"]:
            for n_est in param_grid["n_estimators"]:
                for min_leaf in param_grid["min_data_in_leaf"]:
                    fold_scores = []

                    for train_idx, test_idx in kf.split(X, y if target != "pct_time_high" else None):
                        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                        # dictionary of parameters
                        model_kwargs = dict(
                            learning_rate=lr,
                            max_depth=md,
                            n_estimators=n_est,
                            min_data_in_leaf=min_leaf,
                            random_state=RANDOM_SEED
                        )
                        if model_class == lgb.LGBMClassifier:
                            model_kwargs["class_weight"] = "balanced"

                        model = model_class(**model_kwargs)

                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_test, y_test)],
                            callbacks=[early_stopping(10), log_evaluation(0)]
                        )
                        # For LGBMClassifier: by default returns class labels (0/1), not probabilities.
                        # For LGBMRegressor: returns continuous numeric predictions.
                        if target == "pct_time_high":
                            score = metric_fn(y_test, model.predict(X_test))
                        else:
                            # ROC AUC expects continuous probabiltiies between 0-1, if you feed it 0/1 you get either AUROC=0.5 (random) or 1.0 (perfect separation) with no nuance.
                            # predict_proba(X) → probabilities for each class (e.g. [P(class=0), P(class=1)]).
                            # [:, 1] selects the second column, i.e. the probability of the positive class (class=1).
                            preds_proba = model.predict_proba(X_test)[:, 1] 
                            score = metric_fn(y_test, preds_proba)

                        fold_scores.append(score)

                    mean_score = np.mean(fold_scores)

                    # append dictionary rows to tuning_rows Python list
                    tuning_rows.append({
                        "learning_rate": lr,
                        "max_depth": md,
                        "n_estimators": n_est,
                        "min_data_in_leaf": min_leaf,
                        "mean_score": mean_score
                    })

                    # Update best params
                    if (target != "pct_time_high" and mean_score > best_score) or (
                        target == "pct_time_high" and mean_score < best_score
                    ):
                        best_score = mean_score
                        best_params = {
                            "learning_rate": lr, 
                            "max_depth": md,
                            "n_estimators": n_est, 
                            "min_data_in_leaf": min_leaf
                        }
                        best_fold_scores = fold_scores.copy()
                        
    # save tuning log for this target
    # tuning_rows list turned into a DataFrame, column headers are automatically included
    log_file = TUNE_DIR / "tuning_logs" / f"{target}_tuning_log.csv"
    pd.DataFrame(tuning_rows).to_csv(log_file, index=False)

    # save CV results (fold scores for the winning parameter set)
    cv_csv_file = TUNE_DIR / f"{target}_cv_results.csv"
    pd.DataFrame({"fold": list(range(1, 6)), "score": best_fold_scores}).to_csv(cv_csv_file, index=False)

    # save best params into dictionary
    best_params_all[target] = best_params

# Save all best hyperparameters to JSON
# JSON is the right format to store per-target param dicts; easy to load in feature_importance.py and train_final_models.py.
with open(TUNE_DIR / "best_params.json", "w") as f:
    json.dump(best_params_all, f, indent=4)

print("Hyperparameter tuning complete. Check hyperparameter_tuning_runs/ folder.")