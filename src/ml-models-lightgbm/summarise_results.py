"""
summarise_results.py

Title: Portfolio-Ready Summary

Summary:
- Combines CV results, best hyperparameters, and feature importance
- Produces a training_summary.txt for portfolio or reporting
Outputs:
- 
"""

import pandas as pd
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = SCRIPT_DIR / "deployment_models"
TUNE_DIR = SCRIPT_DIR / "hyperparameter_tuning_runs"
FEATURE_DIR = SCRIPT_DIR / "feature_importance_runs"

TARGETS = ["max_risk", "median_risk", "pct_time_high"]

# Load best params
with open(TUNE_DIR / "best_params.json") as f:
    best_params_all = json.load(f)

summary_file = DEPLOY_DIR / "training_summary.txt"
with open(summary_file, "w") as f:
    for target in TARGETS:
        f.write(f"=== Target: {target} ===\n")
        # CV results
        cv_df = pd.read_csv(TUNE_DIR / f"{target}_cv_results.csv")
        f.write(f"Mean CV score: {cv_df['score'].mean():.4f} ± {cv_df['score'].std():.4f}\n")
        # Best hyperparameters
        f.write("Best hyperparameters:\n")
        for k, v in best_params_all[target].items():
            f.write(f"  {k}: {v}\n")
        # Top features
        feat_df = pd.read_csv(FEATURE_DIR / f"{target}_feature_importance.csv")
        f.write("Top 10 features:\n")
        for i, row in feat_df.head(10).iterrows():
            f.write(f"  {row['feature']}: {row['importance']:.4f}\n")
        f.write("\n")

print("Summary file created in deployment_models/training_summary.txt")