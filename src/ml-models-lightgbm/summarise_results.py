"""
summarise_results.py

Title: Portfolio-Ready Summary

Summary:
- Combines CV results (from tune_models.py), best hyperparameters (from best_params.json), and feature importance (from feature_importance.py).
- Produces a clean text summary for each target model into a single file (training_summary.txt) for portfolio or reporting
- Portfolio-ready, can show to others without needing them to run code or inspect raw CSVs.
Outputs:
- training_summary.txt
"""
#-------------------------------------------------------------
# Imports
#-------------------------------------------------------------
import pandas as pd
import json
from pathlib import Path

#-------------------------------------------------------------
# Directories
#-------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = SCRIPT_DIR / "deployment_models"
TUNE_DIR = SCRIPT_DIR / "hyperparameter_tuning_runs"
FEATURE_DIR = SCRIPT_DIR / "feature_importance_runs"

TARGETS = ["max_risk", "median_risk", "pct_time_high"]

#-------------------------------------------------------------
# Load best params
#-------------------------------------------------------------
# Loads the dictionary of “winning” hyperparameters (from tune_models.py).
with open(TUNE_DIR / "best_params.json") as f:
    best_params_all = json.load(f)

#-------------------------------------------------------------
# Create summary file
#-------------------------------------------------------------
# Creates a new text file inside deployment_models/.
summary_file = DEPLOY_DIR / "training_summary.txt"
with open(summary_file, "w") as f:
    # Loop through targets
    for target in TARGETS:
        # Starts a section for each target
        f.write(f"=== Target: {target} ===\n")
        # Cross-validation results
        # Loads the 5-fold CV results for this target, calculates mean ± standard deviation of the fold scores.
        cv_df = pd.read_csv(TUNE_DIR / f"{target}_cv_results.csv")
        f.write(f"Mean CV score: {cv_df['score'].mean():.4f} ± {cv_df['score'].std():.4f}\n")
        # Best hyperparameters
        # Writes out the tuned hyperparameters for this target
        f.write("Best hyperparameters:\n")
        for k, v in best_params_all[target].items():
            f.write(f"  {k}: {v}\n")
        # Top features
        # Loads the feature importance CSV for this target, selects the top 10 features by importance.
        feat_df = pd.read_csv(FEATURE_DIR / f"{target}_feature_importance.csv")
        f.write("Top 10 features:\n")
        for i, row in feat_df.head(10).iterrows():
            f.write(f"  {row['feature']}: {row['importance']:.4f}\n")
        f.write("\n")

print("Summary file created in deployment_models/training_summary.txt")