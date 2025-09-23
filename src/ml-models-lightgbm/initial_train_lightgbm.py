"""
initial_train_lightgbm.py

Title: LightGBM Quick Test Run On Patient-Level Dataset

Summary:
- Model initialisation setup and create both classifier (max_risk, median_risk) and regressor (pct_time_high)
- Quick test run on a small subset of training data (10 patients)
- Verifys pipeline before coding full production run on full dataset next
"""

# -----------------------------
# Imports
# -----------------------------
import pandas as pd # For loading and manipulating CSV dataset
from sklearn.model_selection import KFold # If we want to implement cross-validation (in this quick test we aren’t actually looping folds).
import lightgbm as lgb # ML library to create classifier or regressor models.
from pathlib import Path  # For dynamic file paths

# -----------------------------
# Configuration
# -----------------------------
# Get the directory of this script
SCRIPT_DIR = Path(__file__).resolve().parent

# Build path to CSV relative to project root
CSV_PATH = SCRIPT_DIR.parent.parent / "data" / "processed-data" / "news2_features_patient.csv"
TARGETS = ["max_risk", "median_risk", "pct_time_high"] # Loop through each outcome we want to predict
N_FOLDS = 5 # Number of folds for cross-validation (not fully used in this quick test, but needed later).

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(CSV_PATH) # Reads CSV into a DataFrame
print(f"Loaded dataset with shape: {df.shape}") # Prints number of columns and rows (100, 44)

# Feature columns (drop identifier (subject_id) + targets)
# Model should not see the "answers" from the targets when training
# .tolist() converts column names into Python list
feature_cols = df.drop(columns=["subject_id"] + TARGETS).columns.tolist()

# -----------------------------
# Loop through targets
# -----------------------------
# Loops through all 3 target variables so we can train a separate model for each.
for target_name in TARGETS:
    print(f"\n### Quick test for target: {target_name}") # Print which target is currently being tested.
    
    X = df[feature_cols] # Input feature (everything except targets + identifier).
    y = df[target_name] # Target labels (the column we want the model to predict).

    # Pick classifier vs regressor
    # LightGBM has different classes for regression vs classification.
    if target_name == "pct_time_high": # pct_time_high is continuous → regression
                                       # max_risk & median_risk are ordinal / categorical → classification.
        model = lgb.LGBMRegressor(random_state=42)  # regression
    else:
        model = lgb.LGBMClassifier(random_state=42)  # classification
    # random_state=42 → ensures reproducible results (same random splits every time), stabilises all internal randomness inside LightGBM for reproducibility.


    # -----------------------------
    # Quick test run: use only first 10 rows
    # -----------------------------
    X_small, y_small = X.iloc[:10], y.iloc[:10] # Take first 10 rows from X and y, .iloc[:10] → selects rows 0–9.
    model.fit(X_small, y_small)  # Fit tiny dataset just to check pipeline
                                 # Trains the LightGBM model on this small subset.
                                 # The goal is not to get meaningful predictions, just to check for errors (feature mismatch, data type issues).
    
    preds = model.predict(X_small) # Makes predictions on same tiny dataset
    print(f"First 10 predictions for {target_name}: {preds}") # Prints to confirm it runs and outputs