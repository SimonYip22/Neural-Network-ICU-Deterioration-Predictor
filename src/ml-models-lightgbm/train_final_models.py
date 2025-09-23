"""
train_final_models.py

Title: 

Summary:
- Trains one final model per target on entire dataset
- Saves deployment-ready .pkl files and optional summary logs
Outputs:
- 
"""

import pandas as pd
import lightgbm as lgb
from pathlib import Path
import joblib
import json

#-------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR.parent.parent / "data/processed-data/news2_features_patient.csv"
DEPLOY_DIR = SCRIPT_DIR / "deployment_models"
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
TUNE_DIR = SCRIPT_DIR / "hyperparameter_tuning_runs"

TARGETS = ["max_risk", "median_risk", "pct_time_high"]
RANDOM_SEED = 42

#-------------------------------------------------------------
# Load dataset
#-------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
feature_cols = df.drop(columns=["subject_id"] + TARGETS).columns.tolist()

with open(TUNE_DIR / "best_params.json") as f:
    best_params_all = json.load(f)

#-------------------------------------------------------------
# Train final model per target
#-------------------------------------------------------------
for target in TARGETS:
    print(f"Training final model for {target}")
    X = df[feature_cols]
    y = df[target]

    if target == "max_risk":
        y = (y == 3).astype(int)
        model_class = lgb.LGBMClassifier
    elif target == "median_risk":
        y = (y == 2).astype(int)
        model_class = lgb.LGBMClassifier
    else:
        model_class = lgb.LGBMRegressor

    model = model_class(**best_params_all[target], random_state=RANDOM_SEED,
                        class_weight="balanced" if target != "pct_time_high" else None)
    model.fit(X, y)

    # Save model
    joblib.dump(model, DEPLOY_DIR / f"{target}_final_model.pkl")

print("Final deployment-ready models saved in deployment_models/ folder.")