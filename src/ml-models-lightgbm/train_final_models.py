"""
train_final_models.py

Title: Trained Final Deployment-Ready Models

Summary:
- Trains one final model per target on entire dataset
- Saves final, full-data, deployment-ready .pkl files
- In tuning/feature importance, models were trained and evaluated on cross-validation folds only.
- That helps us pick the best hyperparameters, but the actual models from CV aren’t “the one” we’ll deploy.
- For deployment, we want one final model trained on the entire dataset, using the chosen best hyperparameters.
- This maximises data usage (no holdout left unused) and gives us the strongest model for real-world prediction.
Outputs:
- 3x {target}_model.pkl (full LightGBM model (trees, splits, learned parameters), configured with best hyperparameters found during tuning, trained on the entire dataset (not just CV folds)).
"""
#-------------------------------------------------------------
# Imports
#-------------------------------------------------------------
import pandas as pd
import lightgbm as lgb
from pathlib import Path
import joblib
import json

#-------------------------------------------------------------
# Directories
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
# Reads patient-level features into df
df = pd.read_csv(DATA_PATH)
feature_cols = df.drop(columns=["subject_id"] + TARGETS).columns.tolist()

# Loads dictionary of target → best hyperparameters (same as in feature importance).
with open(TUNE_DIR / "best_params.json") as f:
    best_params_all = json.load(f)

#-------------------------------------------------------------
# Train final model per target
#-------------------------------------------------------------
for target in TARGETS:
    print(f"Training final model for {target}")
    X = df[feature_cols]
    y = df[target]

    # Binary classification and model selection
    if target == "max_risk":
        y = (y == 3).astype(int)
        model_class = lgb.LGBMClassifier
    elif target == "median_risk":
        y = (y == 2).astype(int)
        model_class = lgb.LGBMClassifier
    else:
        model_class = lgb.LGBMRegressor

    # Train model on full dataset with best parameters for that target
    model = model_class(**best_params_all[target], random_state=RANDOM_SEED,
                        class_weight="balanced" if target != "pct_time_high" else None) # Adds class_weight="balanced" for classification tasks to handle imbalanced labels. Rare events get more weight in training.
    model.fit(X, y) # take entire dataset and train once

    # Save model ask .pkl file using joblib
    joblib.dump(model, DEPLOY_DIR / f"{target}_final_model.pkl")

print("Final deployment-ready models saved in deployment_models/ folder.")