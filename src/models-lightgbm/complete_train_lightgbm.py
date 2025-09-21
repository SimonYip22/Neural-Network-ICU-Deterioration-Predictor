"""
complete_train_lightgbm.py

Title: Full LightGBM Training Script with 5-Fold Cross-Validation

Summary:
- Trains LightGBM models on patient-level features for three targets:
  1. max_risk (classification)
  2. median_risk (classification)
  3. pct_time_high (regression)
- Uses 5-fold cross-validation to ensure stable performance estimates.
- Implements early stopping to prevent overfitting.
- Logs evaluation metrics per fold (Accuracy / AUROC for classification, RMSE for regression).
- Prepares for saving models (Step 4) and full validation/interpretability workflow (Step 5–6).
- Key outputs:
  1. 
"""

# -----------------------------
# Imports
# -----------------------------
import pandas as pd                                                                 # loading/manipulating data
from sklearn.model_selection import KFold                                           # cross-validation
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error       # evaluate model performance
import lightgbm as lgb                                                              # ML model
from pathlib import Path                                                            # dynamic, reproducible file paths
import joblib                                                                       # for saving/loading models

# -----------------------------
# Configuration
# -----------------------------
# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR.parent.parent / "data" / "processed-data" / "news2_features_patient.csv"
MODEL_DIR = SCRIPT_DIR / "saved_models"                                                             # Define folder to save trained LightGBM models (3 targets × 5 folds = 15 files total)

# Ensure output folder exists
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Targets
TARGETS = ["max_risk", "median_risk", "pct_time_high"]              # The three outcome variables we want to predict
N_FOLDS = 5                                                         # Number of folds for cross-validation (5-fold CV used for stable performance estimates on small datasets)
RANDOM_SEED = 42                                                    # Seeds for reproducibility, fixes all sources of randomness (e.g., shuffling for KFold, LightGBM’s internal randomness)

# -----------------------------
# Load dataset + features
# -----------------------------
df = pd.read_csv(CSV_PATH)
feature_cols = df.drop(columns=["subject_id"] + TARGETS).columns.tolist()             # X = all numeric features except patient IDs/targets, y = target column (handled inside the loop per target)

# -----------------------------
# KFold Setup
# -----------------------------
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)                  # Split = 5 folds, shuffle = random order of patients to prevent ordering bias, seed = reproducible fold splits
                                                                                  
for target_name in TARGETS:                                                           # Loop through targets to train a separate model for each target automatically without repeating code
    print(f"\nTraining for target: {target_name}")                                    # Print statement shows which target is currently being processed.
    
    X = df[feature_cols]                                                              # DataFrame (2D) including all feature columns (inputs) for this target.
    y = df[target_name]                                                               # DataFrame (technically Series as only one column so 1D) including the target column (output) that the model will learn to predict.
    
    # Choose classifier vs regressor                                                  # metric_fn sets the evaluation metric to use when checking performance, allows the same loop to handle both types of prediction problems seamlessly
    if target_name == "pct_time_high":                          
        model_class = lgb.LGBMRegressor                                               # LGBMRegressor → regression for continuous targets (pct_time_high)
        metric_fn = mean_squared_error                                                # Evaluation metric mean_squared_error → regression targets
    else:
        model_class = lgb.LGBMClassifier                                              # LGBMClassifier → classification for categorical/ordinal targets (max_risk, median_risk)
        metric_fn = roc_auc_score                                                     # Evaluation metric oc_auc_score (or accuracy_score) → classification targets

    fold_results = []                                                                 # empty list to store results for each fold, after training each fold append the performance metric (e.g., AUROC, MSE) to this list

    # Rotate through all folds, so each patient is in a test set exactly once.
    # Each iteration rotates which fold is used as validation.
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), 1):                 # kf.split(X) → generates indices for train and test splits for each fold of the cross-validation
                                                                                      # enumerate(..., 1) → gives fold_idx starting at 1 (useful for naming/logging)
                                                                                      # The indexes map to positions in the DataFrame, not the actual content.
                                                                                      # The first variable (train_index) is assigned all the rows indices that are not in the current fold (a list of 80 integars which tell Python which rows to select from your df).
                                                                                      # The second variable (test_index) is assigned the rows indices in the current fold.
        
        # .iloc takes the row numbers from train_idx / test_idx and extracts the corresponding rows
        # Ensures training data never overlaps with test data in a fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]                         # features for training (80 patients) and testing (20 patients in current fold)
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]                         # labels/targets for training (80 patients) and testing (20 patients in current fold)
        
        model = model_class(random_state=RANDOM_SEED)                                 # Instantiates new lightGBM model (classifier or regressor) for this fold, seed ensures reproducibility
        # Train the model on the training fold
        model.fit(
            X_train, y_train,                                                         # DataFrame and Series we are training the model on (patients not in the current fold)
            eval_set=[(X_test, y_test)],                                              # eval_set → monitors performance on the test/validation current fold during training.
            early_stopping_rounds=10,                                                 # early_stopping_rounds=10 → stops training if the model doesn’t improve on the validation set for 10 rounds (prevents overfitting)
            verbose=False                                                             # verbose=False → suppresses detailed training output in terminal
        )
        
        preds = model.predict(X_test)                                                 # Generates predictions for the current test fold (X_test)
                                                                                      # These predictions will be compared with y_test to evaluate performance.
        if target_name == "pct_time_high":
            score = metric_fn(y_test, preds)                                          # Regression target (pct_time_high) → mean_squared_error.                                          
        else:
            # For classifiers: ROC-AUC or accuracy
            if metric_fn == roc_auc_score and len(y_test.unique()) == 1:              # Classification targets → usually roc_auc_score.
                score = accuracy_score(y_test, preds)                                 # Classification targets → usually roc_auc_score.
            else:
                score = metric_fn(y_test, preds)
        
        print(f"Fold {fold_idx} score: {score:.4f}")
        fold_results.append(score)
        
        # Save fold model
        model_path = MODEL_DIR / f"{target_name}_fold{fold_idx}.pkl"
        joblib.dump(model, model_path)

    # Summarise CV Performance
    mean_score = sum(fold_results) / N_FOLDS
    print(f"\nAverage {target_name} score across {N_FOLDS} folds: {mean_score:.4f}")