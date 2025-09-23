"""
complete_train_lightgbm.py

Title: Full LightGBM Training Script with Cross-Validation

Summary:
- Trains LightGBM models on patient-level features for three targets:
  1. max_risk (binary classification, 5-fold CV - high risk vs not high risk)
  2. median_risk (binary classification, 5-fold CV)
  3. pct_time_high (regression, 5-fold CV)
- Cross-validation uses StratifiedKFold for classification and KFold for regression
- Implements early stopping to prevent overfitting
- Explicitly encodes valid classes for classification targets (so LightGBM expects all labels).
- Logs evaluation metrics per fold (Accuracy / AUROC for classification, RMSE for regression).
- Saves trained models, CV results, feature importances, and a summary log.
- Fully reproducible and portfolio-ready, with interpretable outputs.
Outputs:
  1. 15 trained models (.pkl) → (5 folds × 3 targets)
  2. 3 per-target CV result CSVs (*_cv_results.csv) → one per target
  3. 15 feature importance CSVs (*_fold{fold_idx}_feature_importance.csv) → one per fold per target
  4. 1 training summary text file (training_summary.txt) → cumulative summary for all targets
  = 34 files in saved_models/
"""

# -----------------------------
# Imports
# -----------------------------
import pandas as pd                                                                                 # loading/manipulating data
import numpy as np                                                                                  # Import numpy for argmax function
from sklearn.model_selection import KFold                                                           # cross-validation
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error                       # evaluate model performance
from sklearn.model_selection import StratifiedKFold                                                 # StratifiedKFold for classification to preserve class ratios across folds
import lightgbm as lgb                                                                              # ML model
from lightgbm import early_stopping, log_evaluation                                                 # use callback functions
from pathlib import Path                                                                            # dynamic, reproducible file paths
import joblib                                                                                       # for saving/loading models
import csv                                                                                          # read from and write to CSV files

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
TARGETS = ["max_risk", "median_risk", "pct_time_high"]                                              # The three outcome variables we want to predict
RANDOM_SEED = 42                                                                                    # Seeds for reproducibility, fixes all sources of randomness (e.g., shuffling for KFold, LightGBM’s internal randomness)

# -----------------------------
# Load dataset + features
# -----------------------------
df = pd.read_csv(CSV_PATH)
feature_cols = df.drop(columns=["subject_id"] + TARGETS).columns.tolist()                           # X = all numeric features except patient IDs/targets, y = target column (handled inside the loop per target)

# -----------------------------
# Preprocessing: Combine Classes
# -----------------------------
# max_risk: Combine 0,1,2 → 2 (not high risk), keep 3 as high risk
if "max_risk" in df.columns:
    df["max_risk"] = df["max_risk"].replace({0: 2, 1: 2})   
# median_risk: Combine 0 → 1 (low risk), keep 2 as high risk                                               
if "median_risk" in df.columns:
    df["median_risk"] = df["median_risk"].replace({0: 1})

# -----------------------------
# Training Loop
# -----------------------------                                                                    
for target_name in TARGETS:                                                                         # Loop through targets to train a separate model for each target automatically without repeating code
    print(f"\nTraining for target: {target_name}")                                                  # Print statement shows which target is currently being processed.
    
    X = df[feature_cols]                                                                            # DataFrame (2D) including all feature columns (inputs) for this target.
    y = df[target_name]                                                                             # DataFrame (technically Series as only one column so 1D) including the target column (output) that the model will learn to predict.

    # -----------------------------
    # Binary Conversion for Classification Targets
    # -----------------------------
    if target_name == "max_risk":
        y = (y == 3).astype(int)                                                                    # Convert 2→0 (not high risk), 3→1 (high risk)
        print(f"Binary class distribution: {pd.Series(y).value_counts().sort_index()}")             # Show 0/1 distribution
    elif target_name == "median_risk":
        y = (y == 2).astype(int)                                                                    # Convert 1→0 (low risk), 2→1 (high risk) 
        print(f"Binary class distribution: {pd.Series(y).value_counts().sort_index()}")             # Show 0/1 distribution

    # -----------------------------
    # Cross-Validation Setup
    # ----------------------------- 
    # Number of folds = 5 for all targets
    n_folds = 5
    
    # Decide CV splitter depending on the target
    if target_name == "pct_time_high":
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)                        # Regular K-Fold cross-validation splitter for regression
                                                                                                    # shuffle=True randomises the row order before splitting, which prevents bias                                                                                              
    else:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)              # Classification (max_risk, median_risk), preserves class ratios across folds in each split.
  
    # -----------------------------
    # Model Setup
    # -----------------------------
    # Choose model class (classifier vs regressor)
    # Explicitly define classes depending on the target
    # Ensures all classes are expected 
    if target_name == "pct_time_high":                          
        model = lgb.LGBMRegressor(random_state=RANDOM_SEED)                                   # LGBMRegressor → regression for continuous targets (pct_time_high)
        metric_fn = mean_squared_error                                                              # Evaluation metric mean_squared_error → regression targets (measures squared difference between predicted and true values)    
    else:
        model = lgb.LGBMClassifier(random_state=RANDOM_SEED,                                        # Binary classification for both max_risk and median_risk
                                   class_weight="balanced")  
        metric_fn = roc_auc_score                                                                   # Evaluation metric oc_auc_score → classification targets (good for imbalanced binary classification, it isn’t fooled by always predicting the majority class)
        
    fold_results = []                                                                               # empty list to store results for each fold, after training each fold append the performance metric (e.g., AUROC, MSE) to this list

    # -----------------------------
    # Cross-Validation Loop
    # -----------------------------
    # Rotate through all folds, so each patient is in a test set exactly once.
    # Each iteration rotates which fold is used as validation.
    # kf.split(X) → generates indices for train and test splits for each fold of the cross-validation
    # For regression (pct_time_high), kf.split(X, None) works.
    # For classification, kf.split(X, y) ensures each fold sees all classes proportionally, in both X and y.
    # enumerate(..., 1) → gives fold_idx starting at 1 (useful for naming/logging)
    # The indexes map to positions in the DataFrame, not the actual content.
    # The first variable (train_index) is assigned all the rows indices that are not in the current fold (a list of 80 integars which tell Python which rows to select from your df).
    # The second variable (test_index) is assigned the rows indices in the current fold.
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y if target_name != "pct_time_high" else None ), 1):                                      
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]                                       # features for training (80 patients) and testing (20 patients in current fold)
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]                                       # labels/targets for training (80 patients) and testing (20 patients in current fold)
                                                                                                    # .iloc takes the row numbers from train_idx / test_idx and extracts the corresponding rows
                                                                                                    # Ensures training data never overlaps with test data in a fold
        # Build a dictionary of training parameters
        fit_params = {                                                                              # Always include X, y, eval_set, and callbacks (common to all tasks).
            "X": X_train,                                                                           # X DataFrame we are training the model on (patients not in the current fold)
            "y": y_train,                                                                           # y Series we are training the model on (patients not in the current fold)
            "eval_set": [(X_test, y_test)],                                                         # eval_set → monitors performance on the test/validation current fold during training.
            "callbacks": [early_stopping(10), log_evaluation(0)]                                    # early_stopping(10) stops training if the model doesn’t improve on the validation set for 10 rounds (prevents overfitting)
        }                                                                                           # log_evaluation(0) = silence, disables all logging output

        # Train the model on the training fold
        model.fit(**fit_params)                                                                     # .fit(**fit_params) → The ** operator unpacks the dictionary into keyword arguments.

        # Generate predictions for the current test fold (X_test)
        preds = model.predict(X_test)                                                               # These predictions will be compared with y_test to evaluate performance.                                                                                          

        # Calculate evaluation metric
        if target_name == "pct_time_high":
            score = metric_fn(y_test, preds)                                                        # Direct MSE calculation for regression
        else:
            # Handle different prediction formats for classification
            if preds.ndim > 1 and preds.shape[1] > 1:
                preds_labels = np.argmax(preds, axis=1)                                             # Convert multiclass probabilities to labels
            else:
                preds_labels = preds.round().astype(int)                                            # Convert binary probabilities to 0/1 labels

            # # Fallback for edge case where test fold has only one class
            if metric_fn == roc_auc_score and len(y_test.unique()) == 1:
                score = accuracy_score(y_test, preds_labels)                                        # Use accuracy when ROC-AUC undefined
                print(f"  Warning: Only one class in test fold, using accuracy instead of ROC-AUC")
            else:
                score = metric_fn(y_test, preds_labels)                                             # Standard ROC-AUC calculation

        # Prints the performance metric for this fold (ROC-AUC, accuracy, or RMSE depending on the task)
        print(f"Fold {fold_idx} score: {score:.4f}")    
        # Saves the metric for this fold into the list fold_results, keep scores to compute averages. 
        # After all 5 folds, fold_results will look like: [0.81, 0.76, 0.79, 0.83, 0.78]                                           
        fold_results.append(score)                                                                  
        
        # Save individual fold model (15 .pkl files in total) for reproducibility, debugging, or later ensemble use.
        model_path = MODEL_DIR / f"{target_name}_fold{fold_idx}.pkl"
        joblib.dump(model, model_path)
                                                                                 
        # -----------------------------
        # Feature Importance Export (.csv)
        # -----------------------------
        # LightGBM exposes feature importance via model.feature_importances_.
        # Each fold per target has its own feature importance CSV (15).
        feat_imp_file = MODEL_DIR / f"{target_name}_fold{fold_idx}_feature_importance.csv"          # Creates a Path object for saving feature importance for the current target and fold.
        feat_importances = pd.DataFrame({                                                           # Creates a DataFrame with two columns:
            "feature": feature_cols,                                                                # "feature" → names of all input features in each row
            "importance": model.feature_importances_                                                # "importance" → the feature importance values computed by LightGBM for this trained model (model.feature_importances_).
        })                                                                                          # Each value is a number representing how much the model relied on that feature to make predictions.
        feat_importances.sort_values(by="importance", ascending=False, inplace=True)                # Sorts the DataFrame in descending order of importance (most important at top), inplace=True → modifies the DataFrame in place without creating a new object.
        feat_importances.to_csv(feat_imp_file, index=False)                                         # Saves the sorted feature importance DataFrame to CSV at the path defined above, index=False → prevents pandas from writing row numbers to the CSV.

    # -----------------------------
    # Cross-Validation Summary
    # -----------------------------
    mean_score = sum(fold_results) / n_folds                                                       # Average performance across folds is more representative
    std_score = np.std(fold_results)                                                               # Standard deviation of fold scores
    print(f"\nAverage {target_name} score: {mean_score:.4f} ± {std_score:.4f}")                    # Report mean ± std

    # -----------------------------
    # Save CV Results to CSV (.csv)
    # -----------------------------
    # Save per-target CV results (3)
    # Each CSV includes: fold number and its score, average score for that target
    # After the loop, we end up with 3 CSV files, one per target
    results_file = MODEL_DIR / f"{target_name}_cv_results.csv"                                      
    with open(results_file, mode='w', newline='') as f:                                             
        writer = csv.writer(f)                                                                      
        writer.writerow(["fold", "score"])                                                          # Writes the header row to the CSV file. "fold" → column for the fold number, "score" → column for the performance metric of that fold.
        for idx, score in enumerate(fold_results, 1):                                               # Loops over the list fold_results
            writer.writerow([idx, score])                                                           # Writes individual fold’s index and score as a row in the CSV.
        writer.writerow(["mean", mean_score])                                                       # Adds row with the mean score across all folds.
        writer.writerow(["std", std_score])                                                         # Add standard deviation

    # -----------------------------
    # Training Summary Log (.txt)
    # -----------------------------
    # Structured log after training (1), summarising: 
    # Dataset shape, Target, Mean CV score, Top 10 features per target
    log_file = MODEL_DIR / "training_summary.txt"                                                   # Defines a single text file to append a structured summary of training per target.
    with open(log_file, "a") as f:                                                                  # open file in append mode so new summaries are added without overwriting previous content.
        f.write(f"Target: {target_name}\n")                                                         # Writes the target name (e.g., max_risk) as a header in the summary file.
        f.write(f"Dataset shape: {X.shape}\n")                                                      # Records the shape of the feature matrix X for this target. (100, 40) → 100 patients × 40 features.
        f.write(f"Mean CV score: {mean_score:.4f}\n")
        if target_name != "pct_time_high":
            f.write(f"Class distribution: {pd.Series(y).value_counts().sort_index().to_dict()}\n")  # Log final class balance
        f.write("Top 10 features:\n")
        top10 = feat_importances.head(10)                                                           # Selects the first 10 rows from the sorted feat_importances DataFrame (highest importance first).
        for i, row in top10.iterrows():                                                             # Loops over the top 10 features and writes each feature name and importance. iterrows() iterates through each row in the DataFrame.
            f.write(f"  {row['feature']}: {row['importance']:.4f}\n")                 
        f.write("\n")

print("\nTraining completed! Check saved_models/ folder for outputs.")   