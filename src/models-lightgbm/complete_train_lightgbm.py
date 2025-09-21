"""
complete_train_lightgbm.py

Title: Full LightGBM Training Script with 5-Fold Cross-Validation

Summary:
- Trains LightGBM models on patient-level features for three targets:
  1. max_risk (classification)
  2. median_risk (classification)
  3. pct_time_high (regression)
- Uses 5-fold cross-validation to ensure stable performance estimates (trains 5 folds × 3 targets = 15 models).
- Implements early stopping to prevent overfitting.
- Logs evaluation metrics per fold (Accuracy / AUROC for classification, RMSE for regression).
- Saves all 15 models, CV results, feature importances, and a summary log. Outputs:
  1. 15 trained models (.pkl) → 5 folds × 3 targets
	2. 3 per-target CV result CSVs (*_cv_results.csv) → one per target
	3. 15 feature importance CSVs (*_fold{fold_idx}_feature_importance.csv) → one per fold per target
	4. 1 training summary text file (training_summary.txt) → cumulative summary for all targets
- Fully reproducible and portfolio-ready, with interpretable outputs.
"""

# -----------------------------
# Imports
# -----------------------------
import pandas as pd                                                                                 # loading/manipulating data
from sklearn.model_selection import KFold                                                           # cross-validation
from sklearn.model_selection import StratifiedKFold                                                 # stratified version of KFold from scikit-learn, safer splitter for classification tasks
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error                       # evaluate model performance
import lightgbm as lgb                                                                              # ML model
from lightgbm import early_stopping, log_evaluation                                                 # use callback functions
from pathlib import Path                                                                            # dynamic, reproducible file paths
import joblib                                                                                       # for saving/loading models
import csv                                                                                          # read from and write to CSV files
import numpy as np                                                                                  # handling arrays

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
N_FOLDS = 5                                                                                         # Number of folds for cross-validation (5-fold CV used for stable performance estimates on small datasets)
RANDOM_SEED = 42                                                                                    # Seeds for reproducibility, fixes all sources of randomness (e.g., shuffling for KFold, LightGBM’s internal randomness)

# -----------------------------
# Load dataset + features
# -----------------------------
df = pd.read_csv(CSV_PATH)
feature_cols = df.drop(columns=["subject_id"] + TARGETS).columns.tolist()                           # X = all numeric features except patient IDs/targets, y = target column (handled inside the loop per target)

# -----------------------------
# KFold / StratifiedKFold Setup
# -----------------------------
# StratifiedKFold is a special cross-validation splitter that keeps class proportions (0/1 labels) balanced across folds.
# KFold randomly splits rows → some folds might end up with only 0s or only 1s. Then ROC-AUC fails.
# StratifiedKFold ensures each fold preserves the same proportion of 0s and 1s as the full dataset. That avoids the error.  
# Used on classification model as target values are binary (0/1).
                                                                            
for target_name in TARGETS:                                                                         # Loop through targets to train a separate model for each target automatically without repeating code
    print(f"\nTraining for target: {target_name}")                                                  # Print statement shows which target is currently being processed.
    
    X = df[feature_cols]                                                                            # DataFrame (2D) including all feature columns (inputs) for this target.
    y = df[target_name]                                                                             # DataFrame (technically Series as only one column so 1D) including the target column (output) that the model will learn to predict.
    
    # Choose CV splitter depending on target type
    if target_name == "pct_time_high":
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)                        # Use plain KFold, no classes to stratify: Split = 5 folds, shuffle = random order of patients to prevent ordering bias, seed = reproducible fold splits
    else:
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)              # StratifiedKFold, to ensure each fold has a fair mix of 0s and 1s (target values are ordinal/binary)

    # -----------------------------
    # Model Setup
    # -----------------------------
    # Choose model class (classifier vs regressor)
    if target_name == "pct_time_high":                          
        model_class = lgb.LGBMRegressor                                                             # LGBMRegressor → regression for continuous targets (pct_time_high)
        metric_fn = mean_squared_error                                                              # Evaluation metric mean_squared_error → regression targets
    else:
        model_class = lgb.LGBMClassifier                                                            # LGBMClassifier → classification for categorical/ordinal targets (max_risk, median_risk)
        metric_fn = roc_auc_score                                                                   # Evaluation metric oc_auc_score (or accuracy_score) → classification targets

    fold_results = []                                                                               # empty list to store results for each fold, after training each fold append the performance metric (e.g., AUROC, MSE) to this list

    # Rotate through all folds, so each patient is in a test set exactly once.
    # Each iteration rotates which fold is used as validation.
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):                            # kf.split(X, y) → generates indices for train and test splits for each fold of the cross-validation
                                                                                                    # KFold.split() → only needs X (random partitioning by rows), StratifiedKFold.split() → requires both X and y (partitioning by rows while preserving the class balance in y, strafifies based on y).
                                                                                                    # enumerate(..., 1) → gives fold_idx starting at 1 (useful for naming/logging)
                                                                                                    # The indexes map to positions in the DataFrame, not the actual content.
                                                                                                    # The first variable (train_index) is assigned all the rows indices that are not in the current fold (a list of 80 integars which tell Python which rows to select from your df).
                                                                                                    # The second variable (test_index) is assigned the rows indices in the current fold.
        
        # .iloc takes the row numbers from train_idx / test_idx and extracts the corresponding rows
        # Ensures training data never overlaps with test data in a fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]                         # features for training (80 patients) and testing (20 patients in current fold)
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]                         # labels/targets for training (80 patients) and testing (20 patients in current fold)
        
        model = model_class(random_state=RANDOM_SEED,                                 # Instantiates new lightGBM model (classifier or regressor) for this fold, seed ensures reproducibility
                            class_weight="balanced"                                   # Makes classifier model aware of both 0 and 1 labels
                            )                                 
        # Train the model on the training fold
        model.fit(
            X_train, y_train,                                                         # DataFrame and Series we are training the model on (patients not in the current fold)
            eval_set=[(X_test, y_test)],                                              # eval_set → monitors performance on the test/validation current fold during training.
            callbacks=[
                early_stopping(10),                                                   # Stops training if the model doesn’t improve on the validation set for 10 rounds (prevents overfitting)
                log_evaluation(0)],                                                   # 0 = silence, disables all logging output
                classes=[0, 1]                                                        # Ensures both classes are always expected
        )

        preds = model.predict(X_test)                                                 # Generates predictions for the current test fold (X_test)
                                                                                      # These predictions will be compared with y_test to evaluate performance.
        if target_name == "pct_time_high":
            score = metric_fn(y_test, preds)                                          # Regression target (pct_time_high) → mean_squared_error.                                          
        else:
            # For classifiers: ROC-AUC or accuracy
            if metric_fn == roc_auc_score and len(y_test.unique()) == 1:              # Classification targets → usually roc_auc_score, it measures how well the model separates positive vs negative cases across all thresholds.
                score = accuracy_score(y_test, preds)                                 # More informative than raw accuracy (which just measures percentage correct).
            else:                                                                     # However, in small test sets, soemtimes all patients are the same class, so ROC-AUC cannot be computed, because it needs both positives and negatives to calculate the curve.
                 score = metric_fn(y_test, preds)                                     # If test fold has only one class → fall back to accuracy_score (at least gives you some metric, rather than crashing).
        
        print(f"Fold {fold_idx} score: {score:.4f}")                                  # Prints the performance metric for this fold (ROC-AUC, accuracy, or RMSE depending on the task).
                                                                                      # {fold_idx} → tells you which fold you’re on (1–5).
                                                                                      # {score:.4f} → formats the score to 4 decimal places (e.g., 0.8123).
        fold_results.append(score)                                                    # Saves the metric for this fold into the list fold_results, keep scores to compute averages.
                                                                                      # After all 5 folds, fold_results will look like: [0.81, 0.76, 0.79, 0.83, 0.78]
        
        # Save fold model
        model_path = MODEL_DIR / f"{target_name}_fold{fold_idx}.pkl"                  # Example: max_risk_fold1.pkl
        joblib.dump(model, model_path)                                                # joblib.dump(model, model_path) → saves the trained LightGBM model object to disk.
                                                                                      # So after training, we have 15 model files total: 5 folds × 3 targets = 15 saved .pkl files.
                                                                                      # Persist the model for reproducibility, debugging, or later ensemble use.                                                                                    
    # Summarise CV Performance
    mean_score = sum(fold_results) / N_FOLDS                                          # Average performance across all folds
    print(f"\nAverage {target_name} score across {N_FOLDS} folds: {mean_score:.4f}")  # Example output: Average max_risk score across 5 folds: 0.7940
                                                                                      # The average score across folds is more stable and representative.

    # -----------------------------
    # Save results to CSV 
    # -----------------------------
    # Save per-target CV results
    # Each CSV includes: fold number and its score, average score for that target
    # After the loop, we end up with 3 CSV files, one per target
    results_file = MODEL_DIR / f"{target_name}_cv_results.csv"                         # Creates new CSV file where results will be saved.
    with open(results_file, mode='w', newline='') as f:                                # newline='' → ensures that the CSV module handles line breaks correctly across platforms (prevents blank lines on Windows), with statement: Automatically closes the file when done, even if an error occurs.
        writer = csv.writer(f)                                                         # Creates a CSV “writer” object that will write rows of data to the file f.
        writer.writerow(["fold", "score"])                                             # Writes the header row to the CSV file. "fold" → column for the fold number, "score" → column for the performance metric of that fold.
        for idx, score in enumerate(fold_results, 1):                                  # Loops over the list fold_results
            writer.writerow([idx, score])                                              # Writes each fold’s index and score as a row in the CSV.
        writer.writerow(["mean", mean_score])                                          # Adds a final row with the average score across all folds.

    # -----------------------------
    # Feature importance logging
    # -----------------------------
    # LightGBM exposes feature importance via model.feature_importances_.
    # Each fold per target has its own feature importance CSV.
    feat_imp_file = MODEL_DIR / f"{target_name}_fold{fold_idx}_feature_importance.csv" # Creates a Path object for saving feature importance for the current target and fold.
    feat_importances = pd.DataFrame({                                                  # Creates a DataFrame with two columns:
        "feature": feature_cols,                                                       # "feature" → names of all input features in each row
        "importance": model.feature_importances_                                       # "importance" → the feature importance values computed by LightGBM for this trained model (model.feature_importances_).
    })                                                                                 # Each value is a number representing how much the model relied on that feature to make predictions.
    feat_importances.sort_values(by="importance", ascending=False, inplace=True)       # Sorts the DataFrame in descending order of importance (most important at top), inplace=True → modifies the DataFrame in place without creating a new object.
    feat_importances.to_csv(feat_imp_file, index=False)                                # Saves the sorted feature importance DataFrame to CSV at the path defined above, index=False → prevents pandas from writing row numbers to the CSV.

    # -----------------------------
    # Documentation
    # -----------------------------
    # Structured log after training, summarising: 
    # Dataset shape, Target, Mean CV score, Top 10 features per target
    log_file = MODEL_DIR / "training_summary.txt"                                      # Defines a single text file to append a structured summary of training per target.
    with open(log_file, "a") as f:                                                     # open file in append mode so new summaries are added without overwriting previous content.
        f.write(f"Target: {target_name}\n")                                            # Writes the target name (e.g., max_risk) as a header in the summary file.
        f.write(f"Dataset shape: {X.shape}\n")                                         # Records the shape of the feature matrix X for this target. (100, 40) → 100 patients × 40 features.
        f.write(f"Mean CV score: {mean_score:.4f}\n")
        f.write("Top 10 features:\n")
        top10 = feat_importances.head(10)                                              # Selects the first 10 rows from the sorted feat_importances DataFrame (highest importance first).
        for i, row in top10.iterrows():                                                # Loops over the top 10 features and writes each feature name and importance. iterrows() iterates through each row in the DataFrame.
            f.write(f"  {row['feature']}: {row['importance']}\n")                 
        f.write("\n")