"""
prepare_patient_dataset.py

Title: Dataset Preparation for LightGBM

Summary:
- Safety and sanity checks for news2_features_patient.csv
- Handles patient-level features and multiple target variables
- Ensures compatibility for LightGBM ML model

"""
# -----------------------------
# Imports
# -----------------------------
import pandas as pd # main data handling
from sklearn.model_selection import KFold # tool for splitting dataset into k folds (train/test) for cross-validation

# -----------------------------
# Configuration
# -----------------------------
CSV_PATH = "../../data/processed-data/news2_features_patient.csv"  # input CSV
TARGETS = ["max_risk", "median_risk", "pct_time_high"]
N_FOLDS = 5  # for cross-validation

# -----------------------------
# Load dataset
# -----------------------------
# Loads preprocessed patient-level csv data from news2_features_patient.csv, check rows and shape
df = pd.read_csv(CSV_PATH)
print(f"Loaded dataset with shape: {df.shape}")

# -----------------------------
# Safety checks
# -----------------------------
# Check that all features are numeric and no missing values
print("Data types of all columns:")
print(df.dtypes)
print("Missing values per column:")
print(df.isna().sum())

# Separate features (X) and target (y)
features = df.drop(columns=["subject_id", "max_risk", "median_risk", "pct_time_high"]) # Remove identifiers ("subject_id") and target columns ("max_risk", "median_risk", "pct_time_high")
                                                                                       # We don't want the model to see the asnwers (y) in X, we only give it the input features to predict the target.
target = df["max_risk"] # pick a target (label the model is trying to predict) to start with for first iteration of loop
                        # Later, the loop will overwrite target with median_risk and pct_time_high.

# -----------------------------
# Prepare features and loop through targets
# -----------------------------
# For feature columns, remove identifier column ("subject_id") and target columns ("max_risk", "median_risk", "pct_time_high")
# We don't want the model to see the answers (y) in X, we only give it the input features to predict the target.
# .columns.tolist() converts the remaining column names from a pandas index object to a Python list.
# Later we want to use a Python list to select columns (X = df[feature_cols]).
feature_cols = df.drop(columns=["subject_id"] + TARGETS).columns.tolist()

# Loop through target variables list
# We want to train seperate models for each target
for target_name in TARGETS:
    print(f"\nPreparing dataset for target: {target_name}")
    
    # Separate features (X) and target (y)
    X = df[feature_cols] # List of input features the model will use (everything except subject ID and targets)
    y = df[target_name] # The output target variable (the “answer key” the model is trying to learn).
    
    # Final safety check, prevent errors before training
    if X.isna().sum().sum() > 0: # Check for missing values
                                 # X.isna() → returns a boolean DataFrame: True where values are NaN.
                                 # First .sum() sums each column, producing a Series with the number of NaNs per column
                                 # Second .sum() sums that Series to get total number of NaNs in the whole DataFrame.
        print("Warning: Missing values found in features!")
    # Check data types in each column of X, returns True if that column is numeric
    # all([...]) returns True only if all columns are numeric
    if not all([pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes]):
        print("Warning: Non-numeric feature detected!")
    
    # -----------------------------
    # 5-Fold Cross-Validation Setup
    # -----------------------------
    # KFold splits the dataset into N_FOLDS=5 groups (folds).
    # n_splits = number of folds → how many groups the dataset is split into.
    # shuffle=True randomises the patient order before splitting to reduce bias.
    # random_state=42 ensures reproducibility (you get the same splits every time).
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    # KFold does stratified rotation: each patient is assigned to exactly one fold, so no patient is ever skipped.
    # Over the 5 folds, each patient will be in the test set once and in the training set 4 times.
    
    fold_idx = 1
    # Loop over each fold (5)
    # kf.split(X) returns row indices for training (train_index) and test sets (text_index) for each fold.
    # The first variable (train_index) is assigned all the rows that are not in the current fold.
    # The second variable (test_index) is assigned the rows in the current fold.
    # Each iteration of the loop returns train_index (training indices of rows) and test_index (testing indices of rows)
    for train_index, test_index in kf.split(X):
        # .iloc[train_index] / .iloc[test_index] selects the actual rows for X and y from DataFrame according to the indices KFold generated.
        # So the X feature columns and y target column each get both the training index of patients (4 indices), and the test index of patients (1 index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        print(f"Fold {fold_idx}: Train shape {X_train.shape}, Test shape {X_test.shape}") # .shape shows the dimensions of a DataFrame or array: (rows, columns)
                                                                                          # Train will have 80 rows, many columns (e.g. hr_mean). Test will have 20 rows, 1 column (the target)
                                                                                          # Helps verify the splits: the training set has 4 folds (80 patients), the test set has 1 fold (20 patients).
        fold_idx += 1 # counts the fold number for easier tracking.
    
    # At this stage, X_train, y_train, X_test, y_test are ready for model training
    # For each fold in cross-validation, you end up with 4 objects:
        # X_train → DataFrame (rows = training patients, cols = features).
        # y_train → Series (rows = training patients, values = labels/answers).
        # X_test → DataFrame (rows = test patients, cols = features).
        # y_test → Series (rows = test patients, values = labels/answers).