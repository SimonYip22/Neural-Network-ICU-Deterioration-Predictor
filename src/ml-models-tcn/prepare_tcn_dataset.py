"""
prepare_tcn_dataset.py

Title: Dataset Preparation for TCN

Summary:
- Input: news2_features_timestamp.csv (timestamp-level EHR features with vitals, labs, obs, missingness flags)
- Preprocessing steps:
    1. Sorts records by subject_id and charttime to ensure chronological sequences.
	2. Splits patients into train/validation/test sets at the patient level (with stratification by outcome).
	3. Normalises continuous features using z-score scaling (fit on training set only).
	4. Pads/truncates sequences to a fixed max_seq_len, producing both sequences and binary masks.
Outputs:
- Tensors (prepared_datasets/):
    - train.pt, val.pt, test.pt → sequence tensors of shape (num_patients, max_seq_len, num_features)
    - train_mask.pt, val_mask.pt, test_mask.pt → masks of shape (num_patients, max_seq_len)
- Deployment objects for reproducibility (deployment_models/preprocessing/):
    - standard_scaler.pkl → trained z-score normalisation scaler
    - padding_config.json → JSON file with max_seq_len + feature ordering
"""

#-------------------------------------------------------------
# Imports
#-------------------------------------------------------------
import pandas as pd       # DataFrames: load and manipulate CSVs
import numpy as np        # Numerical ops: arrays, NaNs, padding
import torch              # Deep learning: save tensors + masks
import json               # Save config files (e.g., padding, feature order)
import joblib             # Save/load preprocessing objects (e.g., StandardScaler)
from pathlib import Path  # File/directory paths in a clean, OS-independent way
from sklearn.preprocessing import StandardScaler  # z-score normalisation for continuous features
from sklearn.model_selection import train_test_split  # patient-level train/val/test splits

#-------------------------------------------------------------
# Directories
#-------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
# Input dir
DATA_PATH = SCRIPT_DIR.parent.parent / "data/processed-data/news2_features_timestamp.csv"
PATIENT_FEATURES_PATH = SCRIPT_DIR.parent.parent / "data/processed-data/news2_features_patient.csv"

# Output dirs
# Folder for training/validation/test tensors and masks 
PREPARED_DIR = SCRIPT_DIR / "prepared_datasets"
PREPARED_DIR.mkdir(parents=True, exist_ok=True)

# Folder for scalers and padding config (deployment/preprocessing/ objects)
SCALER_DIR = SCRIPT_DIR / "deployment_models/preprocessing"
SCALER_DIR.mkdir(parents=True, exist_ok=True)

#-------------------------------------------------------------
# Load dataset, sort & make copy
#-------------------------------------------------------------
print("[INFO] Loading dataset...")
# load the timestamped features CSV
df = pd.read_csv(DATA_PATH)

# sort by patient and time ensures sequences are chronologically aligned
df = df.sort_values(['subject_id', 'charttime'])

# Make a copy of the timestamp-level dataframe
df = df.copy()

#-------------------------------------------------------------
# Merge patient-level outcomes/features
#-------------------------------------------------------------
# max_risk, median_risk, pct_time_high doesn't exist in the timestamp data, but we need it to compare.
# Load patient-level outcomes (from Phase 3 dataset)
patient_features_df = pd.read_csv(PATIENT_FEATURES_PATH)

# Merge into timestamp-level dataframe copy to avoid overwriting original
df = df.merge(patient_features_df, on="subject_id", how="left")

#-------------------------------------------------------------
# Convert targets to binary for stratification
#-------------------------------------------------------------
df["max_risk_binary"] = df["max_risk"].apply(lambda x: 1 if x > 2 else 0) # collapse [0,1,2]=0; [3]=1 (0=not-high, 1=high).
df["median_risk_binary"] = df["median_risk"].apply(lambda x: 0 if x == 1 else 1) # collapse [0,1]=0; [2]=1 (0=low, 1=medium).

#-------------------------------------------------------------
# Step 1: Patient-level stratification
#-------------------------------------------------------------
# Stratification is to ensure that the splits aren’t wildly imbalanced (keeps class balance).
# Only needs to be done on one “anchor” target. Use max_risk because it’s the rarer, more clinically severe outcome → the one most vulnerable to imbalance.
# Generate patient splits stratified on this anchor, then reuse the same patient splits for median_risk and regression tasks.
print("[INFO] Preparing patient-level stratification...")

# Get the list of unique patients. We will split by patients, not rows, prevents time-leakage (same patient cannot appear in train and test)
patient_ids = df["subject_id"].unique()

# Build a patient-level label: for each patient take their max_risk. This yields one label per patient for stratification.
patient_labels = df.groupby("subject_id")["max_risk_binary"].max()

#-------------------------------------------------------------
# Patient-level split (train/val/test)
#-------------------------------------------------------------
# First split: 70% patients → train, 30% → temporary pool (val+test)
# Second split: splits that 30% into 15% val, 15% test
# random_state=42: ensures reproducibility (same split every run)
# stratify=...: balances the ratio of high vs low risk patients across sets
train_ids, temp_ids, train_labels, temp_labels = train_test_split(
    patient_ids, 
    patient_labels.loc[patient_ids],
    test_size=0.3, # 30% goes to temp pool
    random_state=42, 
    stratify=patient_labels.loc[patient_ids] # preserve class ratio
)
val_ids, test_ids, _, _ = train_test_split(
    temp_ids, 
    temp_labels,
    test_size=0.5, # split the 30% pool evenly = 15% val, 50% test
    random_state=42, 
    stratify=temp_labels
)

# Saves three lists of patient IDs in a dictionary so we can later create per-split tensors.
# We save splits for reproducibility.
splits = {"train": train_ids, "val": val_ids, "test": test_ids}
print("[INFO] Split sizes:", {k: len(v) for k, v in splits.items()})
print("[INFO] Stratification preserved binary balance across splits.")

#-------------------------------------------------------------
# Step 2: Feature/target separation
#-------------------------------------------------------------
# Decide which columns will be inputs into the model and which are not, then we group the dataframe rows by patient into sequences.
# Columns we do NOT want to treat as model inputs
id_cols = ["subject_id", "stay_id", "charttime"]
target_cols = ["max_risk", "median_risk", "pct_time_high"]
extra_target_cols = ["max_risk_binary", "median_risk_binary"] # added binary versions for stratification

# Identify all columns excluding patient IDs, timestamps, outcomes)
candidate_cols = [c for c in df.columns if c not in id_cols + target_cols + extra_target_cols]

# Convert consciousness_label to binary numeric (0 = Alert, 1 = Not Alert) as it is the only categorical label we want to keep
df["consciousness_label"] = df["consciousness_label"].apply(lambda x: 0 if x == "Alert" else 1)

# Drop categorical text columns (e.g. risk)
drop_cols = [c for c in candidate_cols if df[c].dtype == "object"]

# Identify all feature columns 
feature_cols = [c for c in candidate_cols if c not in drop_cols]

# Split into continuous vs binary features (do not need to z-score scale binary features)
# (binary = only takes values 0 or 1 across the dataset)
binary_cols = [c for c in feature_cols if set(df[c].dropna().unique()).issubset({0, 1})]
continuous_cols = [c for c in feature_cols if c not in binary_cols]

print(f"[INFO] Selected {len(feature_cols)} features")
print(f"[INFO] - {len(continuous_cols)} continuous")
print(f"[INFO] - {len(binary_cols)} binary")
print(f"[INFO] Dropped categorical cols: {drop_cols}")

#-------------------------------------------------------------
# Step 3: Normalisation (z-score on continuous only)
#-------------------------------------------------------------
# Fit scaler on continuous features from training patients only (compute mean/std for each feature using only training patients), so the reference will always be from the training data.
# No leakage (scaler only sees training patients, and doesn't use other patients to influence the scaling).
# .fit_transform() means we both fit and transform (replace training values with z-scores) the training set
scaler = StandardScaler()
df.loc[df["subject_id"].isin(train_ids), continuous_cols] = scaler.fit_transform(
    df.loc[df["subject_id"].isin(train_ids), continuous_cols]
)

# Apply same z-score transformation to val/test patients
# Now we have the same full dataframe of all patients, all continuous features scaled, binary features untouched. 
for ids in [val_ids, test_ids]:
    df.loc[df["subject_id"].isin(ids), continuous_cols] = scaler.transform(
        df.loc[df["subject_id"].isin(ids), continuous_cols]
    )
# Ensure numeric dtypes post scaling but pre tensor creation
# Convert continuous columns to float32
df[continuous_cols] = df[continuous_cols].astype(np.float32)
# Convert binary columns to float32 (in case they are object dtype)
df[binary_cols] = df[binary_cols].astype(np.float32)

# Save the scaler for later reproducibility (deployment, inference)
# file is a sklearn object with the means and std for every feature (in training set), and the transformation logic: z = (x - mean)/std
joblib.dump(scaler, SCALER_DIR / "standard_scaler.pkl")

#-------------------------------------------------------------
# Step 4: Group DataFrame into per-patient sequences
#-------------------------------------------------------------
# Group the z-score normalisated dataframe rows into per-patient sequences (list of feature vectors)
# Now each patient_id row maps to a 2D NumPy array (timesteps × features)
# TCN expects sequences, not random rows. We need per-patient time series.
# Per-patient sequences dictionary contains key = patient ID, value = 2D NumPy array
sequences = {
    pid: df[df["subject_id"] == pid][feature_cols].values
    for pid in patient_ids
}

#-------------------------------------------------------------
# Step 5: Padding / truncation
#-------------------------------------------------------------
# TCN needs fixed-length sequences per batch, ICU stays are variable, so for some sequences we truncate and some we pad, keeping tensor size consistent.
# Truncate sequences longer than MAX_SEQ_LEN → keeps the tensor size consistent.
# Pad sequences shorter than MAX_SEQ_LEN with zeros
# Mask tells the model which positions are real vs padding (1 = real, 0 = padded), loss functions ignore padding so it doesn't affect training.
MAX_SEQ_LEN = 96  # 96h = 4 days ICU stay, captures critical time windows without excessive truncation 
feature_dim = len(feature_cols)

def make_patient_tensor(pid, max_len=MAX_SEQ_LEN):
    # Pulls all rows (timestamps) for a patient and converts into a NumPy array (timesteps × num_features).
    patient_df = df[df["subject_id"] == pid][feature_cols].to_numpy()
    seq_len = len(patient_df)

    if seq_len >= max_len:
        arr = patient_df[:max_len] # truncate if sequences longer than MAX_SEQ_LEN
        mask = np.ones(max_len, dtype=np.float32)
    else:
        arr = np.zeros((max_len, feature_dim)) # pad sequences shorter than MAX_SEQ_LEN with zeros (keep shape)
        arr[:seq_len] = patient_df
        mask = np.zeros(max_len, dtype=np.float32) # creates corresponding mask of 1s (real) and 0s (padding), so for every timestamp (all 96) there is a corresponding 0 or 1.
        mask[:seq_len] = 1.0

    # Returns a PyTorch tensor for the TCN.
	# Output shape for every patient: (MAX_SEQ_LEN, num_features) for the sequence tensor, (MAX_SEQ_LEN,) for the mask tensor (1D array).
    return torch.tensor(arr, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

#-------------------------------------------------------------
# Step 6: Save tensors + masks
#-------------------------------------------------------------
# We convert our numeric numpy arrays into tensors using PyTorch

# Loop through all patients in that split
for split_name, ids in splits.items():
    tensors, masks = [], []     # Collects all patients into two big lists: tensors and masks
    for pid in ids:
        arr, mask = make_patient_tensor(pid)
        tensors.append(arr)     # arr: padded/truncated sequence (shape = (96, num_features))
        masks.append(mask)      # mask: mask array (shape = (96,)), with 1 = real timestep, 0 = padding

    # Stacks them → 3D arrays for the whole split:
    tensors = torch.stack(tensors)   # (num_patients, 96, feature_dim)
    masks = torch.stack(masks)       # (num_patients, 96)

    # Save into prepared dataset
    torch.save(tensors, PREPARED_DIR / f"{split_name}.pt")
    torch.save(masks, PREPARED_DIR / f"{split_name}_mask.pt")
    print(f"[INFO] Saved {split_name}: {tensors.shape}")

#-------------------------------------------------------------
# Step 7: Save padding / feature configuration
#-------------------------------------------------------------
# Saves metadata so we know how the tensors were created
padding_config = {
    "max_seq_len": MAX_SEQ_LEN, # 96
    "feature_cols": feature_cols, # exact input features (kept for reproducibility)
    "target_cols": target_cols # outcomes available (max_risk, median_risk, pct_time_high)
}
# padding_config.json + standard_scaler.pkl make the preprocessing reproducible
with open(SCALER_DIR / "padding_config.json", "w") as f:
    json.dump(padding_config, f, indent=2)

print("[INFO] Step 1 complete: datasets + scaler + config saved.")