"""
tcn_training_script.py

Title: Full TCN Training Loop Script

Summary:

Ouput:

"""
# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
import torch                                                # core PyTorch library
import torch.nn as nn                                       # neural network modules                    
from torch.utils.data import TensorDataset                  # wraps tensors (features + masks + targets) into a dataset that PyTorch can iterate over.
from torch.utils.data import DataLoader                     # handles batching, shuffling, and parallel loading.
from pathlib import Path                                    # handles filesystem paths in a clean, cross-platform way.
import pandas as pd                                         # data manipulation and analysis (loading CSVs, DataFrames).
import joblib                                               # save and load scalers (StandardScaler) and preprocessing objects.                            
import json                                                 # Read/write JSON files.
from tcn_model import TCNModel                              # TCN architecture implementation (class we defined in tcn_model.py)

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
# All key parameters for training 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"     # Use GPU if available, else CPU
BATCH_SIZE = 32                                             # Number of patient sequences processed in one forward/backward pass.
EPOCHS = 50                                                 # Number of complete passes through the training dataset. Multiple epochs → the model gradually adjusts weights to reduce loss.
LR = 1e-3                                                   # Learning rate for the Adam optimizer. Controls how much to change the model in response to estimated error each time the model weights are updated.
EARLY_STOPPING_PATIENCE = 7                                 # Monitors a validation metric (e.g., ROC-AUC or RMSE). If metric does not improve for 7 epochs, training stops early to avoid overfitting.

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "prepared_datasets"
SCALER_DIR = SCRIPT_DIR / "deployment_models/preprocessing"

# Create new directory to save trained models if it doesn't exist
MODEL_SAVE_DIR = SCRIPT_DIR / "trained_models"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Load datasets
# -------------------------------------------------------------
# ---- Function to Load Dataset Splits ----
def load_split(split_name):
    x = torch.load(DATA_DIR / f"{split_name}.pt")           # Load sequence tensor 
    mask = torch.load(DATA_DIR / f"{split_name}_mask.pt")   # Load corresponding mask tensor
    return x, mask                                          # The function returns both x and mask for a given split

# ---- Load the Dataset Splits using load_split() ----
# x_train, x_val, x_test → input sequences for training, validation, and testing.
# mask_train, mask_val, mask_test → masks to ignore padded timesteps during model computation and loss.
x_train, mask_train = load_split("train")
x_val, mask_val = load_split("val")
x_test, mask_test = load_split("test")

# ---- Load Target Column Names ----
# Load target column names from padding_config.json to identify which columns are targets
with open(SCALER_DIR / "padding_config.json") as f:
    config = json.load(f)

target_cols = ["max_risk_binary", "median_risk_binary", "pct_time_high"] # target columns in patient-level dataframe (used the binary versions for classification tasks)

# -------------------------------------------------------------
# Load true patient-level targets
# -------------------------------------------------------------
# Already have the features (x_train, x_val, x_test) from prepare_tcn_dataset.py
# Only need the patient-level CSV here to build the target tensors (y_train_max, y_train_median, y_train_reg).
# Paths
PATIENT_FEATURES_PATH = SCRIPT_DIR.parent.parent / "data/processed-data/news2_features_patient.csv"
SPLITS_PATH = SCALER_DIR / "patient_splits.json"

# Load patient-level outcomes
patient_df = pd.read_csv(PATIENT_FEATURES_PATH).set_index("subject_id")


# Recreate binary targets (same logic as in prepare_tcn_dataset.py)
patient_df["max_risk_binary"] = patient_df["max_risk"].apply(lambda x: 1 if x > 2 else 0) # collapse [0,1,2]=0; [3]=1 (0=not-high, 1=high).
patient_df["median_risk_binary"] = patient_df["median_risk"].apply(lambda x: 1 if x == 2 else 0) # collapse [0,1]=0; [2]=1 (0=low, 1=medium).

# Load saved patient ID splits
with open(SPLITS_PATH) as f:
    patient_splits = json.load(f)

# Helper function to extract rows for given patient IDs in the split, and the 3 target columns
def get_targets(split_ids):
    df_split = patient_df.loc[split_ids, target_cols]   # selects just the split patient rows from patient-level dataframe (patient_df), keeping only the target columns
    # convert each target column into a PyTorch tensor (contains values for each pateint in the split)
    return (
        torch.tensor(df_split["max_risk_binary"].values, dtype=torch.float32),
        torch.tensor(df_split["median_risk_binary"].values, dtype=torch.float32),
        torch.tensor(df_split["pct_time_high"].values, dtype=torch.float32),
    )

# ---- Build target tensors for each split ----
# Need tensors for target outcomes to compare with model predictions during training (x_train[i], mask_train[i]) → (y_train_max[i], y_train_median[i], y_train_reg[i])
# use patient_splits.json to get patient IDs for each split
train_ids = patient_splits["train"]
val_ids = patient_splits["val"]
test_ids = patient_splits["test"]

# get_targets() returns 3 tensors per split (max, median, regression targets)
# this gets repeated for train, val, test splits
# now we have tensors for all 3 targets in all 3 splits
y_train_max, y_train_median, y_train_reg = get_targets(train_ids)
y_val_max, y_val_median, y_val_reg = get_targets(val_ids)
y_test_max, y_test_median, y_test_reg = get_targets(test_ids)

# Print shapes to verify, each tensor shape should match number of patients in each split
print("[INFO] Targets loaded:")
print(" - train:", y_train_max.shape, y_train_median.shape, y_train_reg.shape) # 70
print(" - val:", y_val_max.shape, y_val_median.shape, y_val_reg.shape) # 15
print(" - test:", y_test_max.shape, y_test_median.shape, y_test_reg.shape) # 15

# -------------------------------------------------------------
# TensorDatasets & DataLoaders
# -------------------------------------------------------------
"""
TensorDataset is a PyTorch utility.
- It bundles multiple tensors (inputs + masks + labels) together so that they can be indexed as a unit.
DataLoader 
- Takes the dataset and feeds it to the model in mini-batches during training/validation.
Why we need this:
- Training on the entire dataset at once (full batch) is too memory-heavy.
- Mini-batching allows: More efficient GPU/CPU usage, smoother gradient estimates (stochastic gradient descent), the possibility to shuffle → avoids learning artefacts from data order.
"""

# Each sample (patient) in train_dataset is a tuple containing 5 tensors: (input sequence, mask, max target, median target, regression target)
train_dataset = TensorDataset(x_train, mask_train, y_train_max, y_train_median, y_train_reg)
val_dataset = TensorDataset(x_val, mask_val, y_val_max, y_val_median, y_val_reg)

# batch_size=BATCH_SIZE: 32 → each training step sees 32 patients at once.
# shuffle=True: randomises order each epoch → prevents overfitting to sequence of data.
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) # For validation we set shuffle=False (keep stable evaluation).

# -------------------------------------------------------------
# Model
# -------------------------------------------------------------
# build TCN with 3 convolutional blocks (64 → 64 → 128 filters), followed by a dense head of size 64, and made it ready to train on GPU/CPU.
# x_train is of shape (num_patients, max_seq_len, num_features), shape[2] is num_features (171)
# num_features=feature_dim into mdoel so it knows dimensions of each input vector
feature_dim = x_train.shape[2]
model = TCNModel(num_features=feature_dim,      # the size of the input feature vector at each timestep (171).
                 num_channels=[64, 64, 128],    # defines residual block architecture
                 head_hidden=64                 # Size of the dense hidden layer after pooling.
                 ).to(DEVICE)                   # moves model to GPU if available, else CPU

# -------------------------------------------------------------
# Training Logic Setup: Losses & Optimiser
# -------------------------------------------------------------
# Each task needs its own loss function:
criterion_max = nn.BCEWithLogitsLoss()          # Binary Cross-Entropy with Logits Loss (raw model output) for binary classification tasks (max and median risk).
criterion_median = nn.BCEWithLogitsLoss()
criterion_reg = nn.MSELoss()                    # Mean Squared Error, standard choice for continuous outputs. Penalizes larger errors more strongly.

# Optimiser
# Adam = Adaptive Moment Estimation, updates all trainable weights of model based on computed gradients from backpropagation.
# LR = 1e-3, a common starting learning rate for Adam.
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Schedular monitora validation metric (val loss here). If no improvement for 3 epochs, reduce LR by factor of 0.5.
# Helps model converge better by taking smaller steps when progress stalls.
# mode="min" → because we want to minimise loss.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

# -------------------------------------------------------------
# Training Loop with Validation
# -------------------------------------------------------------
best_val_loss = float("inf")        # validation loss starts as infinity, so any first real validation loss will be smaller and count as “best”.
patience_counter = 0                # counts how many consecutive epochs the validation loss has not improved.

# ---- Epoch Loop ----
# epoch = one full pass over the training dataset.
# run multiple epochs (10–100+) so the model gradually learns.
for epoch in range(1, EPOCHS + 1):
    # set model to training mode (activates dropout, batch norm updates, etc.).
    model.train()
    total_loss = 0.0
    # ---- Mini-batch Loop ----
    # iterate over mini-batches from train_loader
    for x_batch, mask_batch, y_max, y_median, y_reg in train_loader:
        # Move data to GPU/CPU
        x_batch = x_batch.to(DEVICE)
        mask_batch = mask_batch.to(DEVICE)
        y_max = y_max.to(DEVICE)
        y_median = y_median.to(DEVICE)
        y_reg = y_reg.to(DEVICE)

        optimizer.zero_grad()                     # reset old gradients 
        outputs = model(x_batch, mask_batch)      # forward pass: get model predictions for the batch

        # Calculate losses for 3 tasks
        loss_max = criterion_max(outputs["logit_max"], y_max)
        loss_median = criterion_median(outputs["logit_median"], y_median)
        loss_reg = criterion_reg(outputs["regression"], y_reg)

        # Combine losses (simple sum, equal weighting)
        loss = loss_max + loss_median + loss_reg

        # Backward pass (backpropagation)
        loss.backward()                                                     # compute gradient of the loss with respect to every trainable parameter in the model.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    # gradient clipping to prevent exploding gradients
        optimizer.step()                                                    # update model weights (based on computed gradients), moving them in a direction that decreases the loss.
        
        # Accumulate loss for averaging  
        # ensures you sum up loss weighted by batch size.
        # You get the average training loss across all patients.
        total_loss += loss.item() * x_batch.size(0)

    avg_train_loss = total_loss / len(train_loader.dataset)

    # ---- Validation phase ----
    # set model to evaluation mode (disables dropout, batch norm updates, etc.)
    # Same as training, but no gradients or optimiser step.
    # You only compute validation loss → tells you how well the model generalises to unseen patients.
    model.eval()
    val_loss = 0.0
    with torch.no_grad():             # disable gradient computation (saves memory, speeds up)
        for x_batch, mask_batch, y_max, y_median, y_reg in val_loader:
            x_batch = x_batch.to(DEVICE)
            mask_batch = mask_batch.to(DEVICE)
            y_max = y_max.to(DEVICE)
            y_median = y_median.to(DEVICE)
            y_reg = y_reg.to(DEVICE)

            outputs = model(x_batch, mask_batch)
            loss_max = criterion_max(outputs["logit_max"], y_max)
            loss_median = criterion_median(outputs["logit_median"], y_median)
            loss_reg = criterion_reg(outputs["regression"], y_reg)

            loss = loss_max + loss_median + loss_reg
            val_loss += loss.item() * x_batch.size(0)

    avg_val_loss = val_loss / len(val_loader.dataset)

    # ---- Scheduler step ----
    # If avg_val_loss hasn’t improved for a while, it automatically lowers the learning rate to make training more fine-grained.
    scheduler.step(avg_val_loss)

    # ---- Progress reporting ----
    # track learning across epochs.
    # Normally, train loss ↓ and val loss ↓ if model learns well.
    # If train ↓ but val ↑, that’s overfitting.
    print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    # ---- Early stopping ----
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_SAVE_DIR / "tcn_best.pt") # If validation loss improves → save model.
    else:
        patience_counter += 1                               # No improvement → increment counter.
        if patience_counter >= EARLY_STOPPING_PATIENCE:     # If no improvement for 7 epochs → stop training (saves time, avoids overfitting).
            print(f"Early stopping at epoch {epoch}")
            break

print("Training complete. Best model saved to tcn_best.pt")     # tcn_best.pt stores the best model weights.

print(y_train_max.unique())
print(y_train_median.unique())
print(y_train_reg.min(), y_train_reg.max())

print(torch.isnan(x_train).any(), torch.isinf(x_train).any())