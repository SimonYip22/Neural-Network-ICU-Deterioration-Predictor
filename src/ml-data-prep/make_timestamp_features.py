"""
make_timestamp_features.py

Generates timestamp-level ML features from news2_scores.csv:
- Handles missing values (LOCF + missingness flags)
- Create carried-forward flags
- Computes rolling windows (1h, 4h, 24h)
- Computes time since last observation per vital
- Encodes escalation/risk labels into numeric format

Output: news2_features_timestamp.csv (ML-ready)
"""

# ------------------------------
# Imports
# ------------------------------
import pandas as pd
from pathlib import Path

# ------------------------------
# Config: file paths
# ------------------------------
DATA_DIR = Path("../data/processed-data")
INPUT_FILE = DATA_DIR / "news2_scores.csv"
OUTPUT_FILE = DATA_DIR / "news2_features_timestamp.csv"

# ------------------------------
# Step 1: Load & Sort
# ------------------------------
def load_and_sort_data(input_file: Path) -> pd.DataFrame:
    # Load CSV
    df = pd.read_csv(input_file) #reads the CSV into a DataFrame (df)

    # Ensure charttime is datetime
    # Converts the charttime column from string (like "2180-07-23 14:00:00") into datetime objects.
    df['charttime'] = pd.to_datetime(df['charttime'])

    # Sort by patient and time, ensures all rows are in chronological order per patient.
    # sorted by subject_id (unique patient), stay_id (hospital stay), and charttime (timestamp of observation).
    # reset_index(drop=True) → resets row numbering from 0…n-1 and discards the old index.
    df = df.sort_values(by=['subject_id', 'stay_id', 'charttime']).reset_index(drop=True)

    return df # return clean and sorted DataFrame

if __name__ == "__main__":
    df = load_and_sort_data(INPUT_FILE)
    print("Data loaded and sorted. Sample:")
    print(df.head())

# ----------------------------------------------------
# Step 2: Create missingness flags before filling
# ----------------------------------------------------
def add_missingness_flags(df: pd.DataFrame) -> pd.DataFrame:
    vitals = [
        "respiratory_rate", "spo2", "supplemental_o2",
        "temperature", "systolic_bp", "heart_rate",
        "level_of_consciousness", "co2_retainer"
    ]
    # loops through each vital sign column
    for v in vitals:
        flag_col = f"{v}_missing" # name of new flag column created
        df[flag_col] = df[v].isna().astype(int) # checks if value is NaN, returns a boolean, then converts to int (1 if NaN, else 0)
                                                # store in new column df[flag_col]
    return df 

if __name__ == "__main__":
    df = load_and_sort_data(INPUT_FILE) # df is loaded and sorted, then returned
    df = add_missingness_flags(df) # missingness flag function is called here, and df is called and updated
    print("Data with missingness flags. Sample:")
    print(df.head())

# -------------------------------------
# Step 3: LOCF forward-fill per subject
# -------------------------------------
# Missingness flags already created in step 2 so the ML model knows which values were originally missing
def apply_locf(df: pd.DataFrame) -> pd.DataFrame:
    vitals = [
        "respiratory_rate", "spo2", "supplemental_o2",
        "temperature", "systolic_bp", "heart_rate",
        "level_of_consciousness", "co2_retainer"
    ]
    # Group by subject_id and stay_id to ensure filling is done within each patient's hospital stay
    # Then .ffill() and .bfill() are applied inside each group independently.
    
    # Forward-fill per subject_id + stay_id
    # if row missing, fill with last available value
    df[vitals] = df.groupby(["subject_id", "stay_id"])[vitals].ffill()

    # Also backfill the very first missing values (first row per patient)
    # if first row missing, fill with next available value 
    df[vitals] = df.groupby(["subject_id", "stay_id"])[vitals].bfill()

    return df


if __name__ == "__main__":
    df = load_and_sort_data(INPUT_FILE)
    df = add_missingness_flags(df)
    df = apply_locf(df)
    print("Data after LOCF applied. Sample:")
    print(df.head(20))

# -------------------------------------
# Step 4: Create carried-forward flags
# -------------------------------------
# Marks which non-NaN values in the final dataset are actually imputed from LOCF instead of observed vitals
# Using the _missing flags from Step 2 as ground truth, this avoids mislabeling repeated natural values as carried-forward.
# Missingness flags are before filling, carried-forward flags are after filling
def add_carried_forward_flags(df: pd.DataFrame) -> pd.DataFrame:
    vitals = [
        "respiratory_rate", "spo2", "supplemental_o2",
        "temperature", "systolic_bp", "heart_rate",
        "level_of_consciousness", "co2_retainer"
    ]

    for v in vitals:
        carried_col = f"{v}_carried" # name of new carried forward flag column
        missing_col = f"{v}_missing" # name of existing missingness flag column from Step 2
        
        # df[v].notna() → checks if the final value in this column is not NaN (so it exists after filling).
        # (df[missing_col] == 1) → checks if that same row was missing before fill.
        # & → logical AND operator, so both conditions must be true, if value exists and it was missing before
        df[carried_col] = (
            (df[v].notna()) & (df[missing_col] == 1)
        ).astype(int) # Convert boolean to int (1 if carried forward (LOCF), 0 if observed naturally)

    return df # Returns the same DataFrame with the new _carried columns added.

if __name__ == "__main__":
    df = load_and_sort_data(INPUT_FILE)
    df = add_missingness_flags(df)
    df = apply_locf(df)
    df = add_carried_forward_flags(df)
    print("Data with carried-forward flags. Sample:")
    print(df.head(20))

# -------------------------------------
# Step 5: Compute rolling window features
# -------------------------------------
# Number of vitals = 5 (respiratory_rate, spo2, temperature, systolic_bp, heart_rate)
# Number of windows = 3 (1h, 4h, 24h)
# Number of stats per window = 6 (mean, min, max, std, slope, AUC)
# 5 vitals x 3 windows x 6 stats = 90 new feature columns per row

# NumPy is used for numerical operations like slope and AUC calculations
import numpy as np

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    # These 5 vitals will have rolling windows computed (numeric ones only)
    vitals = ["respiratory_rate", "spo2", "temperature", "systolic_bp", "heart_rate"]
    # Time window sizes in hours (3 total)
    windows = [1, 4, 24]
    # Stats to compute per window (6 total)
    stats = ["mean", "min", "max", "std", "slope", "auc"]

    # Convert charttime to numeric timestamp for slope/AUC calculations
    df['charttime_numeric'] = df['charttime'].astype('int64') / 1e9  # seconds since epoch

    # Loop through every vital and window size
    for v in vitals:
        for w in windows:
            # 
            roll = df.groupby(['subject_id', 'stay_id'])[v].rolling(
                f"{w}H", on='charttime', min_periods=1
            )
            # Mean, min, max → capture magnitude.
            # Std → capture variability.
            # Slope → capture trend/direction.
            # AUC → capture cumulative exposure/risk over time.

            # Compute stats
            df[f"{v}_roll{w}h_mean"] = roll.mean().reset_index(level=[0,1], drop=True)
            df[f"{v}_roll{w}h_min"] = roll.min().reset_index(level=[0,1], drop=True)
            df[f"{v}_roll{w}h_max"] = roll.max().reset_index(level=[0,1], drop=True)
            df[f"{v}_roll{w}h_std"] = roll.std().reset_index(level=[0,1], drop=True)

            # Slope via linear regression (simple approach)
            def slope_func(x):
                if len(x) < 2: return np.nan
                t = np.arange(len(x))
                return np.polyfit(t, x, 1)[0]
            df[f"{v}_roll{w}h_slope"] = roll.apply(slope_func, raw=False).reset_index(level=[0,1], drop=True)

            # AUC (cumulative sum * delta time)
            df[f"{v}_roll{w}h_auc"] = roll.apply(lambda x: np.nansum(x), raw=False).reset_index(level=[0,1], drop=True)

    # Drop temporary numeric timestamp
    df = df.drop(columns=['charttime_numeric'])
    return df