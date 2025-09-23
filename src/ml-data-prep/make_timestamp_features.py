"""
make_timestamp_features.py

Title: Generate Timestamp-Level ML Features From news2_scores.csv

Summary:
- Handles missing values (LOCF + missingness flags)
- Create carried-forward flags
- Computes rolling windows (1h, 4h, 24h)
- Computes time since last observation per vital
- Encodes escalation/risk labels into numeric format

Output:
- news2_features_timestamp.csv (ML-ready)
"""

# ------------------------------
# Imports
# ------------------------------
import pandas as pd       # Provides DataFrame structures and tools to load, manipulate, and analyse tabular data (CSVs).
import numpy as np        # Provides numerical operations, arrays, and functions (e.g., mean, std, linear regression for slope calculations).
from pathlib import Path  # Provides an easy, platform-independent way to handle file paths (used to define input/output CSV paths).
from numpy import trapezoid   # Imports the trapezoidal integration function to compute time-aware area under the curve (AUC) for rolling features.

# ------------------------------
# Config: file paths
# ------------------------------
DATA_DIR_INPUT = Path("../../data/interim-data")       # where the input lives
DATA_DIR_OUTPUT = Path("../../data/processed-data")    # where the output should go

INPUT_FILE = DATA_DIR_INPUT / "news2_scores.csv"
OUTPUT_FILE = DATA_DIR_OUTPUT / "news2_features_timestamp.csv"

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

# -------------------------------------
# Step 5: Compute rolling window features
# -------------------------------------
# Number of vitals = 5 (respiratory_rate, spo2, temperature, systolic_bp, heart_rate)
# Number of windows = 3 (1h, 4h, 24h)
# Number of stats per window = 6 (mean, min, max, std, slope, AUC)
# 5 vitals x 3 windows x 6 stats = 90 new feature columns per row

# NumPy is used for numerical operations like slope and AUC calculations
import numpy as np
from numpy import trapz

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    # These 5 vitals will have rolling windows computed (numeric ones only)
    vitals = ["respiratory_rate", "spo2", "temperature", "systolic_bp", "heart_rate"]
    # Time window sizes in hours (3 total)
    windows = [1, 4, 24] 

    # Loop through every vital and window size
    for v in vitals:
        for w in windows:
            # Create rolling object per patient stay and vital 
            # groupby → keeps each patient’s admission separate
            # .rolling(f"{w}H", on='charttime') → creates a lookback time window of size w hours (e.g. for 1H, at 10:00 it looks back at 9:00-10:00).
            # on='charttime' → tells pandas to use charttime as the time reference.
            # min_periods=1 → if only 1 point exists in that window, still return something (itself, otherwise it’d be NaN).
            roll = df.groupby(['subject_id', 'stay_id']).rolling(
                f"{w}h", on='charttime', min_periods=1
            )[v] # rolling object for df[v] 

            # Mean, min, max → capture magnitude.
            # Std → capture variability.
            # Slope → capture trend/direction.
            # AUC → capture cumulative exposure/risk over time.

            # Compute stats and add them as new columns to df
            # .reset_index(drop=True) → flattens the grouped index back so it lines up with the original dataframe (removes the extra groupby index).
            df[f"{v}_roll{w}h_mean"] = (roll.mean().reset_index(drop=True))
            df[f"{v}_roll{w}h_min"] = (roll.min().reset_index(drop=True))
            df[f"{v}_roll{w}h_max"] = (roll.max().reset_index(drop=True))
            df[f"{v}_roll{w}h_std"] = (roll.std().reset_index(drop=True))

            # Time-aware slope calculation reflects rate of change over real time, not “per row.”
            def slope_func(x):
                if len(x) < 2: 
                    return np.nan # Need at least 2 points to fit a line
                t = x.index.get_level_values("charttime").astype("int64") / 3600e9  # hours
                y = x.values # actual vital measurements 
                # Fit line y = a*t + b, return slope (a) 
                return np.polyfit(t, y, 1)[0] # fits a straight line through the points (t, y) in the form y = a*t + b, return slope a
            df[f"{v}_roll{w}h_slope"] = roll.apply(slope_func, raw=False).reset_index(drop=True) # new column added to df

            # Time-aware AUC captures cumulative exposure/risk over time.
            # Instead of summing values, we integrate over time with the trapezoidal rule
            def auc_func(x):
                if len(x) < 2: 
                    return np.nan
                t = x.index.get_level_values("charttime").astype("int64") / 3600e9  # hours
                y = x.values
                return np.trapezoid(y, t)  # trapezoidal integration of y over time t
            
            df[f"{v}_roll{w}h_auc"] = roll.apply(auc_func, raw=False).reset_index(drop=True)

    return df # return the new df

# -----------------------------------------------
# Step 6: Time Since Last Observation (staleness)
# -----------------------------------------------
# Computes how much time has passed since the previous observation for each vital.
# Helps models know if a reading is fresh or old
# Patients with rapidly changing vitals may have shorter intervals between measurements.
def add_time_since_last_obs(df: pd.DataFrame) -> pd.DataFrame:
    vitals = [
        "respiratory_rate", "spo2", "supplemental_o2",
        "temperature", "systolic_bp", "heart_rate",
        "level_of_consciousness", "co2_retainer"
    ]
    
    # Sort df by patient, stay, and time
    df = df.sort_values(by=['subject_id', 'stay_id', 'charttime'])
    
    # Loop through each vital
    for v in vitals:
        # Compute time difference for each patient stay and store as new column
        df[f"{v}_time_since_last_obs"] = (
            # Groups charttime column by patient and stay
            df.groupby(['subject_id', 'stay_id'])['charttime']
              .diff()  # computes difference between consecutive charttimes, if no previous row, returns NaN
              .dt.total_seconds() / 3600  # convert seconds to hours
        )
        
        # First row per patient/stay will have NaN → fill with 0 instead
        df[f"{v}_time_since_last_obs"] = df[f"{v}_time_since_last_obs"].fillna(0)
        
    return df

# -------------------------------------
# Step 7: Encode risk/escalation labels
# -------------------------------------
# Converts textual escalation/risk labels into numeric ordinal values.
# Model cannot operate on text directly, only with numeric outputs
def encode_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    risk_map = {
        "Low": 0,
        "Low-Medium": 1,
        "Medium": 2,
        "High": 3
    }
    
    # Create a new numeric column, by selecting risk column from df
    # .map() method applies a dictionary to every value in a series (column)
    # The original risk column stays intact, so we still have the textual labels if needed for reference.
    df['risk_numeric'] = df['risk'].map(risk_map)
    
    return df

# -----------------------------------
# Step 8: Save the final DataFrame
# -----------------------------------
# Save the ML-ready features
def save_features(df: pd.DataFrame, output_file: Path):
    df.to_csv(output_file, index=False)
    print(f"ML-ready features saved to {output_file}")

# ---------------
# Main pipeline
# ---------------
def main():
    df = load_and_sort_data(INPUT_FILE)
    df = add_missingness_flags(df)
    df = apply_locf(df)
    df = add_carried_forward_flags(df)
    df = add_rolling_features(df)
    df = add_time_since_last_obs(df)
    df = encode_risk_labels(df)
    save_features(df, OUTPUT_FILE)
    print("Pipeline complete. Sample:")
    print(df.head(10))

if __name__ == "__main__":
    main()