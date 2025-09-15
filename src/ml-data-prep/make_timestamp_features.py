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

# ------------------------------
# Step 2: Create missingness flags
# ------------------------------
def add_missingness_flags(df: pd.DataFrame) -> pd.DataFrame:
    vitals = [
        "respiratory_rate", "spo2", "supplemental_o2",
        "temperature", "systolic_bp", "heart_rate",
        "level_of_consciousness"
    ]
    for v in vitals:
        flag_col = f"{v}_missing"
        df[flag_col] = df[v].isna().astype(int)
    return df


