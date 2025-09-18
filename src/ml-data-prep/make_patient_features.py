'''

make_patient_features.py

Generates patient-level ML features from news2_scores.csv:
- Aggregate vitals per patient timeline (median, mean, min, max per vital).
- Performs patient-specific median imputation.
- Computes % missingness per vital.
- Encodes escalation/risk labels into numeric summary stats.

Output: news2_features_patient.csv (compact, one row per patient, ML-ready summary).
'''
# ------------------------------
# Imports
# ------------------------------
import pandas as pd # main library for working with tabular data (csv → df)
import numpy as np # for numerical operations (e.g. means, medians, handling arrays)
from pathlib import Path # handling file paths in a cleaner, cross-platform way (e.g. Path("data/file.csv") instead of raw strings)

# ------------------------------
# Config: file paths
# ------------------------------
DATA_DIR_INPUT = Path("../../data/interim-data")         
DATA_DIR_OUTPUT = Path("../../data/processed-data")

INPUT_FILE = DATA_DIR_INPUT / "news2_scores.csv"
OUTPUT_FILE = DATA_DIR_OUTPUT / "news2_features_patient.csv"

# ------------------------------
# Step 1: Load & Sort
# ------------------------------
def load_and_sort_data(input_file: Path) -> pd.DataFrame:
    # Load CSV
    df = pd.read_csv(input_file)

    # Ensure charttime is converted to datetime
    df['charttime'] = pd.to_datetime(df['charttime'])

    # Sort by patient, stay and time
    df = df.sort_values(by=['subject_id', 'stay_id', 'charttime']).reset_index(drop=True)

    return df

if __name__ == "__main__":
    df = load_and_sort_data(INPUT_FILE)
    print("Data loaded and sorted. Sample:")
    print(df.head())

# -------------------------------------
# Step 2: Aggregate vitals per patient
# -------------------------------------
# For each patient (subject_id), compute summary statistics of their vitals (median, mean, min, max)
# This produces one row per patient with columns like spo2_mean, hr_max, etc.
def aggregate_patient_vitals(df: pd.DataFrame) -> pd.DataFrame:
    # Columns we want to summarise for each patient
    vitals = [
        "respiratory_rate", "spo2", "supplemental_o2",
        "temperature", "systolic_bp", "heart_rate",
        "level_of_consciousness", "co2_retainer"
    ]

    # Define which statistics to compute for each vital
    agg_funcs = ["median", "mean", "min", "max"]

    # Group by patient ID and apply stats
    # .groupby("subject_id") → collects all rows of vitals that belong to each patient
    # [vitals] → only look at the vital sign columns.
	# .agg(agg_funcs) → apply median, mean, min, max to every vital for that patient.
    df_patient = df.groupby("subject_id")[vitals].agg(agg_funcs)

    # Flatten the MultiIndex columns → (vital, stat) → vital_stat
    # After aggregation pandas gives column names like ("spo2", "mean"), ("spo2", "min"), ("hr", "max") which is multiindex (two-level column names)
    # Columns are flattened into single strings
    df_patient.columns = ["_".join(col) for col in df_patient.columns]

    # Reset index so subject_id becomes a normal column again
    # .groupby() makes subject_id the index, reset it back into a normal column so we have a clean table.
    df_patient = df_patient.reset_index()

    return df_patient