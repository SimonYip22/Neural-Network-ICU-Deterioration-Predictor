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

# -------------------------------------------
# Step 3: Patient-specific Median imputation
# -------------------------------------------
# If a patients vital was measured at least once, step 2 would already have computed values for median, mean, min and max, so step 3 wouldn't do anything.
# If a patient never had a vital recorded at all, their step 2 computes would all be NaN, so fallback to the population median (last resort) as a weak, neutral placeholder.
# It’s the only practical way to make sure every patient has features for that vital, to allow the ML model to train properly.
# Data leakage = when information from other patients “leaks” into a patient’s feature set, artificially boosting model performance but destroying real-world validity.
def impute_missing_values(df_patient: pd.DataFrame) -> pd.DataFrame:
    # Preserve the original aggregated stats (with NaNs intact) created in step 2. 
    # Create a copy of df to use for step 3 that will become safe for ML input.
    df_imputed = df_patient.copy()
    
    # Set of vitals we care about 
    vitals = [
        "respiratory_rate", "spo2", "supplemental_o2",
        "temperature", "systolic_bp", "heart_rate",
        "level_of_consciousness", "co2_retainer"
    ]

    # Loop over each vital 
    for vital in vitals:
        # Identify the median column for this vital
        median_col = f"{vital}_median"
        
        # Compute fallback population median of vital (across all patients)
        population_median = df_imputed[median_col].median()
        
        # For each column belonging to this vital (mean, min, max, etc.)
        # Imputation to each column thats in the list of columns for a vital sign
        for col in [c for c in df_imputed.columns if c.startswith(vital)]:
            # Fill with patient-specific median first (edge cases)
            df_imputed[col] = df_imputed[col].fillna(df_imputed[median_col])
            
            # Then fallback to population median (NaN everywhere)
            df_imputed[col] = df_imputed[col].fillna(population_median)
    
    return df_imputed # return the clone df, fully filled and ML ready

# ---------------------------------------
# Step 4: Compute % missingness per vital
# ---------------------------------------
# Use pre-imputation df (original raw per-row data) to determine for each patient, what fraction of their original rows has a missing value for each vital
# Missingness itself carries signal (e.g., SpO₂ only measured in deteriorating patients).
# Then we attach those percentages to the imputed patient-level table (step 3)
def compute_missingness(df: pd.DataFrame, df_imputed: pd.DataFrame) -> pd.DataFrame:
    vitals = [
        "respiratory_rate", "spo2", "supplemental_o2",
        "temperature", "systolic_bp", "heart_rate",
        "level_of_consciousness", "co2_retainer"
    ]
    
    # Create new DataFrame: missingness % per patient
    missingness = (
        # group raw data by patient (becomes key), each patient becomes a group of rows, look only at vital columns
        df.groupby("subject_id")[vitals]
          # For each patient, count how often each column was NaN
          # .isna() gives True if missing, else False
          # .mean() converts those booleans to proportions (True=1, False=0)
          # values are now fractions of missingness
          .apply(lambda x: x.isna().mean())
          .reset_index() # makes subject_id a normal column again instead of the index.
    )
    
    # Rename columns in the new df to make them clear they are missing percentages
    missingness = missingness.rename(columns={v: f"{v}_missing_pct" for v in vitals})
    
    # Merge back into patient-level features from step 3, rename back to df_patient
    # on="subject_id" ensures we match the missingness values to the correct patient.
	# how="left" keeps all patients from the left DataFrame df_imputed (even if for some reason missingness wasn’t computed).
    # Take all rows from df_imputed, and bring in the matching rows from missingness.
    df_patient = df_imputed.merge(missingness, on="subject_id", how="left")
    
    return df_patient # df_patient now contains aggregated stats + imputed values + missingness percentages, ready for Step 5.

# ---------------------------------------------------
# Step 5: Encode Risk Labels & Compute Summary Stats
# ---------------------------------------------------
# Convert categorical risk labels into numeric values and computes summary statistics per patient for trajectory.
# Summary stats: max risk, median risk, % time at high risk.
def encode_risk_and_summarise(df: pd.DataFrame, df_patient: pd.DataFrame) -> pd.DataFrame:
    # Map risk labels to numbers 
    risk_map = {"Low": 0, "Low-Medium": 1, "Medium": 2, "High": 3}
    # create new df column based on the risk labels for each timestamp
    # .map(risk_map) converts each string label in the df to a numeric value 
    # Numeric encoding allows ML models to process risk as a continuous/ordinal feature instead of text.
    df["risk_encoded"] = df["risk"].map(risk_map) 
    
    # Group all timestamps of a patient together
    # ["risk_encoded"] selects the column of interest.
    # .agg() computes three summary statistics for each patient
    risk_summary = df.groupby("subject_id")["risk_encoded"].agg(
        max_risk="max", # create new column using "max" on risk_encoded
        median_risk="median", # create new column using "median" on risk_encoded
        pct_time_high=lambda x: (x==3).mean()  # create new column with the fraction of timestamps where the patient was at High risk (encoded as 3).
    ).reset_index() # turns subject_id back into a regular column instead of an index.
    
    # Merge into patient-level table
    # on="subject_id" ensures each patient gets the correct risk summary.
	# how="left" keeps all patients in df_patient even if, for some reason, risk_summary is missing a patient.
    df_patient = df_patient.merge(risk_summary, on="subject_id", how="left")
    
    return df_patient # outputs updated patient-level df with risk stats added

# ---------------------------------------
# Step 6: Save patient-level features
# ---------------------------------------
# CSV is ML-ready, one row per patient, including: median, mean, min, max per vital, imputing missing values, % missing per vital, risk summary stats (max, median, % time at high risk)
def save_patient_features(df_patient: pd.DataFrame, output_file: Path):
    # index=False → when DataFrame is saved to csv, the index column is removed, so only the real data columns are contained in the csv
    df_patient.to_csv(output_file, index=False)
    print(f"Patient-level features saved to {output_file}")

# ---------------
# Main pipeline
# ---------------
def main():
    df_raw = load_and_sort_data(INPUT_FILE)
    df_patient_agg = aggregate_patient_vitals(df_raw)
    df_patient_imputed = impute_missing_values(df_patient_agg)
    df_patient_with_missingness = compute_missingness(df_raw, df_patient_imputed)
    df_patient_final = encode_risk_and_summarise(df_raw, df_patient_with_missingness)
    
    # Ensure output directory (processed-data) exists and save
    DATA_DIR_OUTPUT.mkdir(parents=True, exist_ok=True)
    save_patient_features(df_patient_final, OUTPUT_FILE)

    print("Pipeline complete. Sample:")
    print(df_patient_final.head(10))

if __name__ == "__main__":
    main()