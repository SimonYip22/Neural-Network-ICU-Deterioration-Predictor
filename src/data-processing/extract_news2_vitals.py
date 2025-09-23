"""
extract_news2_vitals.py

Title: Raw CSV Data Extraction Script To Compute NEWS2

Summary:
- Extracting and formatting data from csv files to compute NEWS2 vitals and CO2 retainer flag
Outputs:
- news2_with_co2.csv (NEWS2 scores along with CO2 retainer column)
- co2_retainer_details.csv (details of all CO2 retainer patients seperated from NEWS2 scores)
"""

import pandas as pd

# ------------------------------
# 1. Define file paths
# ------------------------------
CHARTEVENTS_PATH = "data/raw-data/icu/chartevents.csv"
DITEMS_PATH = "data/raw-data/icu/d_items.csv"
ADMISSIONS_PATH = "data/raw-data/hosp/admissions.csv"
PATIENTS_PATH = "data/raw-data/hosp/patients.csv"
OUTPUT_PATH = "data/interim-data/news2_vitals_with_co2.csv"       # CSV with True/False for CO2 retainer
DETAILS_PATH = "data/interim-data/co2_retainer_details.csv"       # CSV with ABG details

# ------------------------------
# 2. Define NEWS2 mapping
# ------------------------------
# This dictionary tells pandas which IDs correspond to each vital sign in NEWS2.
NEWS2_ITEMIDS = {
    "heart_rate": [
        220045,   # Heart Rate (core)
        211       # Heart Rate (legacy)
    ],
    "respiratory_rate": [
        220210,   # Resp Rate (core)
        618       # Resp Rate (legacy)
    ],
    "systolic_bp": [
        220179,   # Arterial BP systolic
        220050,   # Non-invasive BP systolic
        225309    # Manual BP systolic
    ],
    "temperature": [
        223761,   # Temperature (C)
        678       # Temperature (F, sometimes used)
    ],
    "spo2": [
        220277,   # SpO2 (pulse ox)
        646       # SpO2 (legacy)
    ],
    "supplemental_o2": [
        223835    # FiO2 / Inspired O2
    ],
    "level_of_consciousness": [
        223900,   # GCS - Verbal
        223901,   # GCS - Motor
        220739    # GCS - Eye
    ],
}

# Nested list comprehension to flatten all itemids into a single list.
# This way we can easily filter chartevents with .isin(all_itemids) to grab all NEWS2 vitals at once.
all_itemids = [i for sublist in NEWS2_ITEMIDS.values() for i in sublist]

# ------------------------------
# 3. Load d_items and chartevents
# ------------------------------
# pandas builds a DataFrame with column names exactly matching the CSV headers
d_items = pd.read_csv(DITEMS_PATH) # metadata (what each itemid means)
chartevents = pd.read_csv(CHARTEVENTS_PATH) # actual time-stamped ICU measurements 

# Merge to add human-readable labels next to each row
# how='left' means “keep all rows in chartevents, add a label column if there’s a match in d_items, if no match exists leave it NaN”
chartevents = chartevents.merge(d_items[['itemid', 'label']], on='itemid', how='left')

# ------------------------------
# 4. Extract NEWS2 vitals and ABGs
# ------------------------------
# Subset filtered containing just the vitals we care about, we are keeping the rows where itemid is in our all_itemids list.
news2_vitals = chartevents[chartevents['itemid'].isin(all_itemids)].copy()

# Find ABG rows (PaCO2 or pH)
abg_rows = chartevents[chartevents['label'].str.lower().str.contains("paco2|ph")].copy()
# Convert charttime to datetime so we can compare times later.
abg_rows['charttime'] = pd.to_datetime(abg_rows['charttime'])

# ------------------------------
# 5. Compute CO2 retainer flag
# ------------------------------
co2_flag = {} # dictionary mapping each subject_id → True/False (CO₂ retainer or not).
co2_details = [] # list of records to save later with PaCO₂ and pH values.
# Loop per patient
for subject_id in abg_rows['subject_id'].unique():
    # Pull their ABG rows (PaCO₂ and pH), sort them by time → ensures we can match values in the right chronological order.
    subject_abgs = abg_rows[abg_rows['subject_id'] == subject_id].sort_values('charttime')
    # Separate PaCO₂ and pH, split rows into two groups
    paco2_rows = subject_abgs[subject_abgs['label'].str.lower().str.contains("paco2")]
    ph_rows = subject_abgs[subject_abgs['label'].str.lower().str.contains("ph")]

    # Default to False, assume patient isn't a CO2 retainer until proven otherwise.
    co2_flag[subject_id] = False

    # For each PaCO₂ reading, find the closest pH in time.
    for _, paco2 in paco2_rows.iterrows():
        # Find closest pH
        if not ph_rows.empty:
            ph_diff = (ph_rows['charttime'] - paco2['charttime']).abs() # difference tin time
            closest_ph_idx = ph_diff.idxmin() # .idxmin() gives the index of the closest pH in time.
            ph_val = ph_rows.loc[closest_ph_idx, 'valuenum'] # Pull its valuenum (the actual numeric pH).
            if paco2['valuenum'] > 45 and 7.35 <= ph_val <= 7.45:
                co2_flag[subject_id] = True # if meets criteria, set dictionary flag to True
                co2_details.append({
                    'subject_id': subject_id,
                    'paco2_time': paco2['charttime'],
                    'paco2': paco2['valuenum'],
                    'ph_time': ph_rows.loc[closest_ph_idx, 'charttime'],
                    'ph': ph_val
                }) # Append a record into co2_details
                break

# ------------------------------
# 6. Apply CO2 flag to NEWS2 vitals
# ------------------------------
# Add a co2_retainer column to the vitals dataframe.
# Each row now has a True/False based on whether that patient was flagged.
# apply(lambda x: ...) looks up each subject_id in the co2_flag dictionary, True if flagged, Flase if not.
news2_vitals['co2_retainer'] = news2_vitals['subject_id'].apply(lambda x: co2_flag.get(x, False))

# ------------------------------
# 7. Save outputs
# ------------------------------
# Pandas automatically uses the dictionary keys as column names.
news2_vitals.to_csv(OUTPUT_PATH, index=False)
print(f"✅ CSV saved to {OUTPUT_PATH}")
# Example columns: subject_id, stay_id, charttime, heart_rate, respiratory_rate, ..., co2_retainer

# Even if empty, there will still be headersin the columns 
co2_details_df = pd.DataFrame(co2_details, columns=["subject_id", "paco2_time", "paco2", "ph_time", "ph"])
co2_details_df.to_csv(DETAILS_PATH, index=False)
print(f"✅ Detailed CO2 retainer info saved to {DETAILS_PATH}")
# Example columns: subject_id, paco2_time, paco2, ph_time, ph