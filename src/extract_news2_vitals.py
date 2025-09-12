import pandas as pd

# ------------------------------
# 1. Define file paths
# ------------------------------
CHARTEVENTS_PATH = "data/icu/chartevents.csv"
DITEMS_PATH = "data/icu/d_items.csv"
ADMISSIONS_PATH = "data/hosp/admissions.csv"
PATIENTS_PATH = "data/hosp/patients.csv"
OUTPUT_PATH = "data/news2_vitals_with_co2.csv"       # CSV with True/False for CO2 retainer
DETAILS_PATH = "data/co2_retainer_details.csv"       # CSV with ABG details

# ------------------------------
# 2. Define NEWS2 mapping
# ------------------------------
NEWS2_ITEMIDS = {
    "heart_rate": [220045],
    "respiratory_rate": [220210],
    "systolic_bp": [220179],
    "temperature": [223761],
    "spo2": [220277],
    "supplemental_o2": [223835],
    "level_of_consciousness": [220739, 223900, 223901],
}

all_itemids = [i for sublist in NEWS2_ITEMIDS.values() for i in sublist]

# ------------------------------
# 3. Load d_items and chartevents
# ------------------------------
d_items = pd.read_csv(DITEMS_PATH)
chartevents = pd.read_csv(CHARTEVENTS_PATH)

# Merge to get labels
chartevents = chartevents.merge(d_items[['itemid', 'label']], on='itemid', how='left')

# ------------------------------
# 4. Extract NEWS2 vitals and ABGs
# ------------------------------
news2_vitals = chartevents[chartevents['itemid'].isin(all_itemids)].copy()

# Find ABG rows (PaCO2 or pH)
abg_rows = chartevents[chartevents['label'].str.lower().str.contains("paco2|ph")].copy()
abg_rows['charttime'] = pd.to_datetime(abg_rows['charttime'])

# ------------------------------
# 5. Compute CO2 retainer flag
# ------------------------------
co2_flag = {}
co2_details = []

for subject_id in abg_rows['subject_id'].unique():
    subject_abgs = abg_rows[abg_rows['subject_id'] == subject_id].sort_values('charttime')
    
    paco2_rows = subject_abgs[subject_abgs['label'].str.lower().str.contains("paco2")]
    ph_rows = subject_abgs[subject_abgs['label'].str.lower().str.contains("ph")]

    # Default to False
    co2_flag[subject_id] = False

    # Compare each PaCO2 to nearest pH by time
    for _, paco2 in paco2_rows.iterrows():
        # Find closest pH
        if not ph_rows.empty:
            ph_diff = (ph_rows['charttime'] - paco2['charttime']).abs()
            closest_ph_idx = ph_diff.idxmin()
            ph_val = ph_rows.loc[closest_ph_idx, 'valuenum']
            
            if paco2['valuenum'] > 45 and 7.35 <= ph_val <= 7.45:
                co2_flag[subject_id] = True
                co2_details.append({
                    'subject_id': subject_id,
                    'paco2_time': paco2['charttime'],
                    'paco2': paco2['valuenum'],
                    'ph_time': ph_rows.loc[closest_ph_idx, 'charttime'],
                    'ph': ph_val
                })
                break

# ------------------------------
# 6. Apply CO2 flag to NEWS2 vitals
# ------------------------------
news2_vitals['co2_retainer'] = news2_vitals['subject_id'].apply(lambda x: co2_flag.get(x, False))

# ------------------------------
# 7. Save outputs
# ------------------------------
news2_vitals.to_csv(OUTPUT_PATH, index=False)
print(f"✅ CSV saved to {OUTPUT_PATH}")

co2_details_df = pd.DataFrame(co2_details)
co2_details_df.to_csv(DETAILS_PATH, index=False)
print(f"✅ Detailed CO2 retainer info saved to {DETAILS_PATH}")