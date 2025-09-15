# helper/inspection script to check for CO2 retainers in chartevents
# extract_news2_vitals.py creates co2_retainer_details.csv, but this script independently verifies the logic and prints how many patients qualify

import pandas as pd

# File paths
CHARTEVENTS_PATH = "data/raw-data/icu/chartevents.csv"
DITEMS_PATH = "data/raw-data/icu/d_items.csv"

# Load d_items to filter out only ABG related itemids
d_items = pd.read_csv(DITEMS_PATH)
abg_items = d_items[d_items['label'].str.lower().str.contains("paco2|ph")]

# Load chartevents
# Ensures valuenum = numeric and charttime = datetime objects.
chartevents = pd.read_csv(CHARTEVENTS_PATH)
chartevents['valuenum'] = pd.to_numeric(chartevents['valuenum'], errors='coerce')
chartevents['charttime'] = pd.to_datetime(chartevents['charttime'], errors='coerce')

# Filter ABG rows to keep only the rows with PaCO2 or pH itemids
abg_rows = chartevents[chartevents['itemid'].isin(abg_items['itemid'])].copy()

# Map itemid -> label
itemid_to_label = dict(zip(d_items['itemid'], d_items['label']))
abg_rows['label'] = abg_rows['itemid'].map(itemid_to_label)

# Store retainers
co2_retainers = {}

# Loop through each patient
for subject_id, group in abg_rows.groupby('subject_id'):
    paco2_df = group[group['label'].str.lower().str.contains('paco2')].sort_values('charttime')
    ph_df = group[group['label'].str.lower().str.contains('ph')].sort_values('charttime')
    # Loop through every PaCO2 row for this patient and treat it as its own dictionary
    for _, pc_row in paco2_df.iterrows():
        # For each PaCO₂ measurement, find pH readings that occurred within 1 hour of that PaCO₂.
        ph_candidates = ph_df[abs(ph_df['charttime'] - pc_row['charttime']) <= pd.Timedelta(hours=1)]
        for _, ph_row in ph_candidates.iterrows():
            # If pH is normal and PaCO₂ > 45 → patient is flagged as CO₂ retainer.
            if 7.35 <= ph_row['valuenum'] <= 7.45 and pc_row['valuenum'] > 45:
                # Store result in dictionary (values + timestamps)
                co2_retainers[subject_id] = {
                    "paco2": pc_row['valuenum'],
                    "ph": ph_row['valuenum'],
                    "charttime_paco2": pc_row['charttime'],
                    "charttime_ph": ph_row['charttime']
                }
                break
        if subject_id in co2_retainers:
            break

print(f"✅ Total CO2 retainers found: {len(co2_retainers)}")
if co2_retainers:
    print("Example entries:")
    for k, v in list(co2_retainers.items())[:5]:
        print(k, v)