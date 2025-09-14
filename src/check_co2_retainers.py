# helper/inspection script to check for CO2 retainers in chartevents
# extract_news2_vitals.py creates co2_retainer_details.csv, but this script verifies the logic

import pandas as pd

# File paths
CHARTEVENTS_PATH = "data/icu/chartevents.csv"
DITEMS_PATH = "data/icu/d_items.csv"

# Load d_items to get ABG itemids
d_items = pd.read_csv(DITEMS_PATH)
abg_items = d_items[d_items['label'].str.lower().str.contains("paco2|ph")]

# Load chartevents
chartevents = pd.read_csv(CHARTEVENTS_PATH)
chartevents['valuenum'] = pd.to_numeric(chartevents['valuenum'], errors='coerce')
chartevents['charttime'] = pd.to_datetime(chartevents['charttime'], errors='coerce')

# Filter ABG rows
abg_rows = chartevents[chartevents['itemid'].isin(abg_items['itemid'])].copy()

# Map itemid -> label
itemid_to_label = dict(zip(d_items['itemid'], d_items['label']))
abg_rows['label'] = abg_rows['itemid'].map(itemid_to_label)

# Store retainers
co2_retainers = {}

for subject_id, group in abg_rows.groupby('subject_id'):
    paco2_df = group[group['label'].str.lower().str.contains('paco2')].sort_values('charttime')
    ph_df = group[group['label'].str.lower().str.contains('ph')].sort_values('charttime')
    
    for _, pc_row in paco2_df.iterrows():
        ph_candidates = ph_df[abs(ph_df['charttime'] - pc_row['charttime']) <= pd.Timedelta(hours=1)]
        for _, ph_row in ph_candidates.iterrows():
            if 7.35 <= ph_row['valuenum'] <= 7.45 and pc_row['valuenum'] > 45:
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