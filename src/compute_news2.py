# --------------------------------------------------------------------
# NEWS2 Scoring Script
# Input: news2_vitals_with_co2.csv (long format)
# Output: news2_scores.csv (wide format with NEWS2 scores)
# --------------------------------------------------------------------

import pandas as pd

# --------------------------------------------------------------------
# 1. Define NEWS2 thresholds for each vital
vital_thresholds = {
    "respiratory_rate": [
        (None, 8, 3),
        (9, 11, 1),
        (12, 20, 0),
        (21, 24, 2),
        (25, None, 3)
    ],
    "spo2_thresholds_normal": [
        (None, 91, 3),
        (92, 93, 2),
        (94, 95, 1),
        (96, None, 0)
    ],
    "spo2_thresholds_hypercapnic": [
        (None, 83, 3),
        (84, 85, 2),
        (86, 87, 1),
        (88, 92, 0),
        (93, 94, 1),
        (95, 96, 2),
        (97, None, 3)
    ],
    "supplemental_o2": [(0, 0, 0), (1, 1, 2)],  # 0=no, 1=yes
    "temperature": [
        (None, 35.0, 3),
        (35.1, 36.0, 1),
        (36.1, 38.0, 0),
        (38.1, 39.0, 1),
        (39.1, None, 2)
    ],
    "systolic_bp": [
        (None, 90, 3),
        (91, 100, 2),
        (101, 110, 1),
        (111, 219, 0),
        (220, None, 3)
    ],
    "heart_rate": [
        (None, 40, 3),
        (41, 50, 1),
        (51, 90, 0),
        (91, 110, 1),
        (111, 130, 2),
        (131, None, 3)
    ],
    "level_of_consciousness": [(0, 0, 0), (1, 1, 3)]  # Alert=0, Not alert=3
}

# ------------------------------
# 2. Function to score individual vitals
def score_vital(vital_name, value, co2_retainer=False, fio2=None, gcs=None):
    """Return NEWS2 score for a single vital."""
    if pd.isna(value):
        return 0
    
    if vital_name == "spo2":
        thresholds = vital_thresholds["spo2_thresholds_hypercapnic"] if co2_retainer else vital_thresholds["spo2_thresholds_normal"]
        for low, high, score in thresholds:
            if (low is None or value >= low) and (high is None or value <= high):
                return score

    if vital_name == "supplemental_o2":
        flag = 0 if fio2 is not None and fio2 <= 0.21 else 1
        for low, high, score in vital_thresholds["supplemental_o2"]:
            if low <= flag <= high:
                return score

    if vital_name == "level_of_consciousness":
        flag = 0 if gcs == 15 else 1
        for low, high, score in vital_thresholds["level_of_consciousness"]:
            if low <= flag <= high:
                return score

    if vital_name in vital_thresholds:
        for low, high, score in vital_thresholds[vital_name]:
            if (low is None or value >= low) and (high is None or value <= high):
                return score

    return 0

# ------------------------------
# 3. Load long-format CSV
df = pd.read_csv("data/news2_vitals_with_co2.csv")

# ------------------------------
# 4. Standardize labels to NEWS2 names
label_mapping = {
    "Temperature": "temperature",
    "O2 saturation pulseoxymetry": "spo2",
    "Respiratory Rate": "respiratory_rate",
    "Heart Rate": "heart_rate",
    "Arterial Blood Pressure systolic": "systolic_bp",
    "Inspired O2 Fraction": "supplemental_o2"
}
for original_label, new_label in label_mapping.items():
    df.loc[df['label'].str.contains(original_label, case=False, na=False), 'label'] = new_label

# Temperature conversion F → C if needed
temp_rows = df['label'] == "temperature"
df.loc[temp_rows, 'valuenum'] = (df.loc[temp_rows, 'valuenum'] - 32) * 5/9

# Supplemental O2 from FiO2
fio2_rows = df['label'] == "supplemental_o2"
df.loc[fio2_rows, 'fio2_fraction'] = df.loc[fio2_rows, 'valuenum'] / 100
df.loc[fio2_rows, 'supplemental_o2'] = df.loc[fio2_rows, 'fio2_fraction'].apply(lambda x: 1 if x > 0.21 else 0)

# ------------------------------
# 5. Process GCS → raw components + total + level of consciousness
gcs_rows = df['label'].str.contains("GCS")
gcs_df = df[gcs_rows].pivot_table(
    index=['subject_id', 'stay_id', 'charttime'],
    columns='label',
    values='valuenum'
).reset_index()
for comp in ['GCS - Eye Opening','GCS - Verbal Response','GCS - Motor Response']:
    if comp not in gcs_df.columns:
        gcs_df[comp] = pd.NA
gcs_df['gcs_total'] = gcs_df[['GCS - Eye Opening','GCS - Verbal Response','GCS - Motor Response']].sum(axis=1, min_count=1)
gcs_df['level_of_consciousness'] = gcs_df['gcs_total'].apply(lambda x: 0 if x==15 else 1)

# ------------------------------
# 6. CO₂ retainer
co2_df = df[['subject_id','stay_id','charttime','co2_retainer']].drop_duplicates()

# ------------------------------
# 7. Pivot vitals → wide format (NEWS2 vitals)
expected_vitals = ["respiratory_rate","spo2","supplemental_o2",
                   "temperature","systolic_bp","heart_rate"]

# Pivot main vitals (valuenum)
wide_df = df.pivot_table(
    index=['subject_id','stay_id','charttime'],
    columns='label',
    values='valuenum'
).reset_index()

# Ensure all expected vitals exist
for col in expected_vitals:
    if col not in wide_df.columns:
        wide_df[col] = pd.NA

# ------------------------------
# 7a. Merge GCS totals and level_of_consciousness only
gcs_rows = df['label'].str.contains("GCS")
gcs_df = df[gcs_rows].pivot_table(
    index=['subject_id','stay_id','charttime'],
    columns='label',
    values='valuenum'
).reset_index()

# Compute total and consciousness
gcs_df['gcs_total'] = gcs_df[['GCS - Eye Opening','GCS - Verbal Response','GCS - Motor Response']].sum(axis=1, min_count=1)
gcs_df['level_of_consciousness'] = gcs_df['gcs_total'].apply(lambda x: 0 if x==15 else 1)

# Merge only the computed columns to avoid duplicates
wide_df = wide_df.merge(
    gcs_df[['subject_id','stay_id','charttime','gcs_total','level_of_consciousness']],
    on=['subject_id','stay_id','charttime'],
    how='left'
)

# ------------------------------
# 7b. Merge CO₂ retainer
co2_df = df[['subject_id','stay_id','charttime','co2_retainer']].drop_duplicates()
wide_df = wide_df.merge(co2_df, on=['subject_id','stay_id','charttime'], how='left')

# ------------------------------
# 7c. Merge supplemental O2 only if not already present
if 'supplemental_o2' not in wide_df.columns:
    supp_o2_df = df.loc[fio2_rows, ['subject_id','stay_id','charttime','supplemental_o2']].drop_duplicates()
    wide_df = wide_df.merge(supp_o2_df, on=['subject_id','stay_id','charttime'], how='left')

# Fill missing supplemental_o2 with 0 (Room air)
wide_df['supplemental_o2'] = wide_df['supplemental_o2'].fillna(0)

# ------------------------------
# 8. Human-readable columns
wide_df['consciousness_label'] = wide_df['level_of_consciousness'].apply(
    lambda x: "Alert" if x==0 else 
              "New-onset confusion (or disorientation/agitation), responds to voice, responds to pain, or unresponsive"
)
wide_df['co2_retainer_label'] = wide_df['co2_retainer'].apply(lambda x: "Yes" if x else "No")
wide_df['supplemental_o2_label'] = wide_df['supplemental_o2'].apply(lambda x: "Supplemental O₂" if x==1 else "Room air")

# ------------------------------
# 8. Human-readable columns
# Consciousness label
if 'level_of_consciousness' not in wide_df.columns:
    wide_df['level_of_consciousness'] = 0
wide_df['consciousness_label'] = wide_df['level_of_consciousness'].apply(
    lambda x: "Alert" if x == 0 else
              "New-onset confusion (or disorientation/agitation), responds to voice, responds to pain, or unresponsive"
)

# CO₂ retainer label
if 'co2_retainer' not in wide_df.columns:
    wide_df['co2_retainer'] = False
wide_df['co2_retainer_label'] = wide_df['co2_retainer'].apply(lambda x: "Yes" if x else "No")

# Supplemental O₂ label
wide_df['supplemental_o2_label'] = wide_df['supplemental_o2'].apply(
    lambda x: "Supplemental O₂" if x == 1 else "Room air"
)

# ------------------------------
# 9. Compute NEWS2 scores per row
def compute_news2(row):
    scores = {}
    total = 0
    for vital in ["respiratory_rate","spo2","supplemental_o2",
                  "temperature","systolic_bp","heart_rate","level_of_consciousness"]:
        scores[vital] = score_vital(vital,
                                    row.get(vital, pd.NA),
                                    co2_retainer=row.get("co2_retainer", False),
                                    fio2=row.get("fio2_fraction", None),
                                    gcs=row.get("gcs_total", None))
        total += scores[vital]

    # Interpret risk
    if total == 0:
        risk = "Low"; freq="Every 12 hrs"; resp="Nurse assessment"
    elif (1<=total<=4) and (3 not in scores.values()):
        risk="Low"; freq="Every 4–6 hrs"; resp="Nurse assessment"
    elif (3 in scores.values()) and (1<=total<=4):
        risk="Low-Medium"; freq="Every hr"; resp="Urgent doctor review"
    elif 5<=total<=6:
        risk="Medium"; freq="Every hr"; resp="Urgent doctor review"
    else:
        risk="High"; freq="Continuous"; resp="Critical care team"

    return pd.Series([total, risk, freq, resp])

wide_df[["news2_score","risk","monitoring_freq","response"]] = wide_df.apply(compute_news2, axis=1)

# ------------------------------
# 10. Save main CSV
wide_df.to_csv("news2_scores.csv", index=False)
print("Saved news2_scores.csv with all timestamps and raw vitals")

# ------------------------------
# 11. Enhanced sanity checks
wide_df['missing_vitals'] = wide_df[["respiratory_rate","spo2","supplemental_o2",
                                     "temperature","systolic_bp","heart_rate",
                                     "level_of_consciousness"]].isna().sum(axis=1)
print("Example rows with missing vitals:")
print(wide_df[wide_df['missing_vitals']>0].head(10))

# Check NEWS2 score ranges
out_of_range = wide_df[wide_df['news2_score']>20]
print("Any NEWS2 scores >20 (should be none):")
print(out_of_range)

# Summarize number of scores per patient
scores_per_patient = wide_df.groupby('subject_id').size().reset_index(name='num_scores')
print("Number of NEWS2 scores per patient:")
print(scores_per_patient.head(10))

# ------------------------------
# 12. Optional per-patient summary CSV
patient_summary = wide_df.groupby('subject_id').agg(
    min_news2_score=('news2_score','min'),
    max_news2_score=('news2_score','max'),
    mean_news2_score=('news2_score','mean'),
    median_news2_score=('news2_score','median'),
    total_records=('news2_score','count')
).reset_index()
patient_summary.to_csv("news2_patient_summary.csv", index=False)
print("Saved per-patient summary CSV: news2_patient_summary.csv")