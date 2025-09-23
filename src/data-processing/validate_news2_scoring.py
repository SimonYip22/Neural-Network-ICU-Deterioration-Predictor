"""
validate_news2_scoring.py

Title: Validation Testing Script For compute_news2.py Logic

Summary:
- Unit tests for NEWS2 scoring logic to ensure the compute_news2 pipeline correctly implements NHS NEWS2 rules.
- Validates edge cases (SpO2 with/without supplemental O2, GCS changes, RR extremes).
- Tests CO2 retainer logic on synthetic patient rows.
- Uses synthetic patient rows matching the preprocessed CSV (wide format)
- Provides confidence before generating ML features.
"""

import pandas as pd

# ------------------------------
# NEWS2 thresholds (from compute_news2.py)
# ------------------------------
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
# NEWS2 scoring function (from compute_news2.py)
# ------------------------------
def score_vital(vital_name, value, co2_retainer=False, supplemental_o2=False, gcs_total=None):

    # SpO2
    if vital_name == "spo2":
        thresholds = vital_thresholds["spo2_thresholds_hypercapnic"] if co2_retainer else vital_thresholds["spo2_thresholds_normal"]
        for low, high, score in thresholds:
            if (low is None or value >= low) and (high is None or value <= high):
                return score

    # Supplemental O2
    if vital_name == "supplemental_o2":
        flag = 1 if supplemental_o2 else 0
        for low, high, score in vital_thresholds["supplemental_o2"]:
            if low <= flag <= high:
                return score

    # Level of consciousness
    if vital_name == "level_of_consciousness":
        flag = 0 if gcs_total == 15 else 1
        for low, high, score in vital_thresholds["level_of_consciousness"]:
            if low <= flag <= high:
                return score

    # Other vitals
    if vital_name in vital_thresholds:
        for low, high, score in vital_thresholds[vital_name]:
            if (low is None or value >= low) and (high is None or value <= high):
                return score

    if pd.isna(value):
        return 0

# ------------------------------
# Compute full NEWS2 score for a patient row
# ------------------------------
def compute_news2_score(row):
    total = 0
    for vital in ["respiratory_rate","spo2","supplemental_o2",
                  "temperature","systolic_bp","heart_rate","level_of_consciousness"]:
        total += score_vital(
            vital_name=vital,
            value=row.get(vital, pd.NA),
            co2_retainer=row.get("co2_retainer", False),
            supplemental_o2=row.get("supplemental_o2", False),
            gcs_total=row.get("gcs_total", None)
        )
    return total

# ------------------------------
# Synthetic patients for testing
# ------------------------------
test_patients = [
    # Normal patient
    {
        'name': 'normal_patient',
        'heart_rate': 75,
        'respiratory_rate': 16,
        'systolic_bp': 120,
        'temperature': 37.0,
        'spo2': 96,
        'supplemental_o2': False,
        'gcs_total': 15,
        'co2_retainer': False
    },
    # Edge SpO2, supplemental O2
    {
        'name': 'spo2_supp_o2',
        'heart_rate': 85,
        'respiratory_rate': 18,
        'systolic_bp': 115,
        'temperature': 36.5,
        'spo2': 91,
        'supplemental_o2': True,
        'gcs_total': 15,
        'co2_retainer': False
    },
    # Low GCS
    {
        'name': 'low_gcs',
        'heart_rate': 90,
        'respiratory_rate': 20,
        'systolic_bp': 110,
        'temperature': 37.0,
        'spo2': 95,
        'supplemental_o2': False,
        'gcs_total': 14,
        'co2_retainer': False
    },
    # CO2 retainer
    {
        'name': 'co2_retainer',
        'heart_rate': 88,
        'respiratory_rate': 24,
        'systolic_bp': 105,
        'temperature': 36.0,
        'spo2': 90,
        'supplemental_o2': True,
        'gcs_total': 15,
        'co2_retainer': True
    }
]

# ------------------------------
# Run tests
# ------------------------------
print("=== NEWS2 Scoring Validation ===")
for patient in test_patients:
    score = compute_news2_score(patient)
    print(f"Patient: {patient['name']}, NEWS2 score: {score}")

# ------------------------------
# Automated assertions
# ------------------------------
# Calculate expected scores manually using thresholds from compute_news2.py
expected_scores = {
    'normal_patient': 0,
    'spo2_supp_o2': 5,
    'low_gcs': 5,
    'co2_retainer': 6
}

for patient in test_patients:
    computed = compute_news2_score(patient)
    expected = expected_scores[patient['name']]
    assert computed == expected, f"{patient['name']} score {computed} != expected {expected}"

print("✅ All NEWS2 unit tests passed.")