import pandas as pd
import os

# List of files to preview (adjust paths if needed)
files = [
    "data/demo_subject_id.csv",
    "data/hosp/admissions.csv",
    "data/hosp/patients.csv",
    "data/icu/chartevents.csv",
    "data/icu/d_items.csv"
]

for f in files:
    if os.path.exists(f):
        df = pd.read_csv(f, nrows=5)  # Read only first 5 rows to save time
        print(f"\nHeaders for {f}:")
        print(df.columns.tolist())
    else:
        print(f"\nFile not found: {f}")
        