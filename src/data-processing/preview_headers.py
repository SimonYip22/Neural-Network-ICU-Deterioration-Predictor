# --------------------------------------------------------------------
# Previewing headers of various CSV files in the data directory so we have correct header titles for use in extract_news2_vitals.py
# --------------------------------------------------------------------

import pandas as pd
import os # gives access to operating system functions, like checking if a file exists before trying to read it.

# List of files to preview (adjust paths if needed)
files = [
    "data/raw-data/demo_subject_id.csv",
    "data/raw-data/hosp/admissions.csv",
    "data/raw-data/hosp/patients.csv",
    "data/raw-data/icu/chartevents.csv",
    "data/raw-data/icu/d_items.csv"
]

for f in files:
    if os.path.exists(f):
        df = pd.read_csv(f, nrows=5)  # Read only first 5 rows to save time
                                      # Reads the CSV into a pandas DataFrame (df)
        print(f"\nHeaders for {f}:")
        print(df.columns.tolist()) # gives all column names (headers) of CSV, converted from pandas index object into a normal python list
    else:
        print(f"\nFile not found: {f}") # if doesn't exist, print this message
        