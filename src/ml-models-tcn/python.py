import pandas as pd
import numpy as np
from pathlib import Path  # File/directory paths in a clean, OS-independent way

SCRIPT_DIR = Path(__file__).resolve().parent
# Input dir
DATA_PATH = SCRIPT_DIR.parent.parent / "data/processed-data/news2_features_timestamp.csv"

# load same df the script uses (adjust path if needed)
df = pd.read_csv(DATA_PATH, low_memory=False)

# candidate/feature cols: use same list your script uses (example below)
ignore = {"subject_id","stay_id","charttime","risk_numeric"}  # adjust to match your script
candidate_cols = [c for c in df.columns if c not in ignore]

non_numeric = [c for c in candidate_cols if not pd.api.types.is_numeric_dtype(df[c])]
print("NON-NUMERIC candidate columns:", non_numeric)

# show problematic values (first 10 uniques) for each non-numeric
for c in non_numeric:
    print("\n===", c, "dtype:", df[c].dtype)
    print(df[c].dropna().unique()[:10])