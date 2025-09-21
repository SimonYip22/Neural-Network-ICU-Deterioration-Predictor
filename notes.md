# Phase 1: Baseline NEWS2 Tracker

---

## Day 1: NEWS2 Data Extraction and Preliminary Scoring

### Pipeline Overview

```text
Raw CSVs (chartevents.csv, etc.)
        ↓
extract_news2_vitals.py
        ↓
news2_vitals.csv
        ↓
check_co2_retainers.py
        ↓
news2_vitals_with_co2.csv + co2_retainer_details.csv
        ↓
compute_news2.py
        ↓
Final NEWS2 scores per patient
```

### Goals
- Extract relevant vital signs from PhysioNet.org MIMIC-IV Clinical Database Demo synthetic dataset for NEWS2 scoring.  
- Identify and flag CO₂ retainers to ensure accurate oxygen scoring.  
- Implement basic NEWS2 scoring functions in Python.  
- Establish a pipeline that is extendable for future interoperability with real-world clinical data.

### What We Did
1. **Dataset Preparation**
   - Downloaded synthetic dataset `mimic-iv-clinical-database-demo-2.2.zip` and unzipped CSV files.  
   - Explored `chartevents.csv` and other relevant CSVs to identify required vitals.  

2. **Data Extraction**
   - Wrote `extract_news2_vitals.py` to extract NEWS2-relevant vitals from all CSVs.  
   - Used `preview_headers` to determine which columns to extract and standardize CSV headers.  
   - Generated `news2_vitals.csv`.

3. **CO₂ Retainer Identification**
   - Created `check_co2_retainers.py` to verify if any patients met CO₂ retainer criteria:
     - PaCO₂ > 45 mmHg  
     - pH between 7.35–7.45  
     - ABG measurements ±1 hour apart  
   - Updated `extract_news2_vitals.py` to include CO₂ retainer status.  
   - Generated:
     - `news2_vitals_with_co2.csv` – vitals with retainer flags  
     - `co2_retainer_details.csv` – patient-specific CO₂ retainer information  

4. **NEWS2 Scoring**
   - Implemented `compute_news2.py` with:
     - Dictionaries defining scoring thresholds for each vital  
     - Functions to compute individual vital scores  
     - Pandas used to process CSV and calculate total NEWS2 scores  

### Reflections
- **Challenges:**
  - Understanding GCS scoring and mapping three separate components to level of consciousness.  
  - Determining FiO₂ representation in dataset (0.21 vs. 21%).  
  - Determining temperature units
  - Grasping complex Python syntax and tuple-based threshold definitions.  
  - Integrating CO₂ retainer logic into NEWS2 oxygen scoring.

- **Solutions & Learnings:**
  - GCS scoring requires summing Eye, Verbal, and Motor responses per timestamp.  
  - FiO₂ can be identified via `Inspired O2 Fraction` in CSV and converted to binary supplemental O₂ indicator.  
  - Temperature was in Fahrenheit (°F) and so `compute_news2.py` includes conversion from °F to °C.
  - Tuples `(min, max, score)` provide flexible, readable threshold definitions for each vital.  
  - CO₂ retainer pipeline ensures accurate NEWS2 oxygen scoring now and for future datasets.  

### Issues Encountered
- Confusion around GCS mapping and timestamp alignment.  
- Initial uncertainty about FiO₂ and temperature units.  
- Need to verify CO₂ retainer thresholds and data format.  
- Feeling overwhelmed by the complexity of clinical data pipelines and Python functions.

### Lessons Learned
- Extracting and standardising clinical data is a critical and time-consuming first step.  
- Structuring data in CSVs with consistent headers simplifies downstream processing.  
- Python dictionaries and tuple-based thresholds are powerful for flexible clinical scoring functions.  
- Documenting assumptions (temperature units, FiO₂ thresholds) is essential for reproducibility.

### Future Interoperability Considerations
- Pipeline designed to support ingestion of FHIR-based EHR data for future integration.  
- Potential extension: map standardized FHIR resources to predictive EWS pipeline for real-world applicability.

### CO₂ Retainer Validation and NEWS2 Scoring Documentation
1. **Objective:** Identify CO₂ retainers to ensure correct oxygen scoring.  
2. **Methodology:**  
   - All ABG measurements in `chartevents.csv` examined.  
   - CO₂ retainer criteria applied: PaCO₂ > 45 mmHg with pH 7.35–7.45 ±1 hour.  
3. **Results:**  
   - No patients in current dataset met CO₂ retainer criteria.  
   - NEWS2 oxygen scoring applied standard thresholds for all patients.  
4. **Future-proofing:**  
   - CO₂ retainer thresholds remain documented in code.  
   - Future datasets will automatically flag and score retainers according to NEWS2 rules.

---

## Day 2: NEWS2 Pipeline Development

### Goals
- Finalise **Phase 1** of the NEWS2 scoring pipeline.
- Ensure robust extraction, computation, and output of NEWS2 scores from raw vital sign data.
- Handle missing data, standardize column names, and prevent errors caused by merging or absent measurements.
- Create clean, wide-format CSV outputs: `news2_scores.csv` (per-timestamp) and `news2_patient_summary.csv` (per-patient summary).

### What We Did
1. **Updated `extract_news2_vitals.py`:**
  - Included missing `systolic_bp` itemids.
  - Added alternate names for vitals to capture all relevant measurements.
  - Produced `news2_vitals_with_co2.csv` with columns:  
    `subject_id, hadm_id, stay_id, caregiver_id, charttime, storetime, itemid, value, valuenum, valueuom, warning, label, co2_retainer`.
2. **Updated `compute_news2.py`:**
  - Pivoted long-format vitals to wide format, ensuring all **expected NEWS2 vitals** (`respiratory_rate, spo2, supplemental_o2, temperature, systolic_bp, heart_rate`) exist as columns.
  - Safely merged **GCS components**, computing `gcs_total` and `level_of_consciousness`.
  - Safely merged **CO₂ retainer** information.
  - Fixed **supplemental O₂** issues:
    - Checked if column exists before filling.
    - Filled missing rows with `0` (Room air).
    - Only merged if not already present to prevent duplication.

3. **Handled duplicates and column conflicts:**
  - Avoided `_x` / `_y` suffixes by careful merge logic:
    - GCS merge only added `gcs_total` and `level_of_consciousness`.
    - Supplemental O₂ merged only if missing.
    - CO₂ retainer merge ensured no overlap.

4. **Added human-readable labels:**
  - `consciousness_label`, `co2_retainer_label`, `supplemental_o2_label`.
  - Ensured columns exist before applying transformations to prevent KeyErrors.
  - The redundancy exists to **ensure the script runs safely** even if:
    - `level_of_consciousness` is missing (no GCS rows for some patients)
    - `co2_retainer` is missing
    - `supplemental_o2` is missing
  - Purpose:
    - Guarantee idempotency
    - Prevent KeyErrors
    - Keep CSV outputs consistent and complete

5. **Computed NEWS2 scores per row:**
  - Applied scoring rules for each vital.
  - Calculated `news2_score`, `risk`, `monitoring_freq`, and `response`.
  - Validated that **no scores exceeded 20**.

6. **Created outputs:**
  - `news2_scores.csv` – full dataset with scores and all vital measurements.
  - `news2_patient_summary.csv` – per-patient summary with `min_news2_score, max_news2_score, mean_news2_score, median_news2_score, total_records`.

7. **Implemented defensive coding & sanity checks:**
  - Missing vitals counted per row (`missing_vitals` column).
  - All merges and transformations check column existence.
  - Default values (0 or False) used for missing data to maintain dataset integrity.

**Phase 1 gave us**:
- news2_scores.csv (per-timestamp scores).
- news2_patient_summary.csv (per-patient aggregates).

### Reflections: 
**Challenges**: 
- KeyError on `supplemental_o2` when merging due to missing FiO₂ measurements.  
- Duplicate columns (`_x`, `_y`) after merges. 
- Missing GCS components for some patients. 
- Missing NEWS2 vitals in pivot.   
**Solutions & Learnings**:
- Conditional merge and default fill (0). Always check column existence before accessing or transforming it in merged datasets.
- Merge only necessary columns, avoid re-merging existing ones. Thoughtful merge design prevents downstream confusion and simplifies CSV outputs.
- Added missing columns with `pd.NA` and computed `gcs_total` safely. Defensive coding is critical when working with real-world clinical data.
- Added all expected vitals as NA before merges. Preemptive handling of expected columns reduces errors during scoring.

### Issues Encountered
- Missing itemids in `extract_news2_vitals.py`.
- KeyError when accessing non-existent supplemental O₂ or GCS columns.
- Duplicate columns after merging GCS and supplemental O₂.
- Variations in vital naming and units.
- Some timestamps had missing vital measurements.

### Lessons Learned
- Always **validate column existence** before transformations or merges.
- Merge only necessary columns to prevent duplicates.
- Filling missing data with safe defaults ensures pipeline stability.
- Defensive coding allows robust handling of incomplete real-world datasets.
- Maintaining clean, standardised column names simplifies both computation and human-readable output.

### Extra Considerations / Documentation Points
- The pipeline now fully supports **Phase 1** outputs and can be run repeatedly on updated CSVs.
- All merges are idempotent – repeated runs will not create duplicates.
- All human-readable labels (`consciousness_label`, `co2_retainer_label`, `supplemental_o2_label`) are always generated.
- **Defensive coding for human-readable labels**:
  - Two blocks exist in the code assigning `consciousness_label`, `co2_retainer_label`, and `supplemental_o2_label`.
  - Redundancy ensures the script runs safely even if some columns are missing (`level_of_consciousness`, `co2_retainer`, `supplemental_o2`).
  - Guarantees idempotency and prevents KeyErrors on incomplete datasets.
  - Best practice: could combine into a single block that creates defaults and assigns labels in one step.
- Outputs `news2_scores.csv` and `news2_patient_summary.csv` are fully consistent with the pipeline’s intended design.
- Next steps (Phase 2) could include visualisation, predictive modeling, or integrating NEWS2 trajectories into a dashboard.

---

# Phase 2: Preprocess Data for ML-Models

---

## Day 3 Notes — Validating NEWS2 Scoring & ML Pipeline Preparation

### Goals
- Validate NEWS2 scoring logic
  - Validate `compute_news2.py` against NHS NEWS2 rules.
  - Test edge cases (SpO₂ thresholds, supplemental O₂, GCS 15 vs 14, RR 20 vs 25).
- Decide on a consistent missing data strategy for timestamp-level and patient-level features.
- Begin planning ML-ready feature extraction (`news2_features_timestamp.csv` and `news2_features_patient.csv`).
  - Understand why we need LOCF, missingness flags, rolling windows, and numeric encodings.
- **Choose an ML model**:
  - Determine which model is optimal for tabular ICU data.
  - Decide preprocessing strategy based on chosen model.

### Overview
**For timestamp-level ML features (news2_features_timestamp.csv)**:

```text
raw long vitals (from MIMIC/ICU)  
    ↓ compute_news2.py  
news2_scores.csv               ← "clinical truth" (all vitals + NEWS2 + escalation labels)  
    ↓ make_timestamp_features.py
news2_features_timestamp.csv   ← "ML ready" (numeric features, missingness flags, encodings)  
```

**For patient-level summary features (news2_features_patient.csv)**:

```text
raw long vitals  
    ↓ compute_news2.py  
news2_scores.csv                ← news2_patient_summary.csv not needed
    ↓ make_patient_features.py  
news2_features_patient.csv      ← ML ready (patient-level aggregates, imputed medians, missingness %)
```

**The difference**:
- Timestamp pipeline → preserves row-by-row dynamics (LOCF, staleness, rolling windows).
-	Patient pipeline → collapses timeline into patient-level summaries (medians, % missing, escalation profile).

### What We Did
#### Step 1: Validating NEWS2 Scoring
- **Action**: Ran validate_news2_scoring.py on test dictionaries.
- **Findings**:
  - Low GCS cases initially produced incorrect scores.
  - The scoring function ignored consciousness because row.get("level_of_consciousness", pd.NA) returned pd.NA.
  -	Other special cases (SpO₂, supplemental O₂) were correctly scored because their thresholds were handled explicitly.
- **Fixes**: Moved `if pd.isna(value): return 0` **to the end of the function**.
- **Outcome**: All unit tests passed, learned the importance of understanding intermediate variables in scoring pipelines.
- The main pipeline did not have these problems as gcs_total is converted into level_of_consciousness before the scoring is called, so there was no missing keys.

#### Step 2: Missing Data Strategy
- **Timestamp-level features**:
  - Use LOCF (Last Observation Carried Forward) to maintain temporal continuity.
  - Add missingness flags (1 if value was carried forward) so models can learn from missing patterns.
  - **Justification**: mimics clinical reality; preserves trends; Tree-based models like LightGBM handle NaNs naturally.
- **Patient-level summary features**:
  - Use median imputation per patient timeline if a vital is missing across some timestamps.
  -	Include % of missing timestamps as a feature.
  -	**Justification**: balances robustness with bias avoidance; prevents skewing min/max/mean statistics.
-	**Key decisions**:
  -	Do not fill population median at timestamp-level (would break temporal continuity).
  -	Only fill median at patient summary level if some timestamps exist; otherwise, leave as NaN or optionally fallback to population median.

#### Step 3: Preparing Timestamp-Level ML Features
**Pipeline (make_timestamp_features.py)**:
1. Start from news2_scores.csv (all vitals + NEWS2 + escalation labels).
  - Parse charttime as datetime.
  - Sort by subject_id, charttime.
2. Create missingness flags for each vital (before fills).
3. LOCF forward-fill per subject (optionally backward-fill for initial missingness or leave as NaN), do not use population median.
4. Create carried-forward flags (binary indicator - 1 if the value came from LOCF). Helps ML distinguish between observed vs assumed stable, exploit missingness patterns (e.g. vitals measured more frequently when patients deteriorate).
5. **Compute rolling windows (1h, 4h, 24h)**: mean,min,max,std,count,slope,AUC.
6. Compute time since last observation (`time_since_last_obs`) for each vital (staleness).
7. Convert textual escalation/risk labels → numeric ordinal encoding (Low=0, Low-Medium=1, Medium=2, High=3) for ML. Keeps things simple - one column, easy to track in feature importance
8. Save news2_features_timestamp.csv.
**Rationale**:
- Trees can leverage trends and missingness.
-	Rolling windows capture short-, medium-, and long-term deterioration patterns.
-	Timestamp features feed ML models like LightGBM directly without further preprocessing.

#### Step 4: Preparing Patient-Level ML Features
**Pipeline (make_patient_features.py)**:
1. Start from news2_scores.csv.
2. **Group by patient**: Aggregate vitals per patient timeline (median, mean, min, max per vital).
3. **Median imputation**: Fill missing values for each vital using patient-specific median (so their profile isn’t biased by others), if a patient never had a vital recorded, fall back to population median.
4. **% Missing per vital**: Track proportion of missing values per vital before imputation (HR missing in 30% of their rows = 0.3), missingness itself may signal clinical patterns (e.g. some vitals only measured in deteriorating patients).
5. **Encode risk/escalation labels**: Ordinal encoding (Low=0, Low-Medium=1, Medium=2, High=3), calculate summary stats per patient: max risk (highest escalation they reached), median risk (typical risk level), % time at High risk (what fraction of their trajectory was spent here).
6. **Output**: news2_features_patient.csv (compact, one row per patient, ML-ready summary).
**Rationale**:
-	Median imputation preserves patient-specific patterns without introducing bias from other patients.
-	% Missing captures signal from incomplete measurement patterns.
-	Ordinal risk encoding simplifies downstream ML model input while retaining interpretability. Together, these three summary features summarise a patient’s escalation profile across their stay. Proportion features (like % high) are standard numeric features (not encoded categories).
-	This is enough for model; don’t need optional metrics like streaks, AUC, or rolling windows for the patient summary.

#### Step 5: ML Model Selection
-	**Options considered**:
  -	Logistic Regression → easy to deploy and explainable but underpowered, tends to underperform on raw time-series vitals.
  -	Deep learning (LSTMs/Transformers) → overkill, prone to overfitting with moderate datasets.
  -	Boosted Trees (XGBoost / LightGBM / CatBoost) → robust for tabular ICU data, handle NaNs, train fast, interpretable.
-	**Decision: LightGBM (Gradient Boosted Decision Tree (GBDT) library)**
  - State-of-the-art for structured tabular data (EHR/ICU vitals is tabular + time-series).
  -	Handles missing values natively (NaNs) → no additional imputation required (simpler pipeline).
  -	Provides feature importances → interpretability for clinical review.
  -	Easy to train/evaluate quickly → allows multiple experiments.
-	**Future extension**:
  -	Neural nets possible if dataset size grows significantly.
  -	Would require additional preprocessing: time-series sequences, padding, normalisation, possibly interpolation.


### Validation Issue & Fix: GCS → Level of Consciousness
**Problem Identified:**
- `score_vital` incorrectly ignored `level_of_consciousness` when computing NEWS2 scores.
- **Reason**:
1. `compute_news2_score` passes `value = row.get("level_of_consciousness", pd.NA)`.
2. If the row dictionary does not contain `level_of_consciousness` yet (common in synthetic test cases), `value=pd.NA`.
3. Original code had `if pd.isna(value): return 0` at the top of `score_vital`.
4. This caused the function to exit **before using `gcs_total` to compute LOC**, so low GCS patients were scored incorrectly.
**Other Contributing Factor:**
- `level_of_consciousness` exists as a key in `vital_thresholds`.  
- The generic “Other vitals” block ran first, attempting to score with `value=pd.NA`, bypassing LOC-specific logic.
**Fix Implemented:**
- Moved `if pd.isna(value): return 0` **to the end of the function**.
- Ensured the LOC-specific block runs **before** the generic “Other vitals” block.  

```python
if vital_name == "level_of_consciousness":
    flag = 0 if gcs_total == 15 else 1
    for low, high, score in vital_thresholds["level_of_consciousness"]:
        if low <= flag <= high:
            return score

if vital_name in vital_thresholds:
    for low, high, score in vital_thresholds[vital_name]:
        if (low is None or value >= low) and (high is None or value <= high):
            return score

if pd.isna(value):
    return 0
```

### Key Reflections & Lessons Learned
- **Validate NEWS2 carefully**:
  - Subtle errors can arise from intermediate variables like level_of_consciousness.
  - Always check how the test harness mirrors the main code.
- **Data pipeline decisions**:
	- Simplify pipeline to focus on what’s necessary for the chosen ML model, not every theoretical feature.
	- Keeping the pipeline simple ensures maintainability and interpretability.
  - Could adapt for other models or neural networks, but only if dataset and project scope increase.
- **ML pipeline considerations**:
	- Timestamp features → for temporal trends.
	-	Patient summary features → for overall risk profile per patient.
	- Missingness flags → signal to the model without biasing values.

### Planned Next Steps
1. Implement `make_timestamp_features.py` using `news2_scores.csv`.
2. Generate `news2_features_timestamp.csv` with LOCF, flags, rolling window stats, ordinal risk encoding.
3. Start aggregating patient-level summary features (`news2_features_patient.csv`) using median imputation + missingness stats in `make_patient_features.py`.
4. Train a baseline LightGBM model to test predictive performance.
5. Document rationale for each preprocessing decision for reflections section.

⸻

## Day 4 Notes - Timestamp-Level ML Features

### Goals
- Implement make_timestamp_features.py to transform news2_scores.csv into news2_features_timestamp.csv, ready for ML modeling with LightGBM.
- Follow 8 planned steps for feature engineering at the timestamp level.  

### Planned 8 Steps
1. Parse charttime → datetime, sort by subject_id & charttime.  
2. Create missingness flags (per vital, before fills).  
3. LOCF forward-fill per subject (optionally backward-fill initial missing).  
4. Create carried-forward flags (1 if value came from LOCF).  
5. Compute rolling windows (1h, 4h, 24h) → mean, min, max, std, count, slope, AUC.  
6. Compute time_since_last_observation (staleness per vital).  
7. Encode risk/escalation labels → numeric ordinal (Low=0, Low-Med=1, Medium=2, High=3).  
8. Save `news2_features_timestamp.csv` (ML-ready).  

Only Step 1 was implemented today; Steps 2–8 remain.  

### What We Did Today
- Completed **Step 1 (Load & Sort)**:
  - Loaded `news2_scores.csv` into a pandas DataFrame.  
  - Converted `charttime` column to proper `datetime` objects.  
  - Sorted rows by `subject_id`, `stay_id`, and `charttime` to enforce chronological order per patient stay.  
  - Verified with a preview (`df.head()`) that the data is clean and ordered.  

### Reflections
#### Challenges
- **Pandas syntax** feels overwhelming
- Spent most of the day revisiting **all previous scripts (`.py`)** in the project to fully annotate them.  
- The main difficulty was **pandas syntax in general** — not just in this step, but across:
  - `.sort_values`, `.reset_index`, `.merge`, `.pivot_table`, `.apply`, `.isin`, `.loc`.  
  - Understanding why certain operations are applied in a specific order.  
  - Figuring out how pandas “thinks” when reshaping or transforming datasets.  
- Felt frustrated at how much time was spent **understanding code** instead of **writing new features**.  
#### Solutions
- Added **inline comments** to all major pandas operations across the codebase.  
- Broke the pipeline into clear **8 steps**, so I can see the bigger picture and where today’s progress fits.  
- Asked targeted questions (e.g. about return type hints, `.apply()`, `os`, `.merge`) to fill conceptual gaps.  
#### Learnings
- **Pandas is its own language**: It’s not just Python, but a layer of syntax for manipulating tabular data.  
- **Order of operations matters**: E.g. missingness flags must precede filling, or else ML won’t distinguish true vs imputed values.  
- **Debugging strategy**: Always print `df.head()` after each major step to confirm changes.  
- **Reflection is progress**: Even if I only implemented Step 1, I deepened my conceptual foundation, which will make Steps 2–8 easier.  

### Extra Considerations
- My pace felt slower than expected, but it was necessary to **slow down and understand the building blocks**.  
- Future steps (e.g. rolling windows, staleness) will require chaining multiple pandas operations — having this stronger foundation will prevent confusion later.  
- Need to balance **practical coding** (keep pipeline moving) with **conceptual grounding** (understanding transformations).  


### 📅 Next Steps (Day 5 Plan)
- Implement **Step 2 (Missingness flags)**:
  - Add `_missing` columns for each vital before LOCF.  
  - Confirm flags align with actual NaNs.  
- If possible, progress into **Step 3 (LOCF imputation)** and **Step 4 (Carried-forward flags)**.  
- Keep using small previews (`.head()`, `.isna().sum()`) to verify correctness.  

---

## Day 5 Notes - Missingness, Carried-Forward Flags & Rolling Features

### Goals
- Continue building `make_timestamp_features.py` pipeline.  
- **Extend Step 2 → Step 5**:
  - **Step 2**: Add missingness flags.
  - **Step 3**: Apply forward-filling (LOCF).
  - **Step 4**: Add carried-forward flags.
  - **Step 5**: Start rolling window features (mean, min, max, std, slope, AUC).  

### What We Did
#### Step 2: Missingness Flags
- Implemented `add_missingness_flags(df)` to generate new columns like `respiratory_rate_missing`, `spo2_missing`, etc.  
- **Logic**: for each vital, `df[v].isna().astype(int)` creates a flag column where `1 = missing` and `0 = observed`.  
- Called after loading + sorting the CSV with `load_and_sort_data(INPUT_FILE)`.  
- Verified output by printing `df.head()`.
#### Step 3: LOCF (Forward- and Back-Fill)
- Wrote `apply_locf(df)` to handle missing values by carrying the last observed measurement forward (`ffill`) within each patient stay (`groupby(['subject_id', 'stay_id'])`).  
- Added an extra `.bfill()` so the very first row of each stay (if missing) is backfilled with the next available measurement.  
- Ensures no missing values remain for the chosen vitals.
#### Step 4: Carried-Forward Flags
- Added `add_carried_forward_flags(df)` to track which values in the filled dataset are real vs imputed.  
- Used missingness flags from Step 2 as ground truth:  
  - Carried = `value is not NaN after fill` **AND** `was missing before fill`.  
- Output = new columns like `respiratory_rate_carried`, `spo2_carried`, etc.  
- This avoids the problem of falsely marking naturally repeated values as carried-forward.
#### Step 5: Rolling Features (in progress)
- Started `add_rolling_features(df)` to compute rolling-window statistics on numeric vitals (`respiratory_rate`, `spo2`, `temperature`, `systolic_bp`, `heart_rate`).  
- **Window sizes**: 1h, 4h, 24h.  
- **Stats**: mean, min, max, std, slope (trend), AUC (cumulative exposure).  
- For each vital × window combination, new feature columns are created, e.g.:
  - `respiratory_rate_roll1h_mean`  
  - `spo2_roll24h_slope`  
- Implemented slope with a simple linear regression on index order; AUC as the cumulative sum over the window.  
- Still clarifying whether slope/AUC should be computed on true timestamps (`charttime_numeric`) or just index order.  

### Reflections
#### Challenges
- **Pandas syntax**:  
  - Still feels overwhelming, especially with groupby, rolling, and applying custom functions.  
  - Feels like "watching a chess grandmaster" without yet knowing the moves.  
- **Redundant flags**:  
  - Initially thought missingness flags already made carried-forward redundant.  
  - **Learned they complement each other**: missing = gaps before filling, carried = which values were filled in.
- **Rolling features**:  
  - Hard to see how loops systematically build columns.  
  - `charttime_numeric` looked confusing since we’re not yet using real timestamps in slope/AUC.
#### Solutions & Learnings
- Breaking code into **bite-sized functions** helps (e.g., Step 2–4 each modular).  
- Printing `df.head()` after each step is essential for debugging.  
- Carried-forward vs missingness flags = subtle but distinct concepts.  
- Nested loops (`for v in vitals, for w in windows`) → systematic way to generate features.  
- Recognised unused code (`rolling_features = []`, `charttime_numeric` placeholder).  

### Next Steps
- Finish **Step 5**:
  - Decide whether slope/AUC should use real timestamps or simple index order.
  - Simplify code by removing unused prep.  
- Validate with small test DataFrame to confirm columns behave as expected.
- **Move on to Step 6**: **time since last observation** once rolling features are stable.  

---

## Day 6 Notes – Rolling Features, Time Since Obs, Risk Encoding

### Goals
- Continue pipeline development (`make_timestamp_features.py`).
- Finalise **Step 5–7**:
  - **Step 5**: Rolling window features (mean, min, max, std, slope, AUC).
  - **Step 6**: Time since last observation (staleness).
  - **Step 7**: Encode escalation/risk labels.
- Add **Step 8**: End-to-end integration to generate the final ML-ready dataset.
- Resolve slope/AUC approach: simple vs time-aware.

### What We Did
#### Step 5 – Rolling Window Features
- Added rolling features for 5 vitals (`HR, RR, SpO₂, Temp, SBP`) across 3 windows (1h, 4h, 24h).  
- Stats per window: `mean, min, max, std, slope, AUC`.  
- Implemented **time-aware slope and AUC**:
  - **Slope** = rate of change per hour, using actual `charttime` gaps.
  - **AUC** = cumulative exposure over time, integrated with trapezoidal rule.
- Example **outputs**:
  - `heart_rate_roll4h_slope` = rate of HR change (bpm/hour).
  - `spo2_roll24h_auc` = total “oxygen exposure” in the last 24 hours.
#### Step 6 – Time Since Last Observation
- Computed `*_time_since_last_obs` for each vital.  
- Captures how stale each measurement is (fresh vs old data).  
- **Example**: HR measured 3h ago → `heart_rate_time_since_last_obs = 3.0`.
#### Step 7 – Encode Risk Labels
- Created `risk_numeric` column mapping text → numbers:
  - Low = 0  
  - Low-Medium = 1  
  - Medium = 2  
  - High = 3  
#### Step 8 – Pipeline Integration
- End-to-end workflow complete:
  1. Load & sort data.  
  2. Add missingness flags.  
  3. Apply LOCF.  
  4. Add carried-forward flags.  
  5. Add rolling window features.  
  6. Add time since last obs.  
  7. Encode risk labels.  
  8. Save final dataset → `news2_features_timestamp.csv`.

### Reflections
#### Challenges
- **Slope (trend) choice**:
  - **Simple slope**: assumes equal spacing (`t = 0,1,2,...`). Works if vitals are frequent/regular, but misleading if sparse (e.g., 2 readings in 5 minutes then none for 8 hours).
  - **Time-aware slope**: uses real timestamps, slope = change per unit time (e.g., HR 100→120 in 30 mins = +40 bpm/hr vs over 12h = +1.7 bpm/hr).
- **AUC (cumulative exposure)**:
  - **Simple AUC**: sum of values only.  
  - **Time-aware AUC**: integrates over time → reflects true burden/exposure (e.g., SpO₂ 92% for 24h is worse than 92% for 1h).
- **Pipeline complexity**:
  - Step 5 added ~90 columns per row. Easy to lose track without systematic naming and notes.
#### Solutions & Learnings
- Adopted **time-aware slope & AUC** → features are both ML-useful and clinically interpretable.  
- LOCF filling made rows “regular,” but we kept real-time slope because clinical interpretability is higher priority.  
- **Key conceptual clarity**:
  - **Slope** = trend per unit time.  
  - **AUC** = exposure burden over time.  
- Validated that missingness vs carried-forward flags are complementary:
  - Missing = gaps before fill.  
  - Carried = synthetic values after fill.  

### Next Steps
1. **Validate outputs**:
   - Print a few patient timelines to ensure slope/AUC match clinical intuition.  
   - Confirm time-since-last-obs is reasonable.  
2. **Efficiency check**:
   - Test runtime on full dataset (Step 5 may be slow).  
3. **Documentation**:
   - Write a short “feature dictionary” describing each class of features.  
4. **Step 9**: Aggregate timestamp-level features → patient-level summary for downstream ML.

---

## Day 7 Notes - Timestamp Pipeline Complete & Patient Features Begun

### Goals
- Run and debug `make_timestamp_features.py` end-to-end.  
- Verify that the generated CSV (`news2_features_timestamp.csv`) is correct.  
- Decide which model (LightGBM vs Neural Network) uses which feature set.  
- **Finalise model roadmap**: V1 = LightGBM on patient-level, V2 = Neural Network (TCN) on timestamp-level.  
- Begin implementing `make_patient_features.py` (Steps 1–2). 

### Overview 
```text
  Raw EHR Data (vitals, observations, lab results)
         │
         ▼
Timestamp Feature Engineering (news2_scores.csv)
 - Rolling statistics (mean, min, max, std)
 - Slopes, AUC, time since last observation
 - Imputation & missingness flags
         │
         ├─────────────► Neural Network Models (v2)
         │              - Input: full time-series per patient
         │              - Can learn temporal patterns, trends, dynamics
         │
         ▼
Patient-Level Feature Aggregation (make_patient_features.py → news2_features_patient.csv)
 - Median, mean, min, max per vital
 - Impute missing values
 - % missing per vital
 - Risk summary stats (max, median, % time at high risk)
 - Ordinal encoding for risk/escalation
         │
         ▼
LightGBM Model (v1)
 - Input: one row per patient (fixed-length vector)
 - Uses aggregated statistics only
 - Cannot handle sequences or variable-length time series
 ```

### What We Did
1. **Ran `make_timestamp_features.py` successfully**:
   - Adjusted file paths to go two levels up (`../../`) because script runs inside `src/ml-data-prep/`.
   - Resolved duplicate index alignment error by using `.reset_index(drop=True)` which flattens everything back to a simple index that lines up exactly with DataFrame’s rows, instead of dropping groupby levels `.reset_index(level=[0,1], drop=True)` as charttime can have duplicates which confuses pandas when it tries to align by index labels.
   - Fixed rolling window warnings (`'H'` → `'h'`).
   - Output CSV generated: `news2_features_timestamp.csv`.
2. **Debugging Learnings**:
   - **File Paths**:  
     ```python
     DATA_DIR_INPUT = Path("../../data/interim-data")
     DATA_DIR_OUTPUT = Path("../../data/processed-data")
     ```
     → Ensures script looks in correct `data/` directories when run from `src/ml-data-prep/`.
   - **Duplicate Index Issue**:  
     - After `groupby().rolling()`, result had a MultiIndex (patient, stay, charttime).
     - Using `.reset_index(level=[0,1], drop=True)` caused misalignment if charttime was duplicated (after LOCF). Pandas cannot reindex on an axis with duplicate labels.
     - Fix: `.reset_index(drop=True)` → guarantees the Series index matches the DataFrame’s row index.
   - **Trapz Deprecation**: `np.trapz` still works but shows warning; recommended future replacement with `np.trapezoid`.
   - **PerformanceWarning**: Adding 90+ columns one-by-one fragments the DataFrame; harmless but could be optimized with `pd.concat`.
3. **Model Roadmap Finalised**:
   - **V1: LightGBM (Gradient Boosted Decision Trees)**  
     - **Input**: `news2_features_patient.csv`.  
     - **Output**: news2_features_patient.csv → LightGBM → AUROC, feature importances.
     - One row per patient, interpretable, strong baseline.
     - Very interpretable for clinicians (median HR, % missing SpO₂, % time high risk).  
   - **V2: Neural Network (TCN – Temporal Convolutional Network)**  
     - **Input**: `news2_features_timestamp.csv`.  
     - **Output**: news2_features_timestamp.csv → TCN → sequence classification (predict escalation).
     - Full time-series per patient, captures sequential deterioration patterns.
     - Demonstrates modern advanced deep learning sequence modeling.  
     - Shows can move from tabular ML → time-series DL progression.
     - More impressive to interviewers / academics (future-proof).
4. **Neural Network Model Selection**:
   - **Considered**: LSTM/GRU, Transformers, TCN.  
   - **Decision: TCN** because it handles long sequences efficiently, avoids vanishing gradients, and trains faster than RNNs.  
   - **Requirements**: sequence padding, normalization, masking for missingness.  
5. **Started `make_patient_features.py`**:
   - **Step 1**: Load CSV, sort by patient/time.  
   - **Step 2**: Aggregate vitals with `.groupby("subject_id").agg(["median","mean","min","max"])`.  
   - Learned how `agg()` outputs a MultiIndex → flattening into `vital_stat` format.  
   - Steps 3–6 (imputation, % missingness, risk encoding, save CSV) still to do.  

 ### Neural Network Model Selection
- **Options considered**:
  - **Recurrent Neural Networks (LSTM / GRU)** → well-suited for sequences but prone to vanishing gradients on long ICU stays, slower to train.
  - **Transformers** → powerful for long sequences, but overkill for moderate dataset size, computationally intensive.
  - **Temporal Convolutional Networks (TCN)** → convolutional sequence modeling, parallelizable, captures long-term dependencies efficiently.
- **Decision: TCN (Temporal Convolutional Network)**
  - Ideal for time-series vitals data with sequential trends.
  - Can handle long sequences without vanishing gradient issues like recurrent neural networks (RNN).
  - Parallel convolutional operations → faster training than sequential RNNs.
  - Compatible with timestamp-level features and missingness flags.
- **Preprocessing requirements**:
  - Sequence padding to unify input lengths.
  - Normalisation of continuous vitals.
  - Optional interpolation or masking for missing values.
  - One-hot encoding of categorical labels if required.
- **Strengths**:
  - Captures temporal patterns and trends across patient stays.
  - Expressive for sequence modeling where LightGBM may miss temporal dynamics.
  - Empirically outperforms LSTM/GRU for moderate-length clinical sequences.
- **Weaknesses / Limitations**:
  - More computationally intensive than tree-based models.
  - Less interpretable than LightGBM feature importances.
  - Requires careful tuning of hyperparameters (kernel size, dilation, layers).
- **Use case in pipeline**:
  - Secondary model after LightGBM to capture fine-grained temporal trends.
  - Useful for sequences where timestamp-level patterns predict escalation more accurately.

### Planned 6 Steps
1. Load input file `news2_scores.csv`.  
2. Aggregate vitals per patient (median, mean, min, max).  
3. Perform patient-specific median imputation (fallback to population median if never observed).  
4. Compute % missingness per vital (fraction of rows missing before imputation).  
5. Encode risk/escalation labels → numeric ordinal (Low=0, Low-Med=1, Medium=2, High=3), then summarise per patient (max risk, median risk, % time at High risk).  
6. Save `news2_features_patient.csv` (one row per patient, ML-ready).  

Only Steps 1-2 were implemented today; Steps 3-6 remain.

### Reflections
#### Challenges
- **Indexing Misalignments**: Rolling window outputs had MultiIndex misaligned with base DataFrame → caused reindexing errors.  
- **Path Confusion**: Needed to carefully reason about relative paths when running scripts inside `src/`.  
- **Flattening MultiIndexes**: Initially confusing to understand multiindexing and how `(vital, stat)` pairs became clean `vital_stat` columns.  
#### Solutions
- Used `.reset_index(drop=True)` to align rolling stats with DataFrame rows.  
- Standardised file paths with `../../` from script location.  
- Flattened MultiIndex columns using `["_".join(col) for col in df.columns]`.  
#### Learnings
- Index ≠ header → the index is the row labels, not the column names.  
- Duplicated timestamps (after LOCF) can break alignment if not flattened.  
- **Timestamp vs Patient-level features serve complementary roles**:  
  - **Timestamp features** = sequence models.  
  - **Patient features** = tree-based baselines.  
- Portfolio-wise, showing both LightGBM and TCN demonstrates breadth (tabular ML + time-series DL).  

### Extra insights
- **Future-proofing with both feature sets ensures robustness and flexibility**:
  - **LightGBM (V1)** → clinician-friendly, interpretable baseline.  
  - **TCN (V2)** → modern DL, captures dynamics.  
- **Timestamp-level features** = richest representation, essential for sequence models / deep learning
- **Patient-level features** = distilled summaries, useful to quickly test simpler models, feature importance or quick baseline metrics.
- Keeping both pipelines means we can mix (hybrid approaches) if needed (e.g., summary features + LSTM on sequences). 
- LightGBM is often deployed first because it’s fast, robust, and interpretable, while the neural network is a v2 that might improve performance. 


### Portfolio story
- **LightGBM (v1)**: We started with patient-level aggregation to establish a baseline model that is interpretable and fast to train. This gives clinicians an overview of which vitals and risk patterns matter most.
- **Neural Network (TCN)(v2)**: Once we had a solid baseline, we moved to a temporal convolutional network to directly learn time-dependent deterioration patterns from patient trajectories. This captures dynamics that aggregated features can’t.

### Next Steps
- Complete `make_patient_features.py`:  
  3. Median imputation (patient-level, fallback to population).  
  4. % missingness per vital.  
  5. Risk encoding & summary stats (max, median, % time at High).  
  6. Save `news2_features_patient.csv`.  
- Then proceed to implement LightGBM baseline training (V1).  
- Prepare timestamp features (already done) for TCN implementation (V2).  

---

## Day 8 Notes - Patient Pipeline Complete & LightGBM Roadmap Planning

### Goals
- Complete patient-level feature extraction (steps 3–6 in `make_patient_features.py`)
- Verify the output (`news2_features_patient.csv`)
- **Plan the next phase**: LightGBM training and validation

### What We Did Today
- Finished steps 3–6 in `make_patient_features.py`:
  - Aggregated vital signs and NEWS2 scores to patient-level
  - Calculated missing data percentages
  - Generated additional derived features (e.g., `pct_time_high`)
- Ran the feature extraction script successfully
  - **Checked the resulting CSV**: `news2_features_patient.csv`
  - Verified that all patient-level features were correctly calculated
- Planned **Phase 3: LightGBM Training + Validation**:
  - Drafted a high-level roadmap including dataset preparation, model initialisation, training, validation, saving, and documentation


### Phase 3: LightGBM Training + Validation Overview
**Goal:** Train a LightGBM model on patient-level features, validate performance, and document results.
**Step 1**: Dataset Preparation
- Load processed patient-level features
- Split data into training and test sets
- Separate features (X) and target labels (y)
**Step 2**: Model Initialisation
- Initialise LightGBM model (classifier or regressor depending on target)
- Define basic parameters (learning rate, number of trees, random seed)
**Step 3**: Model Training
- Fit the model on the training data
- Monitor performance on test/validation set
- Apply early stopping to prevent overfitting
**Step 4**: Model Saving
- Save trained model to a file for later use
- Organize folder structure for reproducibility
**Step 5**: Model Validation
- Load saved model and run predictions on test set
- Calculate evaluation metrics (accuracy, ROC-AUC, RMSE, etc.)
- Optionally visualize feature importance and performance
**Step 6**: Documentation
- Record training and validation metrics
- Summarise feature importances
- Prepare results for portfolio or reporting
**Step 7**: Debugging / Checks
- Verify dataset shapes and target columns
- Ensure feature consistency between training and test sets
- Check for missing or non-numeric values

### Reflections
- **Good progress today**: patient-level feature pipeline is now complete and verified
- Planning Phase 3 helps visualise the workflow and prevents getting stuck mid-training
- Breaking down the steps into dataset preparation, training, validation, and documentation provides a clear roadmap


### Challenges
- Understanding **groupby operations** and **multi-indexing** in pandas
  - Aggregating by `subject_id` while preserving patient-level information
  - Converting multi-index back to a single index for merging and saving
- **Merging dataframes**:
  - Ensuring column names and indexes match for proper alignment
  - Avoiding duplicate columns or misaligned patient IDs
- Indexing and using `subject_id` as a key for all patient-level operations
- Verifying formats and data types after aggregation to ensure downstream compatibility

### Solutions and Learnings
- Learned to carefully check pandas groupby objects and use `.reset_index()` after aggregation
- Verified feature correctness by sampling output rows and comparing with original measurements
- Documented the aggregation and merging workflow for reproducibility
- Recognized the importance of consistent patient ID usage as a key across all transformations
- Confirmed that the CSV output is clean and ready for modeling

### Extras
- **Reviewed potential pitfalls for Phase 3**:
  - Handling missing values in LightGBM
  - Feature scaling considerations
  - Saving models with metadata (columns, feature order) for reproducibility
- Drafted a markdown roadmap for Phase 3 to guide upcoming training and validation
- Consider adding small unit tests for future pipeline stages to catch aggregation/merge errors early

---

# Phase 3: LightGBM Training + Validation 

---

## Day 9 Notes – Patient Dataset Preparation for LightGBM

### Goals
- Begin **Phase 3: LightGBM training and validation (steps 1-2)**
- Focus on dataset preparation and initial model setup and checks, not full training yet
- Ensure reproducibility from the start
- Make notes on potential challenges for full training and validation

### What We Did
1. **Step 1: Dataset Preparation**
  - Create `src/ml-data-prep/train_lightgbm_prep,py`
  - Load `news2_features_patient.csv` into file.
  - **Identify the target variables**:
    - `max_risk` → “Did this patient ever escalate to severe deterioration?” (binary/ordinal classifier style) → classifier model (ordinal: 0–3).
    - `median_risk` → “What was this patient’s typical level of deterioration?” (long-term trajectory classifier) → classifier model (ordinal: 0–3).
    - `pct_time_high` → “How much of the patient’s stay was spent at critical risk?” (continuous regression, severity burden) → regressor (continuous: 0–1).
  - **Decided on looping automation code**: so that each target gets its own trained model and results, and we do not need to manually code 3 almost-identical training runs.
  - **5-fold cross-validation (CV) due to small data size (100 patients)**:
    - 5 equal groups of 20 patients, train on 4 groups, test on remaining 1 group
    - Repeat 5 times, rotating which group is used for testing
    - Average performance across all groups is a more stable estimate vs a standard train/test split (e.g. 70/30 or 80/20).
  - **Separate features (`X`) from the target (`y`)**:
    - X = Features (inputs) → everything the model uses to make predictions (hr_mean, spo2_min, temperature_max, %missing, etc.).
    - y = Target (outputs) → the “answer key” you want the model to learn to predict (e.g., max_risk).
    - During training, the model learns a mapping from X → y.
      - X_train = inputs for training (DataFrame of row = 80 training patients × column = all features)
      - y_train = labels for training (Series of row = 80 training patients x values = their risk labels)
      - X_test = inputs to evaluate model (DataFrame of row = 20 test patients × column = all features)
      - y_test = labels to compare predictions (Series of row = 20 test patients x values = their risk labels)
  - **Check data types and missing values**: 
    - Ensure all features are numeric and compatible with LightGBM (LightGBM quite forgiving, can handle some NaNs internally).
    - Although our preprocessing should have fixed this (imputed NaNs, encoded categorical risk as numeric, dropped non-numerics), always double check before model training.
    - We need safety check as sometimes merges sneak in NaNs, sometimes column types are wrong.
    - If something unexpected pops up, better to catch it before fitting LightGBM.
2. **Step 2: Model Initialisation Setup**
  - Import LightGBM
  - **Initialise a basic LightGBM model (`train_lightgbm.py`)**
    - Used LightGBM default parameters (learning rate, depth, number of trees, etc.)
    - Create both classifier (for max_risk, median_risk) and regressor (for pct_time_high).
      - `LGBMClassifier` for max_risk & median_risk
      - `LGBMRegressor` for pct_time_high
    - Set the seed `random_state=42` for reproducibility.
  - Skip cross-validation entirely because we aren’t evaluating performance yet, and dataset is too small.
3. **Quick Test Run**:
  - Dont need to do a full training loop yet.
  - **Fit the model on a small subset of training data (10 patients) to**:
    - Verify that the pipeline works
    - Check that data formats, shapes, and types are correct (features (X) are numeric, targets (y) are the right shape)
    - Catch any errors with feature alignment or missing values
    - Predictions are generated
  - Ensured that the loop works for all 3 targets automatically
  - This catches pipeline errors before we spend time coding full CV training tomorrow.
4. **Logging and Documentation**:
  - Recorded:
    - Dataset shapes (rows, columns)
    - Features used
    - Any issues encountered (e.g., unexpected NaNs, strange distributions) - none major
  - Document initial observations and notes for Phase 3

### Train/Test Split vs Cross-Validation
#### Initial plan  
- Standard ML workflow uses a **train/test split** (e.g., 70/30 or 80/20).  
- Training set is used to fit the model, test set evaluates generalisation.  
- Works well when datasets are **large** (>10,000).  
- With a big dataset, 20–30% for testing still leaves enough training data to learn robust patterns.  
#### Problem with our dataset:  
- We only have **100 patients** (100 rows after patient-level aggregation).  
- A 70/30 split leaves 30 patients for testing; 80/20 leaves only 20.  
- **This is too small**: metrics like AUROC or accuracy would fluctuate a lot if even 1–2 patients are misclassified.  
- **Result**: unreliable, unstable performance estimates.  
#### Solution: Cross-Validation
- Instead of one split, we use **k-fold cross-validation**.  
- **Process**:  
  1. Split patients into *k* equal groups (folds).  
  2. Train on *k–1* folds, test on the remaining fold.  
  3. Repeat *k* times, rotating which fold is used for testing.  
  4. Average performance across all folds → more stable estimate.  
- Every patient is used for both training and testing (but never in the same round).  
#### Why 5-fold CV?
- **k=5** is a common default:  
  - Balances computational efficiency with robustness.  
  - Each test fold has 20 patients → big improvement over a single 20-patient test set.  
  - Results averaged across 5 runs smooth out randomness.  
- For very tiny datasets, k=10 can be used, but 5-fold is usually enough here.  
- **Decision:** 
  - Use **5-fold cross-validation** for LightGBM training/validation, and optionally hold out ~10 patients as a final untouched test set for a “real-world” check.  

### How the model works
- In supervised machine learning, the model learns a mapping from features (X) → target (y).
- Features (X) = things you give the model as input (heart rate, SpO2, temperature, missingness, etc.)
- Target (y) = the thing you want the model to predict (e.g. max_risk).
- Even though max_risk is already known, the model uses it as the “answer key” during training:

```text
Input features (X) → Model → Predict max_risk
Compare predicted max_risk vs actual max_risk → Adjust model
```

- During training, the model compares its predictions to the real max_risk values and adjusts weights to minimise errors.
- Without max_risk (or whatever target), the model cannot learn, because it has nothing to compare its predictions to.

### KFold logic diagram (100 patients, 5 folds)

```python
100 patients numbered 0–99.

Step 1: Shuffle patients (random order)
Shuffled indices: 23, 45, 12, 7, 56, ... , 99  (total 100)

Step 2: Split into 5 folds (20 patients each)
100 patients → 5 folds (20 patients each)

Fold 1: 23, 45, 12, ..., 78        (test fold for iteration 1)
Fold 2: 7, 56, 34, ..., 81         (test fold for iteration 2)
Fold 3: ...
Fold 4: ...
Fold 5: ...

Step 3: Loop through folds

Iteration 1 (fold_idx=1)
------------------------
Train folds = 2,3,4,5 → 80 patients
Test fold  = 1         → 20 patients

X_train = features for patients in folds 2–5 (80×num_features)
y_train = labels   for patients in folds 2–5 (80×1)

X_test  = features for patients in fold 1 (20×num_features)
y_test  = labels   for patients in fold 1 (20×1)

Iteration 2 (fold_idx=2)
------------------------
Train folds = 1,3,4,5 → 80 patients
Test fold  = 2         → 20 patients

X_train = features for patients in folds 1,3,4,5
y_train = labels   for patients in folds 1,3,4,5

X_test  = features for patients in fold 2
y_test  = labels   for patients in fold 2

Iteration 3 (fold_idx=3)
------------------------
Train folds = 1,2,4,5 → 80 patients
Test fold  = 3         → 20 patients

X_train = features for patients in folds 1,2,4,5
y_train = labels   for patients in folds 1,2,4,5

X_test  = features for patients in fold 3
y_test  = labels   for patients in fold 3

Iteration 4 (fold_idx=4)
------------------------
Train folds = 1,2,3,5 → 80 patients
Test fold  = 4         → 20 patients

X_train = features for patients in folds 1,2,3,5
y_train = labels   for patients in folds 1,2,3,5

X_test  = features for patients in fold 4
y_test  = labels   for patients in fold 4

Iteration 5 (fold_idx=5)
------------------------
Train folds = 1,2,3,4 → 80 patients
Test fold  = 5         → 20 patients

X_train = features for patients in folds 1,2,3,4
y_train = labels   for patients in folds 1,2,3,4

X_test  = features for patients in fold 5
y_test  = labels   for patients in fold 5
```

### Dataset Preparation Output (`prepare_patient_dataset.py`)
- Dataset loaded correctly → shape (100, 44) → 100 patients, 44 columns.
- All columns are numeric (float64 / int64) except two (co2_retainer_min and co2_retainer_max, which are bool but still compatible with LightGBM).
- No missing values left — preprocessing worked properly.
- For each target (max_risk, median_risk, pct_time_high):
- The dataset is being split into 5 folds.
- **Each fold shows the correct sizes**: Train shape (80, 40), Test shape (20, 40) → 80 patients for training, 20 for testing, 40 features in X.
- The loop cycles through all 3 targets automatically, so pipeline is flexible and future-proof.
- The dataset is fully prepared, the cross-validation setup works perfectly.
- Now have X_train, y_train, X_test, y_test ready for model training in the next step.

### Model Initialisation & Test Run Output (`train_lightgbm.py`)
- Quick LightGBM test on 10 patients completed
- Dataset loads correctly → shape (100, 44)
- Feature selection works → X has all numeric features except target columns and subject_id.
- Target separation works → y is selected for each loop.
- Models (classifier/regressor) fit without crashing.
- Predictions generated → pipeline is complete.
- Warnings about “no meaningful features” expected for tiny dataset; safe to ignore

### Reflections
#### Challenges
- **Understanding X and y separation**
  - Confused why both X (features) and y (target) are needed when the target is already known.
- **KFold logic and cross-validation**
  - Confused how `KFold.split(X)` automatically generates `train_index` and `test_index`.
  - Unsure how each iteration selects a different fold for testing and uses the remaining folds for training.
  - Did not initially understand why `fold_idx` starts at 1 and how it corresponds to the test fold.
  - Hard to visualise how all 100 patients are rotated through the 5 folds.
- **Handling X_train, X_test, y_train, y_test**
  - Unclear why there are four separate objects and what each represents in cross-validation.
- **Directory structure and relative paths**
  - Confused why `../../` worked in some scripts but not others, and where the CSV should be located relative to each script.
#### Solutions & Learnings
- **X and y separation**
  - Reviewed supervised learning: X = input features, y = labels/“answer key”.
  - **Insight:** Model requires y to compute loss and adjust weights; without it, training cannot proceed.
- **KFold `.split()` logic**
  - Learned that `.split()` is a generator which, for each iteration:
    1. Assigns one fold as `test_index`.
    2. Automatically uses all remaining folds as `train_index`.
  - Created a diagram mapping 100 patients → 5 folds → train/test sets per iteration.
  - **Insight:** Ensures each patient appears in the test set exactly once across folds, giving full coverage and stable performance estimates.
- **Fold indexing (`fold_idx`)**
  - Serves as a counter for current iteration; tracks which fold is the test fold.
  - **Insight:** Facilitates debugging, logging, and fold-specific analysis.
- **X/y training/test objects**
  - Verified that `X_train`/`y_train` contain features and labels for training folds, `X_test`/`y_test` for the test fold.
  - Checked shapes and sample rows to confirm correctness.
  - **Insight:** Clearly defines what data the model sees during training vs evaluation.
- **Quick visualisation**
  - Diagrammed how `.iloc[train_index]` and `.iloc[test_index]` select the correct rows.
  - **Insight:** Makes cross-validation logic intuitive; reinforces the necessity of looping through folds for small datasets.
- **Directory paths**
  - Used `pathlib.Path(__file__).resolve().parent` to dynamically build paths.
  - **Insight:** Avoids path errors, ensures reproducibility regardless of script location.

### Overall
- Gained confidence in preparing datasets, setting up cross-validation, and creating reproducible ML workflows.
- Pipeline is now robust for looping through all three target variables, ready for full training in the next phase.

### Next Steps
- Implement **full 5-fold CV training on all 100 patients**
- Save trained models and CV results
- Calculate evaluation metrics (accuracy, ROC-AUC, RMSE, etc.)
- Start exploring **feature importance** and preliminary model interpretation
- Extend pipeline to loop automatically for all targets and optionally timestamp-level features in parallel

---

## Day 10 Notes 

### Phase 3: LightGBM Training + Validation (Steps 1–8) Finalised 
**Goal: Train, validate, and document a LightGBM model on patient-level features, producing a polished, credible baseline.**
1. **Dataset Preparation**
  - Load processed patient-level features.
  - Separate features (X) and targets (y).
  - Verify datatypes, shapes, and missing values.
2. **Model Initialisation**
  - Initialise LGBMClassifier (max_risk, median_risk) or LGBMRegressor (pct_time_high).
  - Define baseline parameters and set random seed.
3. **Model Training**
  - Fit model on training folds with early stopping.
  - Monitor performance on validation folds.
4. **Model Saving**
  - Save trained models and results for reproducibility.
  - 3 outcomes x 5 folds = 15 models.
5. **Model Validation**
  - Load saved models, run predictions.
  - Compute metrics (accuracy, ROC-AUC, RMSE).
  - Visualise feature importance and results.
6. **Documentation**
  - Record metrics, shapes, and key outputs.
  - Summarise feature importances for interpretability.
7. **Debugging / Checks**
  - Verify dataset consistency across folds.
  - Ensure features align between training and test sets.
8. **Hyperparameter Tuning + Feature Importance**
  - Tune key parameters (learning rate, depth, trees) for fair performance.
  - Analyse feature importance to explain clinical drivers of prediction.
  - Produces a polished, interpretable, portfolio-ready baseline.
9. **Final Model Training (Deployment-Style Models)**
	-	After validation, train a final single model per target (3 total) on all 100 patients.
	-	**Purpose**:
    - Produces the best possible trained model using all available data.
    -	**Matches real-world practice**: once validated, you don’t throw away data — you train on the full cohort.
    -	Gives you 3 final models you can save, reload, and demonstrate (classifier + regressor).
This makes LightGBM phase complete, credible, and deployment-worthy without unnecessary over-optimisation.
**Why Not Go Further**
- **Ensembling (stacking, blending, bagging multiple LightGBM models)**: adds complexity without new insights → not unique for a portfolio.
- **Nested CV**: more statistically rigorous, but overkill for 100 patients; doesn’t change credibility.
- **Bayesian optimisation / AutoML**: looks flashy, but to recruiters it signals you know how to use a library, not that you understand the fundamentals.
- **Overfitting risk**: with 100 patients, “chasing” tiny gains just makes results unstable and less reproducible.
- **Time sink**: delays me getting to Neural Nets (the unique, impressive part of your project).

### Purpose of Baseline Classical LightGBM ML Model
1. Show I can prepare patient-level data for ML.
2. Provides a baseline classical ML benchmark for patient deterioration prediction.
3. Demonstrates an end-to-end ML workflow, and a credible, well-structured pipeline (data prep → CV → training → saving → validation → documentation → final deployment models).
4. Ensures reproducibility and robustness with cross-validation, and deployment readiness (final models).
5. Adds interpretability through feature importance, crucial in clinical healthcare settings.
6. Establishes a strong baseline Performance benchmark for later comparison with Neural Networks, showing their added value.

### Evaluation Metrics: MSE vs ROC-AUC vs Accuracy  
#### 1. **Mean Squared Error (MSE)**
- **Type:** Regression metric (continuous outcomes).
- **Definition:**  
  \[
  \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  \]  
  where \(y_i\) = true value, \(\hat{y}_i\) = predicted value.  
- **What it measures:**  
  - The *average squared difference* between predictions and actuals.  
  - Penalises large errors much more heavily than small ones (because of the square).  
- **How it evaluates:**  
  - Lower MSE = better fit.  
  - Perfect model → MSE = 0.  
- **Use case:**  
  - Regression tasks (e.g., predicting %time_high for a patient).  
  - Good when you care about magnitude of errors.  

#### 2. **Accuracy**
- **Type:** Classification metric (discrete categories).  
- **Definition:**  
  \[
  \text{Accuracy} = \frac{\text{# correct predictions}}{\text{total # predictions}}
  \]  
- **What it measures:**  
  - The proportion of predictions that are exactly correct.  
- **How it evaluates:**  
  - 0–1 range (or %).  
  - Example: if model got 85 out of 100 patients’ classes right → accuracy = 0.85.  
- **Limitations:**  
  - Misleading for *imbalanced datasets*.  
    - If 95% of patients are “low risk”, a dumb model that predicts “low risk” for everyone gets 95% accuracy, but is useless.  
- **Use case:**  
  - Quick baseline check for balanced classification problems.  
  - Less informative when classes are skewed.  

#### 3. **ROC-AUC (Receiver Operating Characteristic – Area Under Curve)**
- **Type:** Classification metric (binary or multiclass with extensions).  
- **Definition:**  
  - Plots **True Positive Rate (Sensitivity)** vs **False Positive Rate (1 – Specificity)** at different thresholds.  
  - AUC = area under that ROC curve (0–1).  
- **What it measures:**  
  - **Discrimination ability**: how well the model separates positive from negative classes.  
  - AUC = probability that the model assigns a higher score to a randomly chosen positive case than to a randomly chosen negative case.  
- **How it evaluates:**  
  - 0.5 = no better than random guessing.  
  - 1.0 = perfect discrimination.  
- **Strengths:**  
  - Works well even when classes are imbalanced.  
  - Evaluates the **ranking of predictions**, not just “hard” class labels.  
- **Use case:**  
  - Preferred metric for classifiers when outcomes are rare (e.g., detecting deteriorating patients).  
  - More informative than accuracy in medicine because it accounts for both sensitivity and specificity across thresholds.  

#### Summary Table

| Metric      | Task Type       | Range      | Goal   | Strengths                                   | Weaknesses |
|-------------|-----------------|------------|--------|---------------------------------------------|-------------|
| **MSE**     | Regression      | [0, ∞)     | Lower  | Penalises big errors; sensitive to scale.   | Hard to interpret clinically (units²). |
| **Accuracy**| Classification  | [0,1]      | Higher | Simple, intuitive.                          | Misleading with imbalanced data. |
| **ROC-AUC** | Classification  | [0.5,1]    | Higher | Robust to imbalance; measures discrimination. | Harder to intuitively explain to non-tech audience. |

#### For this project:
- Use **MSE** for `pct_time_high` (regression).  
- Use **ROC-AUC** for `max_risk` and `median_risk` (classification).  
- Fall back to **accuracy** only when a fold has a single class (so ROC-AUC is undefined).  


### What we did 
`complete_train_lightgbm.py`
 - sets up a loop for each target variable, selects the appropriate model type and metric, and prepares a list to store the results from cross-validation folds. It’s the first step in a modular, reusable training pipeline.



confused about this:
No – each saved LightGBM model file doesn’t store the X and y data. It only stores the trained model itself:
	•	The model contains all the information LightGBM learned from training on the X and y for that fold (splits, leaf values, weights, etc.).
	•	When you load the model later, you can use it to make predictions on new X data.

So the actual X_train and y_train aren’t stored in the file – only the learned parameters.

Workflow recap for clarity:
	1.	For fold 1 of max_risk:
	•	Train model on X_train_fold1 and y_train_fold1.
	•	Save model → saved_models/max_risk_fold1.txt.
	2.	Repeat for folds 2–5 and other targets.

Later, when predicting:
	•	Load each model.
	•	Pass in new X data (e.g., the corresponding test fold or external data).
	•	Get predictions.

This is why the saved files are small – they don’t contain the raw data, only the trained LightGBM model parameters.


1️⃣ train_idx / test_idx
	•	These are arrays of row numbers (integers) that tell Python which rows to select from your DataFrame.
train_index = [0,1,2,4,5,6,...,99]   # 80 integers
test_index  = [3,10,15,21,...,95]   # 20 integers
They map to positions in the DataFrame, not the actual content.

⸻
2️⃣ X_train / X_test after .iloc
X_train = X.iloc[train_idx]
X_test  = X.iloc[test_idx]
	•	.iloc takes the row numbers from train_index / test_index and extracts the corresponding rows.
	•	The resulting X_train and X_test retain the original row indices internally (you can see them in X_test.index) but most operations ignore them unless you explicitly print .index.

  	•	The term “index” refers to row positions in the original DataFrame, not a separate object in X_test or y_test.
	•	You only need to know the indices if you want to trace which exact patients are in which fold.


Feed to Model
  	•	X_train / y_train → model learns patterns
	•	X_test / y_test → model monitors performance for early stopping
	•	Each fold repeats with a different 20-patient test set and 80-patient training set.

   Repeat for All Folds
   	•	Every patient eventually appears in the test fold exactly once.
	•	This ensures full coverage for reliable performance estimation.

key points
  	1.	train_index / test_index → integers mapping rows in original dataset.
	2.	.iloc → extracts actual feature/label rows based on these indices.
	3.	X_train / y_train → used to train the model
	4.	X_test / y_test → used to evaluate/monitor the model per fold
	5.	Indices are preserved internally, but mostly ignored during training.


✅ Summary:
	•	Training → X_train, y_train (model knows these patients)
	•	Evaluation → X_test, y_test (model must generalise, doesn’t know these)
	•	Cross-validation ensures each patient is used for both training (4 times) and testing (once), without overlap within a fold.

  	•	X_train = the questions (heart rate, SpO₂, etc.)
	•	y_train = the answers (risk level)

During training, LightGBM guesses the answers → compares with y_train → updates itself to do better.
If you didn’t give it y_train, it would have nothing to compare against and wouldn’t learn anything.


Does the model reset between folds?
✅ Yes, the model resets at the start of every fold.
	•	Each fold trains a fresh LightGBM model (new instance of LGBMClassifier or LGBMRegressor).
	•	It is only trained on that fold’s X_train/y_train.
	•	Once trained and evaluated, it’s saved (or discarded), and the next fold starts from scratch.

So there isn’t one “master model” gradually seeing all 100 patients — it’s actually 5 separate models (one per fold).

Why reset?

If you didn’t reset:
	•	By the time you get to Fold 5, the model would have already seen all 100 patients in previous folds.
	•	That would defeat the purpose of having a “test set” → the model would indirectly know the answers.
	•	Evaluation would be optimistically biased and untrustworthy.

Resetting ensures:
	•	In each fold, the test patients are truly unseen.
	•	The performance metric reflects generalisation, not memory.

What you end up with
	•	5 trained models (e.g. max_risk_fold1.pkl, max_risk_fold2.pkl, …).
	•	5 scores (one per fold).
	•	A final average score across folds → this is the unbiased estimate of performance.

Step 1: What happens in k-fold CV
	•	For each outcome (max_risk, median_risk, pct_time_high), we train 5 models (one per fold).
	•	That’s why you’ll see 15 saved models total (3 outcomes × 5 folds).

So yes → at the end you’ve got 15 trained model files.

⸻

Step 2: Do we combine them into one model?

🔑 No — the models are not combined into one.
	•	The point of cross-validation is evaluation, not deployment.
	•	The goal is to estimate how well the model type (e.g. LightGBM classifier) will generalise to unseen patients.
	•	You average the 5 fold scores → that gives you a robust performance metric.

⚡ In summary:
	•	Cross-validation → gives you 15 models total.
	•	They are not combined; their scores are averaged.
	•	Afterwards, you usually train 1 final model per outcome on all the data.
	•	So you’ll finish with evaluation results (from 15 models) and 3 final models for deployment.