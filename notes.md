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

1. **Validating NEWS2 Scoring**
  - **Action**: Ran validate_news2_scoring.py on test dictionaries.
  - **Findings**:
    - Low GCS cases initially produced incorrect scores.
    - The scoring function ignored consciousness because row.get("level_of_consciousness", pd.NA) returned pd.NA.
    -	Other special cases (SpO₂, supplemental O₂) were correctly scored because their thresholds were handled explicitly.
  - **Fixes**: Moved `if pd.isna(value): return 0` **to the end of the function**.
  - **Outcome**: All unit tests passed, learned the importance of understanding intermediate variables in scoring pipelines.
  - The main pipeline did not have these problems as gcs_total is converted into level_of_consciousness before the scoring is called, so there was no missing keys.

2. **Missing Data Strategy**
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

3. **Preparing Timestamp-Level ML Features**

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
  8.	Save news2_features_timestamp.csv.

  **Rationale**:
  - Trees can leverage trends and missingness.
  -	Rolling windows capture short-, medium-, and long-term deterioration patterns.
  -	Timestamp features feed ML models like LightGBM directly without further preprocessing.

4. **Preparing Patient-Level ML Features**

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


5. **ML Model Selection**
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
- Reason:
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

