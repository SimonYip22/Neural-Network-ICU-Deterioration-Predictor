# Phase 1: Baseline NEWS2 Tracker

## Day 1: NEWS2 Data Extraction and Preliminary Scoring

## Pipeline Overview

Raw CSVs (chartevents.csv, etc.)
        │
        ▼
extract_news2_vitals.py
        │
        ▼
news2_vitals.csv
        │
        ▼
check_co2_retainers.py
        │
        ▼
news2_vitals_with_co2.csv + co2_retainer_details.csv
        │
        ▼
compute_news2.py
        │
        ▼
Final NEWS2 scores per patient

## Goals
- Extract relevant vital signs from PhysioNet.org MIMIC-IV Clinical Database Demo synthetic dataset for NEWS2 scoring.  
- Identify and flag CO₂ retainers to ensure accurate oxygen scoring.  
- Implement basic NEWS2 scoring functions in Python.  
- Establish a pipeline that is extendable for future interoperability with real-world clinical data.

## What We Did
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

## Reflections
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

## Issues Encountered
- Confusion around GCS mapping and timestamp alignment.  
- Initial uncertainty about FiO₂ and temperature units.  
- Need to verify CO₂ retainer thresholds and data format.  
- Feeling overwhelmed by the complexity of clinical data pipelines and Python functions.

## Lessons Learned
- Extracting and standardising clinical data is a critical and time-consuming first step.  
- Structuring data in CSVs with consistent headers simplifies downstream processing.  
- Python dictionaries and tuple-based thresholds are powerful for flexible clinical scoring functions.  
- Documenting assumptions (temperature units, FiO₂ thresholds) is essential for reproducibility.

## Future Interoperability Considerations
- Pipeline designed to support ingestion of FHIR-based EHR data for future integration.  
- Potential extension: map standardized FHIR resources to predictive EWS pipeline for real-world applicability.

## CO₂ Retainer Validation and NEWS2 Scoring Documentation
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
