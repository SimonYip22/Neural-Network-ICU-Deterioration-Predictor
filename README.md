# EWS-Predictive-Dashboard
Python-based ICU Early Warning System (EWS) predictive dashboard. Uses time-series vitals to detect patient deterioration, benchmarked against NEWS2, with real-time alerts, trend visualisations, and a dual CLI/FastAPI interface. Portfolio-ready, deployed, and clinically-informed.

```text
Raw ICU Vitals (long format, MIMIC-style)
   └─> compute_news2.py
         ├─ Input: raw vitals CSV
         ├─ Action: compute NEWS2 scores per timestamp
         └─ Output: news2_scores.csv
               (wide format with vitals, NEWS2 score, escalation labels)

news2_scores.csv
   └─> make_timestamp_features.py
         ├─ Action:
         │   ├─ Apply LOCF per vital
         │   ├─ Add missingness flags
         │   ├─ Compute rolling window stats (1h/4h/24h)
         │   ├─ Compute time-since-last-observation
         │   └─ Encode risk/escalation as ordinal numeric
         └─ Output: news2_features_timestamp.csv
               (ML-ready timestamp-level features)

news2_features_timestamp.csv
   └─> make_patient_features.py
         ├─ Action:
         │   ├─ Aggregate per patient
         │   ├─ Compute median, mean, min, max, std per vital
         │   ├─ Include % missingness per vital
         │   └─ Optionally include derived trend features
         └─ Output: news2_features_patient.csv
               (ML-ready patient-level summary features)

ML Model (LightGBM)
   ├─ Input: news2_features_timestamp.csv and/or patient-level features
   ├─ Action:
   │   ├─ Train predictive model for deterioration / escalation
   │   ├─ Use timestamp trends + missingness flags
   │   └─ Evaluate performance (AUROC, precision-recall, etc.)
   └─ Output: predictions, feature importances, evaluation metrics
```

“We compute rolling window features over 1h, 4h, and 24h intervals. Mean, min, max, and std capture the magnitude and variability of vitals. Slope gives the trend — whether the vital is rising or falling and how fast. AUC measures cumulative exposure, i.e., how much and for how long a patient has experienced abnormal values. These features provide temporal context for the ML model, so it doesn’t just see isolated values but also their trajectory over time.”