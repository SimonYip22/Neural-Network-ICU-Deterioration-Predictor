# Neural Network (TCN) Time-Series ICU Deterioration Predictor (PyTorch & LightGBM) 📈⏳ 

Python-based ICU deterioration predictor leveraging advanced time-series ML modeling using TCN Neural Network (PyTorch) on timestamp-level vitals and LightGBM on patient-level features, capturing temporal trends, missingness, and sequence dynamics. 

Benchmarked against NEWS2, with real-time alerts and dual CLI/FastAPI deployment for real-time monitoring.

Portfolio-ready, deployed, and clinically-informed.

**Tech stack**: python, pandas, NumpPy, LightGBM, PyTorch

**Pipeline**
```text
Raw ICU Vitals (long format, MIMIC-style)
   └─> compute_news2.py
         ├─ Input: raw vitals CSV
         ├─ Action: compute NEWS2 scores per timestamp
         └─ Output: news2_scores.csv (wide format with vitals, NEWS2 score, escalation labels), news2_patient_summary.csv (patient-level summary)

news2_scores.csv
   └─> make_timestamp_features.py
         ├─ Action:
         │   ├─ Aggregate per patient
         │   ├─ Add missingness flags
         │   ├─ Apply LOCF per vital
         │   ├─ Compute carried-forward flags
         │   ├─ Compute rolling window stats (1h/4h/24h)
         │   ├─ Compute time-since-last-observation
         │   └─ Encode risk/escalation as ordinal numeric
         └─ Output: news2_features_timestamp.csv
               (ML-ready timestamp-level features)

news2_scores.csv
   └─> make_patient_features.py
         ├─ Action:
         │   ├─ Aggregate per patient
         │   ├─ Compute median, mean, min, max per vital
         │   └─ Include % missingness per vital
         └─ Output: news2_features_patient.csv
               (ML-ready patient-level summary features)
```

# Timestamp features rationale
- We compute rolling window features over 1h, 4h, and 24h intervals. 
   - Mean, min, max capture the magnitude and variability of vitals. 
   - Slope gives the trend — whether the vital is rising or falling and how fast. 
   - AUC measures cumulative exposure, i.e., how much and for how long a patient has experienced abnormal values. 
- These features provide temporal context for the ML model, so it doesn’t just see isolated values but also their trajectory over time.


# LightGBM vs Neural Network (TCN) Pipeline
```text
ML Model (LightGBM)
   ├─ Input: news2_features_patient.csv 
   │     ├─ Median, mean, min, max per vital
   │     ├─ Impute missing values
   │     ├─ % missing per vital
   │     └─ Risk summary stats (max, median, % time at high risk)
   ├─ Action:
   │     ├─ Train predictive model for deterioration / escalation
   │     ├─ Use timestamp trends + missingness flags
   │     └─ Evaluate performance (AUROC, precision-recall, etc.)
   └─ Output: predictions, feature importances, evaluation metrics

ML Model (Neural Network, TCN)
   ├─ Input: news2_features_timestamp.csv
   │     ├─ Timestamp-level vitals & rolling features (mean, min, max, std, slopes, AUC)
   │     ├─ Missingness flags
   │     ├─ Carried-forward flags  
   │     └─ Time since last observation
   ├─ Action:
   │     ├─ Train predictive model for deterioration / escalation
   │     ├─ Learn temporal patterns, trends, and interactions
   │     ├─ Can handle sequences of variable length per patient
   │     └─ Evaluate performance (AUROC, precision-recall, calibration)
   └─ Output: 
         ├─ Predictions per timestamp or per patient
         ├─ Learned feature embeddings / attention weights (if applicable)
         └─ Evaluation metrics
```

# LightGBM vs Neural Network (TCN) Pipeline Visualisation
```text
  Raw EHR Data (vitals, observations, lab results)
         │
         ▼
Timestamp Feature Engineering (news2_scores.csv)
 - Rolling statistics (mean, min, max, std)
 - Slopes, AUC, time since last observation
 - Imputation & missingness flags
         │
         ├─────────────► TCN Neural Network Model (v2)
         │              - Input: full time-series per patient
         │              - Can learn temporal patterns, trends, dynamics
         │
         ▼
Patient-Level Feature Aggregation (make_patient_features.py → news2_features_patient.csv)
 - Median, mean, min, max per vital
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

 # Model Comparison: LightGBM vs Neural Network (V1 & V2)

| Aspect | LightGBM (V1) | Temporal Convolutional Network (TCN) (V2) |
|--------|-------------------|-------------------|
| **ML Model Name / Type** | LightGBM (Gradient Boosted Decision Trees) | Temporal Convolutional Network (TCN)(Neural network) |
| **V1 / V2** | V1: uses patient-level features, baseline interpretable patient summary (classic tabular ML) | V2: uses timestamp-level features, advanced sequence modeling (modern deep learning) |
| **Input Datasets** | `news2_features_patient.csv` (patient-level summaries) | `news2_features_timestamp.csv` (time series of vitals, missingness flags) |
| **Optional Inputs** | Timestamp features could be added later for hybrid model | Patient-level summary features from `news2_features_patient.csv` can be appended but not mandatory |
| **Reason for this input choice** | LightGBM is a tree-based model: handles static features and aggregates well; does not naturally model temporal sequences | Neural networks (LSTM/TCN) can model temporal trends, sequences, and interactions over time; need full timestamp features to exploit sequential information |
| **Why two different models** | LightGBM: fast, interpretable (feature importance), strong baseline. | Neural network: captures temporal dynamics, can potentially improve predictive performance on time-series deterioration<br>Complements LightGBM; addresses potential limitations of static patient summaries by using sequential information in timestamp features |
| **Strengths** | - Handles missing values gracefully.<br>- Fast training and inference.<br>- Provides feature importances.<br>- Works well with tabular summary features. | - Models temporal trends and interactions.<br>- Can capture subtle patterns in sequences of vitals.<br>- Potentially better performance on real-time deterioration prediction. |
| **Weaknesses / Limitations** | - Ignores sequence and timing of events.<br>- May lose some granularity of patient trajectory.<br>- Cannot capture interactions over time. | - Requires more computation and tuning.<br>- Harder to interpret.<br>- Sensitive to missing data; requires careful imputation or masking. |
| **Output** | Predictions per patient, feature importances, evaluation metrics (AUROC, PR-AUC, etc.) | Predictions per timestamp or per patient trajectory, evaluation metrics (AUROC, PR-AUC, potentially time-dependent metrics) |
| **Use case / Deployment** | Baseline model; interpretable; fast deployment; can be used for early warning systems using summary features | Advanced model for final deployment or v2 experimentation; may be integrated in real-time monitoring dashboards for continuous deterioration prediction |


Portfolio narrative framing (objective and honest)

Here’s how you can present this:
	1.	State the limitation upfront:
	•	“Synthetic dataset contains very few high-risk events; patient-level deterioration classification targets were largely zero. Standard classification tasks were infeasible.”
	2.	Pivot your narrative to learnable outcomes:
	•	LightGBM: Predict patient-level NEWS2 / continuous risk burden, analyze feature importances to show clinical insights.
	•	TCN: Predict timestamp-level NEWS2 trends to capture dynamic risk evolution.
	3.	Metrics and comparison:
	•	Report regression metrics (RMSE, R², MAE).
	•	Compare to simple baselines (mean NEWS2, last observation carried forward) to show your model improves predictive performance.
	•	Highlight trend detection and feature influence, which is a clinically relevant skill.
	4.	Why this is still strong for a portfolio:
	•	Demonstrates data wrangling, preprocessing, CV, feature engineering, ML pipeline, model selection, hyperparameter tuning, and neural networks.
	•	Shows clinical insight (feature importance, temporal trends).
	•	Recruiters and technical reviewers care about how you solved real-world limitations, not just “predicted rare events.”