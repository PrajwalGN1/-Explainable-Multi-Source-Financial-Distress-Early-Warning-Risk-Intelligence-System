# Phase 2 Model & Experimental Artifact Manifest

**Project:** Explainable SME Financial Distress Early-Warning & Risk Intelligence System  
**Audit Date:** October 12, 2026  
**Auditor Roles:** Senior Data Scientist, ML Engineer, Financial Risk Analytics Specialist, ML Research Auditor  
**Branch:** `upgrade/v1.2-model-validation`  
**Phase:** Phase 2 (Model Validation, Stacking Refactoring & Early-Warning Reliability)  

---

## 1. Overview & Artifact Isolation Policy

To guarantee research integrity and full backward compatibility, all Phase 2 model checkpoints, preprocessor pipelines, calibrators, evaluation metrics, and threshold configurations are stored **in a dedicated experimental directory (`models/phase2/`) completely isolated from the original baseline models (`models/ews/`, `models/bankruptcy/`, `models/anomaly/`, `models/fusion/`)**. 

Historical baseline artifacts and Phase 1 test checkpoints remain unmodified for long-term audit comparison.

---

## 2. Phase 2 Model Artifacts Directory (`models/phase2/`)

| Artifact Name | File Path | Format / Size | Description & Mathematical Specifications | Provenance / Split |
| :--- | :--- | :---: | :--- | :--- |
| **Primary EWS Model** | `models/phase2/best_ews_model.joblib` | Joblib / ~450 KB | `RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)` | Fitted strictly on Train ($T \le 10$) |
| **EWS Preprocessor** | `models/phase2/ews_preprocessor.joblib` | Joblib / ~15 KB | `EWSPreprocessor` encapsulating median imputation and `RobustScaler` across 163 temporal panel columns | Fitted strictly on Train ($T \le 10$) |
| **Probability Calibrator**| `models/phase2/ews_calibrator.joblib` | Joblib / ~5 KB | Platt Scaling (`CalibratedClassifierCV(method='sigmoid', cv='prefit')`) | Fitted strictly on Val ($T \in [11, 12]$) |
| **Threshold Configuration**| `models/phase2/threshold_config.json` | JSON / ~400 B | Operational threshold $\tau^* = 0.080$, critical alert 0.170, high alert 0.130, risk score bands | Optimized on Val ($T \in [11, 12]$) |
| **Evaluation Metrics** | `models/phase2/ews_metrics.json` | JSON / ~2.5 KB | Validation metrics, holdout test metrics at $\tau^*=0.08$ and 0.50, Wilson 95% CI, Brier scores | Validated on Val ($T=11,12$) & Test ($T=13$) |
| **Model Manifest** | `models/phase2/model_manifest.json` | JSON / ~800 B | Model metadata, training periods, proxy status, ranking discrimination, and deployment status | System registry |

---

## 3. Experimental Research Reports & CSV Tables (`reports/phase2/`)

| Artifact Name | File Path | Format / Rows | Key Contents & Metrics |
| :--- | :--- | :---: | :--- |
| **Candidate Comparisons** | `reports/phase2/candidate_model_comparisons.csv` | CSV / 8 rows | Comprehensive validation metrics comparing Logistic Regression, Random Forest (depths 5, 8), and XGBoost under weighted and unweighted regimes. |
| **Proxy Contamination Experiments** | `reports/phase2/proxy_contamination_experiments.csv` | CSV / 4 rows | Empirical results for Experiment A (Contaminated Stacking, Test ROC 0.6588), Experiment B (Clean Primary Model, Test ROC 0.7783), and Experiment C (Leak-Free OOF Stacking, Test ROC 0.7863). |
| **Validation Threshold Analysis** | `reports/phase2/validation_threshold_analysis.csv` | CSV / 7 rows | Full confusion matrices, precision, recall, F1, F2, specificity, and review workloads across candidate cutoffs $\{0.05, 0.08, 0.10, 0.13, 0.17, 0.50\}$ on $T \in [11, 12]$. |
| **Test Threshold Analysis** | `reports/phase2/test_threshold_analysis.csv` | CSV / 7 rows | Out-of-period holdout test performance across candidate cutoffs on untouched $T=13$. |

---

## 4. Audit Deliverables Inventory (`reports/audit/`)

| Deliverable Name | File Path | Scope |
| :--- | :--- | :--- |
| **Phase 2 Model Comparison** | `reports/audit/phase2_model_comparison.md` | Candidate architectures, proxy ablation, OOF stacking, and primary model selection rationale. |
| **Phase 2 Threshold Analysis** | `reports/audit/phase2_threshold_analysis.md` | Validation threshold optimization, untouched test results, Wilson score CI, and risk score mapping. |
| **Phase 2 Validation Report** | `reports/audit/phase2_validation_report.md` | Comprehensive validation audit, temporal splits, calibration diagnostics, and operational surveillance analysis. |
| **Phase 2 Leakage Review** | `reports/audit/phase2_leakage_review.md` | Matrix of all 8 leakage vectors, preprocessor isolation, target horizons, and remediated risks. |
| **Phase 2 Artifact Manifest** | `reports/audit/phase2_artifact_manifest.md` | This document: complete registry of models, configurations, and verification instructions. |

---

## 5. Preservation of Baseline Artifacts

The following original baseline directories and files were preserved without modification:
- `models/ews/best_ews_model.joblib` (Original Phase 1 baseline checkpoint)
- `models/ews/ews_metrics.json` (Original evaluation report recording 0% recall at threshold 0.50)
- `models/bankruptcy/` (Cross-sectional Taiwan bankruptcy model and metrics)
- `models/anomaly/` (Dataset C Isolation Forest model and metadata)
- `models/fusion/` (Baseline multi-source fusion configuration)
- `reports/model_evaluation/ablation_study.csv` (Original ablation study)
- `reports/model_evaluation/model_comparison.csv` (Original comparison table)

---

## 6. Reproducibility & Verification Instructions

### 6.1 Re-running Empirical Experiments
To reproduce all model comparisons, proxy ablations, threshold optimizations, and holdout test evaluations:
```bash
python scripts/phase2_model_experiments.py
```

### 6.2 Executing Automated Regression & Validation Test Suite
To execute the complete 41-test suite (including 10 Phase 2 validation tests and 31 Phase 1 regression tests):
```bash
python -m pytest tests/ -v
```

### 6.3 Loading Phase 2 Checkpoint in Python
```python
import joblib
from src.preprocessing.preprocessors import EWSPreprocessor

model = joblib.load("models/phase2/best_ews_model.joblib")
preprocessor = EWSPreprocessor.load("models/phase2/ews_preprocessor.joblib")
calibrator = joblib.load("models/phase2/ews_calibrator.joblib")

# Inference on new panel data
X_scaled, _ = preprocessor.transform(df_new)
calibrated_prob = calibrator.predict_proba(X_scaled)[:, 1]
operating_distress_predictions = (calibrated_prob >= 0.08).astype(int)
```
