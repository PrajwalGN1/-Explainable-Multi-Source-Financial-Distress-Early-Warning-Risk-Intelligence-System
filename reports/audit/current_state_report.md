# Current State Audit & System Inventory Report

**Project:** Explainable SME Financial Distress Early-Warning & Risk Intelligence System  
**Audit Date:** October 12, 2026  
**Auditor Roles:** Senior Data Scientist, ML Engineer, Financial Risk Analytics Specialist, ML Research Auditor, Software Architect, QA Engineer  
**Audit Scope:** Repository Structure, Version Control Status, Codebase Architecture, Verified Functionality, Test Suite Results, and Feature Inventory  

---

## 1. Executive Context & Audit Mandate

This report establishes the baseline technical and scientific state of the existing **Financial Distress Early-Warning System (EWS)**. The codebase was developed to synthesize longitudinal corporate panel data, cross-sectional bankruptcy indicators, and unsupervised anomaly detection into an interpretable credit risk intelligence platform.

This audit evaluates the codebase strictly against empirical evidence in source files, datasets, trained model artifacts, serialized configurations, evaluation metrics, test execution results, and runtime environments. Features are evaluated based on verified functionality rather than declarative statements in documentation.

---

## 2. Repository Topology & Environment Status

### 2.1 Workspace & Version Control
- **Project Root:** `D:\Financial Distress`
- **Version Control Status:** **INITIALIZED (Phase 1 Baseline Established)**.
  - *Baseline Commit:* `509c866` (`baseline: initial state prior to Phase 1 audit remediation`)
  - *Active Working Branch:* `upgrade/v1.1-audit-fixes`
  - *Rollback Protection:* Guaranteed via Git baseline branch isolation.
- **Uncommitted Changes:** Phase 1 remediations active on branch `upgrade/v1.1-audit-fixes` (dashboard band synchronization, calibrated alert rules, thread-safe API database caching).

### 2.2 Execution Runtime & Dependencies
- **Active Python Environment:** Python 3.13.5 (`C:\Python313\python.exe`)
- **Package Manager:** pip / conda available.
- **Dependency File:** `requirements.txt` (14 packages specified: Streamlit, FastAPI, Uvicorn, Pydantic, Pandas, NumPy, SciPy, Scikit-Learn, XGBoost, Plotly, PyYAML, Joblib, Pytest, HTTPX).
- **Missing Declared Dependencies:** The `shap` package is referenced throughout project documentation and scripts but is **absent** from `requirements.txt` and not installed in the active environment.
- **Test Runner Observations:** Invoking the standalone `pytest` CLI triggered a deprecated Anaconda script wrapper (`C:\Users\DELL\anaconda3\Scripts\pytest-script.py`), whereas running `python -m pytest` executed cleanly under Python 3.13.5.

### 2.3 Directory Structure & Artifact Layout

```text
D:\Financial Distress\
├── .gitignore                      # Git ignore patterns (404 bytes)
├── README.md                       # Comprehensive project documentation (16.9 KB)
├── requirements.txt                # Python dependencies (215 bytes)
├── Dockerfile                      # Container build specification (625 bytes)
├── docker-compose.yml              # Multi-container orchestration (583 bytes)
├── LICENSE                         # MIT License (1,117 bytes)
├── app.py                          # Streamlit analytics dashboard (49.2 KB, 1,067 lines)
├── api.py                          # FastAPI REST microservice (7.0 KB, 164 lines)
├── config/
│   ├── config.yaml                 # Core system, data paths, and model hyperparameters (1.9 KB)
│   └── risk_config.yaml            # Risk category thresholds, alert rules, recommendations (2.5 KB)
├── Data/ & Data/raw/
│   ├── Financial Distress.csv      # Dataset A: Tehran Stock Exchange panel (2.2 MB, 3,672 rows, 86 cols)
│   ├── data.csv                    # Dataset B: Taiwan Economic Journal bankruptcy (11.5 MB, 6,819 rows, 96 cols)
│   └── bankruptcy_Test_X.csv       # Dataset C: Polish unlabelled test partition (6.2 MB, 5,000 rows, 65 cols)
├── Data/processed/
│   ├── companies_risk_intelligence.json # Precomputed risk dossiers for 422 firms (27.0 MB)
│   └── watchlist.csv               # Prioritized corporate surveillance table (32.8 KB, 422 rows)
├── models/
│   ├── ews/                        # Temporal EWS model, calibrator, preprocessor, metrics
│   ├── bankruptcy/                 # Bankruptcy model, calibrator, preprocessor, metrics
│   ├── anomaly/                    # Isolation Forest model, metadata
│   └── fusion/                     # Stacking meta-model, metadata
├── reports/
│   ├── data_audit/                 # Data summary, missing values, duplicates, profile JSON
│   ├── eda/                        # Distributions, class balance, top discriminators, PNG plots
│   ├── model_evaluation/           # Ablation study, experiment results, model comparison
│   └── explainability/             # Global feature importances
├── scripts/
│   ├── check_json.py, corr_analysis.py, deep_audit.py, inspect_distribution.py
│   ├── quick_inspect.py, refresh_intelligence_db.py, setup_dirs.py
│   └── smoke_test_dashboard.py, test_shap.py, test_thresholds.py, verify_native_shap.py
├── src/
│   ├── pipeline.py                 # End-to-end training and inference orchestrator (21.4 KB, 414 lines)
│   ├── config/settings.py          # YAML configuration loader
│   ├── data/                       # Loaders, validators, data auditing, EDA
│   ├── preprocessing/              # Leakage-safe EWS & Bankruptcy preprocessors
│   ├── features/                   # Temporal engine, financial health engine, feature registry
│   ├── models/                     # EWS, Bankruptcy, Anomaly, Platt calibration
│   ├── fusion/                     # Two-level risk fusion & stacking meta-learner
│   ├── explainability/             # ModelExplainer (native booster vs linear approximation)
│   ├── risk_engine/                # Trajectory engine, deterministic alerts, recommendations
│   └── utils/                      # Logging and serialization utilities
└── tests/
    ├── conftest.py                 # Path injection fixture
    ├── test_api.py                 # 8 API endpoint tests
    ├── test_features_and_health.py # 4 feature and health score tests
    ├── test_fusion_and_scoring.py  # 4 fusion, trajectory, and alert tests
    └── test_leakage_prevention.py  # 4 temporal lag and preprocessor leakage tests
```

---

## 3. Entry Points & Operational Commands

The project defines four execution entry points:

| Workflow | Command | Entry File | Verification Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Full Pipeline Training** | `python src/pipeline.py` | `src/pipeline.py` | **Implemented and Verified** | Executes end-to-end: loads 3 datasets, fits 3 models, trains meta-model, computes ablation, outputs 27MB JSON DB. |
| **Streamlit Dashboard** | `streamlit run app.py` | `app.py` | **Implemented and Verified** | Launches dark-mode dashboard with 10 diagnostic views on port 8501. Verified via smoke test. |
| **FastAPI REST Service** | `uvicorn api:app --reload --port 8000` | `api.py` | **Implemented and Verified** | Serves Pydantic-validated endpoints (`/health`, `/predict`, `/company/{id}`, `/watchlist`, `/risk-summary`). |
| **Automated Test Suite** | `python -m pytest tests/ -v` | `tests/` | **Implemented and Verified** | 31 unit and integration tests passing in 3.59s total runtime (expanded from 20 in Phase 1). |



---

## 4. Empirical Test Suite Execution Results

Running the automated test suite with Python 3.13.5 produced the following verified output:

```text
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Financial Distress
plugins: anyio-4.12.0

tests/test_features_and_health.py::test_feature_registry_lookup PASSED   [  5%]
tests/test_features_and_health.py::test_financial_health_score_bounds PASSED [ 10%]
tests/test_features_and_health.py::test_financial_health_edge_cases_and_clipping PASSED [ 15%]
tests/test_features_and_health.py::test_financial_health_dataset_a_row PASSED [ 20%]
tests/test_leakage_prevention.py::test_lag_features_are_strictly_backward_looking PASSED [ 25%]
tests/test_leakage_prevention.py::test_rolling_statistics_are_strictly_backward_looking PASSED [ 30%]
tests/test_leakage_prevention.py::test_forward_target_alignment PASSED   [ 35%]
tests/test_leakage_prevention.py::test_preprocessor_does_not_leak_test_statistics PASSED [ 40%]
tests/test_fusion_and_scoring.py::test_risk_fusion_bounds PASSED         [ 45%]
tests/test_fusion_and_scoring.py::test_trajectory_engine PASSED          [ 50%]
tests/test_fusion_and_scoring.py::test_early_warning_alerts PASSED       [ 55%]
tests/test_fusion_and_scoring.py::test_recommendation_engine PASSED      [ 60%]
tests/test_api.py::test_health_endpoint PASSED                           [ 65%]
tests/test_api.py::test_model_info_endpoint PASSED                       [ 70%]
tests/test_api.py::test_predict_endpoint_valid PASSED                    [ 75%]
tests/test_api.py::test_predict_endpoint_invalid_inputs PASSED           [ 80%]
tests/test_api.py::test_company_profile_endpoint PASSED                  [ 85%]
tests/test_api.py::test_company_profile_not_found PASSED                 [ 90%]
tests/test_api.py::test_watchlist_endpoint PASSED                        [ 95%]
tests/test_api.py::test_risk_summary_endpoint PASSED                     [100%]

============================== 20 passed in 39.93s ==============================
```

### Dashboard Smoke Test Verification
Execution of `python scripts/smoke_test_dashboard.py`:
- Verified cached loaders in `app.py`:
  - `load_risk_db()`: 422 company profiles loaded successfully.
  - `load_watchlist()`: 422 rows loaded successfully.
  - `load_ablation_study()`: 6 experiment rows loaded successfully.
  - `load_global_importance()`: 25 feature rows loaded successfully.
- Streamlit runtime deprecation warning observed:
  `Please replace use_container_width with width. use_container_width will be removed after 2025-12-31.`

---

## 5. Comprehensive Feature & Capability Inventory

Status Categories:
- **Implemented and verified:** Code exists, runs, produces valid results, and has passing test/runtime verification.
- **Implemented but unverified:** Code exists and runs, but scientific validity or operational correctness is unproven.
- **Partially implemented:** Feature exists in code, but key components are missing, hardcoded, or disconnected.
- **Broken:** Code produces invalid results, crashes, or contains internal contradictions.
- **Missing:** Required capability is completely absent.
- **Not applicable:** Out of scope for this architecture.

| Capability / Module | Status | Evidence (File, Class, Method) | Actual Finding / Limitation |
| :--- | :--- | :--- | :--- |
| **Data Ingestion & Loading** | **Implemented and verified** | `src/data/loaders.py`: `DataLoader.load_all` | Successfully loads all 3 CSVs from `Data/raw/`. Enforces type cleanliness. |
| **Data Validation Rules** | **Partially implemented** | `src/data/validators.py`: `DataValidator` | Validation methods exist and check monotonicity, but are **never invoked** in `src/pipeline.py`. |
| **Chronological Time Split** | **Implemented and verified** | `src/models/ews_model.py`: `prepare_data` | Train: $T \le 10$, Val: $T \in [11, 12]$, Test: $T=13$. Strict temporal ordering observed. |
| **Forward Target Construction ($t \to t+1$)** | **Implemented and verified** | `src/features/temporal_features.py`: `construct_forward_target` | Associates features at $t$ with distress at $t+1$. Right-censored rows correctly marked NaN. |
| **Temporal Lag & Delta Engineering** | **Partially implemented** | `src/features/temporal_features.py`: `generate_temporal_features` | Lags, deltas, momentum, and rolling means are leak-free, but applied **only to 8 hardcoded features** (`x1, x2, x5, x10, x25, x48, x64, x81`). The remaining 75 features remain static. |
| **Feature Registry** | **Partially implemented** | `src/features/feature_registry.py`: `FeatureRegistry` | Maps 12 ratios from Dataset B and 5 variables from Dataset A. However, financial interpretations for `x81, x48, x25, x64` are **unsubstantiated conjectures** based on correlation rather than publisher documentation. |
| **Financial Health Diagnostics Engine** | **Broken** | `src/features/financial_features.py`: `compute_for_dataset_a_row` | Computes dimensional subscores (profitability, liquidity, leverage, solvency) by mapping `x48, x25, x64, x81` into sigmoid formulas without accounting provenance. |
| **Preprocessors (Imputation & Scaling)** | **Implemented and verified** | `src/preprocessing/preprocessors.py`: `EWSPreprocessor`, `BankruptcyPreprocessor` | Fitted exclusively on training sets. Transforms validation and test sets without leaking test distribution moments. Note: Treats discrete sector code `x80` as continuous. |
| **Temporal EWS Model (Dataset A)** | **Implemented but unverified** | `src/models/ews_model.py`: `TemporalEWSModel` | Compares Logistic Regression, Random Forest, and XGBoost. Selects Random Forest (`ROC-AUC = 0.7783`, `PR-AUC = 0.1946`). **Critical defect:** At 0.5 decision threshold, Test Recall is **0.0%** (misses all 11 distress events). |
| **Cross-Sectional Bankruptcy Model (Dataset B)** | **Implemented and verified** | `src/models/bankruptcy_model.py`: `BankruptcyModel` | Stratified holdout split. Random Forest achieves `ROC-AUC = 0.9455`, `Precision = 72.2%`, `Recall = 29.55%`. |
| **External Anomaly Detection (Dataset C)** | **Implemented and verified** | `src/models/anomaly_model.py`: `AnomalyOODModel` | Isolation Forest fitted on 5,000 Polish records. Generates [0, 1] normalized outlier score and PSI. |
| **Two-Level Multi-Source Fusion** | **Broken** | `src/pipeline.py`: lines 76–95, `src/fusion/risk_fusion.py` | Stacking meta-model requires bankruptcy and anomaly inputs for Dataset A firms. Because schemas differ, the pipeline **invented pseudo-proxy formulas** on `x64` and `x81` to manufacture synthetic inputs. Result: Meta-model ROC-AUC dropped from 0.7783 to 0.6588. |
| **Probability Calibration** | **Implemented and verified** | `src/models/calibration.py`: `ProbabilityCalibrator` | Sigmoid Platt scaling fitted on validation fold. Reduces Brier score from 0.0626 to 0.0387. However, causes threshold compression where probabilities rarely exceed 0.20. |
| **Explainable AI (SHAP)** | **Broken** | `src/explainability/shap_explainer.py`: `explain_instance_shap` | Pipeline claims native TreeSHAP. Because `RandomForestClassifier` was selected (and `shap` library is uninstalled), code falls back to `feature_row * norm_importances`. This is **not SHAP**; it is an ad-hoc unscaled product of feature value and Gini importance. |
| **Longitudinal Trajectory Engine** | **Implemented and verified** | `src/risk_engine/trajectory.py`: `RiskTrajectoryEngine` | Calculates delta, acceleration, consecutive deterioration streaks, and pattern classifications across historical periods. |
| **Deterministic Early Warning Alerts** | **Broken** | `src/risk_engine/alerts.py`: `EarlyWarningSignalEngine` | Alert Rule 1 requires probability $\ge 0.70$ or $\ge 0.50$. Since calibrated probabilities in the portfolio max out at 0.1913, Rule 1 **never fires** for any firm. |
| **Actionable Recommendations** | **Implemented and verified** | `src/risk_engine/recommendations.py`: `RecommendationEngine` | Generates prioritized surveillance protocols with prominent academic research disclaimers. |
| **Streamlit Analytics Dashboard** | **Implemented and verified** | `app.py`, `dashboard/components/` | 10 interactive views with dark-mode styling, responsive Plotly charts, and caching. **Visual defect:** Gauge and timeline charts use hardcoded 0-20, 20-40 bands, causing Critical firms (score 17-19) to display inside the green "Healthy" zone. |
| **FastAPI REST Endpoints** | **Implemented and verified** | `api.py` | 6 functioning endpoints verified with 8 passing integration tests. **Efficiency defect:** Reads 27MB JSON database from disk on every `/company/{id}` request without in-memory caching. |
| **Containerization Deployment** | **Implemented but unverified** | `Dockerfile`, `docker-compose.yml` | Container specifications are syntactically valid and map ports 8501 and 8000. Not verified inside a running Docker daemon in this environment. |

---

## 6. Synthesis of Critical Findings

1. **Working Foundation:** The application startup, data loading, time-aware splitting, test suite, Streamlit dashboard, and FastAPI microservice are functional and demonstrable.
2. **Operational Decision Failure:** The primary EWS model has an operational test recall of **0.0%** at threshold 0.5 because probability calibration compresses outputs into the 0.0–0.2 range without re-calibrating the classification threshold.
3. **Pseudo-Evidence Contamination:** Heterogeneous datasets (Iranian panel vs Taiwan TEJ vs Polish unlabelled) were fused by fabricating synthetic proxy calculations rather than maintaining defensible, decoupled risk scores.
4. **Explainability Defect:** The system advertises TreeSHAP explanations but actually computes `feature_row * Gini_importance` because the selected model is Random Forest and the `shap` package is absent.
5. **UI / Threshold Inconsistency:** Risk category boundaries in `risk_config.yaml` ($>17\%$ is Critical) contradict the visual bands in dashboard charts (0–20 is Healthy), resulting in Critical firms appearing in green zones.
