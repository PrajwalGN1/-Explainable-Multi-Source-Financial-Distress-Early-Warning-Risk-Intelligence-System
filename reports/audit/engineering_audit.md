# Software Engineering & Production Readiness Audit Report

**Project:** Explainable SME Financial Distress Early-Warning & Risk Intelligence System  
**Audit Scope:** PHASE 10 — Audit Software Engineering  
**Evidence Source:** Root configuration files, `src/`, `tests/`, `api.py`, `app.py`, `Dockerfile`, `docker-compose.yml`, test execution logs, and runtime profiling.

---

## 1. Executive Summary: Software Engineering Audit

This audit evaluates the codebase against production-grade software engineering standards, focusing on architecture, dependency management, reproducibility, data validation, serialization, test coverage, API design, memory efficiency, security, and deployment readiness.

### Key Strengths:
- **Clean Modular Architecture:** Well-organized separation of concerns across `src/data`, `src/preprocessing`, `src/features`, `src/models`, `src/fusion`, `src/explainability`, `src/risk_engine`, and `src/utils`.
- **Reproducibility:** Global random seed (`42`) is systematically enforced across configurations, data splitting, model initializations, and tree ensembles.
- **Passing Test Suite:** 20 automated unit and integration tests execute and pass cleanly under Python 3.13.5.
- **Security & Secrets Hygiene:** Zero API keys, passwords, tokens, or confidential database credentials are exposed in source code, configuration files, or logs.

### Key Engineering Deficiencies:
1. **Dependency Discrepancy:** `requirements.txt` omits the `shap` package despite extensive references across documentation and scripts.
2. **Disconnected Data Validators:** `DataValidator` in `src/data/validators.py` implements schema and monotonicity checks, but is **never invoked** within the operational training pipeline (`src/pipeline.py`).
3. **API I/O Bottleneck:** The FastAPI endpoint `/company/{company_id}` re-reads the entire 27 MB `companies_risk_intelligence.json` file from disk on **every single incoming HTTP request**.
4. **Test Suite Coverage Blindspots:** Existing tests validate that functions execute and return bounded values, but fail to assert scientific correctness (e.g., they do not test that operational recall exceeds 0%, nor do they verify that alert rules trigger under empirical probability ranges).

---

## 2. In-Depth Engineering Dimension Audit

### 2.1 Dependency Management & Environment Compatibility
- **Dependency Specification:** `requirements.txt` specifies 14 core packages with lower-bound version pinning (`>=`).
- **Defects:**
  - `shap` is absent from `requirements.txt`.
  - Python 3.13 compatibility: The active environment is Python 3.13.5. Certain scientific C-extensions (like older `shap` binary wheels) have limited official support for Python 3.13 on Windows.
  - The `Dockerfile` pins `FROM python:3.11-slim`, which is more stable for ML workloads than Python 3.13.
- **Recommendation:** Add `shap>=0.44.0` or standardize on XGBoost's built-in native TreeSHAP to eliminate the dependency entirely.

---

### 2.2 Reproducibility & Random Seeds
- **Seed Management:** Centrally declared in `config/config.yaml`:
  `project.random_seed: 42`.
- **Enforcement Across Codebase:**
  - `src/models/ews_model.py`: `LogisticRegression(random_state=42)`, `RandomForestClassifier(random_state=42)`, `XGBClassifier(random_state=42)`.
  - `src/models/bankruptcy_model.py`: `train_test_split(random_state=42)`, all classifiers seeded.
  - `src/models/anomaly_model.py`: `IsolationForest(random_state=42)`.
  - `src/fusion/meta_model.py`: `LogisticRegression(random_state=42)`.
- **Assessment:** **Fully compliant.** Model training and evaluation splits are strictly deterministic.

---

### 2.3 Configuration Management
- **Implementation:** Configuration is externalized into YAML files:
  - `config/config.yaml`: System metadata, paths, data splitting periods, model hyperparameters.
  - `config/risk_config.yaml`: Risk score ranges, category intervals, alert thresholds, surveillance recommendations.
- **Loader:** `src/config/settings.py` provides cached helpers (`get_config()`, `get_risk_config()`).
- **Assessment:** Well-structured. However, as documented in Phase 8, several threshold definitions in `risk_config.yaml` were overridden by hardcoded values in `pipeline.py` and `charts.py`.

---

### 2.4 Data Validation & Pipeline Integration
- **Implementation:** `src/data/validators.py` implements:
  - `validate_dataset_a`: Verifies required columns, infinite values, and per-firm time monotonicity ($\Delta Time > 0$).
  - `validate_dataset_b`: Verifies binary targets and schema.
  - `validate_dataset_c`: Verifies numeric features.
- **Critical Defect:** In `src/pipeline.py`, `DataValidator` is **never imported or called** during `run_full_pipeline()`. Data validation exists as orphaned code.
- **Correction:** Integrate `DataValidator` calls at Step 1 of `RiskIntelligencePipeline.run_full_pipeline`, raising explicit exceptions if input data fails schema or monotonicity constraints.

---

### 2.5 Model Serialization & Artifact Versioning
- **Serialization Format:** Uses `joblib` for scikit-learn and XGBoost estimators; `json` for evaluation metrics and metadata.
- **Artifact Tree:**
  - `models/ews/`: `best_ews_model.joblib`, `ews_calibrator.joblib`, `ews_preprocessor.joblib`, `ews_metrics.json`.
  - `models/bankruptcy/`: `best_bankruptcy_model.joblib`, `bankruptcy_calibrator.joblib`, `bankruptcy_preprocessor.joblib`, `bankruptcy_metrics.json`.
  - `models/anomaly/`: `isolation_forest.joblib`, `anomaly_meta.json`.
  - `models/fusion/`: `meta_model.joblib`, `fusion_meta.json`.
- **Assessment:** Serialization is complete and self-contained. Preprocessors are saved alongside models, ensuring that imputers and scalers are available during inference.
- **Limitation:** Artifacts are overwritten in-place without semantic versioning or checksum validation.

---

### 2.6 Training / Inference Consistency
- **Evaluation:**
  - The `EWSPreprocessor` class enforces identical column ordering during `transform()`:
    ```python
    for col in self.feature_names:
      if col not in X.columns:
        X[col] = 0.0
    X = X[self.feature_names]
    ```
  - Preprocessor parameters (median values, robust scaling medians and IQRs) learned on the training split are applied identically during validation, testing, and production inference.
- **Assessment:** **Consistent and robust against feature order permutation.**

---

### 2.7 API Architecture & Performance Bottleneck (`api.py`)
- **Framework:** FastAPI with Pydantic schema validation.
- **Endpoints:**
  - `GET /health`: System heartbeat.
  - `GET /model-info`: Architectural metadata.
  - `POST /predict`: Real-time risk scoring.
  - `GET /company/{company_id}`: Full corporate dossier.
  - `GET /watchlist`: Filterable corporate table.
  - `GET /risk-summary`: Universe risk statistics.
- **Critical Performance Defect in `api.py` (lines 29–33, 138):**
  ```python
  def get_db() -> Dict[str, Any]:
    if os.path.exists(PROCESSED_DB_PATH):
      with open(PROCESSED_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
    return {"companies": {}, "metadata": {}}


  @app.get("/company/{company_id}")
  def get_company_profile(company_id: str):
    db = get_db()  # <-- RELOADS 27 MB JSON FILE ON EVERY SINGLE REQUEST!
    ...
```
- **Forensic Impact:** Parsing a 27 MB JSON file in Python takes approximately **150–300 ms of CPU time**. Under concurrent requests, this will bottleneck the API and exhaust server CPU.
- **Correction:** Cache the database dictionary in memory at application startup:
  ```python
  _CACHED_DB = None


  def get_db():
    global _CACHED_DB
    if _CACHED_DB is None and os.path.exists(PROCESSED_DB_PATH):
      with open(PROCESSED_DB_PATH, "r", encoding="utf-8") as f:
        _CACHED_DB = json.load(f)
    return _CACHED_DB or {"companies": {}, "metadata": {}}
```

---

### 2.8 Unit & Integration Test Suite Audit

The project contains 4 test files with 20 passing tests:

| Test File | Tests | Focus Areas | Blindspots / Missing Assertions |
| :--- | :---: | :--- | :--- |
| `tests/test_leakage_prevention.py` | 4 | Lag-1/2 backwardness, rolling mean backwardness, forward target alignment, preprocessor isolation. | Strong coverage of temporal mechanics. |
| `tests/test_features_and_health.py` | 4 | Feature registry lookups, health score bounds, edge-case clipping, Dataset A row scoring. | Tests bounded output $[0, 100]$, but does not verify semantic accounting validity. |
| `tests/test_fusion_and_scoring.py` | 4 | Risk fusion bounds, trajectory trend detection, alert signal generation, recommendations. | Tests alert engine with artificial `prob = 0.75`, failing to catch that production probabilities ($<0.20$) never trigger the rule. |
| `tests/test_api.py` | 8 | All 6 FastAPI endpoints, Pydantic 422 error on invalid inputs, 404 on missing company. | Comprehensive API contract coverage. |

---

### 2.9 Containerization & Deployment Configuration
- **Dockerfile:** Based on `python:3.11-slim`, installs `build-essential` and `curl`, copies requirements and source, exposes ports 8501 and 8000.
- **Docker Compose:** Defines two services (`dashboard` on 8501, `api` on 8000) sharing local data and model volumes.
- **Assessment:** Clean, reproducible container setup. Note: Running both Streamlit and FastAPI as separate containers copying the same root directory is standard for local development, but in production, `api` should serve as the backend and `dashboard` as the frontend client.

---

## 3. Summary of Engineering Corrections

1. **In-Memory Caching for API (`P0`):** Eliminate the disk read on `/company/{id}` by caching `companies_risk_intelligence.json` in memory at startup.
2. **Integrate `DataValidator` in Pipeline (`P1`):** Hook `DataValidator` into `pipeline.py` to abort early if data corruption or time non-monotonicity is detected.
3. **Synchronize Dependencies (`P1`):** Add `shap` to `requirements.txt` or adopt XGBoost native TreeSHAP.
4. **Expand Test Suite (`P1`):** Add tests asserting that test recall exceeds 0% and that alert triggers fire under calibrated probability ranges.
