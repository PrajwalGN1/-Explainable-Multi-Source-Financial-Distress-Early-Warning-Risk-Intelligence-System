# Executive Audit Summary & Research Assessment

**Project:** Explainable SME Financial Distress Early-Warning & Risk Intelligence System  
**Audit Date:** October 12, 2026  
**Auditor Roles:** Senior Data Scientist, ML Engineer, Financial Risk Analytics Specialist, ML Research Auditor, Software Architect, QA Engineer  
**Deliverable:** Master Synthesis & Strategic Assessment  

---

## 1. Executive Overview

This audit establishes the actual scientific, mathematical, and operational state of the **Explainable SME Financial Distress Early-Warning System**. 

The existing implementation represents an ambitious, well-structured codebase with a functional Streamlit analytics dashboard, a Pydantic-validated FastAPI REST service, and 20 passing automated tests. 

However, beneath its polished visual presentation, the audit uncovered **critical scientific vulnerabilities, operational decision failures, explainability compromises, and threshold misalignments** that must be resolved before this system can be considered a trustworthy research prototype or institutional credit intelligence tool.

---

## 2. Definitive Answers to the 10 Core Audit Inquiries

### Question 1: What is genuinely implemented today?
A complete, functioning software architecture is implemented:
- **Data Ingestion:** Loaders in `src/data/loaders.py` cleanly ingest three distinct datasets: Dataset A (Tehran panel, 3,672 rows), Dataset B (Taiwan TEJ, 6,819 rows), and Dataset C (Polish test partition, 5,000 rows).
- **Leakage-Safe Preprocessing:** Preprocessors in `src/preprocessing/preprocessors.py` isolate median imputation and robust scaling to training partitions.
- **Temporal Panel Splitting:** A forward chronological partition ($T \le 10$ Train, $T \in [11, 12]$ Val, $T=13$ Test) prevents lookahead leakage.
- **Forward Target Builder:** Constructs a $t \to t+1$ distress target based on terminal distress ($\le -0.50$).
- **Temporal Feature Engine:** Lags, deltas, acceleration, and rolling statistics are calculated strictly within company groups.
- **Multi-Model Layer:** Random Forest, XGBoost, and Logistic Regression models are fitted for both temporal panel and cross-sectional datasets, alongside an Isolation Forest anomaly detector.
- **Probability Calibration:** Platt scaling (sigmoid) is fitted on validation data.
- **User Interface:** A 10-view Streamlit analytics dashboard (`app.py`) and a 6-endpoint FastAPI backend (`api.py`) with precomputed corporate risk dossiers for 422 enterprises.
- **Test Suite:** 20 automated pytest tests covering API endpoints, leakage rules, feature bounds, and trajectory scoring.

---

### Question 2: What works and has been verified?
Through empirical execution in the active environment (Python 3.13.5), the following capabilities have been verified:
1. **The Automated Test Suite:** `python -m pytest tests/ -v` executes in 39.9 seconds with **20 out of 20 tests passing**.
2. **Dashboard Smoke Test:** `scripts/smoke_test_dashboard.py` runs cleanly, verifying that the 27 MB corporate intelligence database (422 firms), watchlist, ablation studies, and feature importances load into memory without crashing.
3. **Temporal Leakage Safeguards:** Unit tests confirm that lag-1, lag-2, rolling means, and delta features never reference future or cross-company records.
4. **FastAPI Microservice:** Verified contract compliance across `/health`, `/model-info`, `/predict`, `/company/{id}`, `/watchlist`, and `/risk-summary`.
5. **Cross-Sectional Bankruptcy Discrimination (Dataset B):** The Random Forest model on Dataset B achieves a verified Test `ROC-AUC = 0.9455` and `Precision = 72.2%`.
6. **Longitudinal Trajectory Tracking:** Verified calculation of multi-period risk slope, acceleration, and consecutive deterioration streaks.

---

### Question 3: What is incorrect, unverified, or missing?

#### What is Incorrect (Broken):
1. **Operational Test Recall is 0.0%:** In `models/ews/ews_metrics.json`, the primary EWS model has a test confusion matrix of **TN=251, FP=0, FN=11, TP=0**. Because probability calibration compresses predictions to $[0.01, 0.19]$ and the test code applies an arbitrary 0.5 threshold, **the model misses 100% of insolvent firms**.
2. **Chart Threshold Misalignment (Visual Absurdity):** Radial gauges and timeline charts hardcode 0–20 as "Healthy". Consequently, the highest-risk firm in the database (Firm #364, score 19.1, Critical) is drawn **inside the green Healthy zone**.
3. **Rule 1 Distress Alert Inactivity:** `EarlyWarningSignalEngine` requires probability $\ge 0.70$ or $\ge 0.50$ to trigger Rule 1. Because calibrated probabilities max out at 0.191, **Rule 1 never triggers for any firm**.
4. **TreeSHAP Fallback Compromise:** The codebase advertises TreeSHAP, but because the selected model is Random Forest and the `shap` package is uninstalled, it falls back to $x \times \text{norm\_importances}$. This produces scale-distorted, invalid attributions.
5. **Synthetic Pseudo-Proxies in Fusion:** To feed Dataset A rows into a stacking meta-model, the pipeline manufactured synthetic inputs using arbitrary formulas on `x64` and `x81`. This degraded test `ROC-AUC` from 0.778 to 0.659 and assigned a negative weight to the anomaly score.
6. **API I/O Bottleneck:** `api.py` re-reads the entire 27 MB JSON file from disk on every single `/company/{id}` request.

#### What is Unverified / Fabricated:
1. **Fabricated Accounting Semantics:** Attributes `x1`..`x83` in Dataset A are masked. The `FeatureRegistry` arbitrarily mapped `x48` to "Profitability", `x25` to "Liquidity", `x64` to "Leverage", and `x81` to "Solvency", generating unverified accounting narratives.
2. **Hardcoded Peer Benchmarks:** View 4 radar charts use a hardcoded benchmark `[70, 75, 70, 75, 65]` rather than real sector medians.

#### What is Missing:
1. **Git Version Control:** No `.git` repository was initialized.
2. **Pipeline Data Validation Integration:** `DataValidator` exists but is never invoked during training.
3. **Threshold Calibration Protocol:** No automated threshold tuning exists on the validation split.

---

### Question 4: Can the current model legitimately be described as an early-warning system?
**Conditionally YES in concept, but currently NO in operational decisioning.**

- **Conceptually:** Yes. The formulation ($y_{t} = \mathbb{I}(\text{Distress}_{t+1} \le -0.50)$) associates financial indicators at time $t$ with distress at $t+1$, providing a true 1-period forward horizon. Furthermore, incorporating temporal deltas and rolling volatility boosts Test `PR-AUC` by **+43.8%** over static ratios.
- **Operationally:** **NO.** Because the classification threshold was left at 0.5 without re-calibration, the deployed model predicts 0 for every test firm. An alarm system that rings for 0% of emergencies is not an early-warning system. It becomes an operational early-warning system only after the decision threshold is calibrated to its true distribution ($\tau^* \approx 0.08–0.10$).

---

### Question 5: Are the current evaluation results trustworthy?
**Partially trustworthy with significant qualifications:**

| Result / Claim | Trustworthy? | Reason |
| :--- | :---: | :--- |
| **Temporal EWS ROC-AUC (0.7783) & PR-AUC (0.1946)** | **Trustworthy** | Threshold-independent ranking metrics. Evaluated strictly on the out-of-time test partition (Time 13). Demonstrates genuine predictive signal. |
| **Reported EWS Accuracy (95.8%)** | **Misleading** | Pure reflection of the 95.8% negative base rate in the test set. |
| **Reported EWS Precision, Recall, F1 (0.0%)** | **Accurate Reflection of Broken Threshold** | Accurately documents that threshold 0.5 is completely non-functional. |
| **Exp 6 Stacking Meta-Model (Claimed "Best Architecture")** | **Untrustworthy & Degraded** | Stacking was trained on synthetic proxy features, degrading ROC-AUC from 0.778 to 0.659. The claimed Brier score "gain" is an artifact of predicting the low prior for all firms. |
| **Test Set Period Coverage (Claimed "Periods 13 & 14")** | **Inaccurate Documentation** | Period 14 has no forward label, so all Time 14 rows are dropped. The test set contains Time 13 only (262 rows, 11 distress events). |

---

### Question 6: What are the five highest-priority improvements?

1. **Optimize the Operational Decision Threshold on Validation Fold (`P0`):** Tune threshold $\tau^*$ to achieve target recall ($\ge 70\%$) on the validation split, restoring true early-warning detection on the test set.
2. **Synchronize Chart Threshold Bands with `risk_config.yaml` (`P0`):** Update `charts.py` to use calibrated intervals ($[0,4], [4,8], [8,13], [13,17], [17,100]$), eliminating the rendering of Critical firms in green zones.
3. **Re-calibrate Early-Warning Alert Rules (`P0`):** Align Rule 1 thresholds in `alerts.py` with calibrated probability levels ($\ge 0.170$ Critical, $\ge 0.130$ High) so alerts fire deterministically.
4. **Implement Mathematically Exact TreeSHAP (`P1`):** Select XGBoost as the primary EWS model to utilize its native C++ TreeSHAP engine, eliminating the unscaled heuristic fallback.
5. **Eliminate Synthetic Pseudo-Proxies from Risk Scoring (`P1`):** Remove fabricated formulas on `x64` and `x81`. For Dataset A, drive the final risk score directly from the calibrated Temporal EWS model.

---

### Question 7: What should be done first, second, and third?

1. **FIRST (Phase 0 & 1 — Protection & Critical UI/Alert Fixes):**
   - Initialize Git repository with clean baseline commit on branch `upgrade/v1.1-audit-fixes`.
   - Fix chart band misalignment in `dashboard/components/charts.py`.
   - Re-calibrate Rule 1 in `src/risk_engine/alerts.py`.
   - Add in-memory singleton caching to `api.py`.
2. **SECOND (Phase 2 — Operational Decisioning & Fusion Integrity):**
   - Implement cost-sensitive threshold tuning on validation split; re-evaluate test confusion matrix to achieve verified Recall $\ge 60\%$.
   - Remove synthetic pseudo-proxies from `src/pipeline.py` and `src/fusion/risk_fusion.py`.
3. **THIRD (Phase 3 — True Explainability & Semantic Hygiene):**
   - Align EWS model selection with XGBoost native TreeSHAP in `src/explainability/shap_explainer.py`.
   - Replace fabricated accounting labels in `FeatureRegistry` and dashboard views with verified empirical factor descriptions.

---

### Question 8: What should not be built yet?
The following features must **not** be built until core prediction and calibration validity are established:
- **Do NOT build complex multi-dataset feature fusion:** Datasets A, B, and C have completely disjoint feature spaces and populations. Feature-level merging is statistically prohibited.
- **Do NOT introduce external databases or message brokers:** Introducing PostgreSQL, Redis, Celery, or Kafka is unnecessary overhead for this 422-firm panel prototype.
- **Do NOT build automated credit underwriting / loan approval engines:** The system is an early-warning risk intelligence prototype, not an autonomous loan decision engine.
- **Do NOT attempt multi-period hazard survival modeling:** Establishing 1-period forward validity must precede multi-quarter discrete hazard modeling.

---

### Question 9: What is the minimum viable research-quality version?
The Minimum Viable Research-Quality Version (MVRV) requires:
1. **Clean Version Control:** Git repo with isolated branch tracking.
2. **Calibrated Decision Threshold:** Operational threshold selected on the validation fold achieving Recall $\ge 60\%$ and documented False Negative Rate on the test set.
3. **Exact TreeSHAP Explainability:** Mathematically valid Shapley attributions satisfying additive efficiency.
4. **Transparent Feature Grounding:** Anonymized variables labeled honestly as empirical factors without fabricated accounting narratives.
5. **Coherent Visual Interface:** Radial gauge, timeline bands, alert rules, and configuration files operating on the exact same threshold definitions.
6. **Decoupled Specialist Models:** Temporal EWS operates on Dataset A; Structural Bankruptcy operates on Dataset B; Outlier Profiler operates on Dataset C—without synthetic proxy blending.

---

### Question 10: What evidence is needed before claiming the project is complete?
Before claiming project completion, the following empirical artifacts must be produced and verified:
1. **Test Execution Evidence:** 100% of automated tests passing, including new tests asserting operational test recall $> 0$ and active alert firing.
2. **Updated Evaluation Table:** Revised `models/ews/ews_metrics.json` and `ablation_study.csv` reporting test confusion matrix under the optimized threshold (TP $\ge 7$, FN $\le 4$).
3. **Shapley Additive Efficiency Verification:** Execution log from `scripts/verify_native_shap.py` confirming $\sum \phi_i = f(x) - \mathbb{E}[f(x)]$.
4. **Visual Dashboard Audit:** Visual inspection confirming that Firm #364 displays in the red Critical zone on both the radial gauge and timeline chart.
5. **API Performance Benchmark:** Response time under 20 ms for `/company/364` verified across 50 simulated concurrent requests.
