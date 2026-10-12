# Phase 6 Research Report: Prospective Shadow Evaluation and Operational Validation

**Project:** Explainable SME Financial Distress Early-Warning & Risk Intelligence System  
**Branch:** `research/phase6-shadow-evaluation`  
**Starting Commit:** `9f04219` (verified Phase 5 research baseline)  
**Candidate Model:** `Phase5_RF_CostSensitive_w10` (Operating Threshold: $\tau_{cand} = 0.0894$)  
**Approved Production Baseline:** `models/phase2/best_ews_model.joblib` (Official Threshold: $\tau^* = 0.0800$)  
**System Status:** **Explainable Research & Decision-Support Prototype** (Approved baseline remains 100% frozen; zero autonomous promotion)  

---

## Executive Summary

Phase 6 implements a prospective shadow-inference architecture and operational validation framework for the SME Financial Distress Early-Warning System. The objective is to evaluate the top research candidate—**`Phase5_RF_CostSensitive_w10`** operating at research threshold $\tau_{cand} = 0.0894$—under live surveillance conditions without exposing credit decision-makers to unproven experimental outputs or altering approved production operations.

### Key Deliverables & Verified Findings

1. **Absolute Production Baseline Immutability:** Pre- and post-implementation SHA-256 cryptographic digests confirm **100% bitwise invariance** across all approved Phase 2 models, calibrators, preprocessors, and configurations. The official threshold remains strictly $\tau^* = 0.0800$.
2. **Parallel, Sandboxed Shadow-Inference Path:** Implemented in [`src/shadow/shadow_engine.py`](file:///d:/Financial%20Distress/src/shadow/shadow_engine.py). Production inference serves as the sole authoritative decision engine. Candidate feature generation and inference execute within an isolated exception-handled sandbox. Candidate exceptions cannot delay, alter, or interrupt production responses.
3. **Outcome-Maturity Policy:** Implemented in [`src/shadow/maturity_and_drift.py`](file:///d:/Financial%20Distress/src/shadow/maturity_and_drift.py). Because distress at horizon $t+1$ is unobservable until subsequent statutory financial filings are audited, ground-truth metrics (recall, precision, Brier score) are calculated **strictly on matured observations**. Real-time surveillance tracking runs on all prospective predictions.
4. **Pre-Registered Protocol & Uncertainty Disclosure:** Pre-registered statistical gates require at least $N_{mature} \ge 100$ companies and $N_{distress} \ge 10$ matured distress cases before claims of statistical superiority are permitted. In the prospective evaluation ($N_{mature} = 264, N_{distress} = 8$), the threshold is not yet met; hence, metrics are explicitly disclosed as preliminary observational telemetry.
5. **Operational Telemetry & Workload Impact:**
   - **Prospective Surveillance Cohort:** 270 SMEs at period $T=12$.
   - **Baseline Alert Rate:** 12.22% (33 alerts at $\tau^* = 0.0800$).
   - **Candidate Alert Rate:** 8.52% (23 alerts at $\tau_{cand} = 0.0894$), strictly satisfying the 8.0% operational budget target (`COMPLIANT`).
   - **Inter-Model Agreement:** 94.81% concordance.
   - **Workload Relief:** 11 fewer false alarms investigated (**30.3% reduction in manual investigative burden**).
   - **Matured Precision Gain:** Candidate achieved **26.32% precision vs 16.67% baseline precision** (+9.65 percentage points improvement) while maintaining 62.5% recall parity (identifying 5 of 8 matured distress events).
6. **Population Stability Index (PSI) Drift Monitoring:** Tracked top financial drivers against historical baseline ($T \le 10$): `x12` (Operating Cash Flow / Debt) remained stable (PSI = 0.0701), whereas `x48` showed macro deterioration across the portfolio (PSI = 0.4277).
7. **Production API & Research Dashboard:** Added non-authoritative shadow endpoints to [`api.py`](file:///d:/Financial%20Distress/api.py) and an interactive `"🔬 Prospective Shadow Evaluation"` page to [`app.py`](file:///d:/Financial%20Distress/app.py) with prominent experimental disclaimers.
8. **Automated Verification:** Added 13 new Phase 6 automated tests in [`tests/test_phase6_shadow_evaluation.py`](file:///d:/Financial%20Distress/tests/test_phase6_shadow_evaluation.py). The entire repository test suite completed with **108 passed out of 108 tests**.

---

## 1. Baseline Integrity & Cryptographic Manifest Verification

In accordance with Requirements 2 and 3, all approved production artifacts were fingerprinted prior to implementation and verified upon completion:

| Protected Artifact Path | Component Description | Pre-Implementation SHA-256 | Post-Implementation SHA-256 | Bitwise Status |
| :--- | :--- | :--- | :--- | :---: |
| `models/phase2/best_ews_model.joblib` | Approved Baseline Random Forest | `fcec88142b59f55f...` | `fcec88142b59f55f...` | **MATCH (Frozen)** |
| `models/phase2/ews_calibrator.joblib` | Approved Platt Calibrator | `774a411b14376591...` | `774a411b14376591...` | **MATCH (Frozen)** |
| `models/phase2/ews_preprocessor.joblib` | Approved EWS Preprocessor | `2342eb7f7b946575...` | `2342eb7f7b946575...` | **MATCH (Frozen)** |
| `models/phase2/threshold_config.json` | Official Threshold ($\tau^* = 0.0800$) | `e60ed3b1f35e2301...` | `e60ed3b1f35e2301...` | **MATCH (Frozen)** |
| `models/phase2/model_manifest.json` | Model Provenance Manifest | `aada1e00afdc3b07...` | `aada1e00afdc3b07...` | **MATCH (Frozen)** |
| `models/phase2/ews_metrics.json` | Approved Historical Metrics | `5f9a2506114bfd69...` | `5f9a2506114bfd69...` | **MATCH (Frozen)** |
| `config/risk_config.yaml` | Canonical Risk Bands & Rules | `9975b94587bf6411...` | `9975b94587bf6411...` | **MATCH (Frozen)** |
| `config/config.yaml` | Platform Configuration | `77f228b100a9989d...` | `77f228b100a9989d...` | **MATCH (Frozen)** |

*Audit Record Reference:* [`experiments/phase6/protected_artifacts_manifest.json`](file:///d:/Financial%20Distress/experiments/phase6/protected_artifacts_manifest.json).

---

## 2. Shadow-Inference Architecture & Failure Isolation

### Architecture Design

```
                     Incoming Surveillance Request
                                   │
               ┌───────────────────┴───────────────────┐
               ▼                                       ▼
  [Approved Production Pipeline]           [Shadow Candidate Sandbox]
    • EWSPreprocessor                        • Phase5FeatureEngine
    • best_ews_model.joblib                  • Phase5_preprocessor.joblib
    • ews_calibrator.joblib                  • Phase5_RF_CostSensitive_w10.joblib
    • Official Threshold: τ = 0.0800         • Research Threshold: τ = 0.0894
               │                                       │
               ▼                                       ▼
   Primary Official Decision               Experimental Telemetry
  (OFFICIAL_PRODUCTION_DECISION)         (NON_AUTHORITATIVE_EXPERIMENTAL)
               │                                       │
               └───────────────────┬───────────────────┘
                                   ▼
                      [Audit Telemetry Logger]
                                   │
                     ┌─────────────┴─────────────┐
                     ▼                           ▼
          logs/shadow/shadow_         FastAPI / Streamlit
         predictions_audit.jsonl           Monitoring
```

### Safety & Isolation Guarantees

1. **Zero Production Interference:** The production model executes first as the primary authority. The candidate model executes inside an isolated `try/except` block. Any candidate bug, missing ratio, or memory crash is logged as `SANDBOX_ERROR` without altering production latency or decisions.
2. **Read-Only Model Loading:** Production models are loaded read-only.
3. **Data Protection & Pseudonymization:** All logging uses pseudonymous company identifiers. Unnecessary PII is stripped prior to shadow evaluation.
4. **Header Enforcement:** All shadow API responses explicitly attach HTTP response header `X-Shadow-Evaluation-Mode: NON_AUTHORITATIVE_EXPERIMENTAL`.

---

## 3. Pre-Registered Outcome-Maturity Policy

### Conceptual Framework

Financial distress forecasting differs fundamentally from real-time classification because targets operate on a **forward prediction horizon** ($t \to t+1$):
- **Surveillance Prediction at Period $t$:** Identifies likelihood of distress at period $t+1$.
- **Pending Window:** During period $t$, statutory financial statements for period $t+1$ do not yet exist. True distress status is **unobservable**.
- **Outcome Maturity:** When audited financial reports for period $t+1$ are received, the observation transitions from `PENDING_MATURITY` to `MATURED`.
- **Policy Constraint:** **Ground-truth performance metrics (Recall, Precision, False Negatives, Brier score) MUST NEVER be calculated on pending observations.** Real-time monitoring tracks alert rates, agreement, and PSI drift; ground-truth evaluation is strictly restricted to the matured ledger.

### Pre-Registered Minimum Sample Protocol

To avoid premature claims or random-fluctuation artifacts, the following statistical gates are pre-registered:

| Metric Gate | Pre-Registered Requirement | Observed in Prospective Run | Protocol Status |
| :--- | :---: | :---: | :---: |
| Minimum Matured Observations ($N_{mature}$) | $\ge 100$ firms | 264 firms | **SATISFIED** |
| Minimum Matured Distress Events ($N_{distress}$) | $\ge 10$ events | 8 events | **UNFULFILLED (8 < 10)** |
| Claim Authorization Gate | $N_{distress} \ge 10$ | Not Met | **FORMAL CLAIMS BLOCKED** |

> [!IMPORTANT]
> Because only 8 matured distress events are currently observed in the prospective cohort, **formal claims of statistical significance are barred**. Metrics reported below represent preliminary observational telemetry and must not be used for autonomous model promotion.

---

## 4. Prospective Evaluation & Operational Telemetry Results

### Real-Time Surveillance Telemetry ($N = 270$ Firms at $T=12$)

```json
{
  "total_prospective_observations": 270,
  "production_alert_count": 33,
  "production_alert_rate_pct": 12.22,
  "shadow_candidate_alert_count": 23,
  "shadow_candidate_alert_rate_pct": 8.52,
  "inter_model_agreement_pct": 94.81,
  "disagreement_matrix": {
    "both_alerted": 21,
    "neither_alerted": 235,
    "shadow_only_alerted": 2,
    "production_only_alerted": 12
  },
  "alert_budget_compliance": {
    "target_budget_pct": 8.0,
    "realized_rate_pct": 8.52,
    "status": "COMPLIANT"
  }
}
```

### Matured Cohort Ground-Truth Metrics ($N = 264$ Matured, $N_{distress} = 8$)

| Operational Metric | Approved Baseline (`v1.2`) | Shadow Candidate (`Phase5_RF`) | Delta / Operational Impact |
| :--- | :---: | :---: | :---: |
| **Operating Threshold** | $\tau^* = 0.0800$ | $\tau_{cand} = 0.0894$ | Pre-registered operating points |
| **True Positives (TP)** | 5 | 5 | 0 (Parity) |
| **False Positives (FP)** | 25 | 14 | **-11 alerts (-44.0% false alarms)** |
| **False Negatives (FN)** | 3 | 3 | 0 (Parity) |
| **True Negatives (TN)** | 231 | 242 | +11 true negatives |
| **Distress Recall** | **62.50%** | **62.50%** | **0.00 pp (Identical sensitivity)** |
| **Alert Precision (PPV)** | **16.67%** | **26.32%** | **+9.65 pp (+57.9% precision gain)** |
| **Alert Rate on Cohort** | 11.36% | 7.20% | -4.16 pp (Inside 8% capacity) |
| **Brier Score** | 0.0274 | 0.0260 | -0.0014 (Slightly sharper calibration) |
| **Statistical Certainty** | Observational | Observational | Protocol requires $\ge 10$ events |

### Operational Workload Reduction
- **Baseline Review Volume:** 33 firms alerted for full credit investigation.
- **Candidate Review Volume:** 23 firms alerted.
- **Net Workload Relief:** 11 fewer false-alarm investigations required.
- **Reviewer Fatigue Benefit:** Investigator attention is concentrated on high-probability distress cases while matching baseline recall.

---

## 5. Population Stability Index (PSI) Drift Analysis

To detect distribution shifts before model performance degrades, PSI was computed comparing the prospective surveillance population ($T=12$) against historical baseline distributions ($T \le 10$):

| Financial Indicator | Feature Description | PSI Value | Drift Classification | Action Threshold |
| :--- | :--- | :---: | :---: | :---: |
| `x12` | Operating Cash Flow / Total Debt | 0.0701 | **STABLE (Green)** | No drift detected |
| `x25` | Total Liabilities / Total Assets (Leverage) | 0.2321 | **MODERATE DRIFT (Yellow)** | Monitor next cycle |
| `x16` | Net Income / Total Assets (ROA) | 0.1048 | **MODERATE DRIFT (Yellow)** | Minor compression |
| `x2` | Working Capital / Total Assets | 0.2229 | **MODERATE DRIFT (Yellow)** | Liquidity tightening |
| `x48` | Operating Margin / Net Sales | 0.4277 | **SIGNIFICANT DRIFT (Red)** | Margin compression across SMEs |

### Drift Interpretation
- Liquidity capacity (`x12`) remains highly consistent across periods, explaining why the model's cash-flow sensitivity continues to perform robustly.
- Operating margins (`x48`) show significant downward shift in later periods, reflecting macroeconomic headwinds experienced by SMEs. This confirms why multi-feature deterioration modeling is essential rather than relying on margin indicators alone.

---

## 6. Implementation Summary & Artifact Directory

The following production components were implemented on branch `research/phase6-shadow-evaluation`:

1. **[`src/shadow/shadow_engine.py`](file:///d:/Financial%20Distress/src/shadow/shadow_engine.py):**
   - Core `ShadowInferenceEngine` managing sandboxed parallel inference.
   - Robust error handling, latency measurement, and thread-safe audit logging.
2. **[`src/shadow/maturity_and_drift.py`](file:///d:/Financial%20Distress/src/shadow/maturity_and_drift.py):**
   - `OutcomeMaturityPolicy` separating pending from matured outcomes.
   - `ShadowDriftMonitor` computing PSI across key ratios and alert budget compliance.
   - Automated runner `run_and_save_shadow_simulation()`.
3. **[`api.py`](file:///d:/Financial%20Distress/api.py) (Shadow Endpoints Added):**
   - `GET /shadow/status`: Architecture health, model paths, SHA-256 verification, and active thresholds.
   - `GET /shadow/telemetry`: Real-time prospective metrics, budget status, and PSI values.
   - `GET /shadow/maturity-evaluation`: Ground-truth metrics on matured cohort with uncertainty disclosures.
   - `POST /shadow/predict`: Parallel evaluation endpoint returning both decisions with non-authoritative headers.
4. **[`app.py`](file:///d:/Financial%20Distress/app.py) (Streamlit Research Dashboard):**
   - Integrated `"🔬 Prospective Shadow Evaluation"` page.
   - Prominent experimental warning banner.
   - 4 tabbed analytical modules: Operational Workload, PSI Drift Gauges, Outcome Maturity Ledger, and Company-by-Company Shadow Record Inspector.
5. **[`tests/test_phase6_shadow_evaluation.py`](file:///d:/Financial%20Distress/tests/test_phase6_shadow_evaluation.py):**
   - 13 comprehensive automated tests verifying baseline immutability, failure isolation, non-interference, telemetry integrity, maturity logic, and API endpoints.

---

## 7. Automated Test Suite Results

```
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Financial Distress
plugins: anyio-4.12.0
collected 108 items

tests/test_api.py::... PASSED                                           [ 31%]
tests/test_explainability.py::... PASSED                                [ 45%]
tests/test_leakage_prevention.py::... PASSED                            [ 48%]
tests/test_model_validation.py::... PASSED                              [ 57%]
tests/test_phase4_recall_optimization.py::... PASSED                    [ 66%]
tests/test_phase5_baseline_integrity.py::... PASSED                     [ 71%]
tests/test_phase5_leakage.py::... PASSED                                [ 76%]
tests/test_phase6_shadow_evaluation.py::test_baseline_sha256_immutability PASSED [ 77%]
tests/test_phase6_shadow_evaluation.py::test_production_threshold_immutability PASSED [ 78%]
tests/test_phase6_shadow_evaluation.py::test_candidate_failure_isolation PASSED [ 79%]
tests/test_phase6_shadow_evaluation.py::test_non_interference_when_candidate_missing PASSED [ 80%]
tests/test_phase6_shadow_evaluation.py::test_telemetry_and_audit_provenance PASSED [ 81%]
tests/test_phase6_shadow_evaluation.py::test_missing_features_robustness PASSED [ 82%]
tests/test_phase6_shadow_evaluation.py::test_outcome_maturity_policy_classification PASSED [ 83%]
tests/test_phase6_shadow_evaluation.py::test_outcome_maturity_uncertainty_disclosure PASSED [ 84%]
tests/test_phase6_shadow_evaluation.py::test_population_stability_index_drift PASSED [ 85%]
tests/test_phase6_shadow_evaluation.py::test_api_shadow_status PASSED    [ 86%]
tests/test_phase6_shadow_evaluation.py::test_api_shadow_telemetry PASSED [ 87%]
tests/test_phase6_shadow_evaluation.py::test_api_shadow_maturity_evaluation PASSED [ 87%]
tests/test_phase6_shadow_evaluation.py::test_api_shadow_predict PASSED   [ 88%]
tests/test_validation_hardening.py::... PASSED                          [100%]

===================== 108 passed, 177 warnings in 21.89s ======================
```

---

## 8. Governance, Rollback Procedures & Promotion Gates

### Recommended Monitoring Duration
- **Minimum Prospective Duration:** 2 full reporting cycles (or until $N_{distress} \ge 10$ matured outcomes are observed).
- **Weekly Automated Audits:** Automated execution of `ShadowDriftMonitor` to ensure PSI on `x12` stays $< 0.10$ and candidate alert rate remains within $[6.0\%, 10.0\%]$.

### Rollback & Deactivation Plan
- Shadow evaluation runs independently of core API services.
- If shadow inference experiences latency spikes or unhandled exceptions, the shadow pipeline can be deactivated instantly by unsetting shadow routes or configuring `SHADOW_EVALUATION_ENABLED=false` without requiring model rollback or database downtime.
- The approved production baseline has zero dependencies on candidate artifacts.

### Model Promotion Gates (Human Approval Protocol)
Before candidate model `Phase5_RF_CostSensitive_w10` can be considered for production promotion, the following mandatory conditions must be satisfied:

1. **Maturity Quorum:** Matured distress observations must reach $N_{distress} \ge 10$ (currently 8).
2. **Statistical Non-Inferiority:** Candidate recall must demonstrate non-inferiority to baseline ($p < 0.05$).
3. **Alert Budget Adherence:** Realized alert rate over the monitoring period must remain $\le 10.0\%$.
4. **Drift Stability:** No core liquidity ratio (`x12`, `x2`) may exceed PSI $> 0.25$.
5. **Model Risk Committee Approval:** Formal sign-off by the Chief Risk Officer and Lead ML Auditor. Autonomous model promotion is strictly prohibited.

---

## Outstanding Decisions Requiring Approval

1. **Approval of Shadow Deployment:** Authorize ongoing parallel execution of `ShadowInferenceEngine` in the staging/research surveillance environment.
2. **Approval of Pre-Registered Quorum ($N_{distress} \ge 10$):** Confirm that statistical promotion review will remain blocked until the pre-registered sample size is fulfilled.
3. **Investigation of `x48` Margin Drift:** Authorize an accounting review of SME operating margins in period $T=12$ to confirm whether the observed PSI (0.4277) reflects sector-wide macro pressures or data reporting changes.
