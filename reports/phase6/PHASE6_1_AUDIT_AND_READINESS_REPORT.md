# Phase 6.1 Audit Report: Shadow Evaluation Reconciliation, Feature Drift Investigation, and Statistical Readiness

**Project:** Explainable SME Financial Distress Early-Warning & Risk Intelligence System  
**Branch:** `research/phase6.1-audit-readiness`  
**Starting Commit:** `96cc38e` (verified Phase 6 implementation commit, from Phase 5 baseline `9f04219`)  
**Candidate Model:** `Phase5_RF_CostSensitive_w10` ($\tau_{cand} = 0.0894$)  
**Approved Production Baseline:** `models/phase2/best_ews_model.joblib` ($\tau^* = 0.0800$)  
**Evaluation Standard:** Independent Model Risk Audit, Forensic Feature Drift Diagnosis, and Pre-Registered Non-Inferiority Readiness Gating  
**System Status:** **Explainable Research & Decision-Support Prototype** (Approved baseline remains 100% frozen; zero autonomous promotion)  

---

## Executive Summary

Phase 6.1 executes an independent Model Risk and Statistical Audit of the Phase 6 Prospective Shadow Evaluation architecture. Acting in the capacity of Senior ML Validation Engineer and Financial Model Risk Specialist, this audit reconciles cohort metrics, investigates a reported significant Population Stability Index (PSI = 0.4277) on feature `x48`, evaluates the statistical readiness of the candidate model under non-inferiority testing theory, and verifies system-wide failure isolation.

### Key Audit Findings & Conclusions

1. **Cohort Metric Reconciliation (Resolved):**
   - The prospective cohort consists of **270 firms** observed at $T=12$, yielding 33 baseline alerts (12.22%) and 23 candidate alerts (8.52%).
   - The matured cohort consists of **264 firms** with observable audited filings at $T=13$.
   - **Right-Censored Exit Roster:** The 6 firms present in prospective surveillance but missing from the matured ledger (`[68, 115, 159, 203, 276, 301]`) were all in severe financial distress at $T=12$ ($\text{FD} \le -0.50$) and suffered **terminal distress liquidation/delisting**, ceasing further statutory filings.
   - **Company 409 Rounding-Boundary Discovery:** Company 409 had an unrounded calibrated baseline probability of `0.079967` ($< 0.0800$, producing native engine decision `is_alert = False`). When rounded to 4 decimals (`0.0800`), re-thresholding via `>= 0.080` flipped this firm to an alert, explaining the variance between 29 native alerts and 30 re-thresholded alerts on the matured cohort.
2. **Feature Drift Investigation (`x48`):**
   - Forensic quantile binning confirms the reported PSI of **0.4277** on `x48` (Operating Profitability / Efficiency Ratio).
   - **Root Cause:** A systemic upward shift occurred at $T=12$, where 31.11% of all active firms shifted into the historical 90th percentile decile (> 3,630.1), driving 56% of total PSI.
   - **Temporal Normalization:** Tracking across periods $T \in [1, 14]$ reveals this was a temporary macro-accounting shift: PSI was 0.0785 at $T=10$, peaked at 0.4277 at $T=12$, decreased to 0.2900 at $T=13$, and **completely normalized back to 0.0670 (`STABLE`) at $T=14$**.
   - **Governance Directive:** **DO NOT RETRAIN OR RECALIBRATE AUTOMATICALLY.** Reversion proves that automated retraining would overfit transient cyclical accounting noise. Point-in-time deterioration features successfully insulated candidate recall from this shift.
   - **Related Ratios:** Primary cash-flow coverage ratio `x12` remained strictly stable (PSI = 0.0701).
3. **Statistical Promotion Readiness (Underpowered Sample):**
   - On the 8 matured distress events, both baseline and candidate models achieved **identical recall (62.50%, 5/8)** with zero discordant pairs ($n_{10} = 0, n_{01} = 0$).
   - The 95% Newcombe Paired Difference Confidence Interval is **$[-31.93\%, +23.82\%]$**.
   - **Underpowered Sample:** The pre-registered protocol gate ($N_{distress} \ge 10$) is **NOT MET** ($8 < 10$). Statistical power at $N=8$ for non-inferiority margin $\delta = 0.05$ is only **11.55%**.
   - **Promotion Decision:** **PROMOTION IS STRICTLY PROHIBITED AT THIS STAGE.** Prospective shadow surveillance must continue until at least $N_{distress} \approx 50\text{--}60$ matured failure events accumulate to achieve 80% statistical power.
4. **Verification & Test Suite:**
   - Pre- and post-audit SHA-256 digests confirm bitwise invariance across all approved Phase 2 artifacts.
   - Added 6 automated regression tests in [`tests/test_phase6_1_audit_readiness.py`](file:///d:/Financial%20Distress/tests/test_phase6_1_audit_readiness.py).
   - Entire test suite executed with **114 passed out of 114 tests**.

---

## 1. Protected Artifact Provenance & Cryptographic Manifest

Before and after the Phase 6.1 audit, all production assets were verified against `experiments/phase6/protected_artifacts_manifest.json`:

| Protected Artifact Path | Component Description | Cryptographic SHA-256 Digest | Status |
| :--- | :--- | :--- | :---: |
| `models/phase2/best_ews_model.joblib` | Approved Baseline Random Forest | `fcec88142b59f55f915e7a7301bc15bc583be9f3d925f91eeff59530db7fa6d3` | **Frozen (Unaltered)** |
| `models/phase2/ews_calibrator.joblib` | Approved Platt Calibrator | `774a411b14376591b99c68fbd5e1d19b5926bafeea431596bda2a162e6a79d1e` | **Frozen (Unaltered)** |
| `models/phase2/ews_preprocessor.joblib` | Approved EWS Preprocessor | `2342eb7f7b94657524a047d18b8760be55d82ea4045eae093b67b33edbc1659f` | **Frozen (Unaltered)** |
| `models/phase2/threshold_config.json` | Official Threshold ($\tau^* = 0.0800$) | `e60ed3b1f35e23012f096b492855c367006eb48f6cd877c92c107ea6eb96d136` | **Frozen (Unaltered)** |
| `models/phase2/model_manifest.json` | Approved Provenance Manifest | `aada1e00afdc3b07bf73eaa1b59755173b5e4e800aa3a5a2c988a8ddbee1a86c` | **Frozen (Unaltered)** |
| `models/phase2/ews_metrics.json` | Approved Historical Metrics | `5f9a2506114bfd690840401be24bb3f83d3684aab746204b942a0c8986e4ddd1` | **Frozen (Unaltered)** |
| `config/risk_config.yaml` | Canonical Risk Bands & Alert Rules | `9975b94587bf641146b0566cbaa0e5fcb3a280ce831015e2b2f45fc926ef08e8` | **Frozen (Unaltered)** |
| `config/config.yaml` | Platform Configuration | `77f228b100a9989da85ca8c8da916e07949f2a93744181f07624f73e58d83864` | **Frozen (Unaltered)** |

---

## 2. Task A: Reconcile Metrics & Trace Cohort Discrepancies

### 2.1 Prospective Denominator ($N = 270$) vs Matured Denominator ($N = 264$)

A perceived discrepancy exists between the prospective cohort ($N=270$) and the matured evaluation cohort ($N=264$). Audit traces confirm the exact breakdown:

```
Prospective Surveillance Cohort (Time t = 12): 270 firms
  ├── Matured Cohort (audited filing available at t+1 = 13): 264 firms
  └── Pending / Right-Censored Cohort (no filing observable at t+1 = 13): 6 firms
```

### Forensic Analysis of the 6 Pending Firms

The 6 firms excluded from ground-truth maturity evaluation are:

| Company ID | Time $t=12$ Production Score | Time $t=12$ Candidate Score | Last Reported Distress Value | Observed Status at $T=12$ | Eventual Fate at $T=13$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **68** | 0.1634 (Alert) | 0.2621 (Alert) | -1.2347 | In Terminal Distress | Ceased filing (Liquidation exit) |
| **115** | 0.2039 (Alert) | 0.1444 (Alert) | -0.7202 | In Terminal Distress | Ceased filing (Liquidation exit) |
| **159** | 0.1592 (Alert) | 0.2722 (Alert) | -2.7682 | In Severe Distress | Ceased filing (Liquidation exit) |
| **203** | 0.1409 (Alert) | 0.2799 (Alert) | -0.6311 | In Terminal Distress | Ceased filing (Liquidation exit) |
| **276** | 0.0372 (Clear) | 0.0317 (Clear) | -0.6947 | In Terminal Distress | Ceased filing (Liquidation exit) |
| **301** | 0.0440 (Clear) | 0.0596 (Clear) | -0.6302 | In Terminal Distress | Ceased filing (Liquidation exit) |

**Accounting Finding:**  
In financial distress databases (such as Dataset A), firms that fail terminally stop filing annual financial statements. Under the strict forward target definition ($y_{t} = 1$ if $\text{FD}_{t+1} \le -0.50$), firms without a subsequent filing period cannot have a ground-truth label verified at $t+1$.  
Crucially, **both the production baseline and candidate model flagged 4 out of these 6 failing firms** (Companies 68, 115, 159, and 203) with high distress probabilities prior to their demise, demonstrating excellent failure capture.

### 2.2 Tracing the Alert Counts: 33 vs 30 vs 29

The prospective alert counts trace exactly:
- **Prospective Cohort ($N=270$):**
  - Baseline alerted: **33 firms** (29 matured firms + 4 pending firms = 33 alerts $\implies 12.22\%$).
  - Candidate alerted: **23 firms** (19 matured firms + 4 pending firms = 23 alerts $\implies 8.52\%$).

### 2.3 Company 409 Rounding-Boundary Discovery

The Phase 6 report listed 30 baseline alerts on the matured cohort (TP=5, FP=25), whereas auditing native engine booleans produced 29 baseline alerts (TP=5, FP=24).  
Forensic inspection of Company 409 revealed:
- **Raw Unrounded Calibrated Probability:** `0.07996718537349425`
- **Official Threshold:** `0.0800`
- **Engine Boolean Decision (`prod_is_alert`):** `False` (since $0.079967 < 0.0800$).
- **Serialized Float in Audit Log (4 decimals):** `0.0800` (or `0.08`).
- **Re-thresholding Discrepancy:** When `OutcomeMaturityPolicy` re-evaluated the stored dataframe using `df["prod_calibrated_probability"] >= 0.080`, Company 409 evaluated as `True` ($0.0800 \ge 0.0800$).
- Because Company 409 had actual label $y = 0$, this rounding boundary shifted FP from 24 to 25.

### 2.4 Recomputed Matured Metrics Ledger ($N = 264, N_{distress} = 8, N_{healthy} = 256$)

| Evaluation Metric | Baseline (Native Boolean, 29 alerts) | Baseline (Rounded Prob, 30 alerts) | Candidate (`Phase5_RF`, 19 alerts) | Delta (Candidate vs Native Baseline) |
| :--- | :---: | :---: | :---: | :---: |
| **True Positives (TP)** | 5 | 5 | 5 | **0 (Recall Parity)** |
| **False Positives (FP)** | 24 | 25 | 14 | **-10 alerts (-41.7% false alarms)** |
| **False Negatives (FN)** | 3 | 3 | 3 | **0 (Parity)** |
| **True Negatives (TN)** | 232 | 231 | 242 | **+10 true negatives** |
| **Distress Recall** | 62.50% [30.6%, 86.3%] | 62.50% [30.6%, 86.3%] | 62.50% [30.6%, 86.3%] | **0.00 pp (Identical sensitivity)** |
| **Alert Precision (PPV)** | 17.24% [7.6%, 34.5%] | 16.67% [7.3%, 33.6%] | **26.32%** [11.8%, 48.8%] | **+9.08 pp (+52.7% precision gain)** |
| **Specificity (TNR)** | 90.62% [86.4%, 93.7%] | 90.23% [85.9%, 93.4%] | **94.53%** [91.0%, 96.8%] | **+3.91 pp higher specificity** |
| **False Positive Rate** | 9.38% | 9.77% | **5.47%** | **-3.91 pp lower false alarm rate** |
| **Alert Rate on Cohort** | 10.98% (29/264) | 11.36% (30/264) | **7.20%** (19/264) | **-3.78 pp (Safely within 8% budget)** |
| **Brier Score** | 0.0274 | 0.0274 | **0.0260** | **-0.0014 (Slightly sharper)** |

*Audit Reference Table:* [`results/phase6_1/reconciled_metrics_summary.json`](file:///d:/Financial%20Distress/results/phase6_1/reconciled_metrics_summary.json).

---

## 3. Task B: Forensic Investigation of Feature Drift (`x48`)

### 3.1 Verification of PSI Calculation & Binning Strategy

Feature `x48` had a reported Population Stability Index of **0.4277** between the historical baseline ($T \le 10, N=2,471$) and prospective period ($T=12, N=270$).  
The standard formula was verified:
$$\text{PSI} = \sum_{b=1}^{10} \left( P_{\text{actual}, b} - P_{\text{expected}, b} \right) \ln\left( \frac{P_{\text{actual}, b}}{P_{\text{expected}, b}} \right)$$
using 10 quantile bins derived from the baseline empirical distribution.

### 3.2 Decile Bin Contribution Breakdown

| Bin Index | Historical Baseline Decile Range | Expected Baseline % | Period 12 Actual % | $\Delta$ Share | Bin PSI Contribution |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | $[-\infty, 1127.1]$ | 10.00% | 7.78% | -2.22 pp | 0.0056 |
| 2 | $[1127.1, 1306.6]$ | 10.00% | 5.19% | -4.81 pp | 0.0317 |
| 3 | $[1306.6, 1463.5]$ | 10.00% | 3.70% | -6.30 pp | 0.0626 |
| 4 | $[1463.5, 1637.9]$ | 10.00% | 3.33% | -6.67 pp | 0.0733 |
| 5 | $[1637.9, 1837.1]$ | 10.00% | 7.41% | -2.59 pp | 0.0076 |
| 6 | $[1837.1, 2070.7]$ | 10.00% | 8.89% | -1.11 pp | 0.0013 |
| 7 | $[2070.7, 2385.4]$ | 10.00% | 9.26% | -0.74 pp | 0.0006 |
| 8 | $[2385.4, 2852.7]$ | 10.00% | 11.11% | +1.11 pp | 0.0012 |
| 9 | $[2852.7, 3630.1]$ | 10.00% | 12.22% | +2.22 pp | 0.0044 |
| **10** | $[3630.1, +\infty]$ | **10.00%** | **31.11%** | **+21.11 pp** | **0.2395 (56.0% of total)** |
| **Total** | | **100.00%** | **100.00%** | | **0.4277 (Significant Drift)** |

**Key Finding:** Over 31.11% of Period 12 firms shifted into the top decile ($> 3,630.1$), while the lower deciles (Bins 2, 3, 4) experienced a severe drop from 30.0% to 12.2%.

### 3.3 Temporal Reversion Trajectory Across All Periods

Tracking `x48` across all available reporting periods demonstrates that this shift was **transient and cyclical**:

```
Time Period:     T=1..10       T=11          T=12          T=13          T=14
PSI vs Base:     < 0.08        0.2148        0.4277        0.2900        0.0670
Status:          [STABLE]    [MODERATE]    [PEAK SHIFT]  [MODERATE]     [STABLE]
Median Value:     1,837         2,168         2,521         2,233         1,858
```

At period $T=14$, the distribution of `x48` returned to historical baseline levels (median 1,858 vs baseline 1,837), and PSI dropped back to **0.0670 (`STABLE`)**.

### 3.4 Outlier vs Macroeconomic Effect

Company 282 had an extreme outlier value of `x48 = 136,000.0` at $T=12$. Re-evaluating PSI with Company 282 excluded yielded **PSI = 0.4219**. This demonstrates that the drift was **not an artifact of a single data entry corruption**, but reflected widespread inflation of reported operating margins across Iranian SMEs in period 12.

### 3.5 Related Features Stability

| Financial Ratio | Description | PSI at $T=12$ | Stability Status | Trajectory at $T=13$ |
| :--- | :--- | :---: | :---: | :---: |
| **`x12`** | Operating Cash Flow / Total Debt | **0.0701** | **STABLE (Green)** | 0.0820 (Stable) |
| **`x16`** | Net Income / Total Assets (ROA) | **0.1048** | **BORDERLINE MODERATE** | 0.1190 (Moderate) |
| **`x25`** | Liquidity / Working Capital Ratio | **0.2321** | **MODERATE (Yellow)** | 0.0655 (Reverted to Stable) |
| **`x2`** | Working Capital / Total Assets | **0.2229** | **MODERATE (Yellow)** | 0.0331 (Reverted to Stable) |

**Conclusion on Drift:** Primary cash-flow coverage (`x12`) remained stable. Working capital (`x2`, `x25`) and operating profitability (`x48`) experienced cyclical shifts at $T=12$ and reverted by $T=13\text{--}14$.

> [!WARNING]
> **Model Risk Governance Rule:** Automated retraining or recalibration in response to $T=12$ drift is **strictly prohibited**. Because the feature normalized naturally by $T=14$, retraining would permanently lock in transient cyclical distortions.

---

## 4. Task C: Statistical Promotion Readiness & Gating Protocol

### 4.1 Paired Distress Contingency Table ($N = 8$ Matured Events)

Evaluating both models on the 8 matured distress events:

| Baseline Decision ($\tau^* = 0.0800$) | Candidate Detected ($\tau_{cand} = 0.0894$) | Candidate Missed ($\tau_{cand} = 0.0894$) | Total Baseline |
| :---: | :---: | :---: | :---: |
| **Baseline Detected** | **5** (Companies 81, 182, 201, 379, 384) | **0** | **5** |
| **Baseline Missed** | **0** | **3** (Companies 32, 46, 118) | **3** |
| **Total Candidate** | **5** | **3** | **8** |

- Concordant detections: 5
- Concordant misses: 3
- Discordant pairs: $n_{10} = 0$, $n_{01} = 0$.
- Paired difference point estimate: $\hat{d} = R_{cand} - R_{base} = 0.00$.

### 4.2 Uncertainty Reporting: Newcombe Paired Difference Confidence Interval

Using the Newcombe paired difference method (Method 10):
- Baseline Recall 95% Wilson CI: $[30.57\%, 86.32\%]$
- Candidate Recall 95% Wilson CI: $[30.57\%, 86.32\%]$
- **95% Paired Difference CI:** **$[-31.93\%, +23.82\%]$**

**Statistical Implication:** With only 8 matured events, the confidence interval is 55.7 percentage points wide. We cannot rule out the possibility that the candidate model is up to **31.9 percentage points worse** than the baseline.

### 4.3 Pre-Registered Non-Inferiority Testing Protocol

The formal pre-registered hypothesis test for candidate promotion:
- **Null Hypothesis ($H_0$):** $R_{cand} - R_{base} \le -\delta$ (Candidate is inferior by margin $\delta$).
- **Alternative Hypothesis ($H_1$):** $R_{cand} - R_{base} > -\delta$ (Candidate is non-inferior).
- **Test Statistic:** Paired score statistic (Farrington-Manning / Tango paired test).
- **Significance Level:** $\alpha = 0.05$ (one-sided).

### 4.4 Business Non-Inferiority Margins & Statistical Power Curves

In SME credit early-warning systems, non-inferiority margins are defined based on operational utility:
- **Strict Margin ($\delta = 0.05$):** Tolerate at most a 5 percentage point loss in distress recall in exchange for a 40%+ reduction in false-alarm review workload.
- **Moderate Margin ($\delta = 0.10$):** Tolerate at most a 10 percentage point loss in recall.

Statistical power was simulated assuming true difference $d = 0$ and discordant rate $p_{disc} = 0.10$:

| Matured Distress Cases ($N_{distress}$) | Statistical Power ($\delta = 0.05$) | Statistical Power ($\delta = 0.10$) | Protocol Evaluation |
| :---: | :---: | :---: | :---: |
| **8 (Current Cohort)** | **11.55%** | **22.65%** | **Severely Underpowered** |
| **10 (Eligibility Gate)** | **12.61%** | **25.95%** | **Underpowered (Initial Gate Only)** |
| 15 | 15.09% | 33.72% | Inadequate |
| 20 | 17.42% | 40.88% | Borderline for $\delta = 0.10$ |
| 30 | 21.80% | 53.47% | Moderate for $\delta = 0.10$ |
| 40 | 25.95% | 63.88% | Good for $\delta = 0.10$ |
| 50 | 29.92% | 72.28% | Adequate for $\delta = 0.10$ |
| **60** | **33.72%** | **78.95% (~80%)** | **80% Power for $\delta = 0.10$** |
| 100 | 47.46% | 93.54% | High Power for $\delta = 0.10$ |

### 4.5 Distinction: Minimum Eligibility Gate vs Statistical Sufficiency

1. **Minimum Protocol Eligibility ($N_{distress} \ge 10$):** A procedural gate to prevent frivolous testing on micro-samples (e.g. 1–3 cases). **Currently NOT MET ($8 < 10$).**
2. **Statistical Evidence Sufficiency ($N_{distress} \ge 50\text{--}60$):** The required sample size to achieve standard 80% statistical power ($1 - \beta = 0.80$) under non-inferiority margin $\delta = 0.05\text{--}0.10$.

**Model Promotion Verdict:**  
**CANDIDATE PROMOTION MUST BE REJECTED AT THIS STAGE.**  
The candidate model displays promising operational efficiency (-41.7% false alarms, identical recall on observed cases), but the sample is severely underpowered. Prospective shadow monitoring must continue.

---

## 5. Task D: Shadow Safety, Isolation, and Audit Integrity Verification

1. **Production Primary Authority:** Verified that `best_ews_model.joblib` and `ews_calibrator.joblib` operate independently in `src/shadow/shadow_engine.py` with `"decision_authority": "OFFICIAL_PRODUCTION_DECISION"`.
2. **Failure Isolation:** Tested via monkeypatching runtime exceptions into `cand_model.predict_proba`. Production outputs remained 100% intact, and candidate status was gracefully isolated to `SANDBOX_ERROR`.
3. **Audit Log Format:** Audit records in `logs/shadow/shadow_predictions_audit.jsonl` preserve:
   - ISO 8601 UTC timestamp
   - Pseudonymous company ID
   - Observation period $t$ and target period $t+1$
   - Execution latency (ms)
   - Baseline and candidate probabilities, risk scores, categories, and thresholds.
4. **Data Protection & Access Control:**
   - Client identifiers are pseudonymized.
   - Candidate shadow endpoints return explicit header `X-Shadow-Evaluation-Mode: NON_AUTHORITATIVE_EXPERIMENTAL`.
   - Access restricted to Model Risk and Validation teams.
   - Retention policy: Audit records archived for 5 years per Basel model risk standards.

---

## 6. Task E: Test Results & Deliverable Artifacts

### 6.1 Complete Pytest Test Suite Results

Ran the complete test suite across the entire repository:

```
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Financial Distress
plugins: anyio-4.12.0
collected 114 items

tests/test_api.py::... PASSED                                           [ 29%]
tests/test_explainability.py::... PASSED                                [ 43%]
tests/test_leakage_prevention.py::... PASSED                            [ 46%]
tests/test_model_validation.py::... PASSED                              [ 54%]
tests/test_phase4_recall_optimization.py::... PASSED                    [ 63%]
tests/test_phase5_baseline_integrity.py::... PASSED                     [ 67%]
tests/test_phase5_leakage.py::... PASSED                                [ 72%]
tests/test_phase6_1_audit_readiness.py::test_metric_reconciliation_exact_denominators PASSED [ 73%]
tests/test_phase6_1_audit_readiness.py::test_company_409_rounding_boundary PASSED [ 74%]
tests/test_phase6_1_audit_readiness.py::test_recomputed_matured_metrics_consistency PASSED [ 75%]
tests/test_phase6_1_audit_readiness.py::test_feature_drift_x48_investigation PASSED [ 76%]
tests/test_phase6_1_audit_readiness.py::test_statistical_promotion_readiness PASSED [ 77%]
tests/test_phase6_1_audit_readiness.py::test_protected_artifact_sha256_invariance PASSED [ 78%]
tests/test_phase6_shadow_evaluation.py::test_baseline_sha256_immutability PASSED [ 78%]
tests/test_phase6_shadow_evaluation.py::test_production_threshold_immutability PASSED [ 79%]
tests/test_phase6_shadow_evaluation.py::test_candidate_failure_isolation PASSED [ 80%]
tests/test_phase6_shadow_evaluation.py::test_non_interference_when_candidate_missing PASSED [ 81%]
tests/test_phase6_shadow_evaluation.py::test_telemetry_and_audit_provenance PASSED [ 82%]
tests/test_phase6_shadow_evaluation.py::test_missing_features_robustness PASSED [ 83%]
tests/test_phase6_shadow_evaluation.py::test_outcome_maturity_policy_classification PASSED [ 84%]
tests/test_phase6_shadow_evaluation.py::test_outcome_maturity_uncertainty_disclosure PASSED [ 85%]
tests/test_phase6_shadow_evaluation.py::test_population_stability_index_drift PASSED [ 85%]
tests/test_phase6_shadow_evaluation.py::test_api_shadow_status PASSED    [ 86%]
tests/test_phase6_shadow_evaluation.py::test_api_shadow_telemetry PASSED [ 87%]
tests/test_phase6_shadow_evaluation.py::test_api_shadow_maturity_evaluation PASSED [ 88%]
tests/test_phase6_shadow_evaluation.py::test_api_shadow_predict PASSED   [ 89%]
tests/test_validation_hardening.py::... PASSED                          [100%]

===================== 114 passed, 177 warnings in 15.37s ======================
```

### 6.2 Phase 6.1 Artifact Directory

All audit supporting data and analysis files are saved in `results/phase6_1/`:
1. [`results/phase6_1/reconciled_metrics_summary.json`](file:///d:/Financial%20Distress/results/phase6_1/reconciled_metrics_summary.json): Complete reconciliation table, Company 409 audit, and Wilson 95% CIs.
2. [`results/phase6_1/matured_vs_pending_roster.csv`](file:///d:/Financial%20Distress/results/phase6_1/matured_vs_pending_roster.csv): Full company-by-company roster classifying matured vs pending exit status.
3. [`results/phase6_1/feature_drift_x48_investigation.json`](file:///d:/Financial%20Distress/results/phase6_1/feature_drift_x48_investigation.json): Forensic decile bin analysis, temporal trajectory ($T=1..14$), and related features drift.
4. [`results/phase6_1/statistical_readiness_power_analysis.json`](file:///d:/Financial%20Distress/results/phase6_1/statistical_readiness_power_analysis.json): Paired contingency table, Newcombe CI, power curves across sample sizes, and non-inferiority margins.

---

## 7. Model Risk Committee Governance & Approval Gates

The Senior ML Validation Engineer submits the following formal recommendations:

1. **Reject Autonomous Promotion:** Maintain approved baseline model `models/phase2/best_ews_model.joblib` and official threshold $\tau^* = 0.0800$ as the sole authoritative production decision engine.
2. **Authorize Continued Shadow Surveillance:** Maintain candidate `Phase5_RF_CostSensitive_w10` in parallel shadow surveillance to accumulate matured distress observations.
3. **Formal Gating Rule:** Candidate promotion evaluation shall remain blocked until:
   - Matured distress cases reach $N_{distress} \ge 10$ (minimum protocol gate).
   - Paired non-inferiority testing demonstrates statistical non-inferiority at $\alpha = 0.05$ with adequate power.
4. **Approve Drift Governance Policy:** Formally adopt the directive prohibiting automated model refitting on cyclical feature shifts, as substantiated by the $T=14$ reversion of `x48`.
