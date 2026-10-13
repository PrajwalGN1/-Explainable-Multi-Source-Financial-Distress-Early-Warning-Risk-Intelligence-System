# Phase 6.9 — Existing-Data Model Improvement and Project Completion Report

**Executive Model Validation & Research Engineering Report**  
**Project:** Explainable Multi-Source SME Financial Distress Early-Warning Risk Intelligence System  
**Evaluation Standard:** Federal Reserve SR 11-7 / OCC 2011-12 Supervisory Guidance on Model Risk Management  
**Branch:** `research/phase6.9-existing-data-model-improvement`  
**Dataset Scope:** Strictly existing dataset at `data/raw/Financial Distress.csv` (Dataset A)  
**Date of Audit:** October 13, 2026  
**Final Status:** `DEMO_APPLICATION_READY` (with `RESEARCH_EVIDENCE_INCONCLUSIVE` on statistical candidate superiority)

---

## Executive Summary

Phase 6.9 completes a comprehensive, empirical research investigation and demonstration system using **strictly the existing dataset** available in the repository (`data/raw/Financial Distress.csv`), without reliance on external commercial banking data or fabricated records.

The investigation was conducted under strict model risk governance:
1. **Production Immutability Protected:** The approved production baseline (`models/phase2/best_ews_model.joblib` operating at official threshold $\tau^* = 0.0800$) and its 7 associated production artifacts were cryptographically locked and verified via SHA-256 digests against `experiments/phase6/protected_artifacts_manifest.json` before and after all experiments.
2. **Comprehensive Empirical Dataset Audit:** A complete audit of Dataset A confirmed 3,672 corporate-period observations across 422 unique SMEs over 14 contiguous annual reporting periods ($1 \le T \le 14$) with **zero missing values** and **zero duplicate records**. Every single distress event ($136 / 136 = 100\%$) occurs at the firm's terminal observed period, after which the firm exits the panel.
3. **Leakage-Safe Chronological Pipeline:** Evaluated using the pre-established chronological splits:
   - **Training:** $T \le 10$ (2,471 rows, 95 forward defaults, 3.84% distress rate).
   - **Validation:** $11 \le T \le 12$ (517 rows, 14 forward defaults, 2.71% distress rate).
   - **Holdout Test:** $T = 13 \to T = 14$ (262 rows, 11 forward defaults, 4.20% distress rate).
   - Preprocessing (`EWSPreprocessor` with median imputation and RobustScaler) was fit **strictly on training data** ($T \le 10$), and probability calibrators were fit on validation folds ($T \in [11, 12]$) via prefit calibration.
4. **Model Comparison under 8% Alert Budget Constraint ($M \le 20$ alerts on test):**
   - **Approved Production Baseline (Budget-Capped, $\tau^* = 0.1072$):** Detected **4 / 11** distress events (**36.36% recall**) with 16 false alarms (**20.00% precision**), achieving ROC-AUC = 0.7961, PR-AUC = 0.1816, Brier score = 0.0382.
   - **Candidate RF Cost-Sensitive w10 (Nominal $\tau = 0.0894$ & Budget-Capped):** Detected **3 / 11** distress events (**27.27% recall**) with 17 false alarms (**15.00% precision**), achieving ROC-AUC = 0.7521, PR-AUC = 0.1410, Brier score = 0.0389.
   - **Interpretable Logistic Regression (Budget-Capped, $\tau = 0.0271$):** Detected **3 / 11** distress events (**27.27% recall**) with 17 false alarms (**15.00% precision**), achieving ROC-AUC = 0.8095, PR-AUC = 0.1670, Brier score = 0.0404.
   - **Standard Unweighted Random Forest (Budget-Capped, $\tau = 0.0680$):** Detected **2 / 11** distress events (**18.18% recall**) with 18 false alarms (**10.00% precision**), achieving ROC-AUC = 0.7342, PR-AUC = 0.1041, Brier score = 0.0445.
5. **False Negative Diagnostics:** A granular audit of all 11 distress firms in the test cohort revealed that missed cases bifurcate into:
   - **Acute Sudden Defaults (5 firms: 28, 60, 84, 225, 376):** Firms that crashed from healthy prior solvency ($FD_{13} > 0$) without prior multi-period ratio deterioration. The production baseline's acute sensitivity detected Firms 28 and 60, whereas the candidate's rolling solvency features suppressed risk scores below alert thresholds.
   - **Chronic Solvency Deteriorations (6 firms: 23, 29, 53, 100, 101, 112):** Firms with prolonged solvency deficits ($FD_{13} \le 0$). Firms 100, 101, and 112 were detected by both baseline and candidate under both nominal and budget-capped thresholds.
6. **Explainability & Causal Governance:** Global and local TreeSHAP attributions identify working capital, cash flow, and retained earnings as leading drivers, accompanied by an explicit regulatory notice: feature importance reflects **statistical predictive associations**, not **causal proof**.
7. **Complete Application:** Both the Streamlit dashboard (`app.py`) and FastAPI backend (`api.py`) have been enhanced with enterprise-wide company selection (1–422), historical period selection, a dedicated Phase 6.9 Research Model Improvement page, and robust input validation returning clean HTTP 404 / UI error notices for invalid entities.
8. **Test Suite:** Phase 6.9 test suite passed 8/8 (100%), and the full repository regression suite passed **217 / 217 tests** (0 failures).

---

## 1. Approved Baseline Cryptographic Protection

Before and after all empirical evaluations, all 8 approved production artifacts were audited for cryptographic immutability:

| Artifact Path | Expected SHA-256 Digest | Verified Status |
|---|---|---|
| `models/phase2/best_ews_model.joblib` | `fcec88142b59f55f915e7a7301bc15bc583be9f3d925f91eeff59530db7fa6d3` | **VERIFIED MATCH** |
| `models/phase2/ews_calibrator.joblib` | `774a411b14376591b99c68fbd5e1d19b5926bafeea431596bda2a162e6a79d1e` | **VERIFIED MATCH** |
| `models/phase2/ews_preprocessor.joblib` | `2342eb7f7b94657524a047d18b8760be55d82ea4045eae093b67b33edbc1659f` | **VERIFIED MATCH** |
| `models/phase2/threshold_config.json` | `e60ed3b1f35e23012f096b492855c367006eb48f6cd877c92c107ea6eb96d136` | **VERIFIED MATCH** |
| `models/phase2/model_manifest.json` | `aada1e00afdc3b07bf73eaa1b59755173b5e4e800aa3a5a2c988a8ddbee1a86c` | **VERIFIED MATCH** |
| `models/phase2/ews_metrics.json` | `5f9a2506114bfd690840401be24bb3f83d3684aab746204b942a0c8986e4ddd1` | **VERIFIED MATCH** |
| `config/risk_config.yaml` | `9975b94587bf641146b0566cbaa0e5fcb3a280ce831015e2b2f45fc926ef08e8` | **VERIFIED MATCH** |
| `config/config.yaml` | `77f228b100a9989da85ca8c8da916e07949f2a93744181f07624f73e58d83864` | **VERIFIED MATCH** |

The production baseline operates at official threshold $\tau^* = 0.0800$. No production artifact was overwritten, retrained, or altered. The complete integrity report is persisted in `results/phase6_9/production_integrity_manifest.json`.

---

## 2. Empirical Dataset & Target Semantics Audit

Inspection of `data/raw/Financial Distress.csv` established the following ground-truth characteristics:

### 2.1 Dimensions and Data Quality
- **Row Count:** 3,672 observations.
- **Column Count:** 86 columns (`Company`, `Time`, `Financial Distress`, `x1`..`x83`).
- **Unique Enterprises:** 422 companies.
- **Reporting Period Span:** Periods 1 to 14 (annual reporting slices).
- **Periods per Company:** Min = 1, Max = 14, Mean = 8.70, Median = 9.0.
- **Missing Values:** Exactly **0** missing cells.
- **Duplicates:** Exactly **0** duplicate `(Company, Time)` pairs.
- **Time Contiguity:** Exactly **0** time gaps across all 422 companies; every company's observed trajectory consists of contiguous reporting intervals $t, t+1, t+2, \dots$.

### 2.2 Target Direction and Semantics
- Column `Financial Distress` is a continuous solvency index:
  - Mean: 1.0403, Std: 2.6522, Min: -8.6317, 25%: 0.1723, Median: 0.5838, 75%: 1.3518, Max: 128.4000.
- **Directional Semantics:** **LOWER values represent greater financial distress**.
  - Statutory distress threshold: $\text{Financial Distress} \le -0.50$ (derived from Article 141 of Iranian Commercial Code, reflecting severe equity impairment).
  - Values $> -0.50$ represent solvent, commercially active operating health.
- **Raw Distress Events:** 136 records (3.70% of total observations).
- **Terminal Exit Dynamics:** All 136 distress events ($100.0\%$) occur at the company's exact final observed reporting period. Upon recording financial distress, the company ceases to file and exits the panel immediately.

### 2.3 Forward Target Construction ($t \to t+1$)
- Because an early warning system must predict future distress before it occurs, the target is constructed as:
  $$Y_{i, t} = \mathbf{1}\{\text{Financial Distress}_{i, t+1} \le -0.50\}$$
- Observations at period $t$ evaluate whether the same firm experiences distress at period $t+1$.
- Total evaluable observations with observed next-period outcome: **3,250 rows**.
- Terminal unobserved rows: **422 rows** (each company's final observation, including 136 terminal distress periods, 269 surviving firms at $T=14$, and 17 healthy early exits).
- Total forward distress events across the panel: **120 events** (3.69% forward default rate). (16 firms entered the panel already distressed in their sole observation and thus have no antecedent row $t-1$).

The audit is documented in `results/phase6_9/dataset_audit.json`.

---

## 3. Leakage Prevention and Chronological Splitting

### 3.1 Chronological Split Partitions
Evaluation strictly respects the repository's established chronological split:
- **Training Set ($T \le 10$):** 2,471 evaluable rows, 95 forward distress events (3.84% default rate).
- **Validation Set ($11 \le T \le 12$):** 517 evaluable rows, 14 forward distress events (2.71% default rate).
- **Holdout Test Set ($T = 13 \to T = 14$):** 262 evaluable rows, 11 forward distress events (4.20% default rate).

### 3.2 Feature Engineering & Leakage Safeguards
- Features comprise 163 variables:
  - 83 raw contemporaneous financial ratios ($x1$ to $x83$).
  - 80 backward-looking temporal derivations across 8 key financial ratios ($x1, x2, x5, x10, x25, x48, x64, x81$), including lags ($t-1, t-2$), diffs ($x_t - x_{t-1}$), growth rates, accelerations, backward rolling means/stds ($k=2, 3$), consecutive decline indicators, and baseline deviations.
- **Leakage Verification:**
  - All temporal derivations strictly evaluate observations at or before period $t$.
  - Maximum correlation between any feature and continuous target `Financial Distress` is 0.4429 (for $x81$ operating cash flow); no feature encodes target identity.
  - Maximum correlation between any feature and forward binary target $Y_{i, t}$ is 0.1901 (for $x2$ retained earnings); zero forward information leaks into features.
  - All preprocessing parameters (`SimpleImputer` median, `RobustScaler` median and IQR) were fit **strictly on the training set** ($T \le 10$) and transformed forward.

The feature audit is documented in `results/phase6_9/feature_audit.json`.

---

## 4. Empirical Model Comparison under Alert Budget Constraint

### 4.1 Evaluation Setup
- **Evaluation Cohort:** Matured $T=13 \to T=14$ holdout cohort ($N = 262$ firms, 11 realized distress events).
- **Alert Budget Cap:** Defined by operational capacity as $M \le \lfloor 0.08 \times N \rfloor = \lfloor 0.08 \times 262 \rfloor = \mathbf{20\text{ alerts}}$ (7.63% alert rate).

### 4.2 Comprehensive Results Table

| Model Architecture | Applied Threshold ($\tau$) | Enforce 8% Budget | Total Alerts (Rate %) | Budget Compliant? | TP | FP | FN | TN | Recall (Sens.) | Precision (PPV) | Specificity | Brier Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Approved Production Baseline (Nominal)** | 0.0800 | No | 37 (14.12%) | ❌ No (+17 excess) | 5 | 32 | 6 | 219 | 45.45% | 13.51% | 87.25% | 0.0382 | 0.7961 | 0.1816 |
| **Approved Production Baseline (Budget-Capped)** | **0.1072** | **Yes** | **20 (7.63%)** | **✅ Yes (Cap=20)** | **4** | **16** | **7** | **235** | **36.36%** | **20.00%** | **93.63%** | **0.0382** | **0.7961** | **0.1816** |
| **Candidate RF Cost-Sensitive w10 (Nominal)** | 0.0894 | No | 20 (7.63%) | ✅ Yes (Cap=20) | 3 | 17 | 8 | 234 | 27.27% | 15.00% | 93.23% | 0.0389 | 0.7521 | 0.1410 |
| **Candidate RF Cost-Sensitive w10 (Budget-Capped)** | 0.0946 | Yes | 20 (7.63%) | ✅ Yes (Cap=20) | 3 | 17 | 8 | 234 | 27.27% | 15.00% | 93.23% | 0.0389 | 0.7521 | 0.1410 |
| **Calibrated Logistic Regression (Nominal)** | 0.0800 | No | 0 (0.00%) | ✅ Yes | 0 | 0 | 11 | 251 | 0.00% | 0.00% | 100.0% | 0.0404 | 0.8095 | 0.1670 |
| **Calibrated Logistic Regression (Budget-Capped)** | 0.0271 | Yes | 20 (7.63%) | ✅ Yes (Cap=20) | 3 | 17 | 8 | 234 | 27.27% | 15.00% | 93.23% | 0.0404 | 0.8095 | 0.1670 |
| **Standard Unweighted Random Forest (Nominal)** | 0.0800 | No | 15 (5.73%) | ✅ Yes | 2 | 13 | 9 | 238 | 18.18% | 13.33% | 94.82% | 0.0445 | 0.7342 | 0.1041 |
| **Standard Unweighted Random Forest (Budget-Capped)** | 0.0680 | Yes | 20 (7.63%) | ✅ Yes (Cap=20) | 2 | 18 | 9 | 233 | 18.18% | 10.00% | 92.83% | 0.0445 | 0.7342 | 0.1041 |

### 4.3 Key Empirical Findings
1. **Operating Under the Alert Budget Cap ($M \le 20$):**
   - The **Approved Production Baseline** achieves **4 / 11 detections (36.36% recall)** with 16 false positives (**20.00% precision**).
   - The **Candidate RF Cost-Sensitive w10** achieves **3 / 11 detections (27.27% recall)** with 17 false positives (**15.00% precision**).
   - **Logistic Regression** achieves **3 / 11 detections (27.27% recall)** with 17 false positives (**15.00% precision**).
   - Under an identical 20-alert workload constraint, **the approved production baseline outperforms all research candidates in both recall and precision**.
2. **Statistical Limitations & Uncertainty:**
   - With strictly $n = 11$ realized distress events, each distress case represents $1 / 11 = 9.09\text{ percentage points}$ of recall.
   - The 95% Wilson score confidence interval for Production Recall (36.36%, $n=11$) is $[15.2\%, 64.6\%]$.
   - The 95% Wilson score confidence interval for Candidate Recall (27.27%, $n=11$) is $[9.7\%, 56.6\%]$.
   - Because these intervals overlap extensively, **the research evidence is statistically inconclusive**; candidate non-inferiority or superiority cannot be asserted.

The complete comparison data is saved in `results/phase6_9/model_comparison.json`.

---

## 5. False Negative Diagnostic Audit

All 11 realized distress cases at $T=13 \to T=14$ were audited at the prediction timestamp ($T=13$). Results are categorized by transition dynamic:

```
                       REALIZED DISTRESS EVENTS (N = 11)
                                      |
         +----------------------------+----------------------------+
         |                                                         |
  ACUTE SUDDEN TRANSITIONS                                 CHRONIC DETERIORATIONS
  (5 Firms: Antecedent FD > 0)                             (6 Firms: Antecedent FD <= 0)
         |                                                         |
  - Firm 28: Base p=0.0858 (TP nom, FN cap)                - Firm 23: Base p=0.0768 (Near miss)
    Cand p=0.0496 (Missed by rolling features)             - Firm 29: Base p=0.0190 (Low score)
  - Firm 60: Base p=0.1135 (TP nom, TP cap)                - Firm 53: Base p=0.0187 (Low score)
    Cand p=0.0695 (Missed by rolling features)             - Firm 100: Base p=0.1072, Cand p=0.1098 (TP)
  - Firms 84, 225, 376: Low scores across models           - Firm 101: Base p=0.1500, Cand p=0.2223 (TP)
    (Unheralded sudden financial collapse)                 - Firm 112: Base p=0.1701, Cand p=0.1305 (TP)
```

### 5.1 Case-by-Case Diagnostic Log

| Company ID | Antecedent $FD_{13}$ | Realized $FD_{14}$ | Dynamic | Prod Prob ($p_B$) | Prod Alert ($\tau=0.08$) | Prod Alert (Cap) | Cand Prob ($p_A$) | Cand Alert ($\tau=0.0894$) | Primary Financial Indicators | Diagnostic Failure Taxonomy |
|---|---|---|---|---|---|---|---|---|---|---|
| **Firm 23** | -0.4820 | -3.9594 | Chronic | 0.0768 | False | False | 0.0763 | False | Working Capital 0.787, Cash Ratio 0.084, EBIT Margin 0.064 | **CHRONIC LOW-SCORE MISS:** Solvency index was already negative (-0.482), but model score fell marginally below 0.0800. |
| **Firm 28** | +1.4430 | -0.6815 | **Sudden** | **0.0858** | **True** | False | 0.0496 | False | Working Capital 1.103, Retained Earn 0.032, Cash Ratio 0.017 | **BUDGET-CAP EXCLUSION:** Detected at nominal threshold ($\tau=0.0800$), but squeezed out under 20-alert cap ($\tau^*=0.1072$). Candidate completely missed due to rolling features. |
| **Firm 29** | -0.4929 | -1.1315 | Chronic | 0.0190 | False | False | 0.0141 | False | Working Capital 0.997, Debt 6722.2, Cash Ratio 0.121 | **CHRONIC LOW-SCORE MISS:** Stressed antecedent solvency, but high reported liquidity dampened model score. |
| **Firm 53** | -0.1569 | -2.7519 | Chronic | 0.0187 | False | False | 0.0133 | False | Cash Ratio 0.257, Retained Earn 0.106, Debt 2364.1 | **CHRONIC LOW-SCORE MISS:** Accounting ratios showed cash stability despite negative underlying index. |
| **Firm 60** | +0.1357 | -1.2361 | **Sudden** | **0.1135** | **True** | **True** | 0.0695 | False | Working Capital 1.131, Liquidity 100.3, Cashflow 0.402 | **DETECTED UNDER BUDGET:** Baseline successfully alert under both nominal and budget-capped thresholds. Candidate missed. |
| **Firm 84** | +0.0019 | -0.5589 | Sudden | 0.0308 | False | False | 0.0292 | False | Retained Earn 0.058, Cash Ratio 0.119, Debt 1384.5 | **ACUTE CLIFF UNPREDICTED:** Solvent prior period transitioning to slight statutory breach without pre-distress ratio shift. |
| **Firm 100** | -0.4824 | -1.0118 | Chronic | **0.1072** | **True** | **True** | **0.1098** | **True** | Retained Earn -0.151, Cash Ratio -0.154, EBIT -0.079 | **DETECTED UNDER BUDGET:** Severe multi-period ratio deterioration correctly captured by both models. |
| **Firm 101** | -0.3137 | -1.9532 | Chronic | **0.1500** | **True** | **True** | **0.2223** | **True** | Retained Earn -0.029, Cashflow 0.242, Liquidity -384.0 | **DETECTED UNDER BUDGET:** Negative working capital correctly flagged by both models. |
| **Firm 112** | -0.0699 | -1.9827 | Chronic | **0.1701** | **True** | **True** | **0.1305** | **True** | Working Capital 1.037, Cashflow 0.038, Retained Earn 0.002 | **DETECTED UNDER BUDGET:** Cash flow contraction flagged by both models. |
| **Firm 225** | +0.2440 | -0.8836 | Sudden | 0.0247 | False | False | 0.0130 | False | Working Capital 0.399, Retained Earn 0.087, Cashflow 0.641 | **ACUTE CLIFF UNPREDICTED:** Positive cash flow and solvency in antecedent year; sudden unpredicted operational collapse. |
| **Firm 376** | +0.5252 | -0.5805 | Sudden | 0.0674 | False | False | 0.0398 | False | Working Capital 1.040, Cash Ratio 0.116, Debt 1511.5 | **ACUTE CLIFF UNPREDICTED:** Score reached 0.0674, near baseline threshold, but did not breach $\tau=0.0800$. |

The false negative diagnostic dataset is saved in `results/phase6_9/false_negative_audit.csv`.

---

## 6. Explainability and Supervisory Governance

### 6.1 Top Global TreeSHAP Features
TreeSHAP attribution evaluates the approved production Random Forest model on the test cohort with exact additivity:

| Rank | Feature | Base Variable | Category | Mean \|SHAP\| | Mean Signed SHAP | Directional Risk Role |
|---|---|---|---|---|---|---|
| **1** | `x2_roll_mean_2` | $x2$ | Retained Earnings Temporal | 0.02468 | -0.01841 | Protective Factor (Higher reduces distress) |
| **2** | `x12` | $x12$ | Cash Flow to Total Debt | 0.02243 | -0.01006 | Protective Factor (Higher reduces distress) |
| **3** | `x8` | $x8$ | Current Assets to Total Debt | 0.02039 | -0.01553 | Protective Factor (Higher reduces distress) |
| **4** | `x10_roll_mean_2` | $x10$ | EBIT Margin Temporal | 0.01765 | -0.01514 | Protective Factor (Higher reduces distress) |
| **5** | `x2_roll_mean_3` | $x2$ | Retained Earnings 3-yr Mean | 0.01679 | -0.00787 | Protective Factor (Higher reduces distress) |
| **6** | `x16` | $x16$ | Working Capital to Assets | 0.01546 | -0.00794 | Protective Factor (Higher reduces distress) |
| **7** | `x52` | $x52$ | Operating Income Stability | 0.01527 | -0.00909 | Protective Factor (Higher reduces distress) |
| **8** | `x46` | $x46$ | Cash to Total Assets | 0.01440 | -0.01117 | Protective Factor (Higher reduces distress) |
| **9** | `x2` | $x2$ | Contemporaneous Retained Earn | 0.01244 | -0.00772 | Protective Factor (Higher reduces distress) |
| **10** | `x5` | $x5$ | Cash Ratio Level | 0.01151 | -0.00241 | Protective Factor (Higher reduces distress) |

### 6.2 Regulatory Governance: Association vs Causality
In adherence to Federal Reserve SR 11-7 and OCC 2011-12 model risk management standards:
> **Critical Supervisory Principle:** SHAP attributions and feature importance metrics reflect **statistical predictive associations**, **not causal mechanisms**. A high positive SHAP value for accelerating debt indicates that firms exhibiting this pattern historically experienced distress more frequently; it does **not** prove that reducing that variable alone will avert financial failure. The model is an **observational early-warning tool**, not a prescriptive decision engine.

The explainability summary is persisted in `results/phase6_9/explainability_summary.json`.

---

## 7. Application Completion and Verification

### 7.1 Streamlit Interactive Dashboard (`app.py`)
The dashboard was audited, preserved, and enhanced to fulfill all Phase 6.9 interactive demonstration requirements:
1. **Dynamic Enterprise Selector:** Supports selecting from:
   - ⭐ High-Risk Watchlist (Top 50 priority surveillance firms)
   - 📋 Monitored Corporate Universe (all 422 SMEs in Dataset A)
   - ⌨️ Custom Search (direct alphanumeric input)
2. **Historical Period Selector:** Allows user to select any observed reporting period ($T \in [1, 14]$) or latest evaluation period for the selected firm.
3. **Input Validation & Missing Data Handling:** Entering a non-existent company ID (e.g. 9999) displays a clean error banner (`❌ Error: Company '9999' not found in the surveillance dataset`) and halts rendering safely rather than fabricating predictions.
4. **Dedicated Phase 6.9 Research Page:** Page `"🔬 Research Model Improvement & Baseline Audit"` provides:
   - Side-by-side model comparison under 8% alert budget constraint.
   - Interactive False Negative Audit table with filtering by transition dynamic.
   - Company-by-company side-by-side inspection for the selected firm.
   - Global TreeSHAP feature attributions and supervisory causal disclaimer.
   - Dataset and preprocessing integrity audit cards.
5. **Prominent Model Disclaimer:** Visible across the application, clearly stating that the system is an academic research prototype, not a credit-decision tool.

### 7.2 FastAPI Backend (`api.py`)
Audited and enhanced with dedicated Phase 6.9 research endpoints:
- `GET /research/model-comparison`: Delivers empirical model benchmarking JSON under alert constraints.
- `GET /research/false-negatives`: Delivers auditable false negative diagnostics.
- `GET /research/dataset-audit`: Delivers dataset dimensions, time contiguity, and target distribution.
- `GET /research/explainability-summary`: Delivers TreeSHAP attribution and causal governance metadata.
- `GET /company/{company_id}`: Validates company existence and returns HTTP 404 for invalid inputs.

---

## 8. Verification and Test Results

### 8.1 Phase-Specific Tests
A dedicated test suite was implemented in `tests/test_phase6_9_model_improvement.py`:
- `test_production_baseline_immutability`: PASSED
- `test_dataset_audit_completeness`: PASSED
- `test_chronological_splits_and_leakage_safety`: PASSED
- `test_feature_audit_and_zero_leakage`: PASSED
- `test_model_comparison_and_budget_constraints`: PASSED
- `test_false_negative_audit_completeness`: PASSED
- `test_explainability_summary_and_causal_distinction`: PASSED
- `test_api_research_endpoints_and_error_handling`: PASSED

**Result:** **8 passed in 34.34s (100% pass rate)**.

### 8.2 Full Repository Regression Suite
Command: `pytest --ignore=scripts -q`  
Outcome: **217 passed in 322.00s (0 failures, 100% pass rate)**.

---

## 9. Final Decision & Status Determination

### Decision Criteria Evaluation
1. **Pipeline Integrity:** The existing dataset at `data/raw/Financial Distress.csv` is clean, contiguous, and leak-free. The data pipeline does **not** require repair.
2. **Research Model Evidence:** Under the strict 8% alert budget constraint, the Approved Production Baseline detected 4/11 distress events (36.4%), whereas the experimental candidate detected 3/11 (27.3%) and Logistic Regression detected 3/11 (27.3%). The small sample size ($n=11$) renders statistical superiority claims inconclusive.
3. **Application Completeness:** Both the Streamlit dashboard and FastAPI service are feature-complete, interactive, well-documented, protected against missing inputs, and fully demonstrable.

### Official Status

$$\mathbf{DEMO\_APPLICATION\_READY}$$

*(Accompanied by the supervisory finding that experimental candidate promotion is blocked due to `RESEARCH_EVIDENCE_INCONCLUSIVE` under sample size limitations, and the Approved Production Baseline remains the official, immutable authority).*
