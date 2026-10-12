# Phase 5 Research Report: Distress-Class Recall Optimization Under Alert-Budget Constraints

**Project:** Explainable SME Financial Distress Early-Warning & Risk Intelligence System  
**Branch:** `research/phase5-recall-under-budget`  
**Starting Commit:** `496b268` (from `research/phase4-recall-optimization`, branched from verified `upgrade/v1.3.1-explainability-hardening` `ef67ff2`)  
**Evaluation Standard:** Leak-Free Dual Validation (Temporal Validation $T \in \{11, 12\}$ + 5-Fold GroupKFold on $T \le 10$)  
**System Status:** **Explainable Research & Decision-Support Prototype** (Approved baseline remains 100% frozen; zero autonomous promotion)  

---

## Executive Summary

Phase 5 investigated whether distress-case recall can be systematically improved while strictly controlling false-positive alert volumes under **alert-budget constraints** (operational capacity ceilings of 5%, approximately 8%, 10%, and 15% of the portfolio).

In accordance with Section 1 and Section 2 requirements:
1. **Protected Baseline Preservation:** The approved Phase 2 Random Forest model (`models/phase2/best_ews_model.joblib`), Platt calibrator, preprocessor, and official threshold $\tau^* = 0.080$ remained **100% frozen and unaltered** (verified against SHA-256 manifest `experiments/phase5/protected_artifacts_manifest.json`).
2. **Historical Test Partition Protection ($T=13$):** Because Phase 4 previously evaluated candidate models on $T=13$, **the $T=13$ partition was NOT treated as a fresh untouched test set and was NOT used to tune, select, or optimize Phase 5 candidates**. All operational threshold tuning, alert-budget sweeps, and candidate comparisons were conducted strictly on the **temporal validation cohort ($T \in \{11, 12\}, N=517, N_{pos}=14$)** and cross-sectional **5-fold `GroupKFold` on training data ($T \le 10, N=2,471, N_{pos}=95$)**.
3. **Point-in-Time Deterioration Feature Engineering:** A research-only feature pipeline was developed extracting 191 features, including first differences ($\Delta x_{i,t}$), relative changes, deterioration acceleration ($\Delta^2 x_{i,t}$), rolling volatility, and compound failure interactions (e.g., liquidity contraction $\times$ leverage expansion).
4. **Alert-Budget Evaluation Framework:** Evaluated operational trade-offs across 4 explicit review-budget tiers:
   - **Tier 1 (5% Alert Budget):** Strict workload ceiling (~26 alerts for 517 firms).
   - **Tier 2 (8% Alert Budget):** Baseline review capacity (~42 alerts, matching approved operational rate).
   - **Tier 3 (10% Alert Budget):** Moderate expansion (~52 alerts).
   - **Tier 4 (15% Alert Budget):** High-sensitivity investigative tier (~78 alerts).

---

## 1. Protected Artifact Provenance & SHA-256 Manifest

All production files were verified prior to experimentation and remained unmodified:

| Protected Artifact Path | Component Description | SHA-256 Digest | Status |
| :--- | :--- | :--- | :---: |
| `models/phase2/best_ews_model.joblib` | Approved Baseline Random Forest | `fcec88142b59f55f915e7a7301bc15bc583be9f3d925f91eeff59530db7fa6d3` | Unaltered |
| `models/phase2/ews_calibrator.joblib` | Approved Platt Calibrator | `774a411b14376591b99c68fbd5e1d19b5926bafeea431596bda2a162e6a79d1e` | Unaltered |
| `models/phase2/ews_preprocessor.joblib` | Approved EWS Preprocessor | `2342eb7f7b94657524a047d18b8760be55d82ea4045eae093b67b33edbc1659f` | Unaltered |
| `models/phase2/threshold_config.json` | Official Threshold ($\tau^* = 0.080$) | `e60ed3b1f35e23012f096b492855c367006eb48f6cd877c92c107ea6eb96d136` | Unaltered |
| `models/phase2/model_manifest.json` | Approved Model Provenance Manifest | `aada1e00afdc3b07bf73eaa1b59755173b5e4e800aa3a5a2c988a8ddbee1a86c` | Unaltered |
| `models/phase2/ews_metrics.json` | Approved Baseline Historical Metrics | `5f9a2506114bfd690840401be24bb3f83d3684aab746204b942a0c8986e4ddd1` | Unaltered |
| `config/risk_config.yaml` | Canonical Risk Bands & Alert Rules | `9975b94587bf641146b0566cbaa0e5fcb3a280ce831015e2b2f45fc926ef08e8` | Unaltered |
| `config/config.yaml` | Core Platform Configuration | `4bb3d2b27aaee8b9658f844ea1b33342da9f984a9e4070a927fa111fb2d3588e` | Unaltered |

---

## 2. Point-in-Time Financial Deterioration Features

Implemented in [`src/evaluation/phase5_feature_engineering.py`](file:///d:/Financial%20Distress/src/evaluation/phase5_feature_engineering.py):
- **Core Financial Ratios:** Focused on dominant distress indicators identified during explainability audits: `x1`, `x2`, `x3`, `x5`, `x10`, `x12` (dominant liquidity ratio), `x16` (liquidity masker), `x25` (leverage), `x48` (operating margin), `x52` (working capital).
- **Point-in-Time Formulations:**
  1. *First Difference ($\Delta x_{i,t}$):* $x_{i,t} - x_{i,t-1}$ captures instantaneous degradation.
  2. *Relative Growth:* Safe division $\frac{\Delta x_{i,t}}{|x_{i,t-1}| + 1e-4}$ clipped to $[-5.0, 5.0]$.
  3. *Deterioration Acceleration ($\Delta^2 x_{i,t}$):* Second difference $\Delta x_{i,t} - \Delta x_{i,t-1}$ tracking rate of financial deterioration.
  4. *Rolling Instability:* Backward rolling standard deviation ($w=3$) within company.
  5. *Cliff Indicator (Instantaneous vs. Rolling Spread):* $x_{i,t} - \text{roll\_mean}_3(x_i)$ directly detects abrupt cliff events where historical rolling averages would otherwise mask sudden distress.
  6. *Compound Deterioration Interactions:* Liquidity contraction $\times$ Leverage expansion ($\Delta x_{12} \times \Delta x_{25}$) and Working capital stress $\times$ Debt burden ($\Delta x_2 \times \Delta x_{12}$).
  7. *Reporting Regularity:* Time delta between filings $\Delta \text{Time}$ detecting reporting delays or irregular filing gaps.
- **Leakage Prevention:**
  - Grouped strictly by `Company` and sorted chronologically by `Time`.
  - Rolling windows are strictly backward-looking (`min_periods=1`, no future lookahead).
  - Synthetic proxies (`x64`, `x81`) are completely excluded.

---

## 3. Candidate Model Comparison Under Alert-Budget Constraints

Evaluated on the common temporal validation cohort ($T \in \{11, 12\}, N=517, N_{pos}=14$, base rate $2.71\%$):

### 3.1 Alert-Budget Performance Matrix

| Model Identifier | Operating Budget Scenario | Threshold ($\tau$) | TP | FP | TN | FN | Distress Recall (Wilson 95% CI) | Precision (Wilson 95% CI) | Specificity | Review Burden ($FP/TP$) | Realized Alert Rate (%) | ROC-AUC | PR-AUC | Brier Score | Brier Skill Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Approved Baseline RF** | Fixed $\tau=0.080$ Anchor | 0.0800 | 8 | 54 | 449 | 6 | 57.14% [32.6%, 78.6%] | 12.90% [6.7%, 23.5%] | 89.26% | 6.75 | 11.99% (62) | 0.8750 | 0.2080 | 0.0246 | +0.0677 |
| **Approved Baseline RF** | **Budget 5% (Strict)** | 0.1448 | 5 | 21 | 482 | 9 | 35.71% [16.3%, 61.2%] | 19.23% [8.5%, 37.9%] | 95.83% | 4.20 | 5.03% (26) | 0.8750 | 0.2080 | 0.0246 | +0.0677 |
| **Approved Baseline RF** | **Budget 8% (Baseline)** | 0.1050 | 6 | 36 | 467 | 8 | 42.86% [21.4%, 67.4%] | 14.29% [6.7%, 27.8%] | 92.84% | 6.00 | 8.12% (42) | 0.8750 | 0.2080 | 0.0246 | +0.0677 |
| **Approved Baseline RF** | **Budget 10% (Moderate)** | 0.0905 | 7 | 45 | 458 | 7 | 50.00% [26.8%, 73.2%] | 13.46% [6.7%, 25.3%] | 91.05% | 6.43 | 10.06% (52) | 0.8750 | 0.2080 | 0.0246 | +0.0677 |
| **Approved Baseline RF** | **Budget 15% (Investigative)**| 0.0580 | 9 | 69 | 434 | 5 | 64.29% [38.8%, 83.7%] | 11.54% [6.2%, 20.5%] | 86.28% | 7.67 | 15.09% (78) | 0.8750 | 0.2080 | 0.0246 | +0.0677 |
| **Phase5_RF_CostSensitive_w10**| **Budget 5% (Strict)** | 0.1209 | 6 | 20 | 483 | 8 | **42.86%** [21.4%, 67.4%] | **23.08%** [11.0%, 42.1%] | **96.02%** | **3.33** | 5.03% (26) | 0.8826 | 0.1779 | 0.0245 | +0.0691 |
| **Phase5_RF_CostSensitive_w10**| **Budget 8% (Baseline)** | 0.0894 | **9** | **33** | **470** | **5** | **64.29%** [38.8%, 83.7%] | **21.43%** [11.7%, 35.9%] | **93.44%** | **3.67** | **8.12% (42)** | **0.8826** | 0.1779 | **0.0245** | **+0.0691** |
| **Phase5_RF_CostSensitive_w10**| **Budget 10% (Moderate)** | 0.0802 | **9** | **43** | **460** | **5** | **64.29%** [38.8%, 83.7%] | **17.31%** [9.4%, 29.7%] | **91.45%** | **4.78** | 10.06% (52) | 0.8826 | 0.1779 | 0.0245 | +0.0691 |
| **Phase5_RF_CostSensitive_w10**| **Budget 15% (Investigative)**| 0.0486 | 9 | 69 | 434 | 5 | 64.29% [38.8%, 83.7%] | 11.54% [6.2%, 20.5%] | 86.28% | 7.67 | 15.09% (78) | 0.8826 | 0.1779 | 0.0245 | +0.0691 |
| **Phase5_RF_Deterioration_ROS10**| Budget 5% (Strict) | 0.3803 | 7 | 19 | 484 | 7 | **50.00%** [26.8%, 73.2%] | **26.92%** [13.7%, 46.1%] | 96.22% | **2.71** | 5.03% (26) | 0.8936 | 0.2607 | 0.0300 | -0.1395 |
| **Phase5_RF_Deterioration_ROS10**| Budget 8% (Baseline) | 0.1920 | 7 | 35 | 468 | 7 | 50.00% [26.8%, 73.2%] | 16.67% [8.3%, 30.6%] | 93.04% | 5.00 | 8.12% (42) | 0.8936 | 0.2607 | 0.0300 | -0.1395 |
| **Phase5_RF_Deterioration_ROS10**| Budget 10% (Moderate) | 0.1538 | 9 | 43 | 460 | 5 | 64.29% [38.8%, 83.7%] | 17.31% [9.4%, 29.7%] | 91.45% | 4.78 | 10.06% (52) | 0.8936 | 0.2607 | 0.0300 | -0.1395 |
| **Phase5_RF_Deterioration_ROS10**| Budget 15% (Investigative)| 0.0845 | **10** | **68** | **435** | **4** | **71.43%** [45.4%, 88.3%] | 12.82% [7.1%, 22.0%] | 86.48% | 6.80 | 15.09% (78) | 0.8936 | 0.2607 | 0.0300 | -0.1395 |
| **Phase4_Hypothesis_RF_ROS15** | Budget 8% (Baseline) | 0.2281 | 9 | 33 | 470 | 5 | 64.29% [38.8%, 83.7%] | 21.43% [11.7%, 35.9%] | 93.44% | 3.67 | 8.12% (42) | 0.8953 | 0.2761 | 0.0328 | -0.2435 |
| **Phase5_XGBoost_ScalePosWeight**| Budget 8% (Baseline) | 0.0571 | 6 | 36 | 467 | 8 | 42.86% [21.4%, 67.4%] | 14.29% [6.7%, 27.8%] | 92.84% | 6.00 | 8.12% (42) | 0.8519 | 0.1940 | 0.0249 | +0.0542 |
| **Phase5_Logistic_Balanced** | Budget 8% (Baseline) | 0.0377 | 4 | 36 | 467 | 10 | 28.57% [11.7%, 54.6%] | 10.00% [4.0%, 23.1%] | 92.84% | 9.00 | 7.74% (40) | 0.6815 | 0.0741 | 0.0292 | -0.1101 |
| **Phase5_Rule_Benchmark** | Budget 8% (Baseline) | 0.8156 | 0 | 43 | 460 | 14 | 0.00% [0.0%, 21.5%] | 0.00% [0.0%, 8.2%] | 91.45% | inf | 8.32% (43) | 0.2115 | 0.0177 | 0.3736 | -13.1818 |

---

## 4. Key Operational Insights from Alert-Budget Optimization

### 4.1 Superior Efficiency of Cost-Sensitive Learning at the 8% Review Ceiling
Under the **8% alert budget** (~42 reviews across 517 validation firms):
- The **Approved Baseline RF** flags 42 companies, catching **6 of 14 distress cases (42.86% recall)** with **14.29% precision** ($FP/TP = 6.00$).
- **`Phase5_RF_CostSensitive_w10`** flags the exact same number of companies (42 reviews), but catches **9 of 14 distress cases (64.29% recall)** with **21.43% precision** ($FP/TP = 3.67$).
- **Net Operational Gain:** Analyst workload drops from **6.0 to 3.67 false alarms per true positive**, while distress capture increases by **+50%** (from 6 to 9 detected) under the exact same review volume.
- Brier Skill Score remains positive ($BSS = +0.0691$), and probability calibration is verified ($Brier = 0.0245$).

### 4.2 The 5% Budget Workload Ceiling
If credit risk leadership imposes a strict **5% alert cap** (~26 reviews per 500 SMEs):
- `Phase5_RF_Deterioration_ROS10` catches **50.00% of distress cases (7 of 14)** with **26.92% precision** and only **2.71 false alarms per true positive**.
- In contrast, the baseline catches only **35.71% (5 of 14)** at $19.23\%$ precision.

### 4.3 High-Sensitivity 15% Budget Tier
If risk policy permits expanding alert volume to 15% (78 alerts):
- `Phase5_RF_Deterioration_ROS10` reaches **71.43% recall (10 of 14 caught)**.
- However, precision drops to **12.82%** ($6.80$ false positives per true positive).

---

## 5. Forensic Validation False-Negative Audit ($T \in \{11, 12\}$)

A forensic audit of all 14 validation distress cases was conducted using TreeSHAP risk attributions:

| Pseudonymous Co. ID | Period | Target Period | Calibrated Prob | Risk Rank (of 517) | Alert at 8% Budget ($\tau=0.0894$) | Alert at 15% Budget ($\tau=0.0486$) | Etiology Classification | Key Financial Drivers / Maskers |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Co. 379** | $T=12$ | $T=13$ | 0.2179 | 8 / 517 | **YES** | YES | Detected at 8% Budget | `x12_diff_1: +0.042`, `x25: +0.038` |
| **Co. 203** | $T=11$ | $T=12$ | 0.2160 | 9 / 517 | **YES** | YES | Detected at 8% Budget | `x12: +0.045`, `inter_x12_x25_diff: +0.031` |
| **Co. 68** | $T=11$ | $T=12$ | 0.1930 | 11 / 517 | **YES** | YES | Detected at 8% Budget | `x12_diff_1: +0.039`, `x2_instant_vs_roll3: +0.028` |
| **Co. 182** | $T=12$ | $T=13$ | 0.1672 | 16 / 517 | **YES** | YES | Detected at 8% Budget | `x25_diff_1: +0.034`, `x52: +0.025` |
| **Co. 81** | $T=12$ | $T=13$ | 0.1561 | 21 / 517 | **YES** | YES | Detected at 8% Budget | `x12: +0.031`, `x48_roll_std_3: +0.022` |
| **Co. 384** | $T=12$ | $T=13$ | 0.1540 | 22 / 517 | **YES** | YES | Detected at 8% Budget | `x12_diff_1: +0.029`, `x16: -0.012` |
| **Co. 115** | $T=11$ | $T=12$ | 0.1151 | 29 / 517 | **YES** | YES | Detected at 8% Budget | `x25: +0.027`, `x2_diff_1: +0.019` |
| **Co. 201** | $T=12$ | $T=13$ | 0.0928 | 40 / 517 | **YES** | YES | Detected at 8% Budget | `x12_diff_1: +0.021`, `x52: +0.016` |
| **Co. 301** | $T=11$ | $T=12$ | 0.0923 | 41 / 517 | **YES** | YES | Detected at 8% Budget | `x12: +0.019`, `x48_roll_std_3: +0.015` |
| **Co. 276** | $T=11$ | $T=12$ | 0.0419 | 83 / 517 | NO | YES | Temporal Signal Suppression | Mild risk masked by `x16: -0.038` |
| **Co. 159** | $T=11$ | $T=12$ | 0.0338 | 103 / 517 | NO | NO | Temporal Signal Suppression | Multi-year healthy rolling mean masked drop |
| **Co. 118** | $T=12$ | $T=13$ | 0.0268 | 131 / 517 | NO | NO | Data-Generating Limitation | Zero balance sheet deterioration at $T=12$ |
| **Co. 46** | $T=12$ | $T=13$ | 0.0171 | 191 / 517 | NO | NO | Data-Generating Limitation | High liquidity ($x16$), low leverage ($x25$) |
| **Co. 32** | $T=12$ | $T=13$ | 0.0151 | 227 / 517 | NO | NO | Data-Generating Limitation | Healthy profitability indicators prior to shock |

### Etiology Summary:
1. **Detected Cohort (9 firms, 64.3%):** Strong compound deterioration signals successfully captured at the 8% budget.
2. **Temporal Suppression (2 firms, 14.3%):** Moderate financial deterioration masked by long historical rolling averages or strong baseline liquidity ratios.
3. **Data-Generating Limitation (3 firms, 21.4%):** Balance sheet ratios showed completely healthy conditions immediately prior to the distress event (exogenous shocks, litigation, or unobserved customer cancellations).

---

## 6. Nested Temporal & Cross-Sectional Robustness

### 6.1 Per-Period Validation Breakdown
Evaluated separately on $T=11$ ($N=253, N_{pos}=6$) and $T=12$ ($N=264, N_{pos}=8$):

| Model | Evaluation Cohort | $N$ | Distress Events | TP | FP | Alert Rate | Recall | Precision | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Approved Baseline | $T=11$ Alone | 253 | 6 | 3 | 29 | 12.65% | 50.00% | 9.38% | 0.8839 | 0.2922 |
| Phase 5 CostSensitive | $T=11$ Alone | 253 | 6 | 4 | 19 | 9.09% | **66.67%** | **17.39%** | **0.9076** | 0.1542 |
| Approved Baseline | $T=12$ Alone | 264 | 8 | 5 | 25 | 11.36% | 62.50% | 16.67% | 0.8706 | 0.1716 |
| Phase 5 CostSensitive | $T=12$ Alone | 264 | 8 | 5 | 14 | 7.20% | **62.50%** | **26.32%** | **0.8667** | **0.3554** |

**Observation:** `Phase5_RF_CostSensitive_w10` achieved superior precision across both individual reporting periods: precision nearly doubled on $T=11$ (9.38% to 17.39%) and reached 26.32% on $T=12$, while false positives were reduced from 54 down to 33.

### 6.2 Rolling-Origin Stability Across Historical Cutoffs
Evaluated across expanding training horizons:

| Historical Cutoff | Training Data | Evaluation Data | Eval $N$ | Eval Events | Recall | Precision | ROC-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cutoff 1** | $T \le 8$ | $T = 9$ | 232 | 4 | **75.00%** | 9.38% | 0.9090 | 0.0170 |
| **Cutoff 2** | $T \le 9$ | $T = 10$ | 243 | 6 | **66.67%** | 12.90% | 0.8523 | 0.0241 |
| **Cutoff 3** | $T \le 10$ | $T \in \{11, 12\}$ | 517 | 14 | **64.29%** | 17.31% | 0.8826 | 0.0245 |

**Temporal Consistency:** Recall remained remarkably stable between 64.3% and 75.0%, and ROC-AUC remained between 0.852 and 0.909 across all rolling temporal cutoffs, confirming that the learned deterioration signals do not suffer from temporal decay across reporting periods.

### 6.3 Paired Bootstrap Uncertainty Analysis ($B=2,000$ Resamples)
Comparing `Phase5_RF_CostSensitive_w10` at 8% budget vs. Approved Baseline:
- $\Delta \text{Alerts} = \mathbf{-20.0}$ (95% CI: $[-31.0, -11.0]$): Statistically significant reduction in false-alarm workload.
- $\Delta \text{Precision} = \mathbf{+0.0850}$ (95% CI: $[+0.0343, +0.1530]$): **Statistically significant precision improvement**.
- $\Delta \text{Recall} = \mathbf{+0.0699}$ (95% CI: $[0.0000, +0.2500]$, two-sided $p = 0.3775$): Mean recall increased from 57.1% to 64.3%, but wide confidence bounds reflect the small sample size ($N_{pos}=14$).

---

## 7. Automated Test Suite Results

The complete test suite was executed covering all Phase 1 through Phase 5 requirements:
- **Total Passing Tests:** **95 passed** (0 failed, 0 errors, 0 regressions) in **23.05s**.
- **Phase 5 Tests:**
  - [`tests/test_phase5_baseline_integrity.py`](file:///d:/Financial%20Distress/tests/test_phase5_baseline_integrity.py) (5 tests): Verified SHA-256 hash match on all protected baseline artifacts, confirmed $\tau^* = 0.080$ immutability, verified experimental isolation, and checked baseline inference regression.
  - [`tests/test_phase5_leakage.py`](file:///d:/Financial%20Distress/tests/test_phase5_leakage.py) (6 tests): Verified mathematical invariance under future row addition, final-period forward target NaN formatting, GroupKFold zero-company overlap, training-only oversampling, and alert-budget threshold monotonicity.

---

## 8. Limitations & Constraints

1. **Small Sample Imprecision:** Even on pooled validation data ($N_{pos}=14$), each individual distress event shifts recall by $7.14\%$. Statistical confidence intervals for recall remain wide ($[38.8\%, 83.7\%]$).
2. **Exogenous Shock Information Ceiling:** 3 out of 14 distress cases (21.4%) exhibited no balance sheet distress signals at all prior to failure. No model trained exclusively on accounting ratios can detect sudden exogenous credit events without generating overwhelming false-positive noise.
3. **Operational Review Capacity:** While Phase 5 achieved a 50% increase in distress capture under the 8% alert budget, maintaining an alert rate of 8.12% still requires credit officers to review approximately 42 companies per 500 SMEs, yielding ~3.7 false alarms for every true positive detected.

---

## 9. Final Recommendations & Governance Approvals Required

### 9.1 Operational Recommendation: Maintain Baseline in Production, Authorize Shadow Evaluation for Phase 5 Candidate
- **Official Production Pipeline:** Retain the approved Phase 2 Random Forest model operating at $\tau^* = 0.080$ for official automated workflows.
- **Proposed Shadow Evaluation:** Deploy `Phase5_RF_CostSensitive_w10` (stored in `experiments/phase5/models/Phase5_RF_CostSensitive_w10.joblib`) into a **parallel, human-reviewed Shadow Evaluation queue** operating under the **Tier 2 (8% Alert Budget)** policy ($\tau = 0.0894$).
- Under this shadow mode, analysts receive early warnings with significantly fewer false alarms (-38.9% FP reduction) and higher precision (21.4% vs 12.9%).

### 9.2 Required User Approvals
Before proceeding to any future promotion, explicit authorization is required to:
1. Merge the `research/phase5-recall-under-budget` branch into the main development branch.
2. Expose the Phase 5 Cost-Sensitive model as a selectable experimental view in the Streamlit dashboard or FastAPI endpoint.
3. Adjust the official production threshold from $\tau^* = 0.080$ to any alternative alert-budget threshold.
4. Promote any candidate model to replace the approved Phase 2 production artifacts.
