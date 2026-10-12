# Phase 2.1 Independent Reproducibility & Verification Audit Report

**Project:** Explainable SME Financial Distress Early-Warning & Risk Intelligence System  
**Audit Date:** October 12, 2026  
**Auditor Roles:** Senior ML Engineer, Financial Risk Modeling Researcher, ML Validation Specialist  
**Branch:** `upgrade/v1.2.1-validation-hardening`  
**Phase:** Phase 2.1 (Validation, Calibration & Reproducibility Hardening)  

---

## 1. Executive Summary & Verification Mandate

To satisfy rigorous ML research auditing standards, this report documents the independent reproduction and verification of the Phase 2 early-warning model. 

The evaluation was executed via [`scripts/verify_phase2_reproducibility.py`](file:///d:/Financial%20Distress/scripts/verify_phase2_reproducibility.py) strictly by loading the saved model artifacts **without retraining the estimator**.

### Audit Finding
**100% PERFECT REPRODUCIBILITY CONFIRMED.** Every recomputed metric, confusion matrix count, probability range, and confidence interval matches the reported Phase 2 reference values with zero discrepancy.

---

## 2. Recomputed Metrics vs. Reported Claims Audit Matrix

| Metric / Dimension | Reported Reference Value | Independently Recomputed Value | Exact Match Status | Audit Note / Mathematical Formula |
| :--- | :---: | :---: | :---: | :--- |
| **Total Test Observations** | 262 | **262** | **MATCH ✅** | $T=13$ out-of-period holdout partition |
| **True Distress Events** | 11 | **11** | **MATCH ✅** | Base prevalence = 4.20% |
| **Solvent Non-Distress Firms**| 251 | **251** | **MATCH ✅** | Base prevalence = 95.80% |
| **True Positives (TP)** | 3 | **3** | **MATCH ✅** | Hits at $\tau^* = 0.080$ |
| **False Positives (FP)** | 17 | **17** | **MATCH ✅** | False alarms at $\tau^* = 0.080$ |
| **False Negatives (FN)** | 8 | **8** | **MATCH ✅** | Missed distress events at $\tau^* = 0.080$ |
| **True Negatives (TN)** | 234 | **234** | **MATCH ✅** | Cleared solvent firms at $\tau^* = 0.080$ |
| **Test Recall** | 27.3% (0.2727) | **27.27% (3 / 11)** | **MATCH ✅** | $\text{Recall} = \text{TP} / (\text{TP} + \text{FN})$ |
| **Test Precision** | 15.0% (0.1500) | **15.00% (3 / 20)** | **MATCH ✅** | $\text{Precision} = \text{TP} / (\text{TP} + \text{FP})$ |
| **Test Specificity** | 93.2% (0.9323) | **93.23% (234 / 251)**| **MATCH ✅** | $\text{Specificity} = \text{TN} / (\text{TN} + \text{FP})$ |
| **Test F1-Score** | 0.1935 | **0.1935** | **MATCH ✅** | Harmonic mean of precision and recall |
| **Test F2-Score** | 0.2344 | **0.2344** | **MATCH ✅** | Weighting recall 2x higher than precision |
| **Test ROC-AUC** | 0.7783 | **0.7783** | **MATCH ✅** | Area under Receiver Operating Characteristic |
| **Test PR-AUC** | 0.1946 | **0.1946** | **MATCH ✅** | Average precision score |
| **Test Brier Score** | 0.0387 | **0.0387** | **MATCH ✅** | Mean squared probability error (Platt scaled) |
| **Test Recall Wilson 95% CI**| [9.8%, 56.6%] | **[9.75%, 56.57%]** | **MATCH ✅** | Asymmetric binomial confidence interval |
| **Portfolio Flagged Count** | 20 / 262 (7.6%) | **20 / 262 (7.6%)** | **MATCH ✅** | Operational credit review burden |

---

## 3. Provenance & Cryptographic Traceability Metadata

All artifacts used in this verification were validated with SHA-256 cryptographic hashes:

### 3.1 Raw Data Fingerprint
- **Dataset File:** `data/raw/Financial Distress.csv`
- **SHA-256 Hash:** `a6a13dd647a5ed14f1aaade81d969f3c20a18cfd0b1a742f071840cf917ee14d`
- **Verified Dimensions:** 3,672 rows, 86 columns, 422 unique SMEs across 14 fiscal periods.

### 3.2 Model & Preprocessing Checkpoint Hashes
- **Primary Model Checkpoint:** `models/phase2/best_ews_model.joblib`  
  *SHA-256:* `0ce80a1c97f485db5da25fb635a9f6fe0210f8a846c4fa315f6c6d042fb142a7`
- **Fitted Preprocessor:** `models/phase2/ews_preprocessor.joblib`  
  *SHA-256:* `a37463f6844ef15ea82d7c57b5fbaf581ff2ba38f906cce75b6ae0bfbf3962df`
- **Platt Calibrator:** `models/phase2/ews_calibrator.joblib`  
  *SHA-256:* `eb2e3a84e27f1013775bfa960fa3ce942503a7d4db3b85db0f26211833d7b876`
- **Threshold Configuration:** `models/phase2/threshold_config.json`  
  *SHA-256:* `2cefdc3866d5b008e3ebf61a1040a4421b5ca49b6a982992d9d95f6ef3f80169`

### 3.3 Verification Runtime Environment
- **Operating System:** Windows 10/11
- **Python Version:** 3.13.5
- **Scikit-Learn Version:** 1.7.0
- **NumPy Version:** 2.2.3
- **Pandas Version:** 2.2.3
- **Pytest Version:** 9.1.1
- **Execution Script:** [`scripts/verify_phase2_reproducibility.py`](file:///d:/Financial%20Distress/scripts/verify_phase2_reproducibility.py)

---

## 4. Workstream D: Probability Calibration Hardening

### 4.1 Calibration against Training Prevalence Reference Baseline
On imbalanced datasets, a low Brier score can be misleading because a trivial constant forecast $\hat{p} = \bar{y}$ achieves a low Brier score. To prevent false confidence, calibration was evaluated using the **Brier Skill Score (BSS)** relative to historical training prevalence ($\bar{y}_{\text{train}} = 0.0384$):
$$BSS = 1 - \frac{BS}{BS_{\text{ref}}}, \quad \text{where } BS_{\text{ref}} = \frac{1}{N}\sum (y_i - \bar{y}_{\text{train}})^2$$

| Split Partition | Probability Evaluated | Brier Score | Ref Prevalence Brier | Brier Skill Score ($BSS$) | Log Loss | CITL | Calibration Slope |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation ($T \in [11, 12]$)** | Raw Random Forest | 0.0512 | 0.0265 | **-0.9335** | 0.1810 | +0.1044 | 1.0292 |
| **Validation ($T \in [11, 12]$)** | **Calibrated (Platt)** | **0.0241** | 0.0265 | **+0.0881** | **0.0979** | **+0.0002** | **1.0716** |
| **Holdout Test ($T=13$)** | Raw Random Forest | 0.0626 | 0.0402 | **-0.5550** | 0.2193 | +0.0960 | 0.7149 |
| **Holdout Test ($T=13$)** | **Calibrated (Platt)** | **0.0387** | 0.0402 | **+0.0370** | **0.1652** | **-0.0169** | **0.7830** |

### 4.2 Key Calibration Findings
1. **Raw Probability Failure:** Uncalibrated tree probabilities exhibited **negative Brier Skill Scores** on both validation (-0.93) and test (-0.55), meaning raw probabilities were statistically inferior to a constant naive prevalence forecast.
2. **Platt Scaling Efficacy:** Calibration flipped the Brier Skill Score into **positive territory** (+0.0881 on val, +0.0370 on test) and cut log loss by **45.9% on validation** and **24.7% on test**.
3. **Calibration-in-the-Large (CITL):** Calibrated probabilities reduced overall bias from +0.1044 down to +0.0002 on validation, and from +0.0960 down to -0.0169 on the test set.

---

## 5. Workstream F: Stability & Uncertainty Analysis

### 5.1 Rolling-Origin Temporal Stability
To verify that ranking power is not an artifact of a single test period, models were evaluated across three chronological expanding cutoffs:

| Cutoff Split | Train Period | Test Period | Train $N$ ($N_{pos}$) | Test $N$ ($N_{pos}$) | Test Prevalence | Test ROC-AUC | Test PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cutoff 1** | $1 \le T \le 8$ | $T = 9$ | 1,996 (85) | 232 (4) | 1.72% | **0.9123** | 0.2440 |
| **Cutoff 2** | $1 \le T \le 9$ | $T = 10$ | 2,228 (89) | 243 (6) | 2.47% | **0.8418** | 0.1042 |
| **Cutoff 3** | $1 \le T \le 10$| $T \in [11, 12]$ | 2,471 (95) | 517 (14) | 2.71% | **0.8828** | 0.2001 |

The model demonstrates sustained high ranking power across historical forward horizons (ROC-AUC 0.84 to 0.91).

### 5.2 Bootstrap Uncertainty on Test Set (1,000 Resamples)
Given the small positive test sample ($N_{pos} = 11$), bootstrap percentile confidence intervals were calculated:
- **Test ROC-AUC:** Point = 0.7783, **95% Bootstrap CI = [0.6491, 0.8848]**
- **Test PR-AUC:** Point = 0.1946, **95% Bootstrap CI = [0.0596, 0.4747]**
- **Test Recall @ $\tau^* = 0.08$:** Point = 27.27%, **Wilson 95% CI = [9.75%, 56.57%]**

---

## 6. Reproducibility Audit Conclusion

Phase 2 model artifacts, preprocessing pipelines, probability calibration, and operational threshold derivations are **completely reproducible**, mathematically consistent, and empirically grounded. Zero discrepancies were detected.
