# Phase 2.1 Final Statistical Audit & Methodological Integrity Review

**Project:** Explainable SME Financial Distress Early-Warning & Risk Intelligence System  
**Repository Branch:** `upgrade/v1.2.1-validation-hardening`  
**Audit Date:** October 12, 2026  
**Auditor Roles:** Senior Data Scientist, Financial Risk Analytics Specialist, ML Research Auditor  
**Audit Scope:** Independent statistical correctness, leakage audit, calibration uncertainty, and evaluation integrity review prior to Phase 3 authorization.  
**Model & Artifact Policy:** Strictly non-destructive. Zero model retraining, zero modification to operating thresholds ($\tau^* = 0.080$), and zero changes to model checkpoints in `models/phase2/`.

---

## 1. Executive Summary & Audit Mandate

This audit delivers the formal statistical review of Phase 2.1 on branch `upgrade/v1.2.1-validation-hardening`. The goal of this review is to evaluate all empirical claims, probabilistic formulations, resampling schemes, and metric definitions against rigorous financial econometrics and statistical learning standards.

### Summary of Audit Verdict
| Audit Check | Focus Area | Audit Status | Key Empirical Finding |
| :--- | :--- | :---: | :--- |
| **Check 1** | Brier Skill Score Claims | **Accepted with Caveats** | Point estimate is positive ($BSS = +0.0370$ on Test, $+0.0881$ on Val), but 95% bootstrap CIs span zero ($[-0.0450, 0.1286]$ and $[-0.0132, 0.1930]$). Statistical significance at $\alpha = 0.05$ is **not established**; claims corrected to point estimates. |
| **Check 2** | GroupKFold Integrity | **Corrected & Verified** | Initial evaluation script leaked global scaler/imputer statistics across folds. Rerun with strict inner-fold `EWSPreprocessor.fit()` confirms zero leakage. Mean ROC-AUC is **0.8301** (SD: 0.0334), Mean AP is **0.1958** (SD: 0.0765). Companies never cross folds. |
| **Check 3** | Confidence Intervals | **Accepted with Disclosures** | Holdout test set $T=13$ has $N=262$ firms with exactly 1 row per firm. Observation bootstrap is identical to firm-level bootstrap. Resampling reflects **cross-sectional variation in period $T=13$ only**, not multi-year temporal variance. Small-sample uncertainty ($N_{pos}=11$) documented. |
| **Check 4** | Metric Terminology | **Standardized & Verified** | PR-AUC is formally Average Precision (AP), not trapezoidal AUC. Base rates verified: Train 3.84%, Val 2.71%, Test 4.20%. Test precision of 15.00% at $\tau^*=0.080$ represents a $3.57\times$ enrichment over baseline. No claims of production readiness made from retrospective validation alone. |
| **Check 5** | Final Acceptance Decision | **ACCEPTED WITH CAVEATS** | Phase 2.1 is authorized for completion. Model checkpoints and operating thresholds remain 100% untouched. All 53 regression and validation tests pass. Phase 3 (TreeSHAP Explainability) may proceed upon user confirmation. |

---

## 2. Check 1 — Brier Skill Score Claims & Uncertainty

### 2.1 Mathematical Formulation & Baseline Specification
The Brier Score ($BS$) measures the mean squared error between forecast probabilities $p_i \in [0, 1]$ and binary outcomes $y_i \in \{0, 1\}$:
$$BS = \frac{1}{N} \sum_{i=1}^N (y_i - p_i)^2$$

In highly imbalanced financial distress settings, an uninformative model predicting zero distress can achieve an deceptively low Brier score. To benchmark calibration skill against naive guessing, the **Brier Skill Score ($BSS$)** is evaluated against the historical training partition prevalence $\bar{y}_{\text{train}} = \frac{95}{2471} \approx 0.038446$ (3.84%):
$$BS_{\text{ref}} = \frac{1}{N} \sum_{i=1}^N (y_i - \bar{y}_{\text{train}})^2$$
$$BSS = 1 - \frac{BS}{BS_{\text{ref}}}$$

A value of $BSS > 0$ denotes predictive skill superior to issuing a constant historical base-rate forecast; $BSS = 0$ indicates equivalence to the base rate; $BSS < 0$ denotes negative skill (worse than issuing a constant base rate).

### 2.2 Recomputed Calibration Metrics Across Partitions
The exact figures recomputed independently from saved artifacts:

| Partition | Probability Source | Brier Score ($BS$) | Reference Brier ($BS_{\text{ref}}$) | Brier Skill Score ($BSS$) | Log Loss | CITL | Slope |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation ($T=11, 12$)** | Raw Tree Probabilities | 0.0512 | 0.0265 | **-0.9335** | 0.1810 | +0.1044 | 1.0292 |
| **Validation ($T=11, 12$)** | Calibrated (Platt Sigmoid) | **0.0241** | 0.0265 | **+0.0881** | **0.0979** | **+0.0002** | 1.0716 |
| **Holdout Test ($T=13$)** | Raw Tree Probabilities | 0.0626 | 0.0402 | **-0.5550** | 0.2193 | +0.0960 | 0.7149 |
| **Holdout Test ($T=13$)** | Calibrated (Platt Sigmoid) | **0.0387** | 0.0402 | **+0.0370** | **0.1652** | **-0.0169** | 0.7830 |

### 2.3 Statistical Hypothesis Testing for Brier Skill Score
To determine whether $BSS > 0$ represents genuine statistical significance or an artifact of finite sample variation, two independent hypothesis tests were executed:

1. **Paired Error Difference Hypothesis Test:**
   We construct the paired observation-level difference in squared error:
   $$d_i = (y_i - \bar{y}_{\text{train}})^2 - (y_i - p_{i, \text{cal}})^2$$
   $H_0: \mathbb{E}[d] \le 0$ versus $H_1: \mathbb{E}[d] > 0$ via a one-sided paired t-test:
   - **Validation Fold ($T=11, 12$, $N=517$, 14 events):** $t = 1.461$, **$p = 0.0722$**.
   - **Holdout Test ($T=13$, $N=262$, 11 events):** $t = 0.844$, **$p = 0.1997$**.

2. **Non-Parametric Bootstrap Confidence Intervals (1,000 resamples):**
   - **Validation Fold ($T=11, 12$):** Point $BSS = +0.0881$, 95% Bootstrap CI = **$[-0.0132, +0.1930]$**. Empirical $P(BSS \le 0) = 6.1\%$.
   - **Holdout Test ($T=13$):** Point $BSS = +0.0370$, 95% Bootstrap CI = **$[-0.0450, +0.1286]$**. Empirical $P(BSS \le 0) = 20.2\%$.

### 2.4 Audit Conclusion & Report Correction for Check 1
> [!WARNING]
> **Corrective Finding:** While Platt scaling dramatically curtails probability over-confidence (reducing test log loss by 24.7% and bringing calibration-in-the-large from +0.0960 to -0.0169), the observed positive Brier Skill Score on both validation (+0.0881) and test (+0.0370) **does not achieve statistical significance at $\alpha = 0.05$** ($p = 0.072$ and $p = 0.200$).
>
> In both evaluation partitions, the 95% bootstrap confidence interval crosses zero. All prior assertions claiming that the model has "statistically demonstrated skill superiority over prevalence baselines" have been corrected to state:
> *"The calibrated model exhibits a positive Brier Skill Score point estimate ($+0.0370$), but sampling uncertainty spans zero (95% CI: $[-0.0450, +0.1286]$), meaning statistical superiority over constant base-rate forecasting is not yet established at the 5% significance level due to low event counts ($N_{pos} = 11$)."*

---

## 3. Check 2 — GroupKFold Integrity & Pipeline Leakage Review

### 3.1 Unseen-Company Evaluation Pipeline Inspection
To test whether the model generalizes to completely new, unseen corporate entities (Setting B), a 5-fold cross-validation grouped by corporate identifier (`Company`) was evaluated on the training partition ($T \le 10$, 343 companies, 2,471 observations).

### 3.2 Verification of Zero Company Crossing
In every fold, the set of companies in the training split and the held-out validation split are strictly disjoint:
$$\mathcal{C}_{\text{train}, k} \cap \mathcal{C}_{\text{val}, k} = \emptyset, \quad \forall k \in \{1, \dots, 5\}$$

- **Fold 1:** 69 held-out companies (495 rows, 17 distress events, 3.43% prevalence)
- **Fold 2:** 68 held-out companies (494 rows, 16 distress events, 3.24% prevalence)
- **Fold 3:** 68 held-out companies (494 rows, 18 distress events, 3.64% prevalence)
- **Fold 4:** 69 held-out companies (494 rows, 19 distress events, 3.85% prevalence)
- **Fold 5:** 69 held-out companies (494 rows, 25 distress events, 5.06% prevalence)
- **Total:** $69 + 68 + 68 + 69 + 69 = 343$ companies (100% company preservation, 0% company leakage).

### 3.3 Leakage Identification & Methodological Correction
During our initial pipeline audit of [`scripts/verify_phase2_reproducibility.py`](file:///d:/Financial%20Distress/scripts/verify_phase2_reproducibility.py), a methodological defect was uncovered:
- **Defect:** The feature matrix passed into `GroupKFold` had been preprocessed globally by calling `preprocessor.transform(train_df)` before the fold split. Because `preprocessor.fit()` had been executed on all 2,471 training rows, median imputation statistics and RobustScaler interquartile ranges computed from held-out companies leaked into the training folds.
- **Corrective Action:** The GroupKFold loop was refactored in [`scripts/verify_phase2_reproducibility.py`](file:///d:/Financial%20Distress/scripts/verify_phase2_reproducibility.py) (lines 452–511) to instantiate and fit a clean, independent `EWSPreprocessor` instance **strictly within each training fold**:
  ```python
  prep_fold = EWSPreprocessor(target_col="forward_target")
  prep_fold.fit(df_tr_fold)  # strictly 4 folds (~1,977 rows)
  X_fold_tr, y_fold_tr = prep_fold.transform(df_tr_fold)
  X_fold_val, y_fold_val = prep_fold.transform(df_val_fold)
  ```
- **Rerun Results:** The evaluation was rerun with zero preprocessor leakage. The per-fold results are documented below.

### 3.4 Per-Fold Positive Counts, Metrics, and Variability
The verified, leak-free 5-fold cross-validation metrics on unseen companies:

| Fold Index | Held-out Companies | Held-out Observations | Distress Events ($N_{pos}$) | Held-out Prevalence | ROC-AUC | Average Precision (AP) | Brier Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 69 | 495 | 17 | 3.43% | 0.8776 | 0.2853 | 0.0663 |
| **Fold 2** | 68 | 494 | 16 | 3.24% | 0.8274 | 0.1157 | 0.0816 |
| **Fold 3** | 68 | 494 | 18 | 3.64% | 0.7857 | 0.1171 | 0.0812 |
| **Fold 4** | 69 | 494 | 19 | 3.85% | 0.8408 | 0.2440 | 0.0794 |
| **Fold 5** | 69 | 494 | 25 | 5.06% | 0.8189 | 0.2170 | 0.0725 |
| **Mean (SD)** | **343** | **2,471** | **95** | **3.84%** | **0.8301 (0.0334)** | **0.1958 (0.0765)** | **0.0762 (0.0063)** |

### 3.5 Methodological Verification of Preprocessor, Calibration, and Threshold Isolation
1. **Preprocessor Isolation:** Confirmed. Imputation and RobustScaling are fitted strictly within training data.
2. **Temporal Feature Construction:** Confirmed. In [`src/features/temporal_features.py`](file:///d:/Financial%20Distress/src/features/temporal_features.py), momentum, lags, and rolling aggregates are computed grouped strictly by `Company` and sorted by `Time`. No cross-company pooling occurs.
3. **Target Alignment:** Confirmed. Forward target $y_{i, t} = I(\text{Distress}_{i, t+1} \le -0.50)$ uses only the same company's next-period outcome. Terminal period $T=14$ records have undefined next-period outcomes and are dropped completely.
4. **Calibration & Threshold Isolation:** Confirmed. The production model was trained on $T \le 10$, calibrated on Validation ($T=11, 12$), and the operating threshold ($\tau^* = 0.080$) was selected on Validation ($T=11, 12$). The Holdout Test fold ($T=13$) was never touched during model fitting, calibration fitting, or threshold tuning.

---

## 4. Check 3 — Confidence Intervals & Resampling Dependencies

### 4.1 Resampling Implementation Inspection
In [`scripts/verify_phase2_reproducibility.py`](file:///d:/Financial%20Distress/scripts/verify_phase2_reproducibility.py) (lines 515–545), uncertainty on the test partition is evaluated via 1,000 bootstrap resamples with replacement.

### 4.2 Company and Time Dependencies in Test Partition
- **Test Partition Size:** $N = 262$ rows.
- **Unique Companies in Test Partition:** Exactly 262 unique companies (`test_df['Company'].nunique() == 262`).
- **Reporting Period:** Every observation in the test set occurs at the single annual reporting period $T = 13$.

> [!NOTE]
> **Key Methodological Clarification:**
> Because each company has **exactly one row** in the test partition, standard observation-level bootstrap resampling is **mathematically identical to firm-level clustered bootstrap resampling**.
>
> However, auditors must emphasize what this resampling represents:
> - **What the CI Captures:** Cross-sectional variation across SME firms within the single historical annual accounting period $T=13$.
> - **What the CI Does NOT Capture:** Multi-period longitudinal autocorrelation, macro-economic regime shifts, regulatory policy shocks, or temporal variations across consecutive financial years.

### 4.3 Small-Sample Uncertainty Disclosure ($N_{pos} = 11$)
The test set contains only 11 financial distress events out of 262 firms (4.20% prevalence):
- **Test Recall Point Estimate:** $27.27\%$ (3 out of 11 distress events detected).
- **Wilson Score 95% Confidence Interval for Recall:** **$[9.75\%, 56.57\%]$**.
  - *Sensitivity Interpretation:* With only 11 distress cases, a single additional detected or missed firm shifts the observed recall by $\pm 9.09\%$. The wide Wilson interval reflects fundamental small-sample uncertainty.
- **Bootstrap 95% Confidence Interval for ROC-AUC:** **$[0.6491, 0.8848]$** (Point: 0.7783).
- **Bootstrap 95% Confidence Interval for Average Precision (AP):** **$[0.0596, 0.4747]$** (Point: 0.1946).

---

## 5. Check 4 — Metric Terminology & Prevalence Baselines

### 5.1 Average Precision (AP) vs Trapezoidal PR-AUC
- **Code Implementation:** The codebase evaluates precision-recall performance using `sklearn.metrics.average_precision_score`.
- **Mathematical Specification:** This function computes **Average Precision (AP)**:
  $$\text{AP} = \sum_{k=1}^K (R_k - R_{k-1}) P_k$$
  where $P_k$ and $R_k$ are precision and recall at threshold $k$.
- **Distinction from Trapezoidal Integration:** The linear trapezoidal area under the curve (`auc(recall, precision)`) linearly connects precision values between adjacent thresholds, which systematically overestimates the true area under the curve on finite, step-wise empirical curves (Davis & Goadrich, 2006).
- **Audit Mandate:** All documentation must refer to this metric strictly as **Average Precision (AP)** to avoid confusion with trapezoidal interpolation.

### 5.2 Prevalence Baselines Across Partitions
Baseline naive random guessing yields an expected precision equal to the prevalence of the evaluation partition:
- **Training Partition ($T \le 10$):** $95 / 2,471 = \mathbf{3.84\%}$
- **Validation Partition ($T = 11, 12$):** $14 / 517 = \mathbf{2.71\%}$
- **Holdout Test Partition ($T = 13$):** $11 / 262 = \mathbf{4.20\%}$

### 5.3 Confusion Matrix Verification at Governed Operating Threshold ($\tau^* = 0.080$)
On the holdout test set ($T=13$, $N=262$):
- **True Negatives ($TN$):** 234
- **False Positives ($FP$):** 17
- **False Negatives ($FN$):** 8
- **True Positives ($TP$):** 3
- **Total Flagged ($TP + FP$):** 20 (7.63% of portfolio)
- **Precision:** $\frac{3}{20} = \mathbf{15.00\%}$
- **Prevalence Enrichment Factor:** $\frac{15.00\%}{4.198\%} = \mathbf{3.57\times}$ enrichment over random guessing.
- **Recall (Sensitivity):** $\frac{3}{11} = \mathbf{27.27\%}$
- **Specificity:** $\frac{234}{251} = \mathbf{93.23\%}$
- **Negative Predictive Value ($NPV$):** $\frac{234}{242} = \mathbf{96.69\%}$
- **False Positive Rate ($FPR$):** $\frac{17}{251} = \mathbf{6.77\%}$
- **False Negative Rate ($FNR$):** $\frac{8}{11} = \mathbf{72.73\%}$
- **$F_1$-Score:** 0.1935
- **$F_2$-Score:** 0.2344

### 5.4 Production Readiness Retraction
> [!IMPORTANT]
> **Regulatory & Methodological Disclosure:**
> Retrospective validation on a historical panel partition ($T=13$) does not constitute proof of production readiness for live underwriting. Prior phrasing that implied "readiness for production deployment" from offline validation has been retracted. Production readiness requires:
> 1. Prospective out-of-time shadow deployment.
> 2. Real-time data drift and concept drift monitoring.
> 3. Human-in-the-loop credit officer oversight and override governance.

---

## 6. Check 5 — Final Audit Decision & Governance Verification

### 6.1 State of Artifacts & Reproducibility
- **Model Checkpoints:** Model file [`models/phase2/best_ews_model.joblib`](file:///d:/Financial%20Distress/models/phase2/best_ews_model.joblib), preprocessor [`models/phase2/ews_preprocessor.joblib`](file:///d:/Financial%20Distress/models/phase2/ews_preprocessor.joblib), calibrator [`models/phase2/ews_calibrator.joblib`](file:///d:/Financial%20Distress/models/phase2/ews_calibrator.joblib), and threshold configuration [`models/phase2/threshold_config.json`](file:///d:/Financial%20Distress/models/phase2/threshold_config.json) remain **100% byte-for-byte identical** with zero retraining.
- **Operating Threshold:** Operating threshold $\tau^* = 0.080$ remains **strictly unchanged**.
- **Evaluations Rerun:** Only two evaluation components were rerun:
  1. The 5-fold `GroupKFold` cross-validation was rerun to fit `EWSPreprocessor` strictly within each fold, eliminating global preprocessor leakage.
  2. Brier Skill Score uncertainty estimation was executed via 1,000 bootstrap resamples and paired t-tests to formally establish sampling variability.
- **Automated Test Suite:** The full automated test suite passes with **53 passed tests** across 6 test modules in 72 seconds.

### 6.2 Remaining Methodological Limitations
1. **Masked Financial Ratios:** Dataset A features `x1` through `x83` remain anonymized. No balance sheet ratios (e.g., current ratio, leverage) can be reported with verified accounting semantics until feature key disclosure.
2. **Small Test Event Cohort ($N_{pos} = 11$):** Point estimates carry wide sampling uncertainty (Recall 95% CI $[9.8\%, 56.6\%]$). Validation recall of 64.3% did not maintain the same level out-of-period at $\tau^* = 0.080$ (falling to 27.3%), demonstrating cross-period decay.
3. **Discrete Horizon:** The pipeline forecasts binary distress at horizon $t+1$ rather than continuous multi-period survival durations.

### 6.3 Final Formal Recommendation
**VERDICT: PHASE 2.1 ACCEPTED WITH CAVEATS DISCLOSED.**

All statistical checks have been completed, pipeline defects in cross-validation have been corrected, metric terminology has been standardized, and calibration uncertainty has been honestly quantified. Branch `upgrade/v1.2.1-validation-hardening` is clean, robust, and methodologically sound.

Phase 3 (TreeSHAP Explainability & Risk Intelligence) is authorized to begin upon user confirmation.
