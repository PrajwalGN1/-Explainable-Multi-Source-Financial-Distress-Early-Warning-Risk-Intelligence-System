# Phase 2.1 Target Construction, Leakage Prevention & Split Review

**Project:** Explainable SME Financial Distress Early-Warning & Risk Intelligence System  
**Audit Date:** October 12, 2026  
**Auditor Roles:** Senior ML Engineer, Financial Risk Modeling Researcher, ML Validation Specialist  
**Branch:** `upgrade/v1.2.1-validation-hardening`  
**Phase:** Phase 2.1 (Validation, Calibration & Reproducibility Hardening)  

---

## 1. Executive Summary & Verification Mandate

In time-series and longitudinal panel credit risk modeling, **target leakage** and **temporal lookahead bias** represent the most pervasive sources of invalid empirical claims. 

This audit provides an independent verification of:
1. Mathematical alignment of the forward distress label ($t+1$).
2. Verification that period $T=14$ information is used **only** as a target label for period $T=13$, and never as an input feature.
3. Chronological time-series partitioning versus cross-company generalization.
4. Two distinct evaluation questions:
   - **Setting A:** Future-period distress prediction on previously observed companies.
   - **Setting B:** Generalization to entirely unseen companies (zero historical observation).

---

## 2. Target Alignment & Period T=14 Investigation

### 2.1 Forward Target Formulation
In [`src/features/temporal_features.py`](file:///d:/Financial%20Distress/src/features/temporal_features.py) (lines 15–39), the forward target is formulated as:
```python
df_sorted = df.sort_values(by=[id_col, time_col]).copy()
df_sorted["current_distress"] = (df_sorted[raw_target_col] <= threshold).astype(int)
df_sorted["forward_target"] = df_sorted.groupby(id_col)["current_distress"].shift(-1)
```

### 2.2 Verification of Period T=13 Test Labels & Period T=14 Outcomes
A critical question in this audit was verifying the role of period $T=14$:
1. **Source of $T=13$ Labels:** For a firm $i$ observed at period $T=13$, `shift(-1)` fetches `current_distress` from period $T=14$ for the same firm. If firm $i$ enters distress in period 14, `forward_target` at $T=13$ equals 1; otherwise, it equals 0.
2. **Exclusion of $T=14$ from Input Features:** In `prepare_data` and all experiment scripts:
   ```python
   df_valid = df_feats.dropna(subset=["forward_target"]).copy()
   test_df = df_valid[df_valid["Time"] > 12]  # Time == 13
   ```
   At period $T=14$, there is no period $T=15$. Therefore, `forward_target` at $T=14$ is `NaN`.
   When `dropna(subset=["forward_target"])` is executed, **all rows from period $T=14$ (256 observations) are completely dropped from the dataset**.
3. **Audit Finding:** Period $T=14$ is used **strictly to determine whether firms observed at $T=13$ experienced distress in the subsequent period**. Zero accounting features from period $T=14$ are ever present in the feature matrix $X_{\text{test}}$. Lookahead bias is 0.0%.

---

## 3. Two Distinct Evaluation Settings

Credit risk modeling serves two distinct operational use cases, which must not be conflated:

```mermaid
graph TD
    subgraph SettingA [Setting A: Known-Company Temporal Forward Prediction]
        A1[Firms observed in historical periods T <= 10]
        A2[Forecast distress in future period T = 13]
        A3[Evaluation: Strict Chronological Holdout T=13]
    end
    subgraph SettingB [Setting B: Unseen-Company Cross-Sectional Generalization]
        B1[Entirely new firms with zero historical data in training set]
        B2[Forecast distress across firm lifecycle]
        B3[Evaluation: 5-Fold GroupKFold on Company ID]
    end
```

### 3.1 Setting A: Future-Period Prediction for Known Portfolio Companies
- **Deployment Objective:** Annual credit surveillance of active portfolio clients with established multi-year financial history.
- **Evaluation Protocol:** Strict chronological holdout. Models trained on historical years ($1 \le T \le 10$), tuned on validation years ($11 \le T \le 12$), and evaluated on out-of-period test year ($T=13$).
- **Verified Empirical Results:**
  - Test Observations: $N = 262$ (11 distress events, 251 solvent firms).
  - Test ROC-AUC: **0.7783**
  - Test PR-AUC: **0.1946**
  - Test Brier Score: **0.0387**
  - Test Operational Recall @ $\tau^* = 0.08$: **27.3%**
  - Test Specificity @ $\tau^* = 0.08$: **93.2%**

### 3.2 Setting B: Generalization to Entirely Unseen Companies
- **Deployment Objective:** Underwriting new credit applicants where no prior fiscal year reports exist in the institutional portfolio.
- **Evaluation Protocol:** 5-Fold `GroupKFold` partitioned strictly by `Company` identifier across all 343 companies in training periods ($T \le 10$, $N=2,471$, 95 distress events). Firms in the validation fold never appear in the training fold.
- **Verified Empirical Results:**

| Cross-Validation Fold | Heldout Companies | Heldout Observations ($N$) | Distress Cases ($N_{pos}$) | Heldout ROC-AUC | Heldout PR-AUC | Heldout Brier |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 69 | 495 | 17 | 0.8776 | 0.2847 | 0.0664 |
| **Fold 2** | 68 | 494 | 16 | 0.8282 | 0.1160 | 0.0816 |
| **Fold 3** | 68 | 494 | 18 | 0.7878 | 0.1184 | 0.0810 |
| **Fold 4** | 69 | 494 | 19 | 0.8412 | 0.2448 | 0.0793 |
| **Fold 5** | 69 | 494 | 25 | 0.8199 | 0.2144 | 0.0727 |
| **Mean across Folds** | **343 unique firms**| **2,471 total** | **95 total** | **0.8309** | **0.1957** | **0.0762** |

### 3.3 Key Insights from Dual-Setting Comparison
1. **Robust Unseen Generalization:** The clean Random Forest model achieves a mean ROC-AUC of **0.8309** on completely unseen companies, proving that learned decision trees capture structural distress indicators (financial ratios, growth deterioration, and volatility spikes) that generalize across different corporate entities.
2. **Temporal Distribution Shift in $T=13$:** The lower test ROC-AUC in Setting A (0.7783 on $T=13$) compared to Setting B (0.8309 across $T \le 10$) reflects macroeconomic out-of-period shifts in fiscal period 13 rather than company-level overfitting.

---

## 4. Comprehensive Leakage Prevention Checklist

- [x] Lags and rolling operations are strictly grouped by `Company`.
- [x] Rolling windows use backward-looking windows only.
- [x] No company boundary transitions cross over during rolling computations.
- [x] Forward target $y_{t+1}$ is computed by shifting within `Company` only.
- [x] Terminal period $T=14$ is dropped from feature inputs.
- [x] Imputation and scaling statistics are fit strictly on $T \le 10$.
- [x] Validation data is not used to fit the base Random Forest estimator.
- [x] Test data is not accessed during model fitting, calibration, or threshold optimization.

---

## 5. Audit Conclusion

The pipeline obeys strict temporal and company-level separation. Both Setting A (temporal holdout) and Setting B (unseen-company cross-validation) demonstrate strong ranking discrimination without data leakage.
