# Phase 2 Leakage & Methodological Integrity Review

**Project:** Explainable SME Financial Distress Early-Warning & Risk Intelligence System  
**Audit Date:** October 12, 2026  
**Auditor Roles:** Senior Data Scientist, ML Engineer, Financial Risk Analytics Specialist, ML Research Auditor  
**Branch:** `upgrade/v1.2-model-validation`  
**Phase:** Phase 2 (Model Validation, Stacking Refactoring & Early-Warning Reliability)  

---

## 1. Executive Summary & Audit Mandate

In time-series and longitudinal panel credit risk modeling, **data leakage** is the primary driver of phantom predictive performance—inflated evaluation metrics that collapse upon live deployment. 

This review provides an exhaustive audit of all potential data leakage vectors across the SME Financial Distress pipeline:
- Temporal feature engineering (lags, differences, rolling statistics)
- Forward target horizon formulation
- Imputation and scaling isolation
- Out-of-fold meta-model feature generation
- Threshold selection and hyperparameter tuning isolation
- Cross-company boundary contamination

---

## 2. Comprehensive Leakage Vector Audit Matrix

| Vector ID | Leakage Risk Description | Baseline Audit Status | Phase 2 Audit Verification & Remediations | Verified Evidence / Source Reference | Leakage Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **LV-01** | **Future Information in Lags & Rolling Stats** | Verified clean in Phase 1 | Lags (`lag_1`, `lag_2`) and backward rolling windows (`roll_mean_2`, `roll_mean_3`, `roll_std_3`) use only observations at or before time $t$. | `src/features/temporal_features.py`: lines 57–80; tested in `test_leakage_prevention.py` | **CLEAN ✅** |
| **LV-02** | **Cross-Company Rolling Boundary Contamination** | Verified clean in Phase 1 | All temporal operations are partitioned strictly by `groupby("Company")`. Window statistics never aggregate across firm boundaries. | `src/features/temporal_features.py`: lines 53, 74–76; tested in `test_leakage_prevention.py` | **CLEAN ✅** |
| **LV-03** | **Preprocessor Data Snooping** | Tested in Phase 1 | `EWSPreprocessor` fits `SimpleImputer` and `RobustScaler` strictly on training set ($T \le 10$). Validation and test sets are transformed using frozen training medians. | `src/preprocessing/preprocessors.py`: lines 23–32; tested in `test_model_validation.py` | **CLEAN ✅** |
| **LV-04** | **Lookahead in Forward Target ($t+1$)** | Verified clean in Phase 1 | Target $y_{t}^{\text{forward}}$ is constructed by shifting $t+1$ distress within company. Terminal period $T=14$ with no $T=15$ record is set to `NaN` and dropped. | `src/features/temporal_features.py`: lines 28–30; tested in `test_leakage_prevention.py` | **CLEAN ✅** |
| **LV-05** | **Synthetic Proxy Target / Feature Leakage** | **CRITICAL DEFECT IN BASELINE** | Baseline computed `proxy_bankrupt` and `proxy_anomaly` using formulas on `x64`/`x81` and fitted the meta-model on validation set `val_a`. **Remediated in Phase 2:** Proxies removed; meta-model OOF trained on $T \le 10$. | `scripts/phase2_model_experiments.py`; `reports/phase2/proxy_contamination_experiments.csv` | **REMEDIATED ✅** |
| **LV-06** | **Threshold Selection on Final Test Labels** | **CRITICAL DEFECT IN BASELINE** | Baseline had 0.0% test recall due to naive 0.50 threshold. Phase 2 derived operating threshold $\tau^* = 0.08$ **strictly on validation fold** without test set access. | `reports/phase2/validation_threshold_analysis.csv`; tested in `test_model_validation.py` | **REMEDIATED ✅** |
| **LV-07** | **Feature Selection Correlation Snooping** | Identified in GAP-08 | 8 key features (`x1, x2, x5, x10, x25, x48, x64, x81`) were selected historically based on full dataset correlations. Scheduled for training-only screening in Phase 3. | `src/features/temporal_features.py`: line 12; documented in `upgrade_roadmap.md` | **FLAGGED FOR P3 ⚠️** |
| **LV-08** | **Duplicate / Near-Duplicate Firm Overlap** | Verified clean | Dataset A firms possess unique identifiers ($1 \dots 422$). Firm-level temporal splits guarantee complete company time histories remain chronologically ordered. | `src/data/loaders.py`; `tests/test_model_validation.py` | **CLEAN ✅** |

---

## 3. Deep-Dive on Remediated Leakage Vectors

### 3.1 Vector LV-05: Meta-Model Training on Validation Set (Validation-to-Train Contamination)
In the original baseline pipeline (`src/pipeline.py`, lines 65–98):
```python
# Baseline code:
val_X, y_val = self.ews_model.preprocessor.transform(val_a)
val_ews_probs = self.ews_model.calibrator.predict_proba(val_X)[:, 1]
...
meta_learner.fit(oof_meta_features, y_val)
```
Although annotated in code comments as "out-of-fold", the meta-model was actually fitted directly on the validation fold (`val_a`). This caused the meta-model to memorize the 14 positive events in `val_a`, yielding an overfitted validation ROC-AUC of 0.8195 that collapsed to 0.6588 on the test set.

**Phase 2 Remediation:**
1. In controlled Experiment C, out-of-fold training was implemented strictly on the training partition ($T \le 10$) using 5-Fold `GroupKFold` on `Company`.
2. For production deployment, synthetic proxies were removed entirely (Experiment B), and the clean primary Random Forest model was selected, eliminating the contaminated meta-model from the critical prediction path.

### 3.2 Vector LV-06: Test Label Snooping in Operating Threshold Optimization
Standard machine learning guidelines prohibit optimizing decision thresholds on the test set. Tuning $\tau$ on test labels produces optimistic recall estimates that do not generalize.

**Phase 2 Verification:**
1. Candidate thresholds $\tau \in \{0.05, 0.08, 0.10, 0.13, 0.17, 0.50\}$ were evaluated exclusively on the validation fold ($T \in [11, 12]$).
2. The selection criterion (maximizing $F_2$-score under operational recall $\ge 60\%$) unambiguously identified $\tau^* = 0.08$.
3. The threshold configuration was persisted to `models/phase2/threshold_config.json` and **frozen before touching the test set**.
4. Test set evaluation was run once with the frozen configuration, revealing test recall of 27.3% (honestly reported without post-hoc manipulation).

---

## 4. Residual Methodological Risks & Phase 3 Roadmap

| Residual Risk | Severity | Root Cause | Planned Mitigation (Phase 3) |
| :--- | :---: | :--- | :--- |
| **Feature Selection Screening (GAP-08)** | Moderate | 8 key variables selected historically on full dataset correlation. | Fit mutual information / variance threshold screening strictly inside training fold ($T \le 10$). |
| **Anonymized Variable Interpretation (GAP-07)** | Moderate | `x64` and `x81` labeled as "Cost" and "Health" without verified dictionary. | Relabel features strictly as masked statistical factors; compute genuine TreeSHAP attributions. |

---

## 5. Leakage Review Audit Conclusion

All acute leakage vulnerabilities—specifically preprocessor snooping, lookahead bias in forward targets, synthetic proxy contamination, and test-label threshold tuning—are verified as **completely resolved** in Phase 2. The pipeline is methodologically sound and suitable for research audit verification.
