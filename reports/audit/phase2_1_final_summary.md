# Phase 2.1 Model Validation, Calibration & Hardening Final Summary Report

**Project:** Explainable SME Financial Distress Early-Warning & Risk Intelligence System  
**Audit Date:** October 12, 2026  
**Auditor Roles:** Senior ML Engineer, Financial Risk Modeling Researcher, ML Validation Specialist, QA Lead  
**Active Git Branch:** `upgrade/v1.2.1-validation-hardening`  
**Phase:** Phase 2.1 (Model Validation, Calibration & Reproducibility Hardening)  

---

## 1. Executive Summary & Verification Outcome

Phase 2.1 was executed to conduct rigorous statistical validation, independent metric recomputation, probability calibration hardening, threshold governance verification, and dual-setting generalization analysis for the early-warning model pipeline.

### Core Audit Outcomes:
1. **100% Metric Reproducibility:** Verified independently via [`scripts/verify_phase2_reproducibility.py`](file:///d:/Financial%20Distress/scripts/verify_phase2_reproducibility.py) from saved artifacts without retraining. Recomputed test ROC-AUC (0.7783), PR-AUC (0.1946), Brier score (0.0387), confusion matrix counts (TN=234, FP=17, FN=8, TP=3), recall (27.27%), precision (15.00%), and specificity (93.23%) match reported claims with **zero discrepancy**.
2. **Calibration Hardened Against Prevalence Baseline:** Brier Skill Score ($BSS$) relative to the historical training prevalence baseline ($\bar{y}_{\text{train}} = 0.0384$) was negative for raw tree probabilities (-0.55 on test), but flipped to **positive skill (+0.0370 on test, +0.0881 on val)** after Platt scaling, reducing log loss by +24.7% and bringing calibration-in-the-large down to -0.0169.
3. **Dual Generalization Settings Evaluated:**
   - *Setting A (Known-Company Temporal Prediction):* Test ROC-AUC = **0.7783**, PR-AUC = **0.1946** on out-of-period test year $T=13$.
   - *Setting B (Unseen-Company Generalization):* 5-Fold `GroupKFold` across 343 firms in $T \le 10$ yielded Mean ROC-AUC = **0.8309** and PR-AUC = **0.1957**, proving strong generalization across independent corporate entities.
4. **Target Alignment Verified:** Period $T=14$ was confirmed to serve strictly as the next-period ground truth target for period $T=13$ observations. When $T=14$ forward targets are computed, $T=14$ rows have undefined next-period targets (`NaN`) and are completely dropped from the feature space, eliminating lookahead bias.
5. **Regression Test Suite:** Expanded to **53 passed tests** (12 new Phase 2.1 validation hardening tests + 10 Phase 2 tests + 31 Phase 1 regression tests).

---

## 2. Changes Implemented in Phase 2.1

| Component / File | Nature of Change | Purpose & Verification |
| :--- | :--- | :--- |
| [`scripts/verify_phase2_reproducibility.py`](file:///d:/Financial%20Distress/scripts/verify_phase2_reproducibility.py) | New Script | Independently recomputes test metrics from saved artifacts without retraining; evaluates Brier Skill Score, CITL, calibration slope, rolling cutoffs, and bootstrap CIs. |
| [`models/phase2/model_manifest.json`](file:///d:/Financial%20Distress/models/phase2/model_manifest.json) | Strengthened Artifact | Added SHA-256 dataset fingerprint, artifact hashes, full 163 feature names, split boundaries, verified metrics with bootstrap CIs, and environment versions. |
| [`api.py`](file:///d:/Financial%20Distress/api.py) | API Enhancement | `/model-info` exposes model manifest metadata (`model_version: "1.2.1-hardened"`, `operating_threshold: 0.08`, `dataset_sha256`) while preserving Phase 1 security and endpoints. |
| [`tests/test_validation_hardening.py`](file:///d:/Financial%20Distress/tests/test_validation_hardening.py) | New Automated Tests | 12 deterministic unit tests covering target alignment, preprocessor isolation, schema consistency, confusion matrix arithmetic, Brier skill score, and GroupKFold isolation. |
| `reports/phase2_1/calibration/` | New Audit Artifacts | Contains `calibration_metrics_table.csv` and `reliability_curves.json`. |
| `reports/phase2_1/thresholds/` | New Audit Artifacts | Contains `validation_governance_table.csv` and `test_retrospective_table.csv`. |
| `reports/phase2_1/stability/` | New Audit Artifacts | Contains `rolling_origin_stability.csv`, `unseen_company_group_cv.csv`, and `uncertainty_summary.json`. |
| [`reports/audit/phase2_1_pipeline_integrity.md`](file:///d:/Financial%20Distress/reports/audit/phase2_1_pipeline_integrity.md) | Deliverable Report | Ten-stage pipeline traceability, schema invariance, and artifact consistency audit. |
| [`reports/audit/phase2_1_leakage_and_split_review.md`](file:///d:/Financial%20Distress/reports/audit/phase2_1_leakage_and_split_review.md) | Deliverable Report | Forward target alignment, period $T=14$ investigation, and dual evaluation settings analysis. |
| [`reports/audit/phase2_1_reproducibility_report.md`](file:///d:/Financial%20Distress/reports/audit/phase2_1_reproducibility_report.md) | Deliverable Report | Full verification results, zero discrepancy confirmation, and cryptographic provenance. |

---

## 3. Before-versus-After Metrics Comparison

| Evaluation Metric | Baseline Initial State | Phase 2 Baseline State | Phase 2.1 Hardened State | Audit Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Pipeline Architecture** | Contaminated Stacking (`x64`/`x81`) | Clean Random Forest | **Clean Random Forest** | Zero synthetic proxy contamination |
| **Test ROC-AUC** | 0.6588 (contaminated) | 0.7783 (reported) | **0.7783 (reproduced)** | 95% Bootstrap CI: [0.6491, 0.8848] |
| **Test PR-AUC** | 0.1294 (contaminated) | 0.1946 (reported) | **0.1946 (reproduced)** | 95% Bootstrap CI: [0.0596, 0.4747] |
| **Operating Threshold** | 0.50 (naive) | 0.080 (val optimized) | **0.080 (governed)** | Grounded in validation $F_2$-score |
| **Test Recall** | 0.0% (11 missed) | 27.3% (3 caught) | **27.27% (3 caught)** | Wilson 95% CI: [9.75%, 56.57%] |
| **Test Precision** | 0.0% | 15.0% | **15.00% (3 / 20)** | 3.6x enrichment over 4.2% base rate |
| **Test Specificity** | 100.0% | 93.2% | **93.23% (234 / 251)** | High specificity maintained |
| **Test Brier Score** | 0.0626 (raw) | 0.0387 (reported) | **0.0387 (reproduced)** | +38.1% improvement over raw score |
| **Brier Skill Score ($BSS$)**| -0.5550 (uncalibrated) | Not Reported | **+0.0370 (positive skill)** | Outperforms constant prevalence forecast |
| **Test Log Loss** | 0.2193 (raw) | Not Reported | **0.1652 (calibrated)** | +24.7% entropy reduction |
| **Unseen Company ROC-AUC** | Not Evaluated | Not Evaluated | **0.8309 (5-fold mean)** | Cross-sectional firm generalization |
| **Automated Tests Passing**| 31 passing | 41 passing | **53 passing** | Zero failures across full suite |

---

## 4. Test Execution Evidence

Full test suite execution executed via PowerShell on branch `upgrade/v1.2.1-validation-hardening`:
```bash
python -m pytest tests/ -v
```

**Output Summary:**
```text
tests/test_api.py (15 tests) ........................... PASSED
tests/test_features_and_health.py (4 tests) ............ PASSED
tests/test_fusion_and_scoring.py (8 tests) ............. PASSED
tests/test_leakage_prevention.py (4 tests) ............. PASSED
tests/test_model_validation.py (10 tests) .............. PASSED
tests/test_validation_hardening.py (12 tests) .......... PASSED

================== 53 passed, 3 warnings in 72.84s (0:01:12) ==================
```

---

## 5. Remaining Limitations & Statistical Disclosures

1. **Small Positive Sample Size ($N_{pos} = 11$ in Holdout $T=13$):**
   - Point estimates carry wide confidence intervals (Recall 95% CI [9.8%, 56.6%], PR-AUC 95% CI [0.060, 0.475]).
   - Achieving $\ge 60\%$ recall on validation fold ($64.3\%$) does not guarantee $\ge 60\%$ recall out-of-period without lowering thresholds and incurring more false alarms.
2. **Feature Anonymization in Dataset A:**
   - Features `x1` through `x83` remain masked. No fabricated accounting balance sheet narratives or radar charts should be treated as verified ground truth until feature decoding is established.
3. **Absence of Survival Duration Modeling:**
   - The current pipeline predicts discrete single-period forward distress ($t \to t+1$). Multi-quarter continuous hazard curves represent a Phase 4 research opportunity.

---

## 6. Phase 2.1 Acceptance Decision

### Acceptance Verification:
- [x] Existing Phase 1 and Phase 2 functionality fully preserved.
- [x] Model checkpoints and artifacts verified without retraining.
- [x] Target alignment and period $T=14$ lookahead absence verified.
- [x] 100% metric reproduction achieved with zero discrepancies.
- [x] Probability calibration proven against historical prevalence baseline.
- [x] Threshold selection governed strictly by validation data.
- [x] Uncertainty and small-sample intervals honestly disclosed.
- [x] Unseen-company generalization verified via 5-Fold GroupKFold (ROC-AUC 0.8309).
- [x] Model manifest updated with cryptographic hashes and full provenance.
- [x] Full automated test suite passes with 53 tests and zero regressions.
- [x] Zero invented data, zero synthetic test inflation, zero unauthorized pushes.

**STATUS: PHASE 2.1 ACCEPTED.**

---

## 7. Recommended Next Phase & Required Prerequisites

### Recommended Next Step: **PHASE 3 (Explainability & Native C++ TreeSHAP Integration)**
Before initiating Phase 3, the following prerequisites are established:
1. Baseline branch `upgrade/v1.2.1-validation-hardening` is clean, reproducible, and tested with 53 passing tests.
2. User authorization is required before branching to `upgrade/v1.3-explainability` or installing native dependencies.
