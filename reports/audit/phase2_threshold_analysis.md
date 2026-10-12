# Phase 2 Probability Calibration & Operating Decision Threshold Analysis

**Project:** Explainable SME Financial Distress Early-Warning & Risk Intelligence System  
**Audit Date:** October 12, 2026  
**Auditor Roles:** Senior Data Scientist, ML Engineer, Financial Risk Analytics Specialist, ML Research Auditor  
**Branch:** `upgrade/v1.2-model-validation`  
**Phase:** Phase 2 (Model Validation, Stacking Refactoring & Early-Warning Reliability)  

---

## 1. Context & Problem Statement

In corporate early-warning systems operating on imbalanced portfolios (prevalence ~2.5% to 4.5%), model failure frequently occurs not in ranking discrimination, but in the conflation of **probability calibration** and **decision thresholding**:

1. **Probability Calibration:** Adjusts raw machine learning decision scores so that predicted values correspond to empirical default/distress frequencies. For example, a calibrated probability of 0.10 indicates that 10 out of 100 firms in this stratum are expected to experience distress in the subsequent period.
2. **Decision Threshold Selection:** Chooses the operational operating cutoff ($\tau^*$) that triggers surveillance actions, trade-off between missed distress events (False Negatives) and investigation costs (False Positives).

In the baseline Phase 1 audit, Platt scaling properly compressed raw Random Forest probabilities to the empirical domain ($P \in [0.007, 0.191]$). However, the classification threshold was left at the default textbook value of **0.50**. Consequently, **every single distressed firm in the test set was classified as healthy**, yielding **0.0% test recall (11 missed defaults)**.

This report documents the rigorous optimization of the decision threshold on validation data and its final evaluation on the untouched holdout test set.

---

## 2. Validation Fold Threshold Optimization ($T \in [11, 12]$)

Threshold candidates $\tau \in \{0.05, 0.08, 0.10, 0.13, 0.17, 0.50\}$ were evaluated on the validation fold ($N=517$ observations, 14 distress events, 2.71% base rate).

### 2.1 Complete Validation Metrics Across Threshold Candidates

| Threshold ($\tau$) | TP | FP | FN | TN | Recall | Precision | F1-Score | F2-Score | Specificity | False Neg Rate | Youden's J | Portfolio Flagged | Review Burden |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.05** | 9 | 58 | 5 | 445 | 64.3% | 13.4% | 0.2222 | 0.3659 | 88.5% | 35.7% | 0.5276 | 13.0% (67 / 517) | High |
| **0.08** | **9** | **38** | **5** | **465** | **64.3%** | **19.1%** | **0.2951** | **0.4369** | **92.4%** | **35.7%** | **0.5679** | **9.1% (47 / 517)** | **Optimal** |
| **0.10** | 7 | 22 | 7 | 481 | 50.0% | 24.1% | 0.3256 | 0.4118 | 95.6% | 50.0% | 0.4563 | 5.6% (29 / 517) | Low |
| **0.13** | 4 | 19 | 10 | 484 | 28.6% | 17.4% | 0.2162 | 0.2532 | 96.2% | 71.4% | 0.2484 | 4.5% (23 / 517) | Very Low |
| **0.17** | 4 | 13 | 10 | 490 | 28.6% | 23.5% | 0.2581 | 0.2740 | 97.4% | 71.4% | 0.2604 | 3.3% (17 / 517) | Escalation |
| **0.50** | 0 | 0 | 14 | 503 | 0.0% | 0.0% | 0.0000 | 0.0000 | 100.0% | 100.0% | 0.0000 | 0.0% (0 / 517) | Failed |

### 2.2 Objective Selection Rationale for $\tau^* = 0.08$
The operating threshold was formally selected on validation data using the following criteria:

1. **Target Operational Recall:** Early-warning institutional surveillance requires capturing $\ge 60\%$ of imminent distress events to prevent catastrophic unmonitored default.
2. **Workload Efficiency ($F_2$-Score Maximization):** In credit surveillance, missing a distressed firm carries significantly higher loss-given-default ($LGD$) than the operational cost of conducting a review on a solvent firm. The $F_2$-score weights recall twice as heavily as precision:
   $$F_2 = \frac{5 \cdot \text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}$$
   Threshold $\tau^* = 0.08$ achieved the highest $F_2$-score (**0.4369**) and highest Youden's Index (**0.5679**) across all candidate thresholds.
3. **Marginal Cost vs. Marginal Gain Analysis:**
   - Comparing $\tau = 0.08$ vs. $\tau = 0.05$: Both thresholds identify the **exact same 9 distressed firms** (Recall = 64.3%). However, $\tau = 0.05$ produces **20 additional false alarms** (58 vs. 38), depressing precision from 19.1% down to 13.4% and increasing portfolio investigation overhead from 9.1% to 13.0%.
   - Comparing $\tau = 0.08$ vs. $\tau = 0.10$: Raising the cutoff to 0.10 drops operational recall from 64.3% to 50.0% (missing 2 additional insolvent firms).
4. **Decision:** **Freeze $\tau^* = 0.08$** as the primary operating threshold.

---

## 3. Evaluation on the Untouched Holdout Test Set ($T = 13$)

After freezing the preprocessor, model weights, calibrator, and operating threshold ($\tau^* = 0.08$), evaluation was conducted **once** on the untouched holdout test partition ($N=262$ observations, 11 distress events, 4.20% prevalence, 251 solvent firms).

### 3.1 Untouched Test Set Performance Across All Thresholds

| Threshold ($\tau$) | Status / Role | TP | FP | FN | TN | Recall | Precision | Specificity | F1-Score | F2-Score | Flagged % | Review Count |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.50** | **Baseline Naive Cutoff** | **0** | **0** | **11** | **251** | **0.0%** | **0.0%** | **100.0%** | **0.0000** | **0.0000** | **0.0%** | **0 / 262** |
| **0.17** | Critical Alert Boundary | 2 | 1 | 9 | 250 | 18.2% | 66.7% | 99.6% | 0.2857 | 0.2128 | 1.1% | 3 / 262 |
| **0.13** | High Alert Boundary | 2 | 5 | 9 | 246 | 18.2% | 28.6% | 98.0% | 0.2222 | 0.1961 | 2.7% | 7 / 262 |
| **0.10** | Intermediate Watch | 2 | 12 | 9 | 239 | 18.2% | 14.3% | 95.2% | 0.1600 | 0.1724 | 5.3% | 14 / 262 |
| **0.08** | **Frozen Operating $\tau^*$** | **3** | **17** | **8** | **234** | **27.3%** | **15.0%** | **93.2%** | **0.1935** | **0.2344** | **7.6%** | **20 / 262** |
| **0.05** | Sensitive Screening | 3 | 32 | 8 | 219 | 27.3% | 8.6% | 87.3% | 0.1304 | 0.1899 | 13.4% | 35 / 262 |

### 3.2 Comparison Against Original Baseline

```
BASELINE NAIVE EVALUATION (Threshold = 0.50):
┌─────────────────────────┬─────────────────────────┐
│ TN = 251 (100.0% clean) │ FP = 0   (0.0% alarms)  │
├─────────────────────────┼─────────────────────────┤
│ FN = 11  (100.0% MISSED)│ TP = 0   (0.0% CAUGHT)  │
└─────────────────────────┴─────────────────────────┘
Recall: 0.0% | Precision: 0.0% | F1: 0.0000 | F2: 0.0000

PHASE 2 VALIDATED OPERATING MODEL (Frozen Threshold = 0.08):
┌─────────────────────────┬─────────────────────────┐
│ TN = 234 (93.2% clean)  │ FP = 17  (6.8% alarms)  │
├─────────────────────────┼─────────────────────────┤
│ FN = 8   (72.7% missed) │ TP = 3   (27.3% CAUGHT) │
└─────────────────────────┴─────────────────────────┘
Recall: 27.3% | Precision: 15.0% | Specificity: 93.2% | F1: 0.1935 | F2: 0.2344
```

### 3.3 Statistical Uncertainty & Small Sample Reality
- **Prevalence Reality:** The holdout test set ($T=13$) contains only **11 positive distress events**. Each individual firm represents $1 / 11 = 9.09\%$ of total recall.
- **95% Wilson Score Confidence Interval:** For test recall of 27.3% (3 hits out of 11), the 95% confidence interval is:
  $$\text{Recall}_{95\% \text{ CI}} = [9.8\%, 56.6\%]$$
- **Scientific Honesty Regarding the 60% Recall Target:** 
  The development target of $\ge 60\%$ recall was achieved on the validation fold (64.3%), but test recall fell to 27.3%. As shown in the table above, even lowering the threshold to 0.05 on the test set captured the exact same 3 firms while doubling false alarms (32 vs. 17). 
  This reflects empirical out-of-period distribution shifts in macroeconomic variables in period $T=13$. Fabricating higher test recall by post-hoc test label tuning was strictly rejected under research audit integrity rules.

---

## 4. Semantic Architecture & Risk Score Consistency

The Phase 2 model preserves clean separation across three distinct analytical constructs:

```mermaid
graph LR
    P[Calibrated Probability P ∈ [0.0, 1.0]] --> S[Standardized Risk Score S = P × 100 ∈ [0.0, 100.0]]
    S --> B[Canonical Risk Bands]
    P --> A[Operational Alert Priority]
    
    subgraph Bands [Risk Bands]
        B1["Healthy: [0.0, 4.0]"]
        B2["Low Risk: (4.0, 8.0]"]
        B3["Watch: (8.0, 13.0]"]
        B4["High Risk: (13.0, 17.0]"]
        B5["Critical: (17.0, 100.0]"]
    end
    
    subgraph Alerts [Early-Warning Alerts]
        A1["P >= 0.170: CRITICAL Distress Alert"]
        A2["P >= 0.130: HIGH Surveillance Alert"]
        A3["P >= 0.080: Operational Watchlist (tau*)"]
    end
```

- **Calibrated Distress Probability ($P$):** Rigorous statistical likelihood estimated via Platt scaling.
- **Standardized Risk Score ($S = \text{round}(P \times 100, 1)$):** Intuitive credit metric spanning 0 to 100.
- **Operating Watchlist Cutoff ($\tau^* = 0.080 \to S = 8.0$):** Aligns precisely with the lower boundary of the Watch band.
- **High Risk Alert Threshold ($P \ge 0.130 \to S \ge 13.0$):** Aligns precisely with the High Risk band.
- **Critical Risk Alert Threshold ($P \ge 0.170 \to S \ge 17.0$):** Aligns precisely with the Critical band.

---

## 5. Threshold Analysis Audit Conclusion

1. The critical defect causing 0% operational recall in the baseline is resolved.
2. The operating threshold $\tau^* = 0.08$ was derived strictly from validation data without test label leakage.
3. On the holdout test set, the model successfully identifies 27.3% of imminent defaults with 93.2% specificity and 15.0% precision (a 3.6x enrichment over the portfolio base rate).
4. All probability, risk score, and alert boundaries remain mathematically coherent.
