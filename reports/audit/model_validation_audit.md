# Model Validation & Data-Leakage Audit Report

**Project:** Explainable SME Financial Distress Early-Warning & Risk Intelligence System  
**Audit Scope:** PHASE 4 (Strict Data-Leakage Audit) & PHASE 6 (Model Validation & Evaluation Audit)  
**Evidence Source:** `src/pipeline.py`, `src/models/`, `src/preprocessing/`, `models/ews/ews_metrics.json`, `models/bankruptcy/bankruptcy_metrics.json`, `reports/model_evaluation/`, and panel test partitions.

---

## 1. Executive Summary: Validation & Leakage Audit

This report presents a forensic investigation of the data partitioning protocols, leakage vulnerabilities, model evaluation standards, probability calibration behavior, and operational decision thresholds across the project's models.

### Key Audit Conclusions:
1. **Core Temporal Splitting is Structurally Leak-Free:** The partition of Dataset A into Train ($T \le 10$), Validation ($T \in [11, 12]$), and Holdout Test ($T=13$) maintains forward chronological time-ordering. Future observations are not used to train earlier models.
2. **Preprocessor Hygiene is Sound:** Imputation medians and `RobustScaler` parameters are fitted exclusively on training splits and applied forward to validation and test folds.
3. **The 0.5 Threshold Disaster (Operational Failure):** The primary Temporal EWS model was calibrated via Platt scaling (sigmoid), compressing predicted probabilities into the range $[0.01, 0.19]$. When evaluated against the default 0.5 classification threshold, **every test observation is predicted as non-distressed**. The test confusion matrix contains **0 True Positives and 11 False Negatives**, yielding **Recall = 0.0%** and **False Negative Rate = 100%**. The reported accuracy of 95.8% is entirely an artifact of class imbalance (251/262 negative base rate).
4. **Generalization Scope (Known Portfolio vs. New Originations):** 83.6% of firms in the holdout test set were present in the training set. The model is validated to monitor *existing known clients over future reporting cycles*, but has **not been validated for underwriting brand-new, unseen applicant firms**.
5. **Stacking Meta-Model Degradation:** The multi-source stacking meta-model (Exp 6) degrades discrimination (`ROC-AUC` dropped from 0.7783 to 0.6588; `PR-AUC` dropped from 0.1946 to 0.1294) because it was trained on synthetic pseudo-proxy formulas rather than real cross-dataset evidence.

---

## 2. PHASE 4 — Strict Data-Leakage Audit

Treating data leakage prevention as a non-negotiable requirement, this section audits 14 potential leakage vectors:

| Leakage Vector | Code Path / Component | Leakage Status | Evidence & Forensic Analysis |
| :--- | :--- | :---: | :--- |
| **1. Target-Related Post-Event Features** | `src/features/temporal_features.py` | **CLEAN** | Features at time $t$ use only predictor variables $x_t$ and historical lags. Post-distress rows (which do not exist anyway due to terminal attrition) are not referenced. |
| **2. Features Derived Directly from Target** | `src/preprocessing/preprocessors.py`: line 17 | **CLEAN** | `Financial Distress`, `current_distress`, `forward_target`, and `forward_target_2period` are explicitly dropped from the feature matrix prior to training and transformation. |
| **3. Future Observations in Historical Features** | `src/features/temporal_features.py`: lines 56–91 | **CLEAN** | Lags use positive integer shifts (`shift(1)`, `shift(2)`). Difference is $x_t - x_{t-1}$. Rolling means use `.rolling(window=k)` on sorted past periods. Future rows $\ge t+1$ are never referenced. |
| **4. Rolling Calculations Across Entities** | `src/features/temporal_features.py`: line 53 | **CLEAN** | All rolling and expanding statistics are applied after `.groupby("Company")`. Feature values never leak across different company entities. |
| **5. Preprocessing Before Train/Test Split** | `src/pipeline.py`: lines 57–60 | **CLEAN** | `preprocessor.fit(train_df)` is called strictly after splitting. `transform()` is called separately on `train_df`, `val_df`, and `test_df`. |
| **6. Oversampling / SMOTE Before Split** | `src/models/ews_model.py`, `bankruptcy_model.py` | **CLEAN** | SMOTE is not used. Class imbalance is addressed via algorithmic weighting (`scale_pos_weight` in XGBoost, `class_weight="balanced"` in RF/Logistic) computed strictly from `y_train`. |
| **7. Duplicate Records Across Partitions** | Whole dataset audit | **CLEAN** | Zero duplicate rows exist in Dataset A or Dataset B. |
| **8. Repeated Companies Across Partitions** | `src/models/ews_model.py`: `prepare_data` | **IDENTIFIED RISK (Not Leakage)** | 219 out of 262 companies in the Test set (83.6%) appear in the Training set. This is methodologically valid for an *ongoing portfolio surveillance early-warning system*, but limits claims about out-of-sample generalization to *newly founded / unseen companies*. |
| **9. Company Identity Memorization Shortcuts** | `src/preprocessing/preprocessors.py`: line 17 | **CLEAN** | The `Company` ID column is explicitly in `drop_cols`. Models do not receive entity IDs as inputs. |
| **10. Global Feature Selection Before Splitting** | `src/features/temporal_features.py`: line 12 | **LEAKAGE SUSPECTED** | The 8 features chosen for temporal engineering (`x1, x2, x5, x10, x25, x48, x64, x81`) were selected based on correlation with the target across the *entire* dataset. |
| **11. Hyperparameter Tuning on Test Set** | `src/models/ews_model.py`: line 64–68 | **CLEAN** | Hyperparameters are fixed configurations (`n_estimators=100, max_depth=5`). Model selection between candidates is based strictly on validation fold metrics (`val_metrics`), not test metrics. |
| **12. Threshold Selection Using Test Labels** | `src/models/ews_model.py`: line 96 | **CLEAN (But Broken)** | The test predictions use a fixed threshold of 0.5 without optimizing against test labels. While clean of threshold leakage, this causes 0% test recall. |
| **13. Test-Time Preprocessing Inconsistencies** | `src/pipeline.py`, `api.py` | **CLEAN** | Serialized preprocessors (`joblib`) ensure identical median imputations and scaling center/scale factors are applied during test and inference. |
| **14. Dashboard Leaking Future Labels** | `src/pipeline.py`: line 266–308, `app.py` | **CLEAN** | Precomputed database dossiers for the dashboard evaluate risk at period $t$ using only observations $\le t$. Ground-truth future labels are not shown as model inputs. |

### Summary of Leakage Corrections Needed:
- **Correction 10:** Automated feature selection must be wrapped inside a cross-validated scikit-learn pipeline fitted strictly on training folds to eliminate pre-split selection bias.

---

## 3. PHASE 6 — Audit of Validation Architecture & Model Evaluation

### 3.1 Validation Design Evaluation

The project implements a **3-way chronological split**:
- **Training Set:** Periods 1 to 10 (2,471 firm-periods, 95 distress events, 3.84% positive rate)
- **Validation Set:** Periods 11 to 12 (517 firm-periods, 14 distress events, 2.71% positive rate)
- **Holdout Test Set:** Period 13 (262 firm-periods, 11 distress events, 4.20% positive rate)

#### Evaluation of Split Design:
- **Chronological Validity:** Appropriate for longitudinal early-warning systems. Evaluates whether historical financial trajectories up to period 10/12 can predict forward distress in future macroeconomic periods (13).
- **Limitation — Sample Size of Positive Test Cases:** Because Period 14 has no subsequent period ($t+1$), all Time 14 rows are dropped. The holdout test set contains only **11 positive failure events**. Metrics like Precision, Recall, and F1 carry wide binomial confidence intervals:
  - If a model correctly identifies 4 out of 11: Recall = 36.4%.
  - If a model correctly identifies 6 out of 11: Recall = 54.5%.
  - If a model correctly identifies 0 out of 11: Recall = 0.0%.

---

### 3.2 Candidate Model Comparison (Validation Fold Performance)

In `src/models/ews_model.py`, three candidate algorithms were trained on the training partition ($T \le 10$) and compared on the validation fold ($T \in [11, 12]$):

| Algorithm | Val Accuracy | Val Precision | Val Recall | Val F1 | Val ROC-AUC | Val PR-AUC | Val Specificity | Val False Negative Rate | Val Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (Balanced) | 96.13% | 12.50% | 7.14% | 0.0909 | 0.8459 | 0.1919 | 98.61% | 92.86% | 0.0335 |
| **Random Forest** (Balanced, Depth=5) | 94.39% | 24.14% | **50.00%** | **0.3256** | **0.8828** | 0.2001 | 95.63% | **50.00%** | 0.0512 |
| **XGBoost** (`scale_pos_weight`, Depth=4) | 95.16% | 23.81% | 35.71% | 0.2857 | 0.8824 | **0.2050** | 96.82% | 64.29% | 0.0359 |

#### Analysis:
- Random Forest was selected as the best model based on validation fold `ROC-AUC + Recall` (0.8828 + 0.5000).
- Random Forest correctly identified 7 out of 14 distress events on the validation fold (50.0% recall, 24.1% precision).

---

### 3.3 Holdout Test Set Evaluation & The 0.5 Threshold Breakdown

The selected Random Forest model was calibrated using Sigmoid Platt scaling fitted on the validation fold, and then evaluated on the holdout test set ($T=13$).

#### The Official Reported Test Metrics (`models/ews/ews_metrics.json`):

```json
"test_metrics": {
  "accuracy": 0.958,
  "precision": 0.0,
  "recall": 0.0,
  "f1": 0.0,
  "roc_auc": 0.7783,
  "pr_auc": 0.1946,
  "specificity": 1.0,
  "false_negative_rate": 1.0,
  "brier_score": 0.0387,
  "confusion_matrix": {
    "tn": 251,
    "fp": 0,
    "fn": 11,
    "tp": 0
  }
}
```

#### Detailed Forensic Analysis of the Failure:
1. **The Model Predicted Zero Positives:** The confusion matrix shows **TN = 251, FP = 0, FN = 11, TP = 0**.
2. **Every Distressed Firm was Missed:** The False Negative Rate is **100.0%**. The model failed to flag a single one of the 11 enterprises that suffered terminal distress in period 14.
3. **Misleading High Accuracy:** The reported accuracy of **95.8%** is purely the consequence of predicting the majority negative class ($251 / 262 = 0.958$).
4. **Why Did This Happen?**
   - Platt scaling (sigmoid calibration) adjusts raw model probabilities to reflect the empirical base rate of the validation fold (~2.7%).
   - Consequently, the calibrated probabilities for almost all test firms range between **1.0% and 19.1%**.
   - The test evaluation code in `src/models/ews_model.py` (line 96) applied a hardcoded classification threshold:
     ```python
     test_preds = (test_cal_probs >= 0.5).astype(int)
     ```
   - Because the maximum calibrated probability across all 262 firms in the test set was approximately **0.191** (19.1%), **not a single observation exceeded 0.5**.
   - Result: All predictions were 0.

#### Discrimination vs. Decisioning:
- **Discrimination is Intact:** `ROC-AUC = 0.7783` and `PR-AUC = 0.1946` confirm that the model's *ranking* capability is reasonably strong (substantially better than random guessing at 4.2% base rate). Distressed firms were assigned higher probabilities than healthy firms.
- **Decisioning Failed:** Because the classification threshold was left at 0.5 rather than tuned to the calibrated probability distribution, the system is functionally inoperative as an alert mechanism at that threshold.

---

### 3.4 Operational Trade-Offs in Credit Risk Surveillance

In corporate credit risk management and SME financial distress surveillance:
- **Cost of a False Negative ($FN$):** A distressed firm is classified as healthy. The credit institution takes no preventive action, leading to loan default, liquidation, and principal loss (potentially hundreds of thousands of dollars per firm).
- **Cost of a False Positive ($FP$):** A healthy firm is flagged for surveillance. A credit risk officer conducts an enhanced review (estimated cost: 2–4 hours of analyst time).
- **Asymmetric Cost Ratio:** In banking credit risk, $C(FN) \gg C(FP)$, typically by a factor of $20:1$ to $50:1$.
- **Required Threshold:** A scientifically sound early-warning system for rare-event credit distress must select an operating threshold that maximizes Recall or optimizes a Cost-Utility function (e.g., target recall $\ge 70\%$, accepting a precision of $15\%–25\%$).
- For this calibrated distribution, the decision threshold should be set at approximately **0.08 to 0.10** (matching the "Watch" / "High Risk" categories), NOT 0.50.

---

### 3.5 Cross-Sectional Bankruptcy Model Evaluation (Dataset B)

From `models/bankruptcy/bankruptcy_metrics.json`, evaluated on the stratified holdout test partition of Dataset B (1,364 firms, 44 bankruptcies, 3.23% positive rate):

| Metric | Random Forest (Best) | XGBoost | Logistic Regression |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | 97.36% | 96.85% | 51.47% |
| **Test Precision** | **72.22%** | 50.00% | 3.83% |
| **Test Recall** | **29.55%** | 31.82% | 59.09% |
| **Test F1-Score** | **0.4194** | 0.3889 | 0.0720 |
| **Test ROC-AUC** | **0.9455** | 0.9353 | 0.5738 |
| **Test PR-AUC** | **0.4401** | 0.3942 | 0.0767 |
| **Test Specificity** | 99.62% | 98.94% | 50.61% |
| **Test False Negative Rate** | 70.45% | 68.18% | 40.91% |
| **Test Brier Score** | **0.0234** | 0.0289 | 0.2705 |

#### Findings on Dataset B:
- Random Forest demonstrates strong discrimination on canonical cross-sectional ratios (`ROC-AUC = 0.9455`, `PR-AUC = 0.4401`).
- Specificity is 99.62% (only 5 false alarms across 1,320 healthy firms).
- However, at threshold 0.5, Recall is only 29.55% (13 TP, 31 FN). Lowering the threshold to ~0.15–0.20 would capture significantly more insolvency events.

---

## 4. PHASE 6 Summary: Required Methodological Upgrades

1. **Threshold Optimization on Validation Fold:** Implement cost-sensitive threshold selection (e.g., Youden's $J$-statistic, $F_2$-score optimization, or fixed recall target $\ge 70\%$) fitted strictly on the validation split and evaluated on the test split.
2. **Eliminate Misleading Test Confusion Matrix:** Replace the 0.5 threshold evaluation in `src/models/ews_model.py` with the calibrated operating threshold.
3. **Multi-Horizon Expansion ($T=14$ Utilization):** To evaluate observations at Period 14, consider training a secondary static insolvency classifier or evaluating retrospectively against external insolvency filings.
4. **Clarify Generalization Boundaries:** Update documentation and reports to state clearly that the primary model generalizes across time for *monitored portfolio firms*, and specify the separate evaluation required for *cold-start applicant underwriting*.
