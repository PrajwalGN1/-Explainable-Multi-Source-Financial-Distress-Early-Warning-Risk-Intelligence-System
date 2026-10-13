# Portfolio Presentation & Technical Interview Brief

**Project Title:** Explainable Multi-Source SME Financial Distress Early-Warning System  
**One-Line Summary:** A regulatory-compliant, leakage-free corporate early-warning platform combining longitudinal panel dynamics with TreeSHAP explainability to detect SME insolvency up to 12 months in advance.  
**Target Roles:** Senior Machine Learning Engineer, MLOps Engineer, Financial Data Scientist, Quantitative Risk Modeler  

---

## 1. Executive Hook: Why This Project Matters

Conventional bankruptcy prediction models operate on static cross-sectional snapshots: they evaluate a company at period $T$ and predict whether liquidation occurs immediately. In commercial banking and credit portfolio management, this approach is fundamentally inadequate. By the time a corporate borrower enters formal insolvency proceedings, recovery rates collapse below 25%.

**The Engineering Challenge:**  
Build an **Early Warning System (EWS)** that detects subtle multi-period balance-sheet deterioration trajectories 12 months in advance, operates under a strict **8% alert budget constraint** (to avoid overwhelming human credit surveillance desks with false alarms), provides **exact local feature attribution** compliant with model risk regulations (SR 11-7), and preserves **100% cryptographic reproducibility**.

---

## 2. Problem Statement & Core Research Question

> **Research Question:**  
> *Does incorporating longitudinal balance-sheet deterioration dynamics (rolling statistics, lags, deterioration streaks) improve forward distress detection compared to static accounting ratios, and can an early-warning model reliably operate within a strict operational alert budget?*

### Real-World Operational Constraint:
In commercial credit surveillance, an institution monitoring 10,000 corporate loans cannot investigate 1,500 red flags per month. The operations desk operates under an **alert budget constraint** (typically $\le 8\%$ of the monitored universe). If a model achieves 80% recall only by generating a 35% false-positive rate, it is operationally useless.

---

## 3. Dataset Topology & Panel Semantics

The project is built on **Dataset A** (Tehran Stock Exchange SME Corporate Panel), preserving authentic financial characteristics without synthetic data fabrication:
- **Dimensions:** 3,672 annual observations across 422 unique listed enterprises over 14 contiguous reporting periods.
- **Data Quality Invariants:** Exactly 0 missing values, 0 infinite values, 0 duplicate `(Company, Time)` pairs, and 0 time gaps.
- **Target Variable (`Financial Distress`):** A continuous solvency index where **lower values indicate greater distress**. Statutory default is defined at $FD \le -0.50$.
- **Absorbing Exit Dynamics:** 136 distress records exist in the panel; **100.0% (136/136)** occur at the firm's exact terminal reporting period, after which the firm exits the panel due to liquidation or delisting.
- **Forward Horizon Target:**  
  $$Y(i, t) = \mathbf{1}\{FD_{i, t+1} \le -0.50\}$$
  Yields 3,250 evaluable observations with 120 realized defaults (3.69% base rate). 422 terminal records without a $t+1$ outcome are strictly excluded (NaN) and never assumed healthy.

---

## 4. Machine Learning Architecture & Temporal Safeguards

```mermaid
flowchart LR
    A["Raw Historical Panel<br/>(Contiguous T=1..14)"] --> B["Temporal Feature Engine<br/>(Lags, Rolling Means, Deltas)"]
    B --> C["Forward Target Construction<br/>(Y_t = 1 if FD_t+1 <= -0.5)"]
    C --> D["Leakage-Safe Chronological Split<br/>Train: T<=10 | Val: 11-12 | Test: 13->14"]
    D --> E["Frozen Approved EWS Model<br/>Balanced RF + Platt Calibrator"]
    E --> F["TreeSHAP Explainer<br/>Exact Additive Local Drivers"]
    F --> G["Multi-Channel Delivery<br/>Streamlit Dashboard + FastAPI"]
```

### Strict Temporal Leakage Protection:
1. **Chronological Splitting:**
   - Training Set: $T \le 10$ (2,471 evaluable rows, 95 defaults).
   - Validation Set: $T \in [11, 12]$ (517 evaluable rows, 14 defaults).
   - Out-of-Time Holdout Cohort: $T = 13 \to T = 14$ (262 rows, 11 realized defaults).
2. **Train-Only Preprocessing:** Imputation medians and `RobustScaler` parameters are fitted strictly on $T \le 10$. Test transformations use frozen statistics with zero update.
3. **Validation-Only Calibration:** Platt sigmoid calibrators are fitted exclusively on validation out-of-fold predictions. Holdout labels are never exposed during calibrator fitting.
4. **Backward-Looking Features:** All rolling windows ($k=2, 3$) and difference deltas reference historical time periods $\le t$. Max train feature correlation with target is $0.1901$ ($x2$), proving zero direct label lookahead.

---

## 5. Model Risk Governance & Cryptographic Protection

To comply with US Federal Reserve SR 11-7 model risk management guidance:
- **8 Protected Artifacts:** Model weights, calibrator, scaler, threshold config, manifest, and metrics are cryptographically locked via SHA-256 hashes.
- **Production Baseline:** `models/phase2/best_ews_model.joblib` operating at official frozen threshold $\tau = 0.0800$.
- **Research Segregation:** Research candidates and the research budget-capped threshold ($\tau^* = 0.1072$) are sandbox-isolated and never overwrite production configuration.

---

## 6. Empirical Evaluation & Alert-Budget Reconciliation

Evaluated on the matured $T=13 \to T=14$ holdout cohort ($N=262$ firms, $D=11$ distress events, $H=251$ non-distress firms). Under an 8% alert budget constraint, the maximum allowable alerts is $\lfloor 0.08 \times 262 \rfloor = 20$.

| Model Architecture / Operating Point | Threshold | Alerts | Alert Rate | Budget Compliant? | TP | FP | FN | TN | Recall (Sens.) | Precision (PPV) | Specificity | Brier Score | ROC-AUC | 95% Wilson CI (Recall) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Approved Baseline (Nominal)** | `0.0800` | 37 | 14.12% | ❌ (Excess +17) | 5 | 32 | 6 | 219 | **45.45%** | 13.51% | 87.25% | 0.0382 | 0.7961 | [21.3%, 72.0%] |
| **Approved Baseline (Budget-Capped)** | `0.1072` | 20 | 7.63% | ✅ YES | 4 | 16 | 7 | 235 | **36.36%** | 20.00% | 93.63% | 0.0382 | 0.7961 | [15.2%, 64.6%] |
| **Candidate Cost-Sensitive RF** | `0.0894` | 20 | 7.63% | ✅ YES | 3 | 17 | 8 | 234 | **27.27%** | 15.00% | 93.23% | 0.0389 | 0.7521 | [9.7%, 56.6%] |
| **Calibrated Logistic Reg. (Nominal)** | `0.0800` | 0 | 0.00% | ✅ YES | 0 | 0 | 11 | 251 | **0.00%** | 0.00% | 100.0% | 0.0404 | 0.8095 | [0.0%, 25.9%] |
| **Calibrated Logistic Reg. (Budget-Capped)**| `0.0271` | 20 | 7.63% | ✅ YES | 3 | 17 | 8 | 234 | **27.27%** | 15.00% | 93.23% | 0.0404 | 0.8095 | [9.7%, 56.6%] |
| **Standard Random Forest (Nominal)** | `0.0800` | 15 | 5.73% | ✅ YES | 2 | 13 | 9 | 238 | **18.18%** | 13.33% | 94.82% | 0.0445 | 0.7342 | [5.1%, 47.7%] |

### Statistical Honesty in Technical Interviews:
When presenting to quantitative interviewers:
- **Small Sample Size:** With $n=11$ realized defaults in the holdout cohort, each detection represents $9.09\%$ of recall.
- **Confidence Interval Overlap:** The 95% Wilson interval for the Baseline Budget-Capped model ([15.2%, 64.6%]) completely overlaps with that of the Candidate model ([9.7%, 56.6%]).
- **Key Takeaway:** We do **not** claim statistical superiority. The candidate model is retained as a research comparator, demonstrating model risk humility and scientific integrity.

---

## 7. False-Negative Diagnostics: The Acute vs. Chronic Dilemma

An audit of the 11 realized test defaults revealed a critical credit risk insight:
1. **Acute Cliff Crashes (5 Firms):** Firms 28, 60, 84, 225, 376 maintained positive solvency at Period 13 ($FD_{13} > 0.0$) before collapsing to statutory failure ($FD_{14} \le -0.50$).
   - The production baseline captured Firms 28 and 60.
   - The cost-sensitive candidate model missed both because its rolling temporal aggregation smoothed out single-period sudden drops.
2. **Chronic Deteriorations (6 Firms):** Firms 23, 29, 53, 100, 101, 112 experienced multi-period solvency erosion ($FD_{13} \le 0.0$).
   - Both models reliably flagged chronic cases (Firms 100, 101, 112).
   - Misses on Firms 23, 29, and 53 were caused by rank displacement under the 20-alert budget cap.

---

## 8. Explainability & Regulatory Guardrails

Powered by **TreeSHAP** (Lundberg et al., 2020) on the 100-tree ensemble:
- **Top Risk Drivers:**
  1. `x2_roll_mean_2` (base variable: $x2$ retained earnings / total assets) — cumulative profitability buffer.
  2. `x12` (gross profit / total assets) — operating margin health.
  3. `x8` (current assets / total assets) — short-term working capital liquidity.
- **Exact Local Additivity:** $\text{base\_value} + \sum \phi_i = f(x)$ verified on all 422 companies.
- **Regulatory Semantics Guardrail:** All UI views and API responses explicitly state that SHAP attributions represent **statistical predictive associations**, not causal counterfactual mechanisms (e.g., artificially increasing retained earnings will not causally guarantee solvency if underlying sales collapse).

---

## 9. Delivery Ecosystem: UI & API

1. **Streamlit Analytics Dashboard (`app.py`):**
   - 12 comprehensive pages with high-aesthetic dark mode and glassmorphism styling.
   - Dynamic enterprise selector covering the full 422-firm universe with input validation.
   - Real-time TreeSHAP waterfall charts, timeline trajectories, and radar charts.
2. **FastAPI Microservice (`api.py`):**
   - 11 production endpoints covering inference (`/predict`), entity dossiers (`/company/{id}`), and research audits (`/research/...`).
   - Fail-closed administrative cache reload secured via `secrets.compare_digest`.
   - Comprehensive Pydantic request validation rejecting malformed payloads with HTTP 422.

---

## 10. Engineering Lessons Learned & Takeaways

1. **Alert Budgets Govern Real-World Utility:** A model with 80% recall is useless if it flags 40% of the loan book. Credit risk modeling is an exercise in constrained optimization under budget ceilings.
2. **Platt Calibration Compression:** Calibrating unweighted models on imbalanced data ($2.7\%$ prior) compresses probabilities into narrow bands ($[0.01, 0.03]$), explaining why static default thresholds (0.0800) emit zero alerts unless adjusted for budget operating points.
3. **Temporal Features Trade Off Sensitivity to Acute Shocks:** Moving averages enhance stability for chronic distress but delay detection of sudden liquidity cliffs.
4. **Reproducibility is Non-Negotiable:** Automated test suites (226 passed tests) and cryptographic SHA-256 baseline hashing ensure that research iterations never corrupt approved production baselines.
