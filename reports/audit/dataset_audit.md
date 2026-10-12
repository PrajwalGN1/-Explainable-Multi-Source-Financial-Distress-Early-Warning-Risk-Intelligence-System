# Comprehensive Dataset & Target Semantics Audit Report

**Project:** Explainable SME Financial Distress Early-Warning & Risk Intelligence System  
**Audit Scope:** PHASE 2 (Dataset Audit) & PHASE 3 (Business Problem & Target Semantics Audit)  
**Evidence Sources:** Active repository files, dataset CSVs, precomputed metadata, academic literature (Ebrahimi et al., TEJ, Polish bankruptcy benchmarks), and pipeline source code.

---

## 1. Inventory of Candidate vs. Actual Datasets

The project specification and documentation reference several potential data sources. This audit cross-references them against actual local workspace files:

| Dataset Identifier | Intended Source / Citation | Expected Role | Local File Status | Actual File Path | Format & File Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dataset A** | Tehran Stock Exchange Panel (Ebrahimi, 2019 / Kaggle) | Primary Longitudinal EWS | **Present Locally** | `Data/raw/Financial Distress.csv` (and duplicate at `Data/Financial Distress.csv`) | CSV, 2,240,427 bytes (2.24 MB) |
| **Dataset B** | Taiwan Economic Journal (TEJ, 1999–2009 / Kaggle) | Structural Bankruptcy Classifier | **Present Locally** | `Data/raw/data.csv` (and duplicate at `Data/data.csv`) | CSV, 11,456,101 bytes (11.46 MB) |
| **Dataset C** | Polish Companies Bankruptcy (Zięba et al. test partition) | External Anomaly / OOD Detection | **Present Locally** | `Data/raw/bankruptcy_Test_X.csv` (and duplicate at `Data/bankruptcy_Test_X.csv`) | CSV, 6,212,713 bytes (6.21 MB) |
| **World Bank Enterprise Surveys** | World Bank SME Surveys | SME Financial Constraints | **Absent** | Not present on disk | N/A |
| **MoSPI ASUSE** | Ministry of Statistics & Programme Implementation (India) | Unincorporated SME Indicators | **Absent** | Not present on disk | N/A |
| **RBI Public Statistics** | Reserve Bank of India Financial Ratios | Macro / Credit Benchmark | **Absent** | Not present on disk | N/A |

> [!IMPORTANT]
> World Bank, MoSPI, and RBI datasets are merely candidate research references mentioned in earlier scoping discussions. They do not exist locally, are not ingested anywhere in code, and must not be claimed as active data sources.

---

## 2. In-Depth Audit of Actual Datasets

### 2.1 Dataset A: Tehran Stock Exchange Financial Distress Panel

#### 2.1.1 Structural Topology & Observation Counts
- **Physical File:** `Data/raw/Financial Distress.csv`
- **Total Records:** 3,672 rows × 86 columns
- **Entity Identifier:** `Company` (discrete integer identifier, values 1 to 422)
- **Time Variable:** `Time` (discrete sequential periods, integer values 1 to 14)
- **Target Variable Column:** `Financial Distress` (continuous numerical index)
- **Feature Columns:** `x1` through `x83` (83 feature attributes)
- **Unique Companies:** 422 unique firms
- **Observations per Company:** 
  - Minimum: 1 observation
  - Median: 9 observations
  - Maximum: 14 observations
- **Missing Values:** 0 missing values across all 3,672 rows and 86 columns (100% complete).
- **Exact Duplicate Rows:** 0 duplicate rows.
- **Duplicate `(Company, Time)` pairs:** 0 duplicates. Each company has at most one record per time period.
- **Time Gaps:** 0 companies exhibit time gaps ($\Delta Time > 1$). Every company's reporting periods are consecutive integers.

#### 2.1.2 Temporal Topology & Attrition Dynamics
A critical finding regarding Dataset A is the distribution of company entry and exit times:

- **Entry Periods (`Time.min()` per company):**
  - Time 1: 242 firms (57.3%)
  - Time 2: 41 firms
  - Time 3: 25 firms
  - Time 4–10: 59 firms
  - Time 11–14: 55 firms (including 18 firms that appear *only* in period 14)
- **Exit Periods (`Time.max()` per company):**
  - Time 14: 280 firms (surviving/right-censored)
  - Time 1–13: 142 firms cease reporting prior to period 14.

#### 2.1.3 Genuineness of Longitudinal Structure
- **Assessment:** **Genuinely longitudinal panel**. Observations represent repeated temporal financial measurements for the same corporate entities across sequential annual reporting cycles.
- **Limitation:** The panel is **unbalanced** due to staggered entry (left-truncation) and firm failure/attrition (right-censoring).

---

### 2.2 Dataset B: Taiwan Economic Journal (TEJ) Bankruptcy Benchmark

#### 2.2.1 Structural Topology & Observation Counts
- **Physical File:** `Data/raw/data.csv`
- **Total Records:** 6,819 rows × 96 columns
- **Entity Identifier:** **None**. Rows represent independent firm-year observations without company tracking IDs.
- **Time Variable:** **None**. Cross-sectional dataset spanning 1999–2009 without explicit timestamps per row.
- **Target Variable Column:** `Bankrupt?` (discrete binary: `0` = non-bankrupt, `1` = bankrupt)
- **Feature Columns:** 95 financial ratio indicators spanning profitability, liquidity, solvency, leverage, turnover, and operational efficiency.
- **Missing Values:** 0 missing values across the entire dataset.
- **Exact Duplicate Rows:** 0 duplicate rows.
- **Constant Features:** Column `Net Income Flag` is completely constant (value `1` for all 6,819 rows). The preprocessor correctly identifies and drops this column.

#### 2.2.2 Class Distribution & Imbalance
- Non-Bankrupt (`0`): 6,599 records (96.77%)
- Bankrupt (`1`): 220 records (3.23%)
- Imbalance Ratio: approximately 30:1.

#### 2.2.3 Longitudinal Structure & Generalizability
- **Assessment:** **Purely cross-sectional**. Because entity IDs and timestamps are absent, it is impossible to construct lag features, delta vectors, or longitudinal trajectories for Dataset B.
- **Applicable Prediction Type:** Static cross-sectional classification only.

---

### 2.3 Dataset C: Polish Industrial Companies Bankruptcy Test Set

#### 2.3.1 Structural Topology & Observation Counts
- **Physical File:** `Data/raw/bankruptcy_Test_X.csv`
- **Total Records:** 5,000 rows × 65 columns
- **Entity Identifier:** `ID` (row index integer from 1 to 5,000)
- **Time Variable:** **None**.
- **Target Variable:** **Unlabelled**. No bankruptcy or distress labels are present.
- **Feature Columns:** `Attr1` through `Attr64` (64 standardized financial ratio attributes from the Polish industrial bankruptcy dataset).
- **Missing Values:** 0 missing values.
- **Data Types:** All features `Attr1`–`Attr64` are continuous floating-point values.

#### 2.3.2 Role and Analytical Validity
- **Assessment:** Unlabelled benchmark partition. Cannot be used for supervised training or validation.
- **Applicable Analytical Task:** Unsupervised Out-of-Distribution (OOD) anomaly detection, distribution drift quantification, or external outlier profiling.

---

## 3. Audit of the Business Problem & Target Semantics (PHASE 3)

### 3.1 What Does the Target Variable in Dataset A Actually Measure?

The column `Financial Distress` in Dataset A is **continuous**, not binary.
Empirical distribution from `Data/raw/Financial Distress.csv`:
- Minimum: -8.6317
- Maximum: +128.4000
- Mean: +1.0403
- Standard Deviation: 2.6522
- Median: +0.5838
- 1st Percentile: -1.6407
- 5th Percentile: -0.4217
- 10th Percentile: -0.0789
- 90th Percentile: +2.6712

#### Directionality:
- **Smaller (more negative) values indicate greater distress.**
- **Larger positive values indicate financial health.**

#### Literature Consensus (Ebrahimi et al., 2019):
In Iranian corporate insolvency research (under Article 141 of the Iranian Commercial Code), a company is legally deemed in financial distress when accumulated losses exceed 50% of its stated capital. The published financial distress score is constructed such that:
$$\text{Status} = \begin{cases} \text{Distressed (1)}, & \text{if } \text{Financial Distress} \le -0.50 \\ \text{Healthy (0)}, & \text{if } \text{Financial Distress} > -0.50 \end{cases}$$

### 3.2 Terminal Distress Mechanics in Dataset A
An exhaustive audit of the 136 distress events in Dataset A reveals a critical operational fact:
- **Number of records with `Financial Distress <= -0.5`:** Exactly 136 records.
- **Unique companies experiencing distress:** Exactly 136 companies.
- **Number of post-distress observations:** **0**. Once a firm experiences `Financial Distress <= -0.5`, it ceases reporting immediately and exits the dataset.

This confirms that in Dataset A, **distress is an absorbing/terminal state**.

```text
Historical Progression of a Distressed Firm:
Period 1 (Healthy) -> Period 2 (Healthy) -> Period 3 (Terminal Distress <= -0.5) -> [EXITS DATASET]
```

### 3.3 Forward-Shifted Target Construction ($t \to t+1$)

To convert this static terminal label into a forward-looking early warning target, the code implements:
$$y_{i, t}^{\text{forward}} = \mathbb{I}(\text{Financial Distress}_{i, t+1} \le -0.5)$$

#### Target Distribution Across Time:
When shifting the target forward within each company:
- Total rows in dataset: 3,672
- Non-Distressed ($y = 0$): 3,130 rows
- Forward Distressed ($y = 1$): 120 rows
- Right-Censored / Terminal ($y = \text{NaN}$): 422 rows

#### Breakdown by Period:

| Time Period | $y = 0$ (Healthy next period) | $y = 1$ (Distress at $t+1$) | $y = \text{NaN}$ (No subsequent period) | Total Rows | Forward Event Rate |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 216 | 14 | 12 (10 terminal distress, 2 attrition) | 242 | 6.09% |
| **2** | 241 | 11 | 19 (17 terminal distress, 2 attrition) | 271 | 4.37% |
| **3** | 256 | 7 | 14 (12 terminal distress, 2 attrition) | 277 | 2.66% |
| **4** | 258 | 6 | 10 (8 terminal distress, 2 attrition) | 274 | 2.27% |
| **5** | 248 | 9 | 9 (6 terminal distress, 3 attrition) | 266 | 3.50% |
| **6** | 237 | 16 | 10 (9 terminal distress, 1 attrition) | 263 | 6.32% |
| **7** | 232 | 9 | 16 (16 terminal distress) | 257 | 3.73% |
| **8** | 223 | 13 | 9 (9 terminal distress) | 245 | 5.51% |
| **9** | 228 | 4 | 13 (13 terminal distress) | 245 | 1.72% |
| **10** | 237 | 6 | 6 (5 terminal distress, 1 attrition) | 249 | 2.47% |
| **11** | 247 | 6 | 6 (6 terminal distress) | 259 | 2.37% |
| **12** | 256 | 8 | 6 (6 terminal distress) | 270 | 3.03% |
| **13** | 251 | 11 | 12 (8 terminal distress, 4 attrition) | 274 | 4.20% |
| **14** | **0** | **0** | **280 (Panel ends at period 14)** | **280** | **N/A** |

#### Critical Target Horizon Findings:
1. **Period 14 Rows Cannot Be Evaluated:** Because the panel terminates at Period 14, forward distress at $T=15$ is unknowable. All 280 observations at Time 14 must be dropped when training or evaluating a $t \to t+1$ forward model.
2. **True Test Set Composition:** In `src/models/ews_model.py`, the holdout test set is specified as `Time > 12`. Because all Time 14 rows have `forward_target == NaN`, the test set consists **solely of Time 13 observations** (262 rows, 11 positive events). The documentation claim of "Periods 13 and 14 in test set" is inaccurate.
3. **Small Sample of Failure Events in Test Set:** The test set contains only **11 positive failure events**. An evaluation metric computed on 11 positive events carries high sample variance. A single classification shift changes recall by $1 / 11 = 9.09\%$.

---

## 4. Multi-Dataset Compatibility Audit

A core requirement of this audit is to evaluate whether integrating Dataset A, Dataset B, and Dataset C is methodologically defensible.

### 4.1 Compatibility Matrix

| Evaluation Dimension | Dataset A (Tehran Panel) | Dataset B (Taiwan TEJ) | Dataset C (Polish Test) | Compatibility Status |
| :--- | :--- | :--- | :--- | :--- |
| **Corporate Population** | Listed non-financial firms on Tehran Stock Exchange | Publicly traded Taiwanese manufacturing/tech firms | Polish industrial manufacturing enterprises | **INCOMPATIBLE** (Completely different sovereign economies) |
| **Time Period** | 14 sequential reporting periods (~2000s–2010s) | 1999–2009 (pooled cross-section) | Multi-year horizon (unspecified test fold) | **INCOMPATIBLE** (Macroeconomic regimes diverge) |
| **Feature Definitions** | Anonymized attributes `x1`..`x83` (masked definitions) | 95 named financial ratios (GAAP/IFRS standard) | 64 standardized ratios `Attr1`..`Attr64` | **INCOMPATIBLE** (Completely disjoint feature sets) |
| **Accounting Standards** | Iranian National Accounting Standards | Taiwanese GAAP (TEJ convention) | Polish Accounting Act / IFRS | **INCOMPATIBLE** (Different asset valuation & depreciation rules) |
| **Target Variable** | Continuous index $\le -0.50$ (Art. 141 distress) | Binary corporate bankruptcy / liquidation | Unlabelled | **INCOMPATIBLE** (Different insolvency legal definitions) |

### 4.2 Audit of the Implemented "Pseudo-Fusion" in `src/pipeline.py`

In `src/pipeline.py` (lines 76–95, 204–218, 295–306), the codebase attempts to train a Stacking Meta-Learner that takes four inputs:
1. `ews_prob` (from Dataset A model)
2. `bankruptcy_prob` (intended from Dataset B model)
3. `anomaly_score` (intended from Dataset C model)
4. `financial_health_risk` (intended from balance sheet ratios)

#### The Offending Code Path:
Because Dataset A does not have the features required by the Bankruptcy Model (Dataset B) or the Anomaly Model (Dataset C), the pipeline could not execute real inference on Dataset A rows. Instead, lines 80–85 and 296–297 implement the following **hardcoded proxy formulas**:

```python
# Simulated cross-sectional profile matching for proxy bankruptcy & anomaly evidence
proxy_bankrupt = float(
    np.clip(
        health_dict["financial_health_risk"] * 0.85
        + (row.get("x64", 0) > 0.5) * 0.15,
        0.0,
        1.0,
    )
)
proxy_anomaly = float(
    np.clip(
        abs(row.get("x81", 0)) * 0.2 + (row.get("x64", 0) > 1.0) * 0.3, 0.0, 1.0
    )
)
```

#### Scientific & Practical Impact:
1. **The Bankruptcy and Anomaly models were never actually used during inference on Dataset A.** Their outputs were simulated using arbitrary linear combinations of `x64` and `x81`.
2. **Collinearity and Negative Weights:** Because `proxy_bankrupt` was derived from `financial_health_risk`, the two features were collinear ($r > 0.85$). In the resulting logistic regression meta-model, `anomaly_score` received a negative weight (-0.9285), implying that greater anomaly reduces financial distress risk.
3. **Severe Performance Degradation:** In the ablation study (`reports/model_evaluation/ablation_study.csv`), the genuine Temporal EWS model achieved `ROC-AUC = 0.7783` and `PR-AUC = 0.1946`. When "fused" with these pseudo-proxies in Experiment 6, `ROC-AUC` collapsed to **0.6588** and `PR-AUC` collapsed to **0.1294** (a 33.5% loss in precision-recall performance).

### 4.3 Scientific Recommendation on Multi-Dataset Architecture
- **Prohibit Feature-Level Concatenation:** Concatenating rows or aligning columns between Dataset A, B, and C is methodologically invalid.
- **Prohibit Fabricated Proxy Formulas:** Replace the pseudo-proxy calculations with genuine model outputs or eliminate cross-dataset stacking on Dataset A.
- **Adopt a Decoupled Specialist Architecture:**
  - **Model 1 (Temporal EWS):** Operates on Dataset A to predict longitudinal panel distress trajectories.
  - **Model 2 (Structural Financial Diagnostic):** Operates on named canonical ratios (Dataset B) when balance sheet line items are supplied.
  - **Model 3 (Outlier Profiler):** Operates as an independent statistical distance tool.
  - **Score-Level Fusion:** Should only combine models when identical input entities are observed across all sources. When evaluating Dataset A firms, the system should rely directly on the calibrated Temporal EWS score and genuine historical indicators without synthetic multi-source dilution.

---

## 5. Empirical Data Dictionary

### 5.1 Dataset A (`Financial Distress.csv`) Data Dictionary

| Column Name | Observed Type | Range / Categories | Missing | Provenance / Semantics | Research Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Company` | `int64` | 1 to 422 | 0 | Unique enterprise identifier | Entity grouping variable |
| `Time` | `int64` | 1 to 14 | 0 | Consecutive annual reporting periods | Temporal sequence index |
| `Financial Distress` | `float64` | -8.6317 to 128.4000 | 0 | Continuous corporate distress index | Target variable ($\le -0.50$ = terminal distress) |
| `x1` .. `x79` | `float64` | Continuous | 0 | Anonymized financial indicators (masked accounting line items/ratios) | Predictor features |
| `x80` | `int64` | 37 discrete codes (1 to 37) | 0 | **Discrete Industry Sector Code** | Categorical predictor feature |
| `x81` .. `x83` | `float64` | Continuous | 0 | Anonymized financial indicators | Predictor features |

### 5.2 Key Canonical Ratios in Dataset B (`data.csv`)

| Column Name | Observed Type | Typical Range | Missing | Direction of Risk | Economic Meaning |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Bankrupt?` | `int64` | {0, 1} | 0 | Target | Corporate bankruptcy declaration |
| `ROA(A) before interest and % after tax` | `float64` | [0.0, 1.0] | 0 | Lower = Higher Risk | Return on Total Assets |
| `Operating Gross Margin` | `float64` | [0.0, 1.0] | 0 | Lower = Higher Risk | Pricing power and gross profitability |
| `Current Ratio` | `float64` | [0.0, 2.75e9] | 0 | Lower = Higher Risk | Short-term debt coverage buffer |
| `Quick Ratio` | `float64` | [0.0, 9.23e9] | 0 | Lower = Higher Risk | Liquid assets to current liabilities |
| `Debt ratio %` | `float64` | [0.0, 1.0] | 0 | Higher = Higher Risk | Total debt burden relative to assets |
| `Borrowing dependency` | `float64` | [0.0, 1.0] | 0 | Higher = Higher Risk | Reliance on interest-bearing debt |
| `Working Capital to Total Assets` | `float64` | [0.0, 1.0] | 0 | Lower = Higher Risk | Operational liquidity headroom |
| `Net Income Flag` | `int64` | {1} | 0 | Neutral | Constant flag (dropped by preprocessor) |
