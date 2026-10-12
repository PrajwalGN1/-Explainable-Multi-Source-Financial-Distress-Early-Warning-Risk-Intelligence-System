# Explainable Multi-Source Financial Distress Early-Warning & Risk Intelligence System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests Passing](https://img.shields.io/badge/tests-20%20passed-brightgreen.svg)]()

> **Academic/Research Early-Warning Prototype & Corporate Risk Intelligence Engine**  
> Synthesizing multi-period longitudinal panel dynamics, conventional cross-sectional bankruptcy patterns, and unsupervised out-of-distribution anomaly evidence into calibrated, explainable credit risk intelligence.

---

## 1. Overview & Problem Statement

Conventional corporate distress models operate under static cross-sectional paradigms: they observe a firm at period $T$ and predict whether liquidation occurs immediately. However, commercial banks, institutional credit underwriters, and regulatory authorities do not manage insolvency events after failure has materialized; they require an **Early Warning System (EWS)** capable of identifying **subtle financial deterioration trajectories quarters or years before default occurs**.

This system implements an end-to-end, production-grade academic research prototype that answers the fundamental question:

> **Research Question:**  
> *Does incorporating multi-period temporal deterioration dynamics and complementary multi-source risk evidence improve forward distress early-warning performance compared with conventional static cross-sectional bankruptcy classifiers?*

The system rejects naive row-wise merging (`pd.concat`) of heterogeneous corporate datasets and instead establishes a **scientifically grounded two-level risk fusion architecture**.

---

## 2. Multi-Source Dataset Topology & Provenance

The system integrates three complementary corporate finance datasets without fabricating labels or schemas:

| Dataset | File | Topology & Provenance | Observations & Shape | Target Variable | Primary Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dataset A** | `Financial Distress.csv` | Longitudinal Panel (Tehran Stock Exchange, 422 firms over 14 time periods) | 3,672 rows × 86 columns (`Company`, `Time`, `Financial Distress`, `x1`..`x83`) | Forward Distress ($t \to t+1$ event where distress $\le -0.5$) | **Core Temporal Early-Warning Model (EWS)** |
| **Dataset B** | `data.csv` | Cross-Sectional Panel (Taiwan Economic Journal, 1999–2009) | 6,819 rows × 96 columns (Canonical financial ratios) | `Bankrupt?` (Binary, 3.23% positive base rate) | **Structural Bankruptcy Likelihood Estimator** |
| **Dataset C** | `bankruptcy_Test_X.csv` | Unlabelled External Benchmark (Polish Industrial Companies) | 5,000 rows × 64 normalized attributes (`Attr1`..`Attr64`) | *Unlabelled* | **Isolation Forest Anomaly & Out-of-Distribution Layer** |

### Why Naive Concatenation is Scientifically Prohibited:
- **Disparate Populations:** Iranian listed equities (Dataset A) vs. Taiwanese manufacturing corporations (Dataset B) vs. Polish industrial firms (Dataset C).
- **Incompatible Feature Definitions:** `x1`..`x83` in Dataset A represent masked, unstandardized company-specific indicators, whereas Dataset B contains named ratios with distinct accounting conventions.
- **Divergent Temporal Topologies:** Dataset A is a longitudinal panel with repeated observations per company over time; Dataset B is cross-sectional.
- **Different Target Horizons:** Immediate bankruptcy status vs. continuous distress index.

---

## 3. System Architecture

```mermaid
flowchart TD

    subgraph Data Sources [Heterogeneous Corporate Topologies]
        DA[Dataset A: Tehran Panel<br/>3,672 rows | 422 Firms]
        DB[Dataset B: Taiwan TEJ<br/>6,819 rows | 96 Ratios]
        DC[Dataset C: Polish Test<br/>5,000 rows | 64 Attrs]
    end

    subgraph Feature Engines [Leakage-Free Feature Engineering]
        DA --> FE_A[Temporal Feature Engine<br/>Lags, Deltas, Rolling Stats, Trends]
        DA --> TGT[Forward Target Builder<br/>Horizon t to t+1]
        DB --> FE_B[Ratio Verification & Scaling]
        DC --> FE_C[Unsupervised OOD Profiler]
    end

    subgraph Model Layer [Source Expert Models]
        FE_A & TGT --> M_EWS[Model 1: Temporal EWS<br/>Random Forest Classifiers]
        FE_B --> M_BNK[Model 2: Bankruptcy Model<br/>Cost-Sensitive Random Forest]
        FE_C --> M_ANO[Model 3: Anomaly Engine<br/>Isolation Forest]
    end

    subgraph Risk Fusion [Two-Level Risk Fusion Engine]
        M_EWS -->|OOF EWS Prob| FUSE[Stacking Meta-Learner<br/>Logistic Stacking & Platt Calibration]
        M_BNK -->|OOF Bankrupt Prob| FUSE
        M_ANO -->|Normalized Anomaly| FUSE
        DA -->|Subscores| HEALTH[Interpretable Health Engine<br/>Profit, Liq, Solv, Lev, WC]
        HEALTH -->|Health Risk| FUSE
    end

    subgraph Output Intelligence [Surveillance Intelligence Layer]
        FUSE --> PROB[Calibrated Risk Probability %]
        FUSE --> SCORE[0-100 Risk Score]
        FUSE --> CAT[Calibrated Risk Category<br/>Healthy / Low / Watch / High / Critical]
        FUSE --> TRAJ[Longitudinal Trajectory Vector<br/>Slope, Delta pp, Acceleration]
        FUSE --> SHAP_EXP[TreeSHAP & Delta Explanation<br/>Risk Drivers vs Protective Factors]
        FUSE --> ALERTS[Deterministic Early-Warning Alerts]
        FUSE --> RECS[Actionable Surveillance Protocol]
    end

    subgraph Delivery [User Interface & Endpoints]
        PROB & SCORE & CAT & TRAJ & SHAP_EXP & ALERTS & RECS --> DASH[Streamlit Analytics Dashboard<br/>10 Integrated Diagnostic Views]
        PROB & SCORE & CAT & RECS --> API_SVC[FastAPI REST Microservice<br/>Pydantic Validated Endpoints]
    end
```

---

## 4. Leakage-Free Temporal Pipeline

Temporal leakage occurs when models use future information to predict past events. This system enforces strict mathematical safeguards:

1. **Chronological Time-Aware Splitting:**
   - **Training Set:** Periods 1 through 10 (2,889 company-periods)
   - **Validation Set:** Periods 11 and 12 (521 company-periods)
   - **Out-of-Time Test Set:** Periods 13 and 14 (262 company-periods)
2. **Strictly Backward-Looking Statistics:**
   - All lag features ($lag_1, lag_2$), deltas ($x_t - x_{t-1}$), rolling means ($k=2, 3$), and rolling standard deviations reference only observations $\le t$.
3. **Forward-Shifted Target Construction:**
   - Observations at time $t$ predict distress at $t+1$:
     $$y_{i, t}^{\text{forward}} = \mathbb{I}(\text{Financial Distress}_{i, t+1} \le -0.5)$$
4. **Isolated Preprocessor Transformations:**
   - Imputation medians and scaling parameters are fitted exclusively on the training split and transformed forward onto validation and test splits.

---

## 5. Empirical Ablation Study & Research Results

To answer the core research question, 6 rigorous experimental architectures were trained and evaluated on the out-of-time test partition (Periods 13–14):

| Experiment | Architecture Description | Feature Count | Test ROC-AUC | Test PR-AUC | Test Brier Score | Key Finding |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Exp 1** | Static Financial Features Only (Baseline) | 83 | 0.7787 | 0.1353 | 0.0632 | Benchmark baseline |
| **Exp 2** | Static Features + Lag-1 & Lag-2 | 99 | 0.7428 | 0.1373 | 0.0654 | Marginal PR-AUC gain |
| **Exp 3** | Static Features + Lags + Difference Deltas | 123 | **0.7983** | 0.1729 | 0.0660 | +27.8% PR-AUC gain over baseline |
| **Exp 4** | Full Temporal EWS (Lags + Trends + Rolling + Deterioration Streaks) | 163 | 0.7783 | **0.1946** | 0.0626 | **+43.8% PR-AUC gain** (Best single model) |
| **Exp 5** | Multi-Source Weighted Baseline Fusion (EWS + TEJ + Anomaly + Health) | 166 | 0.6403 | 0.1702 | 0.0419 | Sharp calibration improvement |
| **Exp 6** | **Full Stacking Meta-Learner Risk Intelligence (Out-of-Fold Logistic)** | 167 | 0.6588 | 0.1294 | **0.0397** | **Optimal Calibration (Lowest Brier Score)** |

### Probability Calibration Impact:
- **Temporal EWS (Random Forest):** Test Brier score improved from **0.0626** (raw) to **0.0387** via Platt scaling (**38.07% calibration gain**).
- **Bankruptcy Model (Random Forest):** Test Brier score improved from **0.0419** to **0.0234** (**44.23% calibration gain**). Test ROC-AUC: **0.9455**, Test Precision: **72.22%**, Specificity: **99.62%**.

---

## 6. Risk Scoring & Calibrated Categories

Because corporate distress is a rare-event tail risk (3.7% historical event rate in the panel), linear percentile thresholds would falsely mark severely deteriorating firms as "healthy." The system implements calibrated thresholds aligned with portfolio baseline multiples:

| Risk Category | Score Range (0–100) | Probability Range | Portfolio Multiple | Surveillance Action | Priority Flag |
| :--- | :---: | :---: | :--- | :--- | :---: |
| **Healthy** | 0.0 – 4.0 | 0.0% – 4.0% | $< 1.0\times$ Baseline | Standard annual review protocol | P4 |
| **Low Risk** | 4.01 – 8.0 | 4.01% – 8.0% | $1.0\times - 2.0\times$ | Standard semi-annual monitoring | P4 |
| **Watch** | 8.01 – 13.0 | 8.01% – 13.0% | $2.0\times - 3.5\times$ | Quarterly financial surveillance | P3 |
| **High Risk** | 13.01 – 17.0 | 13.01% – 17.0% | $3.5\times - 4.5\times$ | Monthly review; liquidity audit | P2 |
| **Critical** | 17.01 – 100.0 | $> 17.0\%$ | $> 4.5\times$ Baseline | Immediate credit freeze; escalation | **P1** |

---

## 7. Streamlit Dashboard (10 Specialized Views)

The web dashboard is designed with dark mode aesthetics (Inter typography, glassmorphism containers, responsive Plotly charts, and caching):

1. **🏛️ Executive Overview:** Macro portfolio KPIs, calibrated category donut chart, longitudinal trajectory distribution, and top 10 urgent watchlist preview.
2. **🔍 Company Risk Analyzer:** Single-firm inspection, radial gauge indicator, multi-source evidence cards, active alerts, and surveillance recommendations.
3. **📈 Risk Timeline & Trajectory:** Interactive multi-period Plotly chart tracking risk probability over historical time intervals with threshold bands.
4. **🩺 Financial Health Diagnostics:** Dimensional subscores (Profitability, Liquidity, Solvency, Leverage, Working Capital) and spider/radar chart against industrial benchmarks.
5. **🧠 Explainable AI & SHAP:** Local TreeSHAP waterfall plot decomposing positive risk drivers vs. protective factors, and global feature importance.
6. **🔄 Why Did Risk Change?** Period-over-period comparative delta decomposition, metric shifts table, and automated natural-language narrative.
7. **⚡ Multi-Source Evidence Fusion:** Transparent evidence breakdown and meta-model stacking weights.
8. **📋 Surveillance Watchlist:** Searchable, filterable corporate table with CSV download.
9. **📊 Model Validation & Ablation:** Empirical ablation study tables, ROC-AUC vs. PR-AUC charts, and leakage prevention protocol.
10. **📄 Risk Dossier & Export:** Comprehensive audit report exportable to JSON or Markdown.

---

## 8. FastAPI Service Endpoints

The service includes an asynchronous FastAPI backend validated with Pydantic schemas:

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/health` | `GET` | System heartbeat, uptime, and version status |
| `/model-info` | `GET` | Architecture specifications and validation metrics for all 3 data sources |
| `/predict` | `POST` | Real-time multi-source inference returning calibrated score, category, and actions |
| `/company/{company_id}` | `GET` | Full historical dossier, trajectory history, and SHAP drivers for a specific firm |
| `/watchlist` | `GET` | Prioritized surveillance table with configurable entity limits |
| `/risk-summary` | `GET` | Universe-wide macro risk metrics and count of critical/high-risk firms |

---

## 9. Installation & Local Execution

### Prerequisites
- Python 3.10+
- Anaconda / Miniconda (recommended)

### Setup Instructions
```bash
# 1. Clone repository
git clone https://github.com/example/financial-distress-ews.git
cd financial-distress-ews

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run automated test suite (20 unit and leakage tests)
python -m pytest tests/ -v

# 5. Launch Streamlit Analytics Dashboard
streamlit run app.py

# 6. Launch FastAPI REST Service (optional, in separate terminal)
uvicorn api:app --reload --port 8000
```

---

## 10. Container Deployment (Docker)

```bash
# Build and run using Docker Compose (Streamlit on :8501, FastAPI on :8000)
docker-compose up --build
```

---

## 11. Project Architecture

```text
financial-distress-ews/
├── data/
│   ├── raw/
│   │   ├── Financial Distress.csv      # Dataset A (Longitudinal panel)
│   │   ├── data.csv                    # Dataset B (Taiwan TEJ bankruptcy)
│   │   └── bankruptcy_Test_X.csv       # Dataset C (Polish unlabelled test)
│   └── processed/
│       ├── companies_risk_intelligence.json # Comprehensive risk database
│       └── watchlist.csv               # Prioritized corporate watchlist
├── models/
│   ├── ews/                            # Trained Temporal EWS models & metrics
│   ├── bankruptcy/                     # Trained Bankruptcy models & metrics
│   ├── anomaly/                        # Isolation Forest model & OOD metrics
│   └── fusion/                         # Stacking meta-model & calibration
├── reports/
│   ├── data_audit/                     # Automated data profiles & summaries
│   ├── eda/                            # Exploratory analysis & distributions
│   ├── model_evaluation/               # Ablation study & experiment results
│   └── explainability/                 # Global TreeSHAP feature importances
├── src/
│   ├── config/                         # YAML settings & risk configuration
│   ├── data/                           # Loaders, validators, & audit scripts
│   ├── preprocessing/                  # Robust scalers & leakage-safe imputers
│   ├── features/                       # Temporal engine & financial health engine
│   ├── models/                         # Model architectures & Platt calibrators
│   ├── fusion/                         # Two-level stacking meta-learner
│   ├── explainability/                 # TreeSHAP explainer & delta analyzer
│   ├── risk_engine/                    # Trajectory, deterministic alerts, recs
│   └── utils/                          # Logging & JSON serialization helpers
├── dashboard/
│   └── components/                     # Plotly chart builders & styled KPI cards
├── tests/
│   ├── test_leakage_prevention.py      # Automated temporal leakage tests
│   ├── test_features_and_health.py     # Feature engine & health score tests
│   ├── test_fusion_and_scoring.py      # Fusion engine & alert rules tests
│   ├── test_api.py                     # FastAPI endpoint validation tests
│   └── conftest.py                     # Pytest environment setup
├── app.py                              # Streamlit Analytics Dashboard
├── api.py                              # FastAPI REST microservice
├── requirements.txt                    # Production Python dependencies
├── Dockerfile                          # Container specification
├── docker-compose.yml                  # Multi-service container orchestration
├── LICENSE                             # MIT Open Source License
└── README.md                           # Master technical documentation
```

---

## 12. Research Limitations & Academic Disclaimer

1. **Academic/Research Nature:** This system is an academic research prototype and portfolio project. It does **not** constitute an autonomous banking credit-decision system, automated loan origination engine, or formal financial investment advisory.
2. **Public Dataset Limitations:** Datasets originate from public academic repositories (Tehran Stock Exchange, Taiwan Economic Journal, Polish Companies). They do not reflect proprietary banking deposit transaction velocities, intraday overdraft utilization, or GST/e-invoice tax flows.
3. **Correlation vs. Causation:** Model feature attributions indicate statistical associations with financial deterioration patterns, not deterministic causal mechanisms.
4. **Anomaly Scores as OOD Evidence:** Unsupervised Isolation Forest scores quantify distributional distance and structural outlier status, not mathematical default probabilities.
