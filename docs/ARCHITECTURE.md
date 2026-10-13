# System Architecture & Technical Specification

**Project:** Explainable Multi-Source SME Financial Distress Early-Warning System  
**Version:** 1.0.0-release  
**Domain:** Computational Corporate Finance, SME Credit Surveillance & Model Risk Governance  
**Governance Standard:** US Federal Reserve SR 11-7 / OCC 2011-12 Supervisory Guidance on Model Risk Management  

---

## 1. High-Level System Architecture

The system is architected as an end-to-end corporate risk intelligence engine. It transforms raw multi-period accounting panels into calibrated, interpretable forward early-warning alerts without lookahead leakage.

```mermaid
flowchart TD
    subgraph Data_Tier ["Data Tier: Contiguous Historical Panel"]
        D_RAW["Dataset A: Tehran SME Financial Distress Panel<br/>3,672 observations • 422 unique enterprises • 14 annual periods"]
        D_SCHEMA["Schema Validation & Contiguity Audit<br/>0 missing • 0 infinite • 0 time gaps • 100% terminal exits"]
        D_RAW --> D_SCHEMA
    end

    subgraph Feature_Tier ["Feature Tier: Leakage-Free Temporal Pipeline"]
        T_TARGET["Forward Target Builder<br/>Y(i,t) = 1 if FD(i,t+1) &le; -0.50 else 0"]
        T_ENGINE["Temporal Feature Engine<br/>Historical Lags (L=1, 2) • Backward Rolling Means (k=2, 3) • Deltas"]
        T_PREP["EWSPreprocessor (Frozen)<br/>Train-only median imputation & RobustScaler (T &le; 10)"]
        D_SCHEMA --> T_TARGET
        D_SCHEMA --> T_ENGINE
        T_ENGINE --> T_PREP
    end

    subgraph Modeling_Tier ["Modeling Tier: Protected Baseline and Evaluation"]
        M_PROD["Approved Production Baseline (Frozen)<br/>Balanced Random Forest (100 trees, depth 5)<br/>SHA-256: fcec8814... • Nominal tau = 0.0800"]
        M_CALIB["Platt Sigmoid Calibrator (Frozen)<br/>Validation-fitted (T in 11, 12)<br/>SHA-256: 774a411b..."]
        M_RESEARCH["Research Benchmark Candidates (Isolated)<br/>- Candidate Cost-Sensitive RF (tau=0.0894)<br/>- Calibrated Logistic Regression (tau=0.0271)<br/>- Standard RF (tau=0.0680)"]
        T_PREP --> M_PROD
        M_PROD --> M_CALIB
        T_PREP -.-> M_RESEARCH
    end

    subgraph Explainability_Tier ["Explainability Tier: TreeSHAP Attribution"]
        SHAP_TREE["TreeSHAP Exact Tree Explainer<br/>Exact Additivity: base_value + sum(shap_values) == f(x)"]
        SHAP_DRIVERS["Top Attribution Drivers<br/>x2: Retained Earnings / Assets • x12: Cash Flow Ratio • x8: Liquidity"]
        SHAP_GOV["Regulatory Non-Causal Semantics<br/>Statistical predictive association &ne; causal counterfactuals"]
        M_PROD --> SHAP_TREE
        SHAP_TREE --> SHAP_DRIVERS
        SHAP_DRIVERS --> SHAP_GOV
    end

    subgraph Delivery_Tier ["Delivery Tier: Client Applications and Services"]
        UI_STREAMLIT["Streamlit Analytics Dashboard (:8501)<br/>12 interactive views • 422 monitored enterprises • Dark-mode glassmorphism"]
        API_FASTAPI["FastAPI REST Microservice (:8000)<br/>11 endpoints • Pydantic validation • Fail-closed admin security"]
        M_CALIB --> UI_STREAMLIT
        M_CALIB --> API_FASTAPI
        SHAP_TREE --> UI_STREAMLIT
        SHAP_TREE --> API_FASTAPI
    end
```

---

## 2. Temporal Data Splitting & Leakage Prevention

Temporal leakage occurs when future information contaminates training features or targets. This architecture enforces strict chronological boundaries:

```mermaid
gantt
    title Chronological Panel Partitioning Strategy
    dateFormat X
    axisFormat T=%s
    section Training Split
    Periods 1 to 10 (2,471 evaluable rows, 95 defaults) :active, 1, 10
    section Validation Split
    Periods 11 to 12 (517 evaluable rows, 14 defaults) :crit, 11, 12
    section Holdout Cohort
    Period 13 predicting Period 14 (262 rows, 11 defaults) :done, 13, 13
    section Terminal Exit
    Period 14 Terminal Exit (262 rows, unobserved t+1) :milestone, 14, 14
```

### Mathematical Invariants:
1. **Intra-Firm Forward Labeling:**  
   $$Y_{i, t} = \begin{cases} 1 & \text{if } \text{Company}_{i, t+1} == \text{Company}_{i, t} \text{ and } \text{Financial Distress}_{i, t+1} \le -0.50 \\ 0 & \text{if } \text{Company}_{i, t+1} == \text{Company}_{i, t} \text{ and } \text{Financial Distress}_{i, t+1} > -0.50 \\ \text{NaN} & \text{if } t \text{ is the terminal observed period for company } i \end{cases}$$
2. **Preprocessor Isolation:**  
   $$\mu_{\text{train}}, \sigma_{\text{train}} = \text{fit}(X_{\text{train}} \mid T \le 10)$$
   $$X_{\text{test}}^{\text{scaled}} = \frac{X_{\text{test}} - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
   Parameters are frozen and never updated during test transformation.
3. **Calibrator Isolation:**  
   Platt scaling sigmoid parameters $A, B$ are estimated exclusively on $X_{\text{val}}, Y_{\text{val}}$ ($T \in [11, 12]$) via maximum likelihood. Holdout test labels ($T=13 \to 14$) are strictly withheld.

---

## 3. Two-Level Risk Fusion Pipeline

To prevent synthetic leakage while providing multi-dimensional evidence, the system fuses temporal risk signals with structural ratio health:

```mermaid
flowchart LR
    subgraph Signal_Inputs ["Signal Inputs"]
        S1["Temporal EWS Signal<br/>Calibrated Probability P_ews in 0 to 1"]
        S2["Structural Bankruptcy Likelihood<br/>Calibrated Probability P_bnk in 0 to 1"]
        S3["Out-of-Distribution Anomaly Score<br/>Isolation Forest Normalized Score in 0 to 1"]
        S4["Composite Financial Health Score<br/>Dimensionally Weighted Index (0-100)"]
    end

    subgraph Fusion_Engine ["Two-Level Fusion Engine"]
        F1["Baseline Weighted Risk Synthesis<br/>Rule-Governed Multi-Criteria Scoring"]
        F2["Stacking Meta-Learner Layer<br/>L2 Logistic Stacking on Out-of-Fold Splits"]
        S1 --> F1
        S2 --> F1
        S3 --> F1
        S4 --> F1
        S1 --> F2
        S2 --> F2
        S3 --> F2
        S4 --> F2
    end

    subgraph Intelligence_Outputs ["Intelligence Outputs"]
        OUT_P["Final Calibrated Risk Probability %"]
        OUT_S["Composite Risk Score (0 - 100)"]
        OUT_C["Calibrated Risk Category<br/>Healthy • Low • Watch • High • Critical"]
        OUT_A["Deterministic Early-Warning Action Protocol"]
        F1 --> OUT_P
        F1 --> OUT_S
        F1 --> OUT_C
        F1 --> OUT_A
        F2 --> OUT_P
        F2 --> OUT_S
        F2 --> OUT_C
        F2 --> OUT_A
    end
```

---

## 4. TreeSHAP Attribution Architecture

Global and local explainability is powered by Lundberg et al. (Nature Machine Intelligence, 2020) TreeSHAP exact tree attribution:

$$\mathbb{E}[f(X)] + \sum_{j=1}^{M} \phi_j(x) = f(x)$$

```mermaid
flowchart TD
    INPUT_ROW["Company Balance Sheet Record at Period T (Vector x)"]
    RF_FOREST["Balanced Random Forest (100 Decision Trees)"]
    INPUT_ROW --> RF_FOREST
    
    subgraph TreeSHAP_Computation ["TreeSHAP Exact Computation"]
        TREES["Traverse 100 Tree Graph Topologies"]
        VALS["Recursive Conditional Expectation via Algorithm 1"]
        TREES --> VALS
    end
    RF_FOREST --> TREES
    
    subgraph Local_Attribution ["Local Attribution Breakdown"]
        POS["Positive Risk Drivers (phi_j &gt; 0)<br/>Deteriorating Solvency, Negative Cash Flows"]
        NEG["Protective Balance Sheet Buffers (phi_j &lt; 0)<br/>Healthy Retained Earnings, Current Asset Liquidity"]
        VALS --> POS
        VALS --> NEG
    end
    
    GOV_ASSERT["SR 11-7 Compliance Guardrail:<br/>Attributions certified as statistical predictive associations, not causal proofs."]
    POS --> GOV_ASSERT
    NEG --> GOV_ASSERT
```

---

## 5. Security & Administrative Access Architecture

Administrative operations (e.g., database cache invalidation via `/admin/reload-db`) are secured with a strict fail-closed security boundary:

```mermaid
sequenceDiagram
    autonumber
    actor Client as Surveillance Administrator
    participant Gateway as FastAPI Router
    participant Auth as Constant-Time Authenticator
    participant Cache as In-Memory Database Singleton

    Client->>Gateway: POST /admin/reload-db (Header: X-Admin-Key)
    alt ADMIN_API_KEY environment variable is unset on server
        Gateway-->>Client: HTTP 503 Service Unavailable (Fail Closed)
    else Missing X-Admin-Key header
        Gateway-->>Client: HTTP 401 Unauthorized
    else Invalid X-Admin-Key header
        Gateway->>Auth: secrets.compare_digest(provided_key, server_key)
        Auth-->>Gateway: False (constant-time verification)
        Gateway-->>Client: HTTP 403 Forbidden
    else Valid X-Admin-Key header
        Gateway->>Auth: secrets.compare_digest(provided_key, server_key)
        Auth-->>Gateway: True
        Gateway->>Cache: clear_db_cache() & load_db(force_reload=True)
        Cache-->>Gateway: Reload complete (422 companies in memory)
        Gateway-->>Client: HTTP 200 OK {"status": "reloaded", "companies": 422}
    end
```

---

## 6. Directory Structure & Component Mapping

```text
financial-distress-ews/
├── data/
│   ├── raw/
│   │   ├── Financial Distress.csv      # Primary panel (3,672 rows, 422 firms)
│   │   ├── data.csv                    # Cross-sectional bankruptcy panel
│   │   └── bankruptcy_Test_X.csv       # Unlabelled industrial benchmark
│   └── processed/
│       ├── companies_risk_intelligence.json # Pre-warmed 422-firm surveillance cache
│       └── watchlist.csv               # Top-50 priority surveillance watchlist
├── models/
│   └── phase2/                         # Cryptographically protected baseline
│       ├── best_ews_model.joblib       # Approved Random Forest (SHA-256: fcec8814...)
│       ├── ews_calibrator.joblib       # Approved Sigmoid Calibrator (SHA-256: 774a411b...)
│       ├── ews_preprocessor.joblib     # Approved Robust Scaler (SHA-256: 2342eb7f...)
│       └── threshold_config.json       # Production operating threshold (0.0800)
├── src/
│   ├── data/loaders.py                 # Multi-source dataset ingestion & schema validation
│   ├── preprocessing/preprocessors.py  # Leakage-safe EWSPreprocessor & imputer
│   ├── features/temporal_features.py   # Lags, rolling statistics, and forward targets
│   ├── explainability/                 # TreeSHAP explainer & local waterfall generators
│   └── fusion/                         # Stacking meta-learner & composite health index
├── dashboard/                          # Plotly visualization components & dark-mode CSS
├── tests/                              # Automated pytest suite (226 tests)
├── app.py                              # Streamlit 12-page risk analytics dashboard
├── api.py                              # FastAPI 11-endpoint REST microservice
├── Dockerfile                          # Production container specification
├── docker-compose.yml                  # Multi-service dashboard + API compose
├── requirements.txt                    # Pinned Python dependencies
└── README.md                           # Master technical documentation
```
