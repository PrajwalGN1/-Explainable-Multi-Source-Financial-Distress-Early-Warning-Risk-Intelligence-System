# 3–5 Minute Live Demonstration Script

**Project:** Explainable Multi-Source SME Financial Distress Early-Warning System  
**Audience:** Hiring Managers, Lead ML Engineers, Quantitative Risk Directors, Portfolio Evaluators  
**Goal:** Deliver a concise, technically rigorous, end-to-end demonstration showcasing machine learning rigor, model risk governance, and full-stack software packaging.

---

## Pre-Demo Checklist (60 Seconds Before Demo)

1. Open Terminal 1 and launch the Streamlit dashboard:
   ```powershell
   streamlit run app.py --server.port=8501
   ```
2. Open Terminal 2 and launch the FastAPI microservice:
   ```powershell
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```
3. Open browser tabs:
   - Tab 1: `http://localhost:8501` (Streamlit Dashboard)
   - Tab 2: `http://localhost:8000/docs` (FastAPI Swagger Interactive Documentation)

---

## Live Walkthrough Timeline (3–5 Minutes)

### Minute 0:00 – 1:00: The Problem & Executive Macro Surveillance
- **Action:** Open Tab 1 (`http://localhost:8501` - *Executive Overview*).
- **Speaking Script:**
  > *"Conventional corporate credit scoring predicts whether a company is bankrupt right now. In commercial lending, that's too late—recovery rates on insolvent borrowers collapse below 25%.*  
  > *This system is an early-warning risk intelligence engine designed to detect financial distress up to 12 months in advance across 422 monitored corporate enterprises.*  
  > *On this Executive Overview screen, you see macro portfolio telemetry: 422 firms, 14 currently classified as Critical, 34 as High Risk, and a calibrated portfolio distress rate of 5.6%.*  
  > *Notice that we don't use arbitrary percentile cuts; our risk bands are calibrated multiples of the empirical distress base rate."*

---

### Minute 1:00 – 2:00: Deep-Dive Company Risk Inspection & TreeSHAP
- **Action:** In the sidebar, select **🔍 Company Risk Analyzer**. Select **Firm #364** (or search via Custom Search).
- **Speaking Script:**
  > *"Let's drill down into an individual borrower—Firm 364.*  
  > *The gauge displays a calibrated forward distress probability of 28.5%, placing it in the Critical tier.*  
  > *Down here, the multi-dimensional radar chart reveals that while working capital appears adequate, solvency and cash-flow ratios are severely depleted.*  
  > *Under banking regulations like Federal Reserve SR 11-7, black-box scores are unacceptable. Here, our TreeSHAP explainer computes exact additive attributions directly from the decision trees.*  
  > *We see exactly which accounting ratios are driving risk—specifically, declining retained earnings ($x2$) and deteriorating cash flow ($x12$). Crucially, our system enforces a regulatory non-causal disclaimer: these are statistical predictive associations, not causal guarantees."*

---

### Minute 2:00 – 3:00: The 8% Operational Alert Budget & Model Comparison
- **Action:** In the sidebar, select **🔬 Research Model Improvement & Baseline Audit**. Click the **📊 Model Comparison & Budget Cap** tab.
- **Speaking Script:**
  > *"Now let's examine the core engineering challenge: the alert budget.*  
  > *In real-world credit surveillance, risk analysts can't investigate 1,000 flagged companies a month. A real bank operates under a capacity ceiling—typically $\le 8\%$ of the monitored book.*  
  > *On our matured out-of-time test cohort of 262 firms with 11 realized defaults, an 8% alert budget allows exactly 20 alerts.*  
  > *Look at this empirical comparison:*  
  > *- Our approved production baseline at its nominal threshold of 0.0800 detects 5 of 11 defaults, but it generates 37 alerts—blowing past the 20-alert operational budget.*  
  > *- When capped to the top 20 alerts ($\tau = 0.1072$), baseline recall adjusts to 4 of 11 (36.4%).*  
  > *- Our experimental cost-sensitive candidate achieves 3 of 11 within budget.*  
  > *Notice that we don't claim our candidate is 'statistically superior': with 11 realized events, the 95% Wilson confidence intervals overlap substantially. We maintain scientific integrity rather than exaggerating performance."*

---

### Minute 3:00 – 4:00: False-Negative Diagnostics: Acute vs. Chronic Distress
- **Action:** Click the **🚨 False Negative Audit & Dynamics** tab.
- **Speaking Script:**
  > *"We didn't just measure aggregate metrics—we audited every single missed default.*  
  > *Our audit revealed two distinct failure modes:*  
  > *1. Chronic deterioration: multi-period solvency decay where both models reliably flagged firms (like Firms 100, 101, and 112).*  
  > *2. Acute cliff crashes: companies like Firm 28 that appeared solvent at $T=13$ and suddenly collapsed at $T=14$.*  
  > *Interestingly, our approved baseline caught Firm 28, but our cost-sensitive candidate missed it. Why? Because the candidate's rolling temporal features smoothed away the sudden single-period drop.*  
  > *This demonstrates a fundamental credit modeling tradeoff: temporal smoothing improves chronic stability but dampens acute shock sensitivity."*

---

### Minute 4:00 – 5:00: FastAPI Microservice, Security & Test Suite
- **Action:** Switch to Tab 2 (`http://localhost:8000/docs`). Expand `/predict` and `/admin/reload-db`.
- **Speaking Script:**
  > *"Finally, the system is fully packaged for production serving via FastAPI.*  
  > *- `/predict` provides real-time calibrated scoring with strict Pydantic input validation.*  
  > *- `/company/{company_id}` returns full historical trajectories with sub-millisecond response from an in-memory singleton cache.*  
  > *- Administrative cache reloads (`/admin/reload-db`) fail closed with HTTP 503 if the server key is unset, and utilize constant-time verification (`secrets.compare_digest`) against timing attacks.*  
  > *The entire repository is backed by 226 automated pytest regression tests passing cleanly, and all 8 protected production model artifacts are cryptographically locked with SHA-256 hashes."*

---

## Anticipated Interview Questions & Rapid Answers

**Q1: Why not just use SMOTE or train a larger deep neural network?**  
> *"In credit risk panel data with only 422 unique firms, synthetic minority oversampling (SMOTE) creates synthetic balance sheets that violate accounting identities (Assets $\neq$ Liabilities + Equity) and introduce temporal lookahead. Tree ensembles with Platt calibration and cost-sensitive loss provide superior calibration and exact TreeSHAP interpretability without fabricating non-existent corporate entities."*

**Q2: Why is the production threshold 0.0800 instead of 0.5?**  
> *"Financial distress is a rare event (3.7% baseline rate). A default threshold of 0.50 assumes symmetric misclassification costs and an balanced prior. At 0.50, almost zero firms would be flagged. A threshold of 0.0800 represents roughly a $2\times$ base-rate odds multiplier, aligning model sensitivity with portfolio monitoring cost structures."*

**Q3: How do you prevent temporal data leakage?**  
> *"We enforce strict chronological out-of-time splits ($T \le 10$ train, $11-12$ val, $13 \to 14$ test). Preprocessors and scalers are fitted exclusively on the training partition. Lags and rolling windows only look backward in time. Forward labels are constructed strictly within individual firm trajectories ($T \to T+1$), and terminal observations without next-period data are excluded rather than assumed healthy."*
