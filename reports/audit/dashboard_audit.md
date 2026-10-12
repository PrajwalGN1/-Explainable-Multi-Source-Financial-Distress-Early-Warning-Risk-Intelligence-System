# Streamlit Analytics Dashboard & User Experience Audit Report

**Project:** Explainable SME Financial Distress Early-Warning & Risk Intelligence System  
**Audit Scope:** PHASE 9 — Audit the Dashboard and User Experience  
**Evidence Source:** `app.py` (1,067 lines), `dashboard/components/charts.py`, `dashboard/components/kpi_cards.py`, `Data/processed/companies_risk_intelligence.json`, `Data/processed/watchlist.csv`, and smoke test execution logs.

---

## 1. Executive Summary: Dashboard & UX Audit

The user interface is an interactive, dark-mode web application built with Streamlit and Plotly, delivering institutional credit risk intelligence across **10 specialized diagnostic views**.

### Summary of Strengths:
- **Rich Aesthetic Design:** Implements high-quality dark mode styling (`#0b0f19` background, Inter typography, glassmorphism containers, custom status badges, responsive Plotly charts).
- **Comprehensive Functionality:** Successfully implements 10 distinct surveillance workflows with seamless company switching across 422 monitored enterprises.
- **Data-Driven Core:** Monitored corporate profiles, historical timelines, alert lists, and watchlists are loaded from real, empirical model outputs rather than random placeholder generators.
- **Fast Interactive Response:** Uses `@st.cache_data` to load precomputed risk dossiers into memory, enabling near-instantaneous navigation between views.

### Critical Defects & Technical Debt Identified:
1. **Visual Gauge & Timeline Inconsistency (Critical UX Defect):** Chart background bands are hardcoded as linear 20-point intervals (`0–20 Healthy`, `20–40 Low Risk`, etc.). This causes the highest-risk firms (scores 17–19, classified as Critical) to be plotted visually inside the green "Healthy" zone.
2. **Streamlit API Deprecation Warnings:** 58 instances of `use_container_width=True` trigger deprecation warnings (`use_container_width will be removed after 2025-12-31; replace with width='stretch'`).
3. **Hardcoded Peer Benchmarks:** In View 4 (Financial Health Diagnostics), the "Benchmark Median" spider polygon uses hardcoded integers `[70, 75, 70, 75, 65]` rather than real sector medians computed from Dataset A.
4. **Memory Footprint:** The application loads a monolithic 27 MB JSON file into memory on startup. While acceptable for a 422-firm research prototype, this will not scale to a production universe of 50,000+ SMEs.

---

## 2. In-Depth Audit of the 10 Diagnostic Views

### View 1: 🏛️ Executive Overview
- **Visual Structure:** Header banner, 5 KPI cards (Universe Size: 422, Critical Alerts: 14, High Risk Firms: 34, Mean Risk: 5.6, Active Signals: 466), calibrated category donut chart, longitudinal trajectory distribution, and urgent watchlist preview table.
- **Data Grounding:** Metrics reflect actual data in `watchlist.csv` and `companies_risk_intelligence.json`.
- **UX Assessment:** Strong visual layout and immediate situational awareness.
- **Defects:** None in data sourcing; donut chart color map correctly reflects risk tiers.

---

### View 2: 🔍 Company Risk Analyzer
- **Visual Structure:** Company header with status badges, radial risk gauge, multi-source evidence breakdown cards, active deterministic early-warning alerts, and surveillance recommendations.
- **Company Selector:** Formatted dropdown in sidebar allowing selection of any of the 422 firms. Defaults to Firm #364 (the top watchlist firm).
- **Critical Defect:** Radial gauge chart (`create_gauge_chart`) displays Firm #364 (score 19.1) inside the green step (0–20) while the title reads "Critical".

---

### View 3: 📈 Risk Timeline & Trajectory
- **Visual Structure:** 4 trajectory KPI cards (Trend, Period Delta pp, Risk Acceleration, Consecutive Increases), interactive multi-period Plotly timeline chart with threshold bands and alert marker stars, and historical reporting period breakdown table.
- **Data Grounding:** Traces genuine historical observations across all available reporting periods for the selected firm (e.g., 8 periods for Firm #364, 14 periods for healthy survivors).
- **Critical Defect:** Background threshold rectangles (`add_hrect`) in `create_timeline_chart` are hardcoded to 0–20 (Healthy), 20–40 (Low Risk), 40–60 (Watch), 60–80 (High), 80–100 (Critical). Firm #364's trajectory never rises above 20%, visually suggesting the company remained healthy throughout its history.

---

### View 4: 🩺 Financial Health Diagnostics
- **Visual Structure:** Multi-dimensional subscore breakdown (Profitability, Liquidity, Solvency, Leverage, Working Capital), composite health score card, and 5-axis spider/radar chart.
- **Scientific Defect:** Subscores are generated using arbitrary sigmoid formulas on unmasked variables `x48, x25, x64, x81`. The radar chart displays canonical accounting labels that have no verified connection to the masked dataset variables.
- **Hardcoding Defect:** The comparison line "Benchmark Median" is hardcoded to `[70, 75, 70, 75, 65]`.

---

### View 5: 🧠 Explainable AI & SHAP
- **Visual Structure:** Expected value baseline card, local waterfall plot decomposing positive risk drivers vs. protective factors, and detailed feature attribution breakdown table.
- **Defect:** Attributions are computed via the fallback heuristic `feature_row * norm_importances`, not true TreeSHAP. Feature names `x48, x25, x64, x81` display fabricated accounting interpretations.
- **Plotting:** Waterfall chart utilizes Plotly `go.Waterfall`, accurately visualizing positive increments in red and negative decrements in green.

---

### View 6: 🔄 Why Did Risk Change?
- **Visual Structure:** Delta KPI cards (Previous Period Risk %, Current Period Risk %, Net Shift pp), comparative metric shifts table, and automated natural-language narrative.
- **Analytical Value:** High. Directly answers the risk committee's question regarding rating trajectory changes.
- **Defect:** Narratives cite fabricated accounting labels for masked indicators.

---

### View 7: ⚡ Multi-Source Evidence Fusion
- **Visual Structure:** Stacking meta-model coefficient table, radar comparison of multi-source inputs, and evidence breakdown progress bars.
- **Defect:** Displays coefficients where `anomaly_score` is negative (-0.9285). The displayed "Bankruptcy Probability" and "Anomaly Score" for Dataset A firms are synthetic proxies computed on `x64` and `x81`.

---

### View 8: 📋 Surveillance Watchlist
- **Visual Structure:** Interactive filter controls (Risk Category multi-select, Priority Tier multi-select, Minimum Risk Score slider), prioritized corporate table, and CSV download button.
- **Functionality:** Filtering is responsive and accurate. CSV export works cleanly.
- **Defect:** None. Functionality is verified and robust.

---

### View 9: 📊 Model Validation & Ablation
- **Visual Structure:** Empirical ablation study table (Experiments 1–6), ROC-AUC vs. PR-AUC bar charts, calibration reliability curves, and leakage prevention protocol checklist.
- **Data Grounding:** Sourced directly from `reports/model_evaluation/ablation_study.csv` and `models/ews/ews_metrics.json`.
- **Defect:** Shows Experiment 6 as the "Best Calibrated" architecture without highlighting that test discrimination collapsed and recall was 0.0%.

---

### View 10: 📄 Risk Dossier & Export
- **Visual Structure:** Full corporate audit report, key risk metrics, active alerts, surveillance history, and export controls (Download JSON Dossier, Download Markdown Summary).
- **Functionality:** Both JSON and Markdown download buttons generate well-formatted client dossiers.

---

## 3. UI Component Code Audit

### 3.1 `dashboard/components/charts.py`
- Contains 4 core Plotly charting functions: `create_gauge_chart`, `create_timeline_chart`, `create_health_radar_chart`, `create_waterfall_chart`.
- **Required Correction in `create_gauge_chart`:**
  Replace hardcoded steps `[0, 20], [20, 40], ...` with dynamic steps sourced from `risk_config.yaml`:
  - Healthy: `[0, 4.0]`
  - Low Risk: `[4.01, 8.0]`
  - Watch: `[8.01, 13.0]`
  - High Risk: `[13.01, 17.0]`
  - Critical: `[17.01, 100.0]`
- **Required Correction in `create_timeline_chart`:**
  Replace `add_hrect` boundaries `[0, 20], [20, 40], ...` with `[0, 4], [4, 8], [8, 13], [13, 17], [17, 100]`.

### 3.2 `dashboard/components/kpi_cards.py`
- Clean HTML/CSS string builders for glassmorphism metric cards and status badges.
- Free of defects.

---

## 4. Modernization & Engineering Recommendations

1. **Fix Deprecated Streamlit Syntax:** Replace all occurrences of `use_container_width=True` with `width="stretch"` to ensure compatibility with Streamlit releases beyond 2025.
2. **Synchronize Visual Chart Steps:** Wire `charts.py` directly to `get_risk_config()` so all chart intervals reflect calibrated portfolio baseline multiples.
3. **Compute Real Sector Benchmarks:** In View 4, calculate empirical medians by grouping Dataset A by industry sector code `x80` rather than using hardcoded values.
4. **Optimize Database Access:** Replace the 27 MB monolithic JSON load with a lightweight SQLite database or indexed parquet partitions for production scale.
