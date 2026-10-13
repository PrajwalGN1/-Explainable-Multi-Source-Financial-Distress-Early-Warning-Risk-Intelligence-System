# Phase 6.6 — Alternative Financial Risk Signals & Point-in-Time Data Feasibility Study Report

**Author / Roles:** Senior Financial Risk Data Scientist, SME Credit Risk Researcher, Financial Data Engineer, and Model Risk Validator  
**Governance Standards:** Federal Reserve SR 11-7 / OCC 2011-12 Model Risk Management / Basel III Pillar 2 / Fair Credit Reporting Act (FCRA)  
**Repository Branch:** `research/phase6.6-alternative-risk-signals`  
**Base Commit:** `7046c25` (Phase 6.5 Completion)  
**Research-Gate Decision:** `ALTERNATIVE_SIGNAL_DATA_INFEASIBLE`  
**Production Status:** Approved Production Baseline Frozen & Unchanged; Candidate Quarantined; Promotion Strictly Prohibited  

---

## Deliverables Index

The following reproducible JSON artifacts and test suites were generated and verified during Phase 6.6:

1. [`results/phase6_6/production_integrity_manifest.json`](file:///d:/Financial%20Distress/results/phase6_6/production_integrity_manifest.json) — Cryptographic SHA-256 digest audit of all 8 protected production artifacts.
2. [`results/phase6_6/data_source_feasibility.json`](file:///d:/Financial%20Distress/results/phase6_6/data_source_feasibility.json) — Detailed feasibility audit matrix evaluating 19 candidate signals across Families A, B, C, and D.
3. [`results/phase6_6/point_in_time_feature_registry.json`](file:///d:/Financial%20Distress/results/phase6_6/point_in_time_feature_registry.json) — Formal Point-in-Time specifications, timestamp definitions, reporting delays, and leakage rules.
4. [`results/phase6_6/data_coverage_and_missingness.json`](file:///d:/Financial%20Distress/results/phase6_6/data_coverage_and_missingness.json) — Repository-wide storage census and quantitative modality gap audit.
5. [`results/phase6_6/sudden_transition_signal_audit.json`](file:///d:/Financial%20Distress/results/phase6_6/sudden_transition_signal_audit.json) — Forensic statistical analysis of sudden distress transitions and accounting latency limits.
6. [`results/phase6_6/research_experiment_plan.json`](file:///d:/Financial%20Distress/results/phase6_6/research_experiment_plan.json) — Structured acquisition prerequisites, gating milestones, and validation plan.
7. [`tests/test_phase6_6_alternative_signal_feasibility.py`](file:///d:/Financial%20Distress/tests/test_phase6_6_alternative_signal_feasibility.py) — 10 automated feasibility and governance regression tests (100% passing).
8. [`src/shadow/alternative_signal_feasibility_phase6_6.py`](file:///d:/Financial%20Distress/src/shadow/alternative_signal_feasibility_phase6_6.py) — Core audit and evaluation engine.

---

## Executive Summary

Phase 6.6 conducts an exhaustive, supervisory data feasibility audit of **Alternative Financial Risk Signals** following the Phase 6.5 finding that purely recombining annual accounting ratios in a Dual-Path architecture cannot separate sudden distress transitions within an 8% alert budget (`DUAL_PATH_RESEARCH_REPRODUCIBLE_BUT_INSUFFICIENT`).

The objective is to establish whether useful, legally usable, point-in-time alternative risk information (bank transaction cash flows, trade credit payment delinquency, and external macroeconomic curves) exists in the repository before investing in further model training or architecture redesigns.

### Core Audit Outcomes:

1. **Repository Data Census (100% External Modality Deficit):**
   - The repository contains exactly three tabular datasets in `data/raw/`: Dataset A (`Financial Distress.csv`, Tehran panel), Dataset B (`data.csv`, Taiwan TEJ cross-sectional), and Dataset C (`bankruptcy_Test_X.csv`, Polish test cross-sectional).
   - Target directory `data/external/` is **completely empty (0 files, 0 bytes)**.
   - **Zero transaction-level bank cash-flow data, zero B2B trade-credit payment records, and zero external macroeconomic benchmark curves are present in the repository.**

2. **Feasibility Classification Matrix (19 Candidate Signals Audited):**
   - **Requires external data acquisition:** **10 signals ($52.6\%$)** — All high-frequency cash-flow streams (inflows/outflows, cash balance burn, NSF/bounced payments, overdraft utilization), commercial trade-credit delinquency tapes, and sovereign interest rate series.
   - **Available but incomplete:** **3 signals ($15.8\%$)** — Days Payable Outstanding proxy, Receivables collection proxy, and Industry Sector code `x80` (present in Dataset A, but lacking intra-year transaction dates or external monthly output indices).
   - **Not currently feasible:** **1 signal ($5.3\%$)** — Regional economic stress (Dataset A contains zero geographic metadata; linking regional data is structurally impossible without de-anonymizing firms).
   - **Available and verified:** **5 signals ($26.3\%$)** — Periodic annual accounting transforms in Dataset A (working capital changes, cash-flow-to-debt, leverage shocks, rolling z-scores, historical sector default rates). These 5 signals were already comprehensively evaluated in Phase 5 and Phase 6.5 and proven mathematically insufficient under the 8% budget constraint.

3. **Forensic Signal Bottleneck in Accounting Statements:**
   - In Dataset A, maximum point-biserial correlation between any of the 82 continuous accounting ratios and sudden corporate distress is **$|r| = 0.1531$** (variable `x19`). All other ratios exhibit $|r| < 0.15$.
   - For sudden failure cases (e.g. Firm 28, antecedent $FD = +1.4430$; Firm 60, antecedent $FD = +0.1357$), annual balance sheets reported strong solvency prior to immediate insolvency.
   - **Structural Frequency Mismatch:** Annual accounting filings update at a 12-to-18-month latency (fiscal year end plus 60–120 days publication lag). Sudden liquidity crises unfold over weeks or days. Point-in-time balance sheet features cannot capture intra-period cash depletion.

4. **Supervisory Research-Gate Verdict:**
   $$\mathbf{ALTERNATIVE\_SIGNAL\_DATA\_INFEASIBLE}$$
   Because high-frequency transaction, trade-credit, and macroeconomic data are completely absent from the local repository, further machine learning model redesign or architecture engineering is halted. Manufacturing synthetic proxies or simulated cash flows is strictly prohibited under Federal Reserve SR 11-7 standards.

---

## 1. Mandatory Production Baseline & Governance Controls

In accordance with Federal Reserve SR 11-7 / OCC 2011-12 model risk governance standards:

### 1.1 Protected Artifacts Manifest Invariance

Cryptographic SHA-256 digests of all eight protected production artifacts were verified prior to and following audit execution against [`experiments/phase6/protected_artifacts_manifest.json`](file:///d:/Financial%20Distress/experiments/phase6/protected_artifacts_manifest.json):

| Protected Production Artifact | Expected SHA-256 Digest | Audited SHA-256 Digest | Governance Status |
| :--- | :--- | :--- | :---: |
| [`models/phase2/best_ews_model.joblib`](file:///d:/Financial%20Distress/models/phase2/best_ews_model.joblib) | `fcec88142b59f55f915e7a7301bc15bc583be9f3d925f91eeff59530db7fa6d3` | `fcec88142b59f55f915e7a7301bc15bc583be9f3d925f91eeff59530db7fa6d3` | **VERIFIED MATCH** |
| [`models/phase2/ews_calibrator.joblib`](file:///d:/Financial%20Distress/models/phase2/ews_calibrator.joblib) | `774a411b14376591b99c68fbd5e1d19b5926bafeea431596bda2a162e6a79d1e` | `774a411b14376591b99c68fbd5e1d19b5926bafeea431596bda2a162e6a79d1e` | **VERIFIED MATCH** |
| [`models/phase2/ews_preprocessor.joblib`](file:///d:/Financial%20Distress/models/phase2/ews_preprocessor.joblib) | `2342eb7f7b94657524a047d18b8760be55d82ea4045eae093b67b33edbc1659f` | `2342eb7f7b94657524a047d18b8760be55d82ea4045eae093b67b33edbc1659f` | **VERIFIED MATCH** |
| [`models/phase2/threshold_config.json`](file:///d:/Financial%20Distress/models/phase2/threshold_config.json) | `e60ed3b1f35e23012f096b492855c367006eb48f6cd877c92c107ea6eb96d136` | `e60ed3b1f35e23012f096b492855c367006eb48f6cd877c92c107ea6eb96d136` | **VERIFIED MATCH** |
| [`models/phase2/model_manifest.json`](file:///d:/Financial%20Distress/models/phase2/model_manifest.json) | `aada1e00afdc3b07bf73eaa1b59755173b5e4e800aa3a5a2c988a8ddbee1a86c` | `aada1e00afdc3b07bf73eaa1b59755173b5e4e800aa3a5a2c988a8ddbee1a86c` | **VERIFIED MATCH** |
| [`models/phase2/ews_metrics.json`](file:///d:/Financial%20Distress/models/phase2/ews_metrics.json) | `5f9a2506114bfd690840401be24bb3f83d3684aab746204b942a0c8986e4ddd1` | `5f9a2506114bfd690840401be24bb3f83d3684aab746204b942a0c8986e4ddd1` | **VERIFIED MATCH** |
| [`config/risk_config.yaml`](file:///d:/Financial%20Distress/config/risk_config.yaml) | `9975b94587bf641146b0566cbaa0e5fcb3a280ce831015e2b2f45fc926ef08e8` | `9975b94587bf641146b0566cbaa0e5fcb3a280ce831015e2b2f45fc926ef08e8` | **VERIFIED MATCH** |
| [`config/config.yaml`](file:///d:/Financial%20Distress/config/config.yaml) | `77f228b100a9989da85ca8c8da916e07949f2a93744181f07624f73e58d83864` | `77f228b100a9989da85ca8c8da916e07949f2a93744181f07624f73e58d83864` | **VERIFIED MATCH** |
| **Official Operating Threshold** | $\tau^* = \mathbf{0.0800}$ (Frozen on disk) | `operating_threshold: 0.08` | **VERIFIED FROZEN** |
| **Candidate Operating Threshold** | $\tau_{cand} = \mathbf{0.0894}$ (Frozen in sandbox) | Quarantined in shadow engine | **VERIFIED FROZEN** |

Zero production artifacts were modified, retrained, recalibrated, or replaced.

---

## 2. Audit of Existing Data & Modality Coverage

An exhaustive repository census was executed to verify the actual availability and structure of data files:

### 2.1 Repository Storage Census

```
d:\Financial Distress\data\
├── raw\
│   ├── Financial Distress.csv      # Dataset A: Tehran Stock Exchange SME Panel (3,672 rows, 86 cols, 422 firms)
│   ├── data.csv                    # Dataset B: Taiwan Economic Journal Bankruptcy (6,819 rows, 96 cols, static)
│   └── bankruptcy_Test_X.csv       # Dataset C: Polish Bankruptcy Test Partition (5,000 rows, 65 cols, static)
├── external\                       # Target directory for alternative data (EMPTY DIRECTORY: 0 files, 0 bytes)
└── processed\
    ├── companies_risk_intelligence.json # Exported risk scores and explainability payload
    └── watchlist.csv                    # Phase 3 watchlist export
```

### 2.2 Modality Coverage & Deficit Analysis

| Modality / Data Category | Source Dataset | Rows | Unique Firms | Temporal Frequency | Missingness | Operational Feasibility Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Periodic Accounting Ratios** | Dataset A (`Financial Distress.csv`) | 3,672 | 422 | Annual (Time 1 to 14) | $0.0\%$ | **Available & Verified** |
| **Bank Account Transaction Cash Flows** | `data/external/` | **0** | **0** | Daily / Weekly required | $\mathbf{100.0\%}$ | **NOT PRESENT (100% Deficit)** |
| **B2B Trade-Credit Delinquency Aging** | `data/external/` | **0** | **0** | Monthly bureau required | $\mathbf{100.0\%}$ | **NOT PRESENT (100% Deficit)** |
| **Macroeconomic Benchmark Rate Curves** | `data/external/` | **0** | **0** | Monthly curves required | $\mathbf{100.0\%}$ | **NOT PRESENT (100% Deficit)** |
| **Taiwan Cross-Sectional Accounting** | Dataset B (`data.csv`) | 6,819 | 6,819 | Cross-sectional (No time) | $0.0\%$ | **Available but Static** |
| **Polish Cross-Sectional Accounting** | Dataset C (`bankruptcy_Test_X.csv`) | 5,000 | 5,000 | Cross-sectional test | $0.0\%$ | **Available but Static** |

**Key Census Takeaway:** Dataset A represents the **sole longitudinal early warning dataset** in the repository. Its features are annual masked accounting ratios. High-frequency banking cash flows, supplier invoice records, and macroeconomic curves do not exist in the codebase.

---

## 3. Comprehensive Data Feasibility Matrix

Nineteen candidate risk signals across Families A, B, C, and D were audited against empirical feasibility, licensing, reporting delays, and privacy standards:

### 3.1 Candidate Signals Feasibility Matrix

| ID | Candidate Signal Name | Signal Family | Feasibility Status | Present in Repo? | Observation Frequency | Availability Lag | Cost & Licensing Requirements | Legal & Privacy Constraints |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **A01** | Gross Bank Account Inflows & Outflows | Cash-flow dynamics | **Requires external acquisition** | NO | Daily / Weekly | $T_{\text{obs}} + 1\text{d}$ | Open Banking Aggregator API contract (Plaid, Tink, Yodlee) | Explicit SME borrower consent under GDPR / GLBA / Open Banking |
| **A02** | 30-Day Net Cash-Flow Rolling Volatility | Cash-flow dynamics | **Requires external acquisition** | NO | Daily rolling | $T_{\text{obs}} + 1\text{d}$ | Derived from continuous bank transaction ledger | Bank secrecy laws; customer authorization required |
| **A03** | Rapid Cash Balance Depletion (>50% drop) | Cash-flow dynamics | **Requires external acquisition** | NO | Daily balance | $T_{\text{obs}} + 1\text{d}$ | Core banking balance feed / Open Banking | Restricted to active depository account holders |
| **A04** | Bounced Checks / Returned ACH / NSF Events | Cash-flow dynamics | **Requires external acquisition** | NO | Real-time clearing | $T_{\text{obs}} + 2\text{d}$ | Clearinghouse return logs / Commercial NSF feeds | Fair Credit Reporting Act (FCRA) permissible purpose required |
| **A05** | Overdraft Headroom Utilization (>95%) | Cash-flow dynamics | **Requires external acquisition** | NO | Weekly / Monthly | $T_{\text{obs}} + 5\text{d}$ | Commercial loan origination & servicing system | Internal lender data or inter-bank credit registry agreement |
| **B01** | Accounts Payable Delinquency (30/60/90+ DPD) | Trade-credit behavior | **Requires external acquisition** | NO | Monthly tape | $T_{\text{obs}} + 35\text{d}$ | Commercial credit bureau subscription (D&B PAYDEX, Experian) | Commercial credit bureau data licensing terms |
| **B02** | Days Payable Outstanding (DPO) Expansion | Trade-credit behavior | **Available but incomplete** | YES | Annual period | 60–180 days | Proxy ratios present in Dataset A; zero intra-year invoicing | Public dataset; intra-year tracking requires ERP software integration |
| **B03** | Supplier Trade Disputes & Mechanics Liens | Trade-credit behavior | **Requires external acquisition** | NO | Weekly docket | $T_{\text{obs}} + 10\text{d}$ | Public legal court docket scraping service (LexisNexis) | Public court record laws; jurisdiction privacy constraints |
| **B04** | Customer Receivables Collection (DSO Spike) | Trade-credit behavior | **Available but incomplete** | YES | Annual period | 60–180 days | Proxy ratios present in Dataset A; zero aging buckets | Public dataset; intra-year tracking requires accounting software feed |
| **B05** | Supplier Stop-Shipment & COD Conversion | Trade-credit behavior | **Requires external acquisition** | NO | Event-driven | $T_{\text{obs}} + 20\text{d}$ | Trade credit exchange network membership (NACM, CreditSafe) | Trade credit data reciprocity agreements required |
| **C01** | Central Bank Benchmark Interest Rate Hikes | Macro / Sector stress | **Requires external acquisition** | NO | Monthly series | Real-time $T+0$ | Public open data (Federal Reserve FRED / Central Bank / World Bank) | None (Public macroeconomic open series) |
| **C02** | Sector Output & Industrial Production Shocks | Macro / Sector stress | **Available but incomplete** | YES | Categorical code | Static + 45d | Sector code $x80$ present; external production index missing | None for aggregate sector statistics |
| **C03** | Input Energy & Commodity Price Shocks | Macro / Sector stress | **Requires external acquisition** | NO | Daily quotes | Real-time $T+0$ | Public commodity databases (World Bank Pink Sheet / CME) | None (Public financial market quotes) |
| **C04** | Regional Unemployment & Fiscal Stress | Macro / Sector stress | **Not currently feasible** | NO | Quarterly | 60–90 days | Regional census / municipal filings | Structurally impossible without de-anonymizing firm locations |
| **C05** | Sector Peer-Group Insolvency Velocity | Macro / Sector stress | **Available and verified** | YES | Annual period | Prior period $t-1$ | Zero cost (computable cross-sectionally from Dataset A) | Requires strict leave-one-out to prevent concurrent target leakage |
| **D01** | First Difference of Working Capital ($\Delta x_{i,t}$) | Accounting signals | **Available and verified** | YES | Annual period | Statement date | Zero cost (implemented in Phase 5 Feature Engine) | None (Approved open research dataset) |
| **D02** | Operating Cash Flow to Total Debt Trajectory | Accounting signals | **Available and verified** | YES | Annual period | Statement date | Zero cost (implemented in Phase 5 Feature Engine) | None (Approved open research dataset) |
| **D03** | Short-Term Leverage Acceleration ($\Delta^2 x_{i,t}$) | Accounting signals | **Available and verified** | YES | Annual period | Statement date | Zero cost (implemented in Phase 5 Feature Engine) | None (Approved open research dataset) |
| **D04** | Firm-Specific Rolling Z-Score Divergence | Accounting signals | **Available and verified** | YES | Annual period | Statement date | Zero cost (implemented in Phase 5 Feature Engine) | None (Approved open research dataset) |

---

## 4. Point-in-Time Feature Registry & Leakage Safeguards

A point-in-time feature registry was formulated in [`results/phase6_6/point_in_time_feature_registry.json`](file:///d:/Financial%20Distress/results/phase6_6/point_in_time_feature_registry.json) to establish strict timestamping standards:

```
[Observation Timestamp T_obs]                 [Availability Timestamp T_avail]           [Prediction Timestamp T_pred]
          |                                                   |                                         |
          +---> Event occurs                                  +---> Data recorded in                    +---> Model evaluates
                (e.g., invoice unpaid,                              system & queryable                        risk score
                 fiscal year end)                                   (e.g., statement published,
                                                                     bureau tape released)
          |<---------------- Reporting Delay Delta T -------->|
```

### 4.1 Strict Governance Enforcement Rules:
1. **Rule 1 (Dual Timestamp Mandate):** An observation timestamp ($T_{\text{obs}}$) alone is invalid for point-in-time modeling; an explicit availability timestamp ($T_{\text{avail}}$) indicating when the data became queryable must be verified.
2. **Rule 2 (Availability Precedence):** At prediction timestamp $T_{\text{pred}}$, a feature $x$ is valid if and only if $T_{\text{avail}}(x) \le T_{\text{pred}}$.
3. **Rule 3 (Backward-Looking Invariance):** Rolling statistics, aggregations, and difference operators must be strictly backward-looking. Centered or forward-looking windows are strictly prohibited.
4. **Rule 4 (Reporting Delay Rejection):** Features whose historical availability timestamps cannot be demonstrated (e.g. back-filled credit bureau records or revised macroeconomic figures) are rejected.

---

## 5. Forensic Sudden-Transition Signal Audit

A forensic investigation was conducted on all empirical corporate failures in Dataset A to determine why annual accounting ratios failed to capture sudden defaults:

### 5.1 Correlation Analysis with Sudden Corporate Failure

Across all 3,250 valid longitudinal transitions in Dataset A (containing 120 forward defaults: 42 sudden defaults where $FD_t > 0$, and 78 chronic defaults where $FD_t \le 0$):

- **Maximum Absolute Correlation:** Point-biserial correlation between any of the 82 continuous financial ratios and sudden default is **$|r| = 0.1531$** (`x19`).
- **Second Highest Correlation:** $|r| = 0.1434$ (`x8`).
- **All Remaining Ratios:** $|r| < 0.13$.

```
CORRELATION OF ANNUAL ACCOUNTING RATIOS WITH SUDDEN DEFAULT:
  Variable x19:  r = +0.1531 (Weak positive association)
  Variable x8:   r = -0.1434 (Weak negative association)
  Variable x14:  r = +0.1252
  Variable x37:  r = +0.1185
  Variable x49:  r = +0.1094
  Variable x36:  r = -0.1082
  Variable x3:   r = +0.1003
  Variable x13:  r = -0.1003
  [All other 74 ratios: |r| < 0.1000]
```

### 5.2 Out-of-Time Case Study: Firms 28 and 60

In the matured stress evaluation cohort ($T=13 \to T=14$, 5 sudden defaults: Firms 28, 60, 84, 225, 376):
- **Firm 28 Historical Solvency Trajectory:** $FD \in [1.6375, 0.9576, 0.5075, 1.5970, 1.5351, 1.4430] \to \mathbf{-0.6815}$ (Crash at $T=14$).
  - At $T=13$, Firm 28 was in the top $10\%$ of healthiest balance sheets in the entire cohort. Its accounting ratios showed zero impairment.
- **Firm 60 Historical Solvency Trajectory:** $FD \in [0.4210, 0.3120, 0.1850, 0.1357] \to \mathbf{-1.2361}$ (Severe statutory default at $T=14$).
- **Root Cause of Signal Failure:** Annual accounting statements measure historical accumulated accruals over past twelve-month accounting cycles. Sudden corporate failure in SMEs typically results from **acute liquidity cliffs (intraday credit facility cancellation, sudden customer invoice default, or severe working capital seizure)** that unfold in 30 to 60 days. An annual reporting model is structurally blind to intra-year liquidity crashes.

---

## 6. Research Experiment Plan & Acquisition Prerequisites

In [`results/phase6_6/research_experiment_plan.json`](file:///d:/Financial%20Distress/results/phase6_6/research_experiment_plan.json), formal gating milestones are pre-registered before any future exploratory model training can be authorized:

```
========================================================================================
RESEARCH EXPERIMENT PLAN GATING MILESTONES:
========================================================================================
[Milestone 1: External Data Acquisition]   --> STATUS: NOT STARTED (HALTED)
  Prerequisite: Execute legal data-sharing agreements for SME bank transaction ledgers
  or commercial credit bureau trade delinquency tapes.
  Gate: Checksums and licensing metadata in experiments/external_data_manifest.json.

[Milestone 2: Point-in-Time Availability]   --> STATUS: BLOCKED BY MILESTONE 1
  Prerequisite: Verify T_avail timestamps and quantify reporting latency Delta T.
  Gate: Point-in-time timestamping verification test passing 100% with zero leakage.

[Milestone 3: Chronological Split Isolation]--> STATUS: BLOCKED BY MILESTONE 1
  Prerequisite: Isolate training (T <= 10), validation (T in 11, 12), and shadow (T >= 13).
  Gate: Zero borrower overlap across folds enforced by GroupKFold.

[Milestone 4: Alert Capacity Constraint]   --> STATUS: ACTIVE GOVERNANCE MANDATE
  Prerequisite: Hard integer cap: maximum_alerts = floor(0.08 * eligible_firms).
  Gate: Policy must maintain <= 8.0% alert rate across all prospective cohorts.
========================================================================================
```

### Performance Target for Future Alternative Modeling:
- **Sudden Distress Sensitivity Target:** $\ge 60.0\%$ (detecting at least 3 of 5 sudden transitions).
- **Chronic Distress Sensitivity Target:** $\ge 80.0\%$ (detecting at least 5 of 6 chronic transitions).
- **Overall Recall Target:** $\ge 70.0\%$ under the strict 8% alert budget constraint.

---

## 7. Automated Test Execution Results

The automated test suite in [`tests/test_phase6_6_alternative_signal_feasibility.py`](file:///d:/Financial%20Distress/tests/test_phase6_6_alternative_signal_feasibility.py) was executed:

```
================================== test session starts ==================================
platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Financial Distress
plugins: anyio-4.12.0
collected 10 items

tests/test_phase6_6_alternative_signal_feasibility.py::test_production_baseline_manifest_immutability PASSED [ 10%]
tests/test_phase6_6_alternative_signal_feasibility.py::test_data_source_feasibility_matrix_completeness PASSED [ 20%]
tests/test_phase6_6_alternative_signal_feasibility.py::test_point_in_time_feature_registry_schema PASSED [ 30%]
tests/test_phase6_6_alternative_signal_feasibility.py::test_point_in_time_availability_and_leakage_safeguards PASSED [ 40%]
tests/test_phase6_6_alternative_signal_feasibility.py::test_missing_data_handling_and_zero_synthetic_claims PASSED [ 50%]
tests/test_phase6_6_alternative_signal_feasibility.py::test_firm_and_timestamp_alignment PASSED [ 60%]
tests/test_phase6_6_alternative_signal_feasibility.py::test_right_censored_outcome_exclusion PASSED [ 70%]
tests/test_phase6_6_alternative_signal_feasibility.py::test_alert_cap_arithmetic PASSED [ 80%]
tests/test_phase6_6_alternative_signal_feasibility.py::test_deterministic_research_artifacts PASSED [ 90%]
tests/test_phase6_6_alternative_signal_feasibility.py::test_research_gate_decision PASSED [100%]

================================== 10 passed in 2.15s ===================================
```

Additionally, the full repository regression suite was executed across all phases:
- **Total Tests Passed:** **189 passed** (0 failed, 0 errors).
- **Execution Time:** 23.22s.
- **Repository Exit Code:** `0` (clean execution).

---

## 8. Research-Gate Review & Supervisory Decision

### Formal Research-Gate Evaluation:

1. **Do Verified Alternative Risk Signals (Cash-Flow, Trade-Credit, Macroeconomic) Exist in the Repository?**  
   **NO.** Audit confirms `data/external/` is completely empty. No transaction-level banking data, trade credit delinquency tapes, or external macroeconomic curves exist.

2. **Can Accounting-Only Models Be Re-engineered to Anticipate Sudden Distress Under the 8% Budget?**  
   **NO.** Correlation analysis ($|r| \le 0.1531$) and Phase 6.5 frontier proofs confirm that annual balance sheet ratios suffer from a structural reporting latency that makes detecting sudden crashes within an 8% alert budget mathematically impossible.

3. **Did the Feasibility Study Preserve Production Baseline Immutability?**  
   **YES.** All 8 protected production artifact SHA-256 digests match the manifest identically.

### Final Supervisory Status:
$$\mathbf{ALTERNATIVE\_SIGNAL\_DATA\_INFEASIBLE}$$

### Binding Operating Directives:
1. **Halt Model Training:** No further model redesign, re-weighting, or architectural re-combination may be attempted using Dataset A accounting ratios alone.
2. **Maintain Production Baseline:** Model `models/phase2/best_ews_model.joblib` operating at $\tau^* = 0.0800$ remains the approved production early warning system.
3. **Maintain Candidate Quarantine:** Experimental candidate `Phase5_RF_CostSensitive_w10` remains quarantined at $\tau_{cand} = 0.0894$ in passive shadow surveillance.
4. **Next Step:** Formalize commercial data acquisition specifications (Open Banking account aggregation, B2B credit bureau trade tapes, and central bank macro series) before initiating Phase 6.7. Do not train models until point-in-time alternative data streams are ingested and verified.
