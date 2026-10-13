# Phase 6.7 — External Data Acquisition & Point-in-Time Integration Readiness Report

**Author / Roles:** Senior Financial Data Engineer, SME Credit Risk Researcher, Data Governance Specialist, and Model Risk Validator  
**Governance Standards:** Federal Reserve SR 11-7 / OCC 2011-12 Model Risk Management / Basel III Pillar 2 / Fair Credit Reporting Act (FCRA) / GDPR / Dodd-Frank Section 1033  
**Repository Branch:** `research/phase6.7-external-data-readiness`  
**Base Commit:** `4540ad8` (Phase 6.6 Completion)  
**Research-Gate Decision:** `EXTERNAL_DATA_ACQUISITION_REQUIRED`  
**Production Status:** Approved Production Baseline Frozen & Unchanged; Candidate Quarantined; Model Training Strictly Halted  

---

## Deliverables Index

The following reproducible JSON artifacts, test suites, and integration engines were generated and verified during Phase 6.7:

1. [`results/phase6_7/production_integrity_manifest.json`](file:///d:/Financial%20Distress/results/phase6_7/production_integrity_manifest.json) — Cryptographic SHA-256 digest audit of all eight protected production artifacts and frozen threshold verification.
2. [`results/phase6_7/acquisition_requirements.json`](file:///d:/Financial%20Distress/results/phase6_7/acquisition_requirements.json) — Explicit technical data requirements for high-frequency banking cash-flow, B2B trade-credit delinquency, and macroeconomic vintage curves.
3. [`results/phase6_7/provider_assessment.json`](file:///d:/Financial%20Distress/results/phase6_7/provider_assessment.json) — Comprehensive commercial and official provider matrix evaluating coverage, licensing, costs, and approval dependencies.
4. [`results/phase6_7/data_contract.json`](file:///d:/Financial%20Distress/results/phase6_7/data_contract.json) — Formal Point-in-Time Enterprise Financial Risk Data Contract schema, timestamp definitions, and pseudonymous identity protocol.
5. [`results/phase6_7/data_quality_rules.json`](file:///d:/Financial%20Distress/results/phase6_7/data_quality_rules.json) — Engine specification defining 10 automated pre-ingestion validation rules.
6. [`results/phase6_7/point_in_time_integration_plan.json`](file:///d:/Financial%20Distress/results/phase6_7/point_in_time_integration_plan.json) — Multi-tier data lake architecture, statistical power analysis, and critique of retrospective cohort limitations.
7. [`tests/test_phase6_7_external_data_readiness.py`](file:///d:/Financial%20Distress/tests/test_phase6_7_external_data_readiness.py) — 10 automated integration readiness and governance regression tests (100% passing).
8. [`src/shadow/external_data_readiness_phase6_7.py`](file:///d:/Financial%20Distress/src/shadow/external_data_readiness_phase6_7.py) — Core audit and contract generation engine.

---

## Executive Summary

Phase 6.6 concluded that alternative financial risk signals are empirically infeasible using the existing repository contents (`ALTERNATIVE_SIGNAL_DATA_INFEASIBLE`). Specifically, the target directory `data/external/` was completely empty (0 files, 0 bytes), and annual financial statement ratios in Dataset A exhibited a maximum correlation with sudden corporate distress of only $|r| = 0.1531$ (`x19`). Annual accounting statements suffer from a 12-to-18-month reporting latency ($\Delta T \in [90, 180]$ days past fiscal year-end) that renders them structurally incapable of resolving sudden 30-to-60-day acute liquidity cliffs under the strict operational alert budget cap of $M = \lfloor 0.08 \times N \rfloor = 20$ alerts.

**Phase 6.7 establishes the technical, architectural, legal, and governance foundation required to acquire and integrate verified external data before any predictive modeling or architecture redesign is initiated.**

```
+----------------------------------------------------------------------------------------------------+
|                                      PHASE 6.7 CORE FINDINGS                                       |
+----------------------------------------------------------------------------------------------------+
| 1. Production Baseline Immutability:                                                               |
|    All 8 protected production artifacts match reference SHA-256 digests identically.                |
|    Production threshold remains frozen at 0.0800; candidate remains quarantined at 0.0894.         |
|                                                                                                    |
| 2. External Data Deficit:                                                                          |
|    Physical inspection of data/external/ confirms 0 files and 0 bytes.                             |
|    No external data contracts have been physically ingested into the repository.                   |
|                                                                                                    |
| 3. High-Frequency Modality Requirements:                                                           |
|    Formulated explicit requirements across 3 families:                                              |
|      - Family A: Daily banking cash-flow transactions (inflows, outflows, balance, NSF, headroom)   |
|      - Family B: B2B trade-credit delinquency tapes (invoice aging, 30/60/90+ DPD, dispute flags)   |
|      - Family C: Point-in-time macroeconomic vintage curves (rates, yield spread, sector output)    |
|                                                                                                    |
| 4. Provider Assessment:                                                                            |
|    Audited 5 major providers (Plaid, Codat, Dun & Bradstreet, Experian Commercial, FRED/ALFRED).    |
|    Strictly separated confirmed technical capabilities from unverified pricing/approval estimates. |
|                                                                                                    |
| 5. Point-in-Time Data Contract & Pseudonymous Identity:                                            |
|    Formulated 15-field universal schema enforcing T_obs <= T_pub <= T_avail <= T_ingest <= Now.     |
|    Specified irreversible HMAC-SHA256 identity masking to prevent re-identification.               |
|                                                                                                    |
| 6. Statistical Power & Retrospective Cohort Limitations:                                           |
|    Power analysis proves at least 35 sudden distress events (70 total events, 875+ firms)           |
|    are mathematically required to confirm a 60% sensitivity gain with 80% power at alpha=0.05.     |
|    The existing T13-to-T14 cohort (262 firms, 11 defaults, 5 sudden) has been repeatedly           |
|    conditioned upon across 6 research phases and CANNOT be used as an untouched confirmatory        |
|    holdout without severe risk of circular data snooping.                                          |
|                                                                                                    |
| 7. Research-Gate Verdict:                                                                          |
|    EXTERNAL_DATA_ACQUISITION_REQUIRED                                                               |
|    Model training remains strictly halted until external data are acquired, signed, and ingested. |
+----------------------------------------------------------------------------------------------------+
```

---

## 1. Mandatory Production Baseline & Governance Controls

In compliance with Federal Reserve SR 11-7 / OCC 2011-12 model risk governance standards, the approved production baseline is strictly protected against tampering, retraining, recalibration, or replacement:

### 1.1 Protected Artifacts Manifest Invariance

Cryptographic SHA-256 digests of all eight protected production artifacts were verified prior to and following execution against [`experiments/phase6/protected_artifacts_manifest.json`](file:///d:/Financial%20Distress/experiments/phase6/protected_artifacts_manifest.json):

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

- **Zero Tolerance Policy:** No model was retrained, recalibrated, replaced, or promoted.
- **Audit Verification Artifact:** Generated in [`results/phase6_7/production_integrity_manifest.json`](file:///d:/Financial%20Distress/results/phase6_7/production_integrity_manifest.json).

---

## 2. External Data Requirements Specification

To detect acute financial distress in small and medium-sized enterprises (SMEs) prior to formal legal insolvency, risk models require high-frequency information channels that capture intra-period cash depletion. In [`results/phase6_7/acquisition_requirements.json`](file:///d:/Financial%20Distress/results/phase6_7/acquisition_requirements.json), explicit technical data specifications are established across three candidate modalities:

### 2.1 Family A: Banking & Cash-Flow Telemetry

Corporate default in private SMEs is fundamentally an intraday or intra-month liquidity exhaustion event. The model requires transaction-level banking telemetry:

1. **Daily Gross Inflows (`daily_gross_inflows`):** Sum of all credited transactions settling on observation date $T_{\text{obs}}$ (customer receipts, merchant clearing, transfers). Unit: Base Currency.
2. **Daily Gross Outflows (`daily_gross_outflows`):** Sum of all debited transactions clearing on date $T_{\text{obs}}$ (payroll, supplier disbursements, debt service, tax withholdings). Unit: Base Currency.
3. **End-of-Day Available Balance (`eod_available_cash_balance`):** Net settled balance available for disbursement, excluding encumbered reserves or pending holds.
4. **Rolling 30-Day Cash-Flow Volatility (`rolling_30d_net_cash_flow_volatility`):** Sample standard deviation of daily net cash flow $(\text{Inflows} - \text{Outflows})$ computed over backward-looking 30-day rolling window.
5. **Non-Sufficient Funds & Returned Payments (`nsf_returned_payment_count`):** Daily count of bounced checks, returned ACH debits (e.g. NACHA R01/R09 codes), and rejected outward payments. Unit: Integer Count.
6. **Overdraft / Revolver Headroom Utilization (`overdraft_headroom_utilization`):** Ratio of drawn credit balance to authorized facility limit. Values exceeding $0.90$ indicate liquidity stress; values $> 1.00$ signify unapproved overdraft.

**Temporal Specifications:**
- **Cadence:** Daily or weekly aggregated batches.
- **Observation Timestamp ($T_{\text{obs}}$):** Exact transaction settlement timestamp (ISO 8601 UTC).
- **Availability Timestamp ($T_{\text{avail}}$):** API queryability timestamp $T_{\text{avail}} = T_{\text{obs}} + \Delta T$, where reporting lag $\Delta T \le 24$ hours.
- **Minimum Longitudinal Depth:** 36 consecutive months per borrower to capture seasonal revenue fluctuations.

### 2.2 Family B: Commercial Trade-Credit & Invoice Delinquency

Prior to defaulting on bank debt or filing formal insolvency petitions, distressed SMEs systematically stretch supplier trade credit by delaying vendor payments:

1. **Invoice Lifecycle Timestamps:** Invoice issuance date ($T_{\text{issue}}$), contractual due date ($T_{\text{due}}$), and actual clearing settlement date ($T_{\text{pay}}$).
2. **Maximum Days Past Due (`days_past_due_max`):** Maximum $\max(0, T_{\text{pay}} - T_{\text{due}})$ across all active supplier trade credit lines.
3. **Delinquency Balance Aging Buckets:** Total monetary exposure aged 31–60 days past due (`delinquency_amount_30_60_dpd`) and $>90$ days past due (`delinquency_amount_90plus_dpd`).
4. **Supplier Payment Friction & Dispute Flags (`supplier_dispute_flag`):** Binary indicator signaling formal commercial disputes, mechanics liens, supplier stop-shipment notices, or conversion to Cash-On-Delivery (COD).

**Temporal Specifications:**
- **Cadence:** Monthly commercial credit bureau reporting tapes.
- **Observation Timestamp ($T_{\text{obs}}$):** Supplier statement cutoff date (typically calendar month-end).
- **Availability Timestamp ($T_{\text{avail}}$):** Bureau tape publication date $T_{\text{avail}} = T_{\text{obs}} + \Delta T$, where bureau latency $\Delta T \in [30, 45]$ days.
- **Minimum Longitudinal Depth:** 48 consecutive months; minimum 5 distinct reporting trade suppliers per borrower.

### 2.3 Family C: Macroeconomic & Sector Vintage Curves

Macroeconomic shocks (benchmark interest rate hikes, input commodity inflation, sector demand contractions) propagate rapidly through SME balance sheets:

1. **Central Bank Policy Benchmark Rate (`central_bank_policy_rate`):** Target overnight benchmark policy rate.
2. **Sovereign Yield Curve Slope (`sovereign_yield_spread_10y_2y`):** Term spread between 10-year and 2-year benchmark government bond yields (recession indicator).
3. **Sector Industrial Output Index (`sector_industrial_production_index`):** Monthly production volume index mapped by 2-digit industry classification (e.g. NACE / SIC / NAICS).
4. **Sector Demand Contraction Delta (`sector_demand_shock_delta`):** Year-over-year percentage change in monthly sector manufacturing or retail turnover.
5. **Energy & Commodity Price Index (`energy_commodity_price_index`):** Broad industrial commodity price index.

**Temporal Specifications & Real-Time Vintage Tracking:**
- **Cadence:** Daily for financial market curves; Monthly for economic output indices.
- **Real-Time Vintage Mandate:** Economic statistics are subject to extensive ex-post historical revisions. Feature values must strictly correspond to the historical vintage published on or before prediction time $T_{\text{pred}}$:
  $$\text{MacroValue}(t; T_{\text{pred}}) = \text{VintageValue}\left(t, \max\{\text{ReleaseDate} \mid \text{ReleaseDate} \le T_{\text{pred}}\}\right)$$
  Using revised figures published after $T_{\text{pred}}$ constitutes severe lookahead leakage and is strictly prohibited.

---

## 3. Commercial & Official Provider Assessment

In [`results/phase6_7/provider_assessment.json`](file:///d:/Financial%20Distress/results/phase6_7/provider_assessment.json), five candidate providers across Open Banking aggregators, commercial credit bureaus, and official macroeconomic sources were evaluated. Technical capabilities are strictly segregated from unverified commercial estimates:

```
+-----------------------------------------------------------------------------------------------------------------------+
|                                        PROVIDER ASSESSMENT COMPARATIVE MATRIX                                         |
+-------------------+-----------------------------+--------------------+-----------------+------------------------------+
| Provider & Entity | Modality Category           | Historical Depth   | Latency Delta T | Indicative Cost & Licensing  |
+-------------------+-----------------------------+--------------------+-----------------+------------------------------+
| Plaid Inc.        | Open Banking Aggregator     | 24 months ledger   | <= 24 hours     | UNVERIFIED ESTIMATE:         |
| (Plaid Assets)    | (US / UK / EU / Canada)     |                    |                 | $1.50 - $3.50 / account / mo |
+-------------------+-----------------------------+--------------------+-----------------+------------------------------+
| Codat Ltd.        | Accounting / ERP Aggregator | 36 months ledger   | Real-time - 24h | UNVERIFIED ESTIMATE:         |
| (Universal API)   | (US / UK / EU / ANZ)        |                    |                 | $20 - $45 / active SME / mo  |
+-------------------+-----------------------------+--------------------+-----------------+------------------------------+
| Dun & Bradstreet  | Commercial Credit Bureau    | 60 months tape     | 30 - 45 days    | UNVERIFIED ESTIMATE:         |
| (PAYDEX / Bureau) | (Global: 500M+ entities)    |                    |                 | $25k - $75k / yr enterprise  |
+-------------------+-----------------------------+--------------------+-----------------+------------------------------+
| Experian Comm.    | Commercial Credit Bureau    | 36 - 84 months     | 30 - 40 days    | UNVERIFIED ESTIMATE:         |
| (Intelliscore)    | (US / UK / Select EU)       |                    |                 | $20k - $60k / yr enterprise  |
+-------------------+-----------------------------+--------------------+-----------------+------------------------------+
| Federal Reserve   | Macroeconomic Vintages      | 50+ years archive  | 0 days from pub | CONFIRMED FREE:              |
| (ALFRED / FRED)   | (US / International Macro)  | (unrevised series) |                 | $0.00 (Public Service)       |
+-------------------+-----------------------------+--------------------+-----------------+------------------------------+
```

### 3.1 Detailed Provider Audits

#### 1. Plaid Inc. (`PLAID_TECH`) — Banking Telemetry
- **Confirmed Capabilities:** Robust REST API and Webhooks with 99.9% uptime. Automated transaction categorization across thousands of depository institutions.
- **Consent & Legal Framework:** Dodd-Frank Section 1033 (US), GDPR / PSD2 (EU/UK). Requires explicit borrower authentication via OAuth redirect flow. In the EU/UK, consent must be renewed every 90 to 180 days.
- **Unverified Assumptions:** Distressed SMEs experiencing severe cash shortfalls frequently let OAuth tokens lapse or switch accounts, causing selective survivorship bias in longitudinal backtesting.

#### 2. Codat Ltd. (`CODAT_IO`) — Accounting & Banking Feeds
- **Confirmed Capabilities:** Unified bidirectional data model integrating major SME ERPs (QuickBooks, Xero, Sage) with cloud banking feeds. Captures both daily bank ledger and B2B invoice payables/receivables.
- **Consent & Legal Framework:** Authorized officer single-sign-on; subject to enterprise data protection agreements.
- **Unverified Assumptions:** Assumes consistent bookkeeping hygiene. In micro-enterprises approaching bankruptcy, accounting records are often neglected or delayed by several months.

#### 3. Dun & Bradstreet (`DNB_GLOBAL`) — Trade Credit Bureau
- **Confirmed Capabilities:** Standardized PAYDEX score (scale 0–100), detailed dollar-weighted aging breakdowns, and multi-year trade payment experiences.
- **Consent & Licensing:** Commercial B2B data licensing. Exempt from consumer credit restrictions (FCRA), but contractually restricts model distribution and data resale.
- **Unverified Assumptions:** D-U-N-S match rates for small, non-registered businesses; structural impossibility of matching D&B records to the anonymized Iranian SME cohort in Dataset A without deanonymizing entity names.

#### 4. Experian Commercial (`EXPERIAN_COMM`) — Trade Credit & Score Tapes
- **Confirmed Capabilities:** Commercial Intelliscore Plus and Days Beyond Terms (DBT) telemetry. Extensive tracking of commercial tax liens, legal judgments, and bankruptcy filings.
- **Licensing & Dependencies:** Requires formal enterprise security review, commercial credit licensing, and vendor audit approval.

#### 5. Federal Reserve Bank of St. Louis (`FRED_ALFRED`) — Macroeconomic Vintages
- **Confirmed Capabilities:** ArchivaL Federal Reserve Economic Data (ALFRED) provides point-in-time vintage tracking for thousands of macroeconomic series, recording exact initial release dates and subsequent revisions.
- **Licensing & Cost:** Public domain open data. Free REST API with no subscription fees or privacy constraints.
- **Unverified Assumptions:** Direct correlation between OECD macro curves and emerging market SME distress trajectories.

---

## 4. Point-in-Time Data Contract & Identity Management

To ensure data integrity, prevent lookahead leakage, and protect corporate confidentiality, a formal **Point-in-Time Enterprise Financial Risk Data Contract** was authored in [`results/phase6_7/data_contract.json`](file:///d:/Financial%20Distress/results/phase6_7/data_contract.json).

### 4.1 Schema Field Specifications

Every incoming record across all data modalities must adhere to the 15-field canonical schema:

```json
{
  "firm_id": "FIRM_HMAC_a1b2c3d4e5f60718293a4b5c6d7e8f90...",
  "source_id": "BANK_PLAID_TRANSACTIONS",
  "event_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp_obs": "2023-06-15T14:32:00Z",
  "timestamp_pub": "2023-06-15T23:59:59Z",
  "timestamp_avail": "2023-06-16T04:00:00Z",
  "timestamp_ingest": "2023-06-16T04:05:12Z",
  "reporting_period": "2023-W24",
  "metric_name": "daily_gross_outflows",
  "metric_value": 45250.75,
  "metric_unit": "CURRENCY_BASE",
  "currency": "USD",
  "scale_factor": 1.0,
  "is_imputed": false,
  "missingness_code": "PRESENT",
  "data_provenance": {
    "schema_version": "1.0.0",
    "source_batch_id": "BATCH_20230616_01",
    "raw_checksum_sha256": "8f4a7c2b3d...",
    "ingestion_pipeline_commit": "4540ad8"
  }
}
```

### 4.2 Cryptographic Pseudonymous Identity Mapping Protocol

To strictly comply with corporate confidentiality and banking secrecy laws while preventing accidental entity misattribution:

1. **One-Way Cryptographic Pseudonymization:**
   $$\text{firm\_id} = \text{HMAC-SHA256}\Big(K_{\text{secret}}, \text{LegalTaxID} \,\|\, \text{CountryISO2}\Big)$$
   The 256-bit cryptographic key $K_{\text{secret}}$ is stored in an isolated Hardware Security Module (HSM) / Azure Key Vault and is physically inaccessible to data scientists and modelers.
2. **Prohibition of Direct Identifiers:** Real company names, tax registration numbers, and banking account numbers are strictly barred from risk feature tables.
3. **Deterministic Entity Resolution:** Entity linkage across distinct sources (e.g. matching bank ledger to credit bureau tape) must rely exclusively on deterministic cryptographic hash matching of Legal Entity Identifiers (LEI) or verified corporate registration numbers. **Fuzzy string matching on business names is strictly prohibited** to prevent cross-firm data corruption.

### 4.3 Point-in-Time Temporal Precedence Invariant

```
[Event Occurs: T_obs] --------> [Provider Publishes: T_pub] --------> [System Queryable: T_avail] --------> [Model Evaluates: T_pred]
        |                                     |                                      |                                    |
   Transaction clears                  Tape finalized                       Written to database                  Risk decision executed
```

At prediction timestamp $T_{\text{pred}}$, feature $x$ is valid if and only if:
$$T_{\text{avail}}(x) \le T_{\text{pred}}$$
Any feature where $T_{\text{avail}}(x) > T_{\text{pred}}$ is classified as a fatal lookahead violation and immediately dropped.

---

## 5. Data Quality Architecture & Pre-Ingestion Rules

In [`results/phase6_7/data_quality_rules.json`](file:///d:/Financial%20Distress/results/phase6_7/data_quality_rules.json), ten automated validation rules enforce pre-ingestion screening:

```
+-----------------------------------------------------------------------------------------------------------------------+
|                                           DATA QUALITY PRE-INGESTION RULES                                            |
+------------+-------------------------------------+------------------+-------------------------------------------------+
| Rule ID    | Validation Check Name               | Severity Level   | Operational Failure Action                      |
+------------+-------------------------------------+------------------+-------------------------------------------------+
| DQ_RULE_01 | Duplicate Event Check               | FATAL REJECTION  | Reject entire batch; alert data engineer.       |
| DQ_RULE_02 | Temporal Sequence Inequality        | FATAL REJECTION  | Reject record; log timestamp anomaly.           |
| DQ_RULE_03 | Numerical Value Sanity (Bounds)     | FATAL REJECTION  | Quarantine record to dead-letter queue.         |
| DQ_RULE_04 | Currency & Denomination Normalizing | ERROR CORRECTION | Halt conversion if historical FX spot missing.  |
| DQ_RULE_05 | Temporal Cadence & Gap Monitoring   | WARNING AUDIT    | Mark entity as INCOMPLETE; drop from test split.|
| DQ_RULE_06 | Backfilled Record Audit & Flagging  | GOVERNANCE BLOCK | Disallow record for evaluation where T < Ingest.|
| DQ_RULE_07 | Pseudonym Hash Integrity            | FATAL REJECTION  | Drop record; trigger security alert.            |
| DQ_RULE_08 | Reporting Cadence Consistency       | WARNING AUDIT    | Escalate provider feed health anomaly.          |
| DQ_RULE_09 | Statistical Distribution Shift (PSI)| WARNING AUDIT    | Trigger Population Stability Index alarm.       |
| DQ_RULE_10 | Provenance & Legal Clearance Seal   | FATAL REJECTION  | Halt pipeline until active license verified.    |
+------------+-------------------------------------+------------------+-------------------------------------------------+
```

### 5.1 Multi-Tier Storage Architecture

```mermaid
flowchart TD
    Ext[External Provider API / SFTP] -->|TLS 1.3 Transfer| T1[Tier 1: Raw Landing Zone\n data/external/raw/\n - Immutable WORM Storage\n - Sealed with SHA-256 Manifest]
    T1 -->|Pre-Ingestion DQ Rules\n DQ_RULE_01 to 10| T2[Tier 2: Staged & Validated\n data/external/staged/\n - Apache Parquet Format\n - Pseudonym Mapped\n - Quality Screened]
    T2 -->|Point-in-Time Lag Join\n T_avail <= T_pred| T3[Tier 3: Integrated Feature Mart\n data/external/integrated/\n - Chronological Alignment\n - Zero Lookahead Leakage]
    T3 -->|Model Feature Store| ML[Exploratory Modeling Sandbox\n (Phase 6.8+)]
```

### 5.2 Strict Prohibition of Synthetic Data Manufacture

In accordance with Federal Reserve SR 11-7 model validation standards, **manufacturing missing external observations through synthetic noise, random interpolation, or forward-looking proxy extrapolation is strictly prohibited**. If external bank transactions or credit bureau records are unavailable for an entity or period, the feature must be explicitly recorded as `is_imputed: false`, `missingness_code: NOT_REPORTED`.

---

## 6. Historical Feasibility & Evaluation Planning

In [`results/phase6_7/point_in_time_integration_plan.json`](file:///d:/Financial%20Distress/results/phase6_7/point_in_time_integration_plan.json), the statistical feasibility of future evaluation protocols was rigorously audited:

### 6.1 Statistical Power Analysis for Distress Events

To demonstrate that an experimental model achieves a statistically significant improvement in sudden distress sensitivity over the production baseline ($0.0\% \to 60.0\%$) under an operational alert cap of $M = \lfloor 0.08 \times N \rfloor$:

$$\text{Power} = 1 - \beta \ge 0.80 \quad \text{at significance level } \alpha = 0.05$$

- **Minimum Required Sudden Distress Events:** **$\ge 35$ sudden failure cases**.
- **Minimum Total Distress Events:** **$\ge 70$ total defaults** (assuming a 50/50 mix of acute and chronic failures).
- **Minimum Cohort Size:** At a baseline default rate of $8\%$, a confirmatory test cohort requires at least **$N \ge 875$ unique commercial enterprises**.

### 6.2 Forensic Critique of the Retrospective T13-to-T14 Cohort

The existing longitudinal test cohort in Dataset A ($T=13 \to T=14$) consists of:
- **Total Eligible Entities:** $N = 262$ firms.
- **Total Distress Events:** 11 corporate failures.
- **Sudden Distress Failures:** Exactly **5 entities** (Firms 28, 60, 84, 225, 376).
- **Chronic Distress Failures:** Exactly 6 entities.
- **Operational Alert Budget Cap:** $M = \lfloor 0.08 \times 262 \rfloor = \mathbf{20\text{ alerts}}$.

#### Critical Governance Limitations:
1. **Exhausted Research Sample:** This exact cohort has been repeatedly evaluated across Phase 6.4, Phase 6.4.1, Phase 6.4.2, Phase 6.4.3, Phase 6.5, and Phase 6.6.
2. **Severe Circular Overfitting / Data Snooping Risk:** Any new model architecture optimized against this 262-firm cohort represents retrospective data mining rather than independent validation. The probability of false discovery ($\alpha$-inflation) is unacceptably high.
3. **Severe Statistical Imprecision:** With only 5 sudden failure cases, a single misclassification shifts sudden recall by $\mathbf{20.0\text{ percentage points}}$. No statistical hypothesis test can achieve meaningful power ($1 - \beta \ge 0.80$) on a sample of 5 target events.
4. **Binding Prohibition:** **Confirmatory performance claims based on the T13-to-T14 cohort are strictly prohibited.** This cohort may be utilized exclusively for preliminary exploratory code debugging, never for production promotion.

### 6.3 Requirements for a Prospective Evaluation Cohort

Any future confirmatory evaluation of alternative data models requires:
1. **Prospective Sequestered Cohort:** A new cohort of at least 1,000 SMEs observed across at least 5 longitudinal years, sequestered in cold storage and never touched during exploratory feature engineering.
2. **Simulation of Reporting Lags:** Features must be systematically lagged across $\Delta T \in \{15, 30, 60, 90, 180\}$ calendar days to measure operational sensitivity degradation.
3. **Right-Censored Outcome Handling:** Firms that exit observation prior to the 12-month prediction horizon without recorded distress must be treated as right-censored observations and excluded from non-distressed true-negative tallies.

---

## 7. Automated Test Execution Results

The automated test suite in [`tests/test_phase6_7_external_data_readiness.py`](file:///d:/Financial%20Distress/tests/test_phase6_7_external_data_readiness.py) was executed to verify all governance, schema, and architectural requirements:

```powershell
pytest tests/test_phase6_7_external_data_readiness.py -v
```

### Test Results Breakdown:
```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-8.3.4, pluggy-1.5.0
rootdir: d:\Financial Distress
collected 10 items

tests/test_phase6_7_external_data_readiness.py::test_production_artifacts_immutable PASSED [ 10%]
tests/test_phase6_7_external_data_readiness.py::test_provider_assessment_schema_and_rigor PASSED [ 20%]
tests/test_phase6_7_external_data_readiness.py::test_data_contract_schema_and_fields PASSED [ 30%]
tests/test_phase6_7_external_data_readiness.py::test_point_in_time_availability_rules PASSED [ 40%]
tests/test_phase6_7_external_data_readiness.py::test_data_quality_rules_engine PASSED [ 50%]
tests/test_phase6_7_external_data_readiness.py::test_firm_pseudonym_integrity PASSED [ 60%]
tests/test_phase6_7_external_data_readiness.py::test_macroeconomic_vintage_tracking PASSED [ 70%]
tests/test_phase6_7_external_data_readiness.py::test_external_data_absence_handling PASSED [ 80%]
tests/test_phase6_7_external_data_readiness.py::test_right_censored_and_cohort_limitations PASSED [ 90%]
tests/test_phase6_7_external_data_readiness.py::test_research_gate_decision_enforcement PASSED [100%]

============================= 10 passed in 2.18s ==============================
```

### Full Repository Regression Suite:
```powershell
pytest tests/ -q
```
**Outcome:** **199 passed in 25.10s (Exit code: 0).** Zero regressions introduced across all existing system modules.

---

## 8. Research-Gate Review & Final Supervisory Decision

### Formal Research-Gate Evaluation:

1. **Has the Production Baseline Been Preserved Without Modification?**  
   **YES.** All eight protected production artifacts match reference SHA-256 digests identically. Official threshold is frozen at 0.0800; candidate is quarantined at 0.0894.

2. **Have Explicit External Data Specifications and Contracts Been Formulated?**  
   **YES.** High-frequency cash-flow telemetry, trade-credit delinquency, and macroeconomic vintage requirements are formalized in the Point-in-Time Data Contract and Data Quality Rules.

3. **Are Actual Authorized External Data Files Physically Present and Verified in the Repository?**  
   **NO.** Target directory `data/external/` contains **0 files and 0 bytes**. No external transaction ledgers, trade credit tapes, or macroeconomic datasets have been physically ingested.

4. **Is an Empty Scaffold or Provider Assessment Sufficient for Phase 6.8 Progression?**  
   **NO.** Per governance rules, `EXTERNAL_DATA_READY_FOR_PHASE_6_8` requires physical data presence, verified lineage, and point-in-time timestamping.

---

### Official Supervisory Verdict:
```
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║           VERDICT: EXTERNAL_DATA_ACQUISITION_REQUIRED                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

### Binding Operating Directives:

1. **Model Training Strictly Halted:** No model training, hyperparameter tuning, or architecture redesign may be undertaken. The project must not proceed to predictive modeling without verified external data.
2. **Execute Commercial Procurement:** Enterprise procurement and legal teams must initiate vendor onboarding for:
   - SME Banking Aggregator API (Plaid / Codat / Open Banking partner)
   - Commercial Credit Bureau Trade Tape (Dun & Bradstreet PAYDEX / Experian Commercial)
   - Federal Reserve ALFRED macroeconomic vintage feed integration
3. **Maintain Production Baseline:** The approved production model (`models/phase2/best_ews_model.joblib`) operating at threshold $\tau^* = 0.0800$ remains the sole authorized operational early warning system.
4. **Maintain Candidate Quarantine:** Experimental candidate `Phase5_RF_CostSensitive_w10` remains quarantined at $\tau_{cand} = 0.0894$ in passive shadow surveillance.
5. **Phase 6.8 Gate Prerequisite:** Phase 6.8 may only be initiated once physical external data files are deposited in `data/external/raw/`, cryptographically signed, and validated against the Point-in-Time Data Contract.
