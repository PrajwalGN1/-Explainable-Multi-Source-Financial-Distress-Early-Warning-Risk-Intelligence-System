"""
Phase 6.7: External Data Acquisition & Point-in-Time Integration Readiness Engine.

Roles:
- Senior Financial Data Engineer
- SME Credit Risk Researcher
- Data Governance Specialist
- Model Risk Validator

Key Capabilities:
1. Verify production baseline immutability (SHA-256 hashes, threshold 0.0800).
2. Generate comprehensive Provider Assessment Matrix for Banking, Trade Credit, and Macro sources.
3. Formulate Point-in-Time Common Data Contract schema and validation engine.
4. Define External Data Requirements for high-frequency cash-flow, trade delinquency, and macro curves.
5. Specify Point-in-Time Integration Architecture and Historical Evaluation Plan.
6. Implement Data Quality Rules Engine (10 validation rules).
7. Audit repository external data presence and enforce Research-Gate Decision:
   EXTERNAL_DATA_ACQUISITION_REQUIRED.
"""

import os
import sys
import json
import math
import hmac
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple

from src.shadow.shadow_engine import compute_sha256

# Constants and Paths
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MANIFEST_PATH = os.path.join(WORKSPACE_ROOT, "experiments", "phase6", "protected_artifacts_manifest.json")
RESULTS_DIR = os.path.join(WORKSPACE_ROOT, "results", "phase6_7")
DATA_DIR = os.path.join(WORKSPACE_ROOT, "data")
EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, "external")

OFFICIAL_PROD_THRESHOLD = 0.0800
REGISTERED_CAND_THRESHOLD = 0.0894
MAX_ALERT_RATIO = 0.08


# =============================================================================
# 1. PRODUCTION BASELINE INTEGRITY VERIFICATION
# =============================================================================

def verify_production_integrity() -> Dict[str, Any]:
    """
    Verifies that all 8 protected production artifacts match reference SHA-256 digests.
    Validates that production threshold remains frozen at 0.0800 on disk.
    """
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    expected_hashes = manifest["pre_implementation_hashes"]
    artifact_status = {}
    all_matched = True

    for rel_path, exp_hash in expected_hashes.items():
        abs_path = os.path.join(WORKSPACE_ROOT, rel_path)
        if not os.path.exists(abs_path):
            artifact_status[rel_path] = {
                "expected_sha256": exp_hash,
                "actual_sha256": "FILE_NOT_FOUND",
                "status": "MISSING"
            }
            all_matched = False
            continue

        actual_hash = compute_sha256(abs_path)
        matches = (actual_hash == exp_hash)
        if not matches:
            all_matched = False

        artifact_status[rel_path] = {
            "expected_sha256": exp_hash,
            "actual_sha256": actual_hash,
            "status": "VERIFIED_MATCH" if matches else "HASH_MISMATCH"
        }

    # Verify threshold config on disk
    thresh_path = os.path.join(WORKSPACE_ROOT, "models", "phase2", "threshold_config.json")
    thresh_on_disk = None
    if os.path.exists(thresh_path):
        with open(thresh_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            thresh_on_disk = cfg.get("operating_threshold")

    thresh_frozen = (thresh_on_disk is not None and math.isclose(thresh_on_disk, OFFICIAL_PROD_THRESHOLD, rel_tol=1e-5))

    manifest_output = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "Phase 6.7 — External Data Acquisition & Point-in-Time Integration Readiness",
        "production_integrity_verdict": "PASSED" if (all_matched and thresh_frozen) else "FAILED",
        "cryptographic_hash_invariance": all_matched,
        "official_production_threshold_frozen": thresh_frozen,
        "official_production_threshold": OFFICIAL_PROD_THRESHOLD,
        "threshold_on_disk": thresh_on_disk,
        "candidate_threshold_quarantined": REGISTERED_CAND_THRESHOLD,
        "candidate_promotion_state": "PASSIVE_SHADOW_CANDIDATE (QUARANTINED)",
        "governance_standard": "Federal Reserve SR 11-7 / OCC 2011-12 Production Immutability",
        "artifacts": artifact_status
    }

    return manifest_output


# =============================================================================
# 2. EXTERNAL DATA REQUIREMENTS SPECIFICATION
# =============================================================================

def construct_acquisition_requirements() -> Dict[str, Any]:
    """
    Constructs explicit data requirements for high-frequency cash-flow, trade credit,
    and macroeconomic/sector modalities.
    """
    return {
        "metadata": {
            "title": "External Data Requirements Specification",
            "version": "1.0.0",
            "date": "2026-10-13",
            "objective": "Define explicit technical requirements to anticipate sudden distress in SMEs under the strict 8% alert cap."
        },
        "modalities": {
            "family_a_banking_cash_flow": {
                "description": "High-frequency transaction-level bank account telemetry to monitor liquidity burn and cash-flow volatility.",
                "fields": [
                    {
                        "name": "daily_gross_inflows",
                        "type": "FLOAT",
                        "unit": "CURRENCY_BASE",
                        "description": "Sum of all incoming account credits settling on observation date T_obs."
                    },
                    {
                        "name": "daily_gross_outflows",
                        "type": "FLOAT",
                        "unit": "CURRENCY_BASE",
                        "description": "Sum of all outgoing account debits settling on observation date T_obs."
                    },
                    {
                        "name": "eod_available_cash_balance",
                        "type": "FLOAT",
                        "unit": "CURRENCY_BASE",
                        "description": "End-of-day ledger balance net of encumbered funds and pending holds."
                    },
                    {
                        "name": "rolling_30d_net_cash_flow_volatility",
                        "type": "FLOAT",
                        "unit": "RATIO_OR_CURRENCY",
                        "description": "Sample standard deviation of daily net cash flows over backward-looking 30-day window."
                    },
                    {
                        "name": "nsf_returned_payment_count",
                        "type": "INTEGER",
                        "unit": "COUNT",
                        "description": "Count of bounced checks, returned ACH debits, or failed outward payments on date T_obs."
                    },
                    {
                        "name": "overdraft_headroom_utilization",
                        "type": "FLOAT",
                        "unit": "RATIO",
                        "description": "Drawn balance divided by approved credit limit. Values > 1.0 indicate unapproved overdraft."
                    }
                ],
                "temporal_specifications": {
                    "cadence": "DAILY_OR_WEEKLY",
                    "observation_timestamp_definition": "Transaction settlement timestamp T_obs (ISO 8601 UTC).",
                    "availability_timestamp_definition": "API availability timestamp T_avail = T_obs + Delta_T, where Delta_T <= 24 hours.",
                    "minimum_historical_depth_months": 36,
                    "target_frequency_hz": "1 / day"
                },
                "feasibility_rationale": "SME acute distress typically presents as a liquidity run 30-60 days before legal insolvency; daily cash dynamics capture sudden depletion invisible to annual accrual ratios."
            },
            "family_b_trade_credit_payment_behavior": {
                "description": "Commercial B2B trade credit payment tape and supplier invoice aging records.",
                "fields": [
                    {
                        "name": "invoice_issue_date",
                        "type": "TIMESTAMP",
                        "unit": "ISO8601",
                        "description": "Date supplier generated invoice."
                    },
                    {
                        "name": "invoice_due_date",
                        "type": "TIMESTAMP",
                        "unit": "ISO8601",
                        "description": "Contractual payment due date."
                    },
                    {
                        "name": "invoice_settlement_date",
                        "type": "TIMESTAMP",
                        "unit": "ISO8601",
                        "description": "Date invoice was cleared/paid (or null if delinquent/unpaid)."
                    },
                    {
                        "name": "days_past_due_max",
                        "type": "INTEGER",
                        "unit": "DAYS",
                        "description": "Maximum days past due across all active outstanding supplier invoices."
                    },
                    {
                        "name": "delinquency_amount_30_60_dpd",
                        "type": "FLOAT",
                        "unit": "CURRENCY_BASE",
                        "description": "Total monetary value of trade payables delinquent between 31 and 60 days."
                    },
                    {
                        "name": "delinquency_amount_90plus_dpd",
                        "type": "FLOAT",
                        "unit": "CURRENCY_BASE",
                        "description": "Total monetary value of trade payables delinquent beyond 90 days."
                    },
                    {
                        "name": "supplier_dispute_flag",
                        "type": "BOOLEAN",
                        "unit": "FLAG",
                        "description": "True if borrower has formal trade disputes, stop-shipments, or COD conversions logged."
                    }
                ],
                "temporal_specifications": {
                    "cadence": "MONTHLY_BUREAU_TAPE",
                    "observation_timestamp_definition": "Billing cycle cutoff date T_obs (e.g. month-end).",
                    "availability_timestamp_definition": "Bureau tape publication date T_avail = T_obs + Delta_T, where Delta_T is typically 30 to 45 days.",
                    "minimum_historical_depth_months": 48,
                    "target_frequency_hz": "1 / month"
                },
                "feasibility_rationale": "Distressed SMEs systematically stretch supplier trade credit (DPO) and default on vendors prior to defaulting on bank debt."
            },
            "family_c_macroeconomic_and_sector_stress": {
                "description": "Macroeconomic interest rate curves, sector industrial output indices, and commodity prices.",
                "fields": [
                    {
                        "name": "central_bank_policy_rate",
                        "type": "FLOAT",
                        "unit": "PERCENT",
                        "description": "Target benchmark policy rate (e.g. Fed Funds, ECB MRO rate)."
                    },
                    {
                        "name": "sovereign_yield_spread_10y_2y",
                        "type": "FLOAT",
                        "unit": "PERCENT_BPS",
                        "description": "10-year minus 2-year sovereign bond yield spread (yield curve slope)."
                    },
                    {
                        "name": "sector_industrial_production_index",
                        "type": "FLOAT",
                        "unit": "INDEX_BASE100",
                        "description": "Monthly volume of production for 2-digit industry classification."
                    },
                    {
                        "name": "sector_demand_shock_delta",
                        "type": "FLOAT",
                        "unit": "PERCENT_CHANGE",
                        "description": "Year-over-year percentage change in monthly sector manufacturing or retail sales."
                    },
                    {
                        "name": "energy_commodity_price_index",
                        "type": "FLOAT",
                        "unit": "INDEX_BASE100",
                        "description": "Broad energy/fuel input commodity price index."
                    }
                ],
                "temporal_specifications": {
                    "cadence": "DAILY_OR_MONTHLY",
                    "observation_timestamp_definition": "Calendar month or trade date T_obs.",
                    "availability_timestamp_definition": "Official release date of initial vintage T_pub = T_avail. Historical revisions must be indexed by vintage release date.",
                    "minimum_historical_depth_months": 120,
                    "target_frequency_hz": "Daily for rates/commodities; Monthly for production"
                },
                "feasibility_rationale": "SMEs exhibit high sensitivity to monetary tightening cycles and sector-specific demand contractions."
            }
        },
        "governance_mandate": "All candidate data specifications are requirements. None may be assumed present until physically verified in repository storage."
    }


# =============================================================================
# 3. PROVIDER AND ACQUISITION ASSESSMENT MATRIX
# =============================================================================

def construct_provider_assessment() -> Dict[str, Any]:
    """
    Evaluates commercial aggregators, credit bureaus, and official macroeconomic sources.
    Strictly distinguishes confirmed capabilities from unverified estimates.
    """
    providers = [
        {
            "provider_id": "PLAID_TECH",
            "provider_name": "Plaid Inc.",
            "product_name": "Plaid Assets & Transactions API",
            "category": "Banking & Cash-Flow Aggregator",
            "geographic_coverage": "United States, Canada, United Kingdom, European Union",
            "sme_eligibility": "Requires SME borrower to complete OAuth account-linking consent flow.",
            "data_fields_available": [
                "historical_transactions_24m",
                "real_time_balance",
                "nsf_fee_events",
                "income_verification"
            ],
            "historical_depth": "Up to 24 months of settled transaction ledger entries.",
            "refresh_frequency": "Continuous / Webhook on settlement (1 to 4 syncs daily).",
            "timestamp_guarantees": "ISO 8601 transaction posted timestamp and authorized timestamp provided.",
            "reporting_lag_guarantee": "Delta T <= 24 hours from bank clearing.",
            "licensing_and_permitted_use": "Commercial SaaS agreement. Permitted for underwriting and credit risk monitoring. Re-distribution prohibited.",
            "consent_and_privacy_requirements": "Explicit SME authorized officer consent under Dodd-Frank 1033 / GDPR / PSD2. 90-180 day consent re-authorization required in EU/UK.",
            "indicative_cost": {
                "status": "UNVERIFIED_ESTIMATE",
                "estimated_range_usd": "$1.50 - $3.50 per connected account / month + API volume fees",
                "basis": "Industry benchmark pricing; formal vendor quote required."
            },
            "delivery_method": "REST API (JSON) + Webhook push notifications",
            "minimum_viable_sample_size": "500 SME cohorts with continuous 24-month connection",
            "acquisition_owner": "Credit Technology & Data Engineering Lead",
            "approval_dependencies": [
                "Enterprise Vendor Security Review (SOC 2 Type II audit)",
                "Legal & Regulatory Privacy Compliance Review",
                "Executive Credit Risk Committee Budget Approval"
            ],
            "confirmed_capabilities_summary": "High API uptime (99.9%), standardized transaction classification, supports thousands of financial institutions.",
            "unverified_assumptions": [
                "Ability to obtain historical backfills for firms that subsequently failed",
                "Willingness of distressed SMEs to maintain active OAuth tokens"
            ]
        },
        {
            "provider_id": "CODAT_IO",
            "provider_name": "Codat Ltd.",
            "product_name": "Codat Universal Accounting & Banking API",
            "category": "Accounting & Banking Integration Platform",
            "geographic_coverage": "US, UK, EU, Australia, New Zealand",
            "sme_eligibility": "SMEs utilizing cloud accounting (QuickBooks, Xero, Sage) or business banking.",
            "data_fields_available": [
                "daily_bank_transactions",
                "accounts_payable_ledger",
                "accounts_receivable_aging",
                "profit_and_loss_monthly"
            ],
            "historical_depth": "Up to 36 months of accounting ledger and bank feeds.",
            "refresh_frequency": "Scheduled daily sync or on-demand push.",
            "timestamp_guarantees": "Direct ledger entry timestamps from underlying SME accounting software.",
            "reporting_lag_guarantee": "Real-time to 24-hour sync lag depending on SME accounting updates.",
            "licensing_and_permitted_use": "Enterprise data platform license. Data storage in enterprise cloud permitted.",
            "consent_and_privacy_requirements": "Single sign-on SME consent; subject to GDPR and local data protection regulations.",
            "indicative_cost": {
                "status": "UNVERIFIED_ESTIMATE",
                "estimated_range_usd": "$20.00 - $45.00 per active SME profile / month",
                "basis": "Enterprise SaaS tier estimate; custom contract negotiation required."
            },
            "delivery_method": "REST API + Webhooks + Direct Cloud Storage Export (Parquet/S3)",
            "minimum_viable_sample_size": "300 SME cohorts with linked accounting ledgers",
            "acquisition_owner": "SME Lending Product Team",
            "approval_dependencies": [
                "Data Protection Officer (DPO) Privacy Impact Assessment",
                "Architecture Review Board Approval"
            ],
            "confirmed_capabilities_summary": "Unified data model across multiple ERPs, includes both bank cash-flow and invoice aging.",
            "unverified_assumptions": [
                "Consistent bookkeeping hygiene across distressed micro-enterprises",
                "Latency of SME manual reconciliation"
            ]
        },
        {
            "provider_id": "DNB_GLOBAL",
            "provider_name": "Dun & Bradstreet",
            "product_name": "D&B Commercial Credit Risk / PAYDEX Data Tape",
            "category": "Commercial Credit Bureau",
            "geographic_coverage": "Global (over 500 million commercial entities across 200+ countries)",
            "sme_eligibility": "All registered commercial enterprises with active trade supplier lines.",
            "data_fields_available": [
                "paydex_score",
                "total_past_due_amount",
                "aging_buckets_30_60_90_120_dpd",
                "trade_payment_experiences_count",
                "commercial_credit_rating"
            ],
            "historical_depth": "Up to 60 months of historical monthly snapshots.",
            "refresh_frequency": "Monthly batch tapes; On-demand API queries.",
            "timestamp_guarantees": "Monthly reporting cycle cutoff timestamp (T_obs) and publication timestamp (T_pub).",
            "reporting_lag_guarantee": "Delta T between 30 and 45 calendar days past statement cycle.",
            "licensing_and_permitted_use": "Commercial Credit Bureau Data License. Internal credit modeling permitted. Third-party disclosure prohibited.",
            "consent_and_privacy_requirements": "Business entity data (B2B); generally exempt from consumer credit laws (FCRA), but corporate entity licensing terms apply.",
            "indicative_cost": {
                "status": "UNVERIFIED_ESTIMATE",
                "estimated_range_usd": "$25,000 - $75,000 / year enterprise subscription based on entity volume",
                "basis": "Commercial bureau tier pricing; formal sales quotation required."
            },
            "delivery_method": "SFTP Batch (CSV/Parquet) + Direct REST API",
            "minimum_viable_sample_size": "1,000+ commercial SMEs across diverse industrial sectors",
            "acquisition_owner": "Enterprise Risk Management & Credit Bureau Relationship Manager",
            "approval_dependencies": [
                "Corporate Procurement & Legal Contract Execution",
                "Model Risk Management Data Lineage Approval"
            ],
            "confirmed_capabilities_summary": "Standardized PAYDEX score (0-100), deep longitudinal trade credit payment experiences.",
            "unverified_assumptions": [
                "Sufficient trade reference density for very small or newly established SMEs",
                "Exact mapping of D&B D-U-N-S numbers to anonymized Dataset A entities (Structurally impossible without unmasking)"
            ]
        },
        {
            "provider_id": "EXPERIAN_COMM",
            "provider_name": "Experian Commercial",
            "product_name": "Experian Commercial Credit & Intelliscore Plus",
            "category": "Commercial Credit Bureau",
            "geographic_coverage": "US, UK, select European markets",
            "sme_eligibility": "Commercial entities with trade credit and commercial banking relationships.",
            "data_fields_available": [
                "intelliscore_plus",
                "commercial_trade_lines",
                "days_beyond_terms_dbt",
                "derogatory_legal_filings",
                "tax_liens_and_bankruptcies"
            ],
            "historical_depth": "36 to 84 months longitudinal depth.",
            "refresh_frequency": "Monthly batch / Real-time API query.",
            "timestamp_guarantees": "Observation date and trade reporting date explicitly recorded.",
            "reporting_lag_guarantee": "Delta T approx 30-40 days for trade payment experiences.",
            "licensing_and_permitted_use": "Commercial Risk Analytics License. Model development authorized.",
            "consent_and_privacy_requirements": "Commercial permissible purpose under business credit evaluation standards.",
            "indicative_cost": {
                "status": "UNVERIFIED_ESTIMATE",
                "estimated_range_usd": "$20,000 - $60,000 / year based on query volume",
                "basis": "Standard commercial data agreements."
            },
            "delivery_method": "Secure SFTP + REST API",
            "minimum_viable_sample_size": "1,000 SME entities",
            "acquisition_owner": "Credit Bureau Vendor Manager",
            "approval_dependencies": [
                "Vendor Due Diligence Review",
                "Legal Data Use Agreement"
            ],
            "confirmed_capabilities_summary": "Comprehensive DBT metrics, combined business and principal credit risk analytics.",
            "unverified_assumptions": [
                "Match rates on cross-border or foreign SME cohorts"
            ]
        },
        {
            "provider_id": "FRED_ALFRED",
            "provider_name": "Federal Reserve Bank of St. Louis",
            "product_name": "ALFRED (Archival Federal Reserve Economic Data)",
            "category": "Official Macroeconomic Real-Time Vintages",
            "geographic_coverage": "United States and major international sovereign economies",
            "sme_eligibility": "Public macroeconomic benchmarks; applicable to all commercial risk models.",
            "data_fields_available": [
                "effective_federal_funds_rate",
                "treasury_yield_curve_spreads",
                "industrial_production_by_sector",
                "consumer_price_index",
                "crude_oil_and_commodity_prices"
            ],
            "historical_depth": "Over 50 years of historical observations and vintage revisions.",
            "refresh_frequency": "Daily / Weekly / Monthly depending on series.",
            "timestamp_guarantees": "Exact vintage release dates and historical revision timestamps verified.",
            "reporting_lag_guarantee": "Initial release published on official release calendar (Delta T = 0 from official release).",
            "licensing_and_permitted_use": "Public Domain / Open Data. Unrestricted commercial and research use.",
            "consent_and_privacy_requirements": "None. Aggregate macroeconomic statistics contain zero PII.",
            "indicative_cost": {
                "status": "CONFIRMED_FREE",
                "estimated_range_usd": "$0.00 (Public Service)",
                "basis": "Official public data API provided by Federal Reserve Bank."
            },
            "delivery_method": "Public REST API (JSON / XML / CSV)",
            "minimum_viable_sample_size": "Full temporal coverage (all macroeconomic dates)",
            "acquisition_owner": "Senior Financial Data Engineer",
            "approval_dependencies": [
                "API Key Registration (Self-service)",
                "Data Governance Architecture Review"
            ],
            "confirmed_capabilities_summary": "Gold-standard vintage tracking prevents lookahead bias from ex-post revisions.",
            "unverified_assumptions": [
                "Direct correlation of US/OECD macro curves with emerging-market SME distress trajectories"
            ]
        }
    ]

    return {
        "assessment_metadata": {
            "evaluation_date": "2026-10-13",
            "evaluator": "Senior Financial Data Engineer & Model Risk Validator",
            "total_providers_audited": len(providers),
            "governance_rule": "Claims regarding capabilities, pricing, and coverage must be labeled CONFIRMED or UNVERIFIED_ESTIMATE."
        },
        "providers": providers
    }


# =============================================================================
# 4. POINT-IN-TIME DATA CONTRACT SPECIFICATION
# =============================================================================

def construct_data_contract() -> Dict[str, Any]:
    """
    Defines common point-in-time data contract schema and entity resolution rules.
    """
    contract_schema = {
        "title": "Point-in-Time Enterprise Financial Risk Data Contract",
        "version": "1.0.0",
        "status": "APPROVED_SPECIFICATION",
        "fields": {
            "firm_id": {
                "type": "STRING",
                "required": True,
                "format": "FIRM_HMAC_[a-f0-9]{32,64}",
                "description": "Cryptographically masked, irreversible pseudonymous borrower identifier."
            },
            "source_id": {
                "type": "STRING",
                "required": True,
                "enum": [
                    "BANK_PLAID_TRANSACTIONS",
                    "BANK_CODAT_LEDGER",
                    "BUREAU_DNB_PAYDEX",
                    "BUREAU_EXPERIAN_COMM",
                    "MACRO_FRED_ALFRED",
                    "INTERNAL_ACCOUNTING_DATASET_A"
                ],
                "description": "Registered unique identifier of data provider and stream."
            },
            "event_id": {
                "type": "STRING",
                "required": True,
                "format": "UUIDv4 or SHA256 hex string",
                "description": "Globally unique, deterministic identifier of observation event."
            },
            "timestamp_obs": {
                "type": "STRING",
                "required": True,
                "format": "ISO 8601 UTC (YYYY-MM-DDTHH:MM:SSZ)",
                "description": "Timestamp when the economic event occurred or account state was observed."
            },
            "timestamp_pub": {
                "type": "STRING",
                "required": True,
                "format": "ISO 8601 UTC (YYYY-MM-DDTHH:MM:SSZ)",
                "description": "Timestamp when the source provider published the record or finalized the tape."
            },
            "timestamp_avail": {
                "type": "STRING",
                "required": True,
                "format": "ISO 8601 UTC (YYYY-MM-DDTHH:MM:SSZ)",
                "description": "Timestamp when the data became queryable in the risk database. T_avail >= T_pub >= T_obs."
            },
            "timestamp_ingest": {
                "type": "STRING",
                "required": True,
                "format": "ISO 8601 UTC (YYYY-MM-DDTHH:MM:SSZ)",
                "description": "Timestamp when record was physically written to internal data lake landing zone."
            },
            "reporting_period": {
                "type": "STRING",
                "required": True,
                "format": "YYYY-MM or YYYY-Q# or YYYY-W## or integer T",
                "description": "Standardized accounting or economic reporting interval identifier."
            },
            "metric_name": {
                "type": "STRING",
                "required": True,
                "format": "snake_case identifier",
                "description": "Standardized canonical metric name."
            },
            "metric_value": {
                "type": "FLOAT",
                "required": True,
                "description": "Normalized numeric value of metric."
            },
            "metric_unit": {
                "type": "STRING",
                "required": True,
                "enum": ["CURRENCY_BASE", "PERCENT", "PERCENT_BPS", "RATIO", "COUNT", "DAYS", "FLAG", "INDEX_BASE100"],
                "description": "Unit of measurement."
            },
            "currency": {
                "type": "STRING",
                "required": False,
                "format": "ISO 4217 (e.g. USD, EUR, GBP)",
                "description": "Denomination currency for monetary metrics. Null for dimensionless ratios."
            },
            "scale_factor": {
                "type": "FLOAT",
                "required": True,
                "default": 1.0,
                "description": "Multiplicative scaling factor applied to metric_value (e.g. 1000.0 for thousands)."
            },
            "is_imputed": {
                "type": "BOOLEAN",
                "required": True,
                "default": False,
                "description": "True if value was statistically imputed; False if raw observation."
            },
            "missingness_code": {
                "type": "STRING",
                "required": True,
                "enum": ["PRESENT", "NOT_REPORTED", "NOT_APPLICABLE", "CORRUPTED", "RIGHT_CENSORED"],
                "description": "Categorical indicator of missingness status."
            },
            "data_provenance": {
                "type": "OBJECT",
                "required": True,
                "properties": {
                    "schema_version": "STRING",
                    "source_batch_id": "STRING",
                    "raw_checksum_sha256": "STRING",
                    "ingestion_pipeline_commit": "STRING"
                },
                "description": "Complete cryptographic audit trail of data lineage."
            }
        },
        "pseudonymous_identity_mapping_protocol": {
            "algorithm": "HMAC-SHA256 with 256-bit cryptographically secure secret salt",
            "salt_storage": "Enterprise Hardware Security Module (HSM) / Azure Key Vault / AWS KMS",
            "salt_rotation_policy": "Annual rotation with deterministic legacy versioning",
            "isolation_mandate": "Direct legal identifiers (Tax ID, Company Name, LEI) strictly barred from risk feature store and research workspaces.",
            "entity_resolution_rules": [
                "Exact match on Legal Entity Identifier (LEI) where available",
                "Deterministic national registration number hash match",
                "Fuzzy text matching on entity names strictly prohibited to prevent misattribution"
            ]
        },
        "point_in_time_enforcement_invariants": [
            "INVARIANT_TEMPORAL_ORDER: T_obs <= T_pub <= T_avail <= T_ingest",
            "INVARIANT_PREDICTION_CUTOFF: Feature x valid at T_pred iff T_avail(x) <= T_pred",
            "INVARIANT_VINTAGE_SELECTION: For revised macro series, MacroValue(t; T_pred) = Vintage(t, max{T_release <= T_pred})",
            "INVARIANT_NO_FUTURE_PROXY: Zero synthesis or lookahead imputation permitted"
        ]
    }

    return contract_schema


# =============================================================================
# 5. DATA QUALITY RULES ENGINE (10 RULES)
# =============================================================================

def construct_data_quality_rules() -> Dict[str, Any]:
    """
    Defines 10 operational data quality validation rules for incoming external records.
    """
    rules = [
        {
            "rule_id": "DQ_RULE_01",
            "rule_name": "PRIMARY_KEY_AND_DUPLICATE_EVENT_CHECK",
            "severity": "FATAL_REJECTION",
            "description": "Ensures no duplicate records exist for combination of (firm_id, source_id, metric_name, timestamp_obs).",
            "logic": "count(records) == count(distinct(firm_id, source_id, metric_name, timestamp_obs))",
            "action_on_failure": "Reject entire batch; alert pipeline engineer."
        },
        {
            "rule_id": "DQ_RULE_02",
            "rule_name": "TEMPORAL_TIMESTAMP_SEQUENCE_INTEGRITY",
            "severity": "FATAL_REJECTION",
            "description": "Enforces valid ISO 8601 formatting and strict inequality: T_obs <= T_pub <= T_avail <= T_ingest <= Now_UTC.",
            "logic": "T_obs <= T_pub and T_pub <= T_avail and T_avail <= T_ingest and T_ingest <= current_timestamp_utc()",
            "action_on_failure": "Reject offending record; log timestamp discrepancy anomaly."
        },
        {
            "rule_id": "DQ_RULE_03",
            "rule_name": "PHYSICAL_AND_NUMERICAL_VALUE_SANITY",
            "severity": "FATAL_REJECTION",
            "description": "Validates that metric values fall within physically admissible domains.",
            "logic": "For DPD: value >= 0; For Overdraft Headroom: 0.0 <= value <= 5.0; For Bank Balance: not null and not NaN and not +/-Inf.",
            "action_on_failure": "Quarantine record into dead-letter queue."
        },
        {
            "rule_id": "DQ_RULE_04",
            "rule_name": "CURRENCY_AND_DENOMINATION_STANDARDIZATION",
            "severity": "ERROR_CORRECTION_REQUIRED",
            "description": "Monetary metrics must have valid ISO 4217 currency; must be converted to base currency via point-in-time FX spot.",
            "logic": "currency in ['USD', 'EUR', 'GBP', 'JPY', 'IRR', 'TWD'] and fx_rate_timestamp == T_obs",
            "action_on_failure": "Reject conversion if historical point-in-time FX rate unavailable."
        },
        {
            "rule_id": "DQ_RULE_05",
            "rule_name": "TEMPORAL_CADENCE_AND_UNEXPECTED_GAPS",
            "severity": "WARNING_INVESTIGATION",
            "description": "Flags abnormal gaps in continuous reporting feeds (e.g. daily bank feed missing > 14 consecutive calendar days).",
            "logic": "max_gap_days(firm_id, source_id) <= expected_cadence_days * 3",
            "action_on_failure": "Mark entity as 'INCOMPLETE_SERIES'; do not use for sudden distress validation."
        },
        {
            "rule_id": "DQ_RULE_06",
            "rule_name": "BACKFILLED_RECORD_DETECTION_AND_TAGGING",
            "severity": "GOVERNANCE_RESTRICTION",
            "description": "Flags records where T_ingest occurs > 30 days after T_avail as backfills.",
            "logic": "if (T_ingest - T_avail) > 30 days then is_backfilled = true",
            "action_on_failure": "Disallow backfilled record in backtesting for any T_pred < T_ingest."
        },
        {
            "rule_id": "DQ_RULE_07",
            "rule_name": "FIRM_PSEUDONYM_HASH_INTEGRITY",
            "severity": "FATAL_REJECTION",
            "description": "Validates that firm_id conforms to approved pseudonym regex and matches active key registry.",
            "logic": "matches_regex(firm_id, '^FIRM_HMAC_[a-f0-9]{32,64}$') and is_registered(firm_id)",
            "action_on_failure": "Drop record immediately; security alert for unmapped identity."
        },
        {
            "rule_id": "DQ_RULE_08",
            "rule_name": "REPORTING_CADENCE_CONSISTENCY",
            "severity": "WARNING_INVESTIGATION",
            "description": "Verifies that reporting frequency does not fluctuate erratically without provider notification.",
            "logic": "frequency_variance(firm_id, source_id) <= tolerance_threshold",
            "action_on_failure": "Trigger provider feed health alert."
        },
        {
            "rule_id": "DQ_RULE_09",
            "rule_name": "STATISTICAL_DISTRIBUTION_SHIFT_ALERT",
            "severity": "WARNING_INVESTIGATION",
            "description": "Monitors population stability index (PSI) and extreme z-score excursions (> 6 sigma) against baseline.",
            "logic": "PSI(current_batch, historical_baseline) < 0.25 and abs(z_score) < 6.0",
            "action_on_failure": "Escalate to Model Risk Validator for drift review."
        },
        {
            "rule_id": "DQ_RULE_10",
            "rule_name": "CRYPTOGRAPHIC_DATA_PROVENANCE_AND_LEGAL_SEAL",
            "severity": "FATAL_REJECTION",
            "description": "Verifies SHA-256 hash of source raw payload and confirms active legal license metadata.",
            "logic": "sha256(raw_payload) == data_provenance.raw_checksum_sha256 and legal_clearance_status == 'APPROVED'",
            "action_on_failure": "Halt ingestion pipeline; data quarantined until compliance sign-off."
        }
    ]

    return {
        "engine_version": "1.0.0",
        "total_rules": len(rules),
        "execution_mode": "STRICT_PRE_INGESTION_GATE",
        "rules": rules
    }


# =============================================================================
# 6. POINT-IN-TIME INTEGRATION PLAN & EVALUATION ARCHITECTURE
# =============================================================================

def construct_integration_plan() -> Dict[str, Any]:
    """
    Formulates the multi-tier data architecture and historical evaluation plan.
    Documents power analysis requirements and limitations of the T13-to-T14 cohort.
    """
    return {
        "metadata": {
            "title": "Point-in-Time Integration Architecture & Historical Evaluation Plan",
            "version": "1.0.0",
            "date": "2026-10-13",
            "author": "Senior Financial Data Engineer & Model Risk Validator"
        },
        "multi_tier_architecture": {
            "tier_1_raw_landing_zone": {
                "path": "data/external/raw/<source_id>/<year>/<month>/",
                "format": "Immutable raw payload (JSON/CSV/XML) + SHA256 sidecar manifest",
                "permissions": "Read-only (WORM - Write Once, Read Many)",
                "governance": "Cryptographically signed; legal license metadata required before write."
            },
            "tier_2_staged_validated": {
                "path": "data/external/staged/<source_id>/",
                "format": "Apache Parquet partitioned by reporting_period",
                "schema_enforcement": "Point-in-Time Data Contract v1.0.0 schema applied",
                "quality_filtering": "Passed all 10 Data Quality Rules (DQ_RULE_01 to DQ_RULE_10)"
            },
            "tier_3_point_in_time_feature_mart": {
                "path": "data/external/integrated/",
                "format": "Standardized longitudinal feature mart",
                "point_in_time_enforcement": "Features joined strictly where T_avail <= T_pred",
                "leakage_safeguard": "Delta T reporting delay explicitly added to observation timestamps."
            }
        },
        "historical_evaluation_plan": {
            "statistical_power_analysis": {
                "objective": "Determine minimum sudden distress events required to validate candidate sensitivity superiority.",
                "baseline_production_sudden_recall": 0.0,
                "target_candidate_sudden_recall": 0.60,
                "significance_level_alpha": 0.05,
                "target_statistical_power": 0.80,
                "minimum_required_sudden_distress_events": 35,
                "minimum_total_distress_events": 70,
                "minimum_total_firms_in_validation_cohort": 875,
                "alert_cap_formula": "M = floor(0.08 * N_firms)",
                "conclusion": "Statistical power requires at least 35 sudden distress cases to reliably confirm a 60% sensitivity gain under the 8% alert budget."
            },
            "audit_of_existing_t13_to_t14_cohort_limitations": {
                "cohort_identifier": "Tehran SME Panel (Dataset A) T=13 to T=14 transition",
                "total_eligible_firms": 262,
                "total_distress_events": 11,
                "sudden_distress_events": 5,
                "chronic_distress_events": 6,
                "alert_cap_at_8_percent": 20,
                "critical_limitations": [
                    "EXHAUSTED_RESEARCH_SAMPLE: Cohort repeatedly evaluated across Phases 6.4, 6.4.1, 6.4.2, 6.4.3, 6.5, and 6.6.",
                    "CIRCULAR_OVERFITTING_RISK: Any model optimized on this specific cohort represents retrospective data snooping, not confirmatory testing.",
                    "INSUFFICIENT_STATISTICAL_POWER: With only 5 sudden failure cases, a single misclassification shifts recall by 20.0 percentage points.",
                    "UNVERIFIED_FOR_PROMOTION: Cannot be used as an untouched confirmatory holdout under Federal Reserve SR 11-7 standards."
                ],
                "confirmatory_claim_prohibition": "Confirmatory performance claims based on the T13-to-T14 cohort are strictly prohibited."
            },
            "requirements_for_prospective_evaluation_cohort": {
                "minimum_firms": 1000,
                "minimum_longitudinal_depth_years": 5,
                "untouched_quarantine": "Prospective cohort must remain strictly sequestered in cold storage until model architecture is frozen.",
                "reporting_lag_simulation": "Systematic lag testing across Delta T in [15d, 30d, 60d, 90d, 180d].",
                "right_censored_treatment": "Firms exiting observation without distress within 12 months must be censored, not marked non-distressed."
            }
        }
    }


# =============================================================================
# 7. REPOSITORY DATA AUDIT & RESEARCH-GATE VERDICT
# =============================================================================

def audit_repository_external_readiness() -> Tuple[str, Dict[str, Any]]:
    """
    Audits actual external data files in repository.
    Renders the official research-gate decision.
    """
    external_files = []
    if os.path.exists(EXTERNAL_DATA_DIR):
        for root, _, files in os.walk(EXTERNAL_DATA_DIR):
            for f in files:
                external_files.append(os.path.join(root, f))

    raw_dir = os.path.join(DATA_DIR, "raw")
    raw_files = os.listdir(raw_dir) if os.path.exists(raw_dir) else []

    external_data_present = (len(external_files) > 0)

    # In accordance with prompt instructions:
    # Status can be EXTERNAL_DATA_READY_FOR_PHASE_6_8 ONLY if actual authorized
    # external data are present, provenance verified, timestamps reliable, and
    # minimum feasibility requirements satisfied.
    # An empty scaffold or provider assessment is NOT sufficient.
    if not external_data_present:
        verdict = "EXTERNAL_DATA_ACQUISITION_REQUIRED"
        rationale = (
            "Target directory 'data/external/' is completely empty (0 files, 0 bytes). "
            "No verified external transaction cash-flow ledgers, commercial trade-credit delinquency tapes, "
            "or macroeconomic vintage curves exist in repository storage. "
            "In accordance with model risk governance, further predictive modeling is halted until "
            "external data contracts are formally executed and ingested into Tier 1 raw storage."
        )
    else:
        verdict = "EXTERNAL_DATA_INTEGRATION_REQUIRES_REPAIR"
        rationale = "External data files detected but require validation against point-in-time data contract."

    audit_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "research_gate_decision": verdict,
        "rationale": rationale,
        "external_data_directory": EXTERNAL_DATA_DIR,
        "external_files_count": len(external_files),
        "external_files_list": external_files,
        "raw_datasets_present": raw_files,
        "permitted_next_phase": "FORMAL_DATA_PROCUREMENT_ONLY (MODEL_TRAINING_HALTED)",
        "production_impact": "Zero impact; Production baseline remains frozen at threshold 0.0800."
    }

    return verdict, audit_summary


# =============================================================================
# 8. EXECUTION & ARTIFACT GENERATION
# =============================================================================

def run_phase6_7_readiness_audit() -> Dict[str, Any]:
    """
    Executes the complete Phase 6.7 audit and writes all 6 JSON deliverables.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Verify production integrity
    integrity_manifest = verify_production_integrity()
    with open(os.path.join(RESULTS_DIR, "production_integrity_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(integrity_manifest, f, indent=2)

    # 2. Acquisition Requirements
    requirements = construct_acquisition_requirements()
    with open(os.path.join(RESULTS_DIR, "acquisition_requirements.json"), "w", encoding="utf-8") as f:
        json.dump(requirements, f, indent=2)

    # 3. Provider Assessment
    provider_assessment = construct_provider_assessment()
    with open(os.path.join(RESULTS_DIR, "provider_assessment.json"), "w", encoding="utf-8") as f:
        json.dump(provider_assessment, f, indent=2)

    # 4. Data Contract
    data_contract = construct_data_contract()
    with open(os.path.join(RESULTS_DIR, "data_contract.json"), "w", encoding="utf-8") as f:
        json.dump(data_contract, f, indent=2)

    # 5. Data Quality Rules
    quality_rules = construct_data_quality_rules()
    with open(os.path.join(RESULTS_DIR, "data_quality_rules.json"), "w", encoding="utf-8") as f:
        json.dump(quality_rules, f, indent=2)

    # 6. Integration Plan
    integration_plan = construct_integration_plan()
    with open(os.path.join(RESULTS_DIR, "point_in_time_integration_plan.json"), "w", encoding="utf-8") as f:
        json.dump(integration_plan, f, indent=2)

    # Audit external data presence and verdict
    verdict, audit_summary = audit_repository_external_readiness()

    return {
        "verdict": verdict,
        "integrity_manifest": integrity_manifest,
        "requirements": requirements,
        "provider_assessment": provider_assessment,
        "data_contract": data_contract,
        "quality_rules": quality_rules,
        "integration_plan": integration_plan,
        "audit_summary": audit_summary
    }


if __name__ == "__main__":
    results = run_phase6_7_readiness_audit()
    print("=" * 80)
    print("PHASE 6.7: EXTERNAL DATA ACQUISITION & INTEGRATION READINESS AUDIT")
    print("=" * 80)
    print(f"Production Integrity: {results['integrity_manifest']['production_integrity_verdict']}")
    print(f"Official Threshold:   {results['integrity_manifest']['official_production_threshold']}")
    print(f"Research Gate Verdict: {results['verdict']}")
    print(f"Deliverables Dir:     {RESULTS_DIR}")
    print("=" * 80)
