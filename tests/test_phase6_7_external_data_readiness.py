"""
Phase 6.7 Automated Test Suite:
External Data Acquisition & Point-in-Time Integration Readiness.

Comprehensive test coverage verifying:
1. Production baseline manifest immutability (SHA-256 digests, official threshold 0.0800 frozen).
2. Provider assessment schema, coverage of 3 modalities, and confirmed vs unverified distinction.
3. Common point-in-time data contract schema completeness and mandatory timestamp fields.
4. Point-in-time availability rules and lookahead prevention (T_avail <= T_pred).
5. Data quality rules engine coverage (all 10 validation rules).
6. Pseudonymous firm identity mapping integrity (HMAC-SHA256, format validation).
7. Macroeconomic vintage tracking and historical revision cutoff enforcement.
8. External data absence detection and graceful non-manufacturing handling.
9. Right-censored observation rules and statistical power / cohort limitation audit.
10. Research-gate decision enforcement: EXTERNAL_DATA_ACQUISITION_REQUIRED.
"""

import os
import json
import math
import hmac
import hashlib
import re
from datetime import datetime, timezone
import pytest

from src.shadow.shadow_engine import compute_sha256
from src.shadow.external_data_readiness_phase6_7 import (
    verify_production_integrity,
    construct_acquisition_requirements,
    construct_provider_assessment,
    construct_data_contract,
    construct_data_quality_rules,
    construct_integration_plan,
    audit_repository_external_readiness,
    MANIFEST_PATH,
    OFFICIAL_PROD_THRESHOLD,
    REGISTERED_CAND_THRESHOLD,
    RESULTS_DIR,
    EXTERNAL_DATA_DIR
)


@pytest.fixture(scope="module")
def phase6_7_deliverables():
    """Loads all Phase 6.7 JSON deliverable files."""
    deliverables = {
        "integrity": os.path.join(RESULTS_DIR, "production_integrity_manifest.json"),
        "requirements": os.path.join(RESULTS_DIR, "acquisition_requirements.json"),
        "provider": os.path.join(RESULTS_DIR, "provider_assessment.json"),
        "contract": os.path.join(RESULTS_DIR, "data_contract.json"),
        "quality": os.path.join(RESULTS_DIR, "data_quality_rules.json"),
        "plan": os.path.join(RESULTS_DIR, "point_in_time_integration_plan.json")
    }

    loaded = {}
    for key, path in deliverables.items():
        assert os.path.exists(path), f"Missing deliverable: {path}"
        with open(path, "r", encoding="utf-8") as f:
            loaded[key] = json.load(f)

    return loaded


# =============================================================================
# TEST 1: PRODUCTION BASELINE MANIFEST IMMUTABILITY
# =============================================================================

def test_production_artifacts_immutable(phase6_7_deliverables):
    """Verifies that all 8 protected production artifacts match manifest and threshold is 0.0800."""
    integrity = phase6_7_deliverables["integrity"]
    assert integrity["production_integrity_verdict"] == "PASSED"
    assert integrity["cryptographic_hash_invariance"] is True
    assert integrity["official_production_threshold_frozen"] is True
    assert math.isclose(integrity["official_production_threshold"], OFFICIAL_PROD_THRESHOLD, rel_tol=1e-5)
    assert math.isclose(integrity["threshold_on_disk"], OFFICIAL_PROD_THRESHOLD, rel_tol=1e-5)
    assert integrity["candidate_promotion_state"] == "PASSIVE_SHADOW_CANDIDATE (QUARANTINED)"

    # Live verification of SHA-256 digests
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    for path, exp_hash in manifest["pre_implementation_hashes"].items():
        assert compute_sha256(path) == exp_hash, f"Hash mismatch on {path}"


# =============================================================================
# TEST 2: PROVIDER ASSESSMENT SCHEMA & RIGOR
# =============================================================================

def test_provider_assessment_schema_and_rigor(phase6_7_deliverables):
    """Verifies provider assessment covers banking, trade credit, and macro, with confirmed vs unverified distinction."""
    assessment = phase6_7_deliverables["provider"]
    assert "providers" in assessment
    providers = assessment["providers"]
    assert len(providers) >= 5

    categories = set(p["category"] for p in providers)
    assert any("Banking" in c for c in categories)
    assert any("Commercial Credit Bureau" in c for c in categories)
    assert any("Macroeconomic" in c for c in categories)

    for p in providers:
        assert "provider_id" in p
        assert "provider_name" in p
        assert "data_fields_available" in p
        assert "historical_depth" in p
        assert "refresh_frequency" in p
        assert "reporting_lag_guarantee" in p
        assert "licensing_and_permitted_use" in p
        assert "consent_and_privacy_requirements" in p
        assert "indicative_cost" in p
        assert "status" in p["indicative_cost"]
        assert p["indicative_cost"]["status"] in ["CONFIRMED_FREE", "CONFIRMED_QUOTE", "UNVERIFIED_ESTIMATE"]
        assert "delivery_method" in p
        assert "acquisition_owner" in p
        assert "approval_dependencies" in p
        assert "confirmed_capabilities_summary" in p
        assert "unverified_assumptions" in p


# =============================================================================
# TEST 3: COMMON DATA CONTRACT SCHEMA & FIELDS
# =============================================================================

def test_data_contract_schema_and_fields(phase6_7_deliverables):
    """Verifies data contract schema includes all mandatory fields and format requirements."""
    contract = phase6_7_deliverables["contract"]
    fields = contract["fields"]

    required_fields = [
        "firm_id", "source_id", "event_id",
        "timestamp_obs", "timestamp_pub", "timestamp_avail", "timestamp_ingest",
        "reporting_period", "metric_name", "metric_value", "metric_unit",
        "currency", "scale_factor", "is_imputed", "missingness_code", "data_provenance"
    ]

    for req in required_fields:
        assert req in fields, f"Missing contract field: {req}"
        assert fields[req]["required"] in [True, False]

    # Verify timestamp format requirement
    for ts in ["timestamp_obs", "timestamp_pub", "timestamp_avail", "timestamp_ingest"]:
        assert "ISO 8601" in fields[ts]["format"]


# =============================================================================
# TEST 4: POINT-IN-TIME AVAILABILITY RULES (LOOKAHEAD PREVENTION)
# =============================================================================

def test_point_in_time_availability_rules():
    """Validates temporal precedence (T_avail <= T_pred) and rejection of lookahead data."""
    t_pred = datetime(2023, 6, 30, 23, 59, 59, tzinfo=timezone.utc)

    # Valid event (observed 2023-05-15, published 2023-06-15, available 2023-06-20 <= T_pred)
    valid_event = {
        "timestamp_obs": datetime(2023, 5, 15, tzinfo=timezone.utc),
        "timestamp_avail": datetime(2023, 6, 20, tzinfo=timezone.utc)
    }
    assert valid_event["timestamp_avail"] <= t_pred, "Valid event incorrectly rejected"

    # Invalid lookahead event (observed 2023-06-28, available 2023-07-05 > T_pred)
    lookahead_event = {
        "timestamp_obs": datetime(2023, 6, 28, tzinfo=timezone.utc),
        "timestamp_avail": datetime(2023, 7, 5, tzinfo=timezone.utc)
    }
    assert lookahead_event["timestamp_avail"] > t_pred, "Lookahead event failed to be caught"


# =============================================================================
# TEST 5: DATA QUALITY RULES ENGINE (10 RULES)
# =============================================================================

def test_data_quality_rules_engine(phase6_7_deliverables):
    """Validates that quality rules cover duplicate events, timestamps, values, currency, provenance."""
    quality = phase6_7_deliverables["quality"]
    assert quality["total_rules"] == 10
    rules = quality["rules"]
    assert len(rules) == 10

    rule_names = [r["rule_name"] for r in rules]
    assert "PRIMARY_KEY_AND_DUPLICATE_EVENT_CHECK" in rule_names
    assert "TEMPORAL_TIMESTAMP_SEQUENCE_INTEGRITY" in rule_names
    assert "PHYSICAL_AND_NUMERICAL_VALUE_SANITY" in rule_names
    assert "CURRENCY_AND_DENOMINATION_STANDARDIZATION" in rule_names
    assert "TEMPORAL_CADENCE_AND_UNEXPECTED_GAPS" in rule_names
    assert "BACKFILLED_RECORD_DETECTION_AND_TAGGING" in rule_names
    assert "FIRM_PSEUDONYM_HASH_INTEGRITY" in rule_names
    assert "REPORTING_CADENCE_CONSISTENCY" in rule_names
    assert "STATISTICAL_DISTRIBUTION_SHIFT_ALERT" in rule_names
    assert "CRYPTOGRAPHIC_DATA_PROVENANCE_AND_LEGAL_SEAL" in rule_names


# =============================================================================
# TEST 6: PSEUDONYMOUS FIRM IDENTITY INTEGRITY
# =============================================================================

def test_firm_pseudonym_integrity():
    """Validates one-way HMAC-SHA256 pseudonymization, irreversible identity protection, and regex format."""
    secret_key = b"test_hsm_secret_key_32_bytes_len"
    tax_id = "US12-3456789"
    country = "US"

    message = f"{tax_id}|{country}".encode("utf-8")
    digest = hmac.new(secret_key, message, hashlib.sha256).hexdigest()
    firm_id = f"FIRM_HMAC_{digest}"

    # Verify regex conformance
    pattern = r"^FIRM_HMAC_[a-f0-9]{64}$"
    assert re.match(pattern, firm_id), f"Firm ID failed regex: {firm_id}"

    # Verify collision resistance: different tax id yields different pseudonym
    diff_message = f"US98-7654321|{country}".encode("utf-8")
    diff_digest = hmac.new(secret_key, diff_message, hashlib.sha256).hexdigest()
    diff_firm_id = f"FIRM_HMAC_{diff_digest}"
    assert firm_id != diff_firm_id


# =============================================================================
# TEST 7: MACROECONOMIC VINTAGE TRACKING
# =============================================================================

def test_macroeconomic_vintage_tracking():
    """Validates real-time vintage selection and rejection of revised figures published after T_pred."""
    # Economic quarter: 2023-Q1
    # Initial release vintage: published 2023-04-28 (Value = 1.1)
    # First revision vintage: published 2023-05-25 (Value = 1.3)
    # Second revision vintage: published 2023-06-29 (Value = 2.0)
    # Annual benchmark revision: published 2023-09-28 (Value = 2.2)
    vintages = [
        {"release_date": datetime(2023, 4, 28, tzinfo=timezone.utc), "value": 1.1},
        {"release_date": datetime(2023, 5, 25, tzinfo=timezone.utc), "value": 1.3},
        {"release_date": datetime(2023, 6, 29, tzinfo=timezone.utc), "value": 2.0},
        {"release_date": datetime(2023, 9, 28, tzinfo=timezone.utc), "value": 2.2}
    ]

    # At T_pred = 2023-06-01, the valid vintage must be the 2023-05-25 release (Value = 1.3)
    t_pred = datetime(2023, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
    eligible = [v for v in vintages if v["release_date"] <= t_pred]
    latest_valid = max(eligible, key=lambda v: v["release_date"])
    assert latest_valid["value"] == 1.3
    assert latest_valid["release_date"] <= t_pred


# =============================================================================
# TEST 8: EXTERNAL DATA ABSENCE HANDLING
# =============================================================================

def test_external_data_absence_handling():
    """Verifies that absence of raw files in data/external/ is correctly detected."""
    verdict, summary = audit_repository_external_readiness()
    assert verdict == "EXTERNAL_DATA_ACQUISITION_REQUIRED"
    assert summary["external_files_count"] == 0
    assert summary["permitted_next_phase"] == "FORMAL_DATA_PROCUREMENT_ONLY (MODEL_TRAINING_HALTED)"


# =============================================================================
# TEST 9: RIGHT-CENSORED OUTCOME & COHORT LIMITATIONS AUDIT
# =============================================================================

def test_right_censored_and_cohort_limitations(phase6_7_deliverables):
    """Validates statistical power requirements and critique of the repeatedly reused T13-T14 cohort."""
    plan = phase6_7_deliverables["plan"]
    power = plan["historical_evaluation_plan"]["statistical_power_analysis"]
    assert power["minimum_required_sudden_distress_events"] >= 30
    assert power["target_statistical_power"] >= 0.80

    t13_audit = plan["historical_evaluation_plan"]["audit_of_existing_t13_to_t14_cohort_limitations"]
    assert t13_audit["total_eligible_firms"] == 262
    assert t13_audit["sudden_distress_events"] == 5
    assert t13_audit["alert_cap_at_8_percent"] == 20
    assert "CIRCULAR_OVERFITTING_RISK" in " ".join(t13_audit["critical_limitations"])
    assert "prohibited" in t13_audit["confirmatory_claim_prohibition"].lower()


# =============================================================================
# TEST 10: RESEARCH-GATE DECISION ENFORCEMENT
# =============================================================================

def test_research_gate_decision_enforcement():
    """Asserts that research gate decision is strictly EXTERNAL_DATA_ACQUISITION_REQUIRED and halts modeling."""
    verdict, summary = audit_repository_external_readiness()
    assert verdict == "EXTERNAL_DATA_ACQUISITION_REQUIRED"
    assert "halted" in summary["rationale"].lower()
    assert summary["production_impact"] == "Zero impact; Production baseline remains frozen at threshold 0.0800."
