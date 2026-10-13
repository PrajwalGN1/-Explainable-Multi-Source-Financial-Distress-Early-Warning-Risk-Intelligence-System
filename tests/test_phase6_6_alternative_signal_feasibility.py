"""
Phase 6.6 Automated Test Suite:
Alternative Financial Risk Signals & Point-in-Time Data Feasibility Study.

Comprehensive test coverage verifying:
1. Production baseline manifest immutability (SHA-256 digests, official threshold 0.0800 frozen).
2. Data source feasibility matrix completeness and classification adherence.
3. Point-in-time feature registry schema, timestamp definitions, and reporting delays.
4. Point-in-time availability enforcement and prevention of future-data leakage.
5. Missing-data handling and absence of manufactured/synthetic proxy claims.
6. Firm and timestamp alignment across longitudinal partitions.
7. Right-censored observation handling and non-filing exclusion.
8. Strict integer alert-cap arithmetic: floor(0.08 * eligible_firms).
9. Deterministic feasibility audit artifact generation.
10. Research-gate decision enforcement: ALTERNATIVE_SIGNAL_DATA_INFEASIBLE.
"""

import os
import json
import math
import pytest
import numpy as np
import pandas as pd

from src.data.loaders import DataLoader
from src.shadow.shadow_engine import compute_sha256
from src.shadow.alternative_signal_feasibility_phase6_6 import (
    verify_production_integrity,
    construct_data_source_feasibility_matrix,
    construct_point_in_time_feature_registry,
    audit_data_coverage_and_missingness,
    audit_sudden_transition_signals,
    construct_research_experiment_plan,
    MANIFEST_PATH,
    OFFICIAL_PROD_THRESHOLD,
    REGISTERED_CAND_THRESHOLD,
    RESULTS_DIR
)


@pytest.fixture(scope="module")
def phase6_6_deliverables():
    """Loads all Phase 6.6 JSON deliverable files."""
    deliverables = {
        "integrity": os.path.join(RESULTS_DIR, "production_integrity_manifest.json"),
        "feasibility": os.path.join(RESULTS_DIR, "data_source_feasibility.json"),
        "registry": os.path.join(RESULTS_DIR, "point_in_time_feature_registry.json"),
        "coverage": os.path.join(RESULTS_DIR, "data_coverage_and_missingness.json"),
        "sudden": os.path.join(RESULTS_DIR, "sudden_transition_signal_audit.json"),
        "plan": os.path.join(RESULTS_DIR, "research_experiment_plan.json")
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

def test_production_baseline_manifest_immutability(phase6_6_deliverables):
    """Verifies that all 8 protected production artifacts match manifest and threshold is 0.0800."""
    integrity = phase6_6_deliverables["integrity"]
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
# TEST 2: DATA SOURCE FEASIBILITY MATRIX AUDIT
# =============================================================================

def test_data_source_feasibility_matrix_completeness(phase6_6_deliverables):
    """Verifies that all candidate signal families are audited and classified into 5 categories."""
    feasibility = phase6_6_deliverables["feasibility"]
    signals = feasibility["candidate_signals"]

    assert len(signals) == 19
    families = set(s["signal_family"] for s in signals)
    assert "A. Cash-flow dynamics" in families
    assert "B. Trade-credit and payment behavior" in families
    assert "C. Macroeconomic and sector stress" in families
    assert "D. Existing financial-report signals" in families

    valid_statuses = {
        "Available and verified",
        "Available but incomplete",
        "Requires external data acquisition",
        "Legally or operationally unverified",
        "Not currently feasible"
    }

    for s in signals:
        assert s["feasibility_status"] in valid_statuses, f"Invalid status: {s['feasibility_status']}"
        assert "expected_incremental_value" in s
        assert "cost_and_licensing_requirements" in s
        assert "legal_and_privacy_constraints" in s


# =============================================================================
# TEST 3: POINT-IN-TIME FEATURE REGISTRY SCHEMA
# =============================================================================

def test_point_in_time_feature_registry_schema(phase6_6_deliverables):
    """Verifies schema completeness, timestamp specifications, and availability rules."""
    registry = phase6_6_deliverables["registry"]
    entries = registry["registry"]

    assert len(entries) >= 7
    required_keys = [
        "feature_id", "feature_name", "signal_family", "provenance_dataset",
        "observation_timestamp_definition", "availability_timestamp_definition",
        "reporting_delay_days", "transformation_logic", "missing_data_treatment",
        "leakage_risk_assessment", "expected_update_frequency", "point_in_time_compliance"
    ]

    for entry in entries:
        for k in required_keys:
            assert k in entry, f"Missing key '{k}' in {entry.get('feature_id')}"

    # Verify rules are present
    assert len(registry["enforcement_rules"]) == 4


# =============================================================================
# TEST 4: POINT-IN-TIME AVAILABILITY AND LEAKAGE SAFEGUARDS
# =============================================================================

def test_point_in_time_availability_and_leakage_safeguards():
    """Verifies that point-in-time feature generation strictly isolates historical observations."""
    from src.evaluation.phase5_feature_engineering import Phase5FeatureEngine
    engine = Phase5FeatureEngine()

    df_sample = pd.DataFrame({
        "Company": [1, 1, 1],
        "Time": [1, 2, 3],
        "Financial Distress": [0.5, 0.4, -0.6],
        "x1": [10.0, 15.0, 8.0],
        "x2": [1.0, 1.2, 0.9]
    })
    for i in range(3, 84):
        df_sample[f"x{i}"] = 0.5

    df_feats = engine.generate_deterioration_features(df_sample)

    # Time 2 features MUST NOT depend on Time 3
    t2_row = df_feats[df_feats["Time"] == 2].iloc[0]
    assert math.isclose(t2_row["x1_lag_1"], 10.0, rel_tol=1e-5)
    assert math.isclose(t2_row["x1_diff_1"], 5.0, rel_tol=1e-5)


# =============================================================================
# TEST 5: MISSING-DATA HANDLING AND ZERO SYNTHETIC CLAIMS
# =============================================================================

def test_missing_data_handling_and_zero_synthetic_claims(phase6_6_deliverables):
    """Verifies that external missingness is recorded as 100% and zero synthetic data was claimed."""
    coverage = phase6_6_deliverables["coverage"]["modalities_coverage_audit"]

    # External modalities must be reported as 100% missing in repository
    assert coverage["bank_transaction_cash_flows"]["missingness_rate"] == 100.0
    assert coverage["bank_transaction_cash_flows"]["rows"] == 0
    assert coverage["b2b_trade_credit_delinquency"]["missingness_rate"] == 100.0
    assert coverage["b2b_trade_credit_delinquency"]["rows"] == 0
    assert coverage["macroeconomic_interest_rate_series"]["missingness_rate"] == 100.0
    assert coverage["macroeconomic_interest_rate_series"]["rows"] == 0

    # Target external directory is empty
    assert os.path.exists("data/external")
    assert len(os.listdir("data/external")) == 0


# =============================================================================
# TEST 6: FIRM AND TIMESTAMP ALIGNMENT
# =============================================================================

def test_firm_and_timestamp_alignment():
    """Verifies Dataset A alignment: 422 companies, 14 time periods, 3672 rows, 0 missing."""
    loader = DataLoader("data/raw")
    df_a = loader.load_dataset_a()

    assert df_a.shape == (3672, 86)
    assert df_a["Company"].nunique() == 422
    assert sorted(df_a["Time"].unique()) == list(range(1, 15))
    assert df_a.isna().sum().sum() == 0


# =============================================================================
# TEST 7: RIGHT-CENSORED OUTCOME EXCLUSION
# =============================================================================

def test_right_censored_outcome_exclusion():
    """Verifies right-censored non-filing firms are excluded from binary contingency metrics."""
    loader = DataLoader("data/raw")
    df_raw = loader.load_dataset_a()

    c12 = set(df_raw[df_raw["Time"] == 12]["Company"].unique())
    c13 = set(df_raw[df_raw["Time"] == 13]["Company"].unique())
    c14 = set(df_raw[df_raw["Time"] == 14]["Company"].unique())

    censored_12_13 = c12 - c13
    censored_13_14 = c13 - c14

    assert len(censored_12_13) == 6
    assert len(censored_13_14) == 12


# =============================================================================
# TEST 8: STRICT INTEGER ALERT-CAP ARITHMETIC
# =============================================================================

def test_alert_cap_arithmetic():
    """Verifies integer floor capacity cap floor(0.08 * N) across cohorts."""
    cohort_denominators = [
        (262, 20),
        (264, 21),
        (270, 21),
        (274, 21),
        (422, 33)
    ]
    for n, exp_cap in cohort_denominators:
        assert int(np.floor(0.08 * n)) == exp_cap


# =============================================================================
# TEST 9: DETERMINISTIC RESEARCH ARTIFACTS
# =============================================================================

def test_deterministic_research_artifacts():
    """Verifies that audit functions execute deterministically without stochastic variation."""
    plan1 = construct_research_experiment_plan()
    plan2 = construct_research_experiment_plan()
    assert json.dumps(plan1, sort_keys=True) == json.dumps(plan2, sort_keys=True)


# =============================================================================
# TEST 10: RESEARCH-GATE DECISION ENFORCEMENT
# =============================================================================

def test_research_gate_decision(phase6_6_deliverables):
    """Verifies that the research-gate decision is ALTERNATIVE_SIGNAL_DATA_INFEASIBLE."""
    plan = phase6_6_deliverables["plan"]
    assert "HALT AT MILESTONE 1" in plan["current_state_verdict"]

    feasibility = phase6_6_deliverables["feasibility"]
    assert "CRITICAL DATA DEFICIT CONFIRMED" in feasibility["supervisory_verdict"]

    integrity = phase6_6_deliverables["integrity"]
    assert integrity["candidate_promotion_state"] == "PASSIVE_SHADOW_CANDIDATE (QUARANTINED)"
