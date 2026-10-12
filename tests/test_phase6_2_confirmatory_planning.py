"""
Phase 6.2 Automated Regression Test Suite:
Extended Shadow Surveillance, Outcome-Maturity Ledger, Multi-Period Drift Persistence,
and Confirmatory Evaluation Planning.

Tests:
1. Right-Censored Ledger Isolation (6 filing-cessation firms kept separate from binary metrics)
2. Native Decision Records Prioritization (Company 409 accurately preserved as alert=False)
3. Multi-Period Drift Persistence & Reversion Governance (x48 peak at T=12, normalized at T=14)
4. Corroborating Ratio Stability (x12 Operating Cash Flow PSI < 0.10)
5. Statistical Confirmatory Power Sensitivity & Sample Size Requirements (N >= 60 for 80% power at delta=0.05)
6. Group Sequential Alpha-Spending Protocol (O'Brien-Fleming interim boundaries at N=20, 40, 60)
7. Production Baseline SHA-256 Invariance (Zero modification to approved Phase 2 artifacts)
"""

import os
import json
import pytest

from src.shadow.shadow_engine import compute_sha256
from src.shadow.confirmatory_evaluation import (
    audit_right_censored_ledger,
    audit_multi_period_drift_persistence,
    design_confirmatory_evaluation_protocol,
    run_extended_shadow_surveillance
)


MANIFEST_PATH = "experiments/phase6/protected_artifacts_manifest.json"


@pytest.fixture(scope="module")
def censored_ledger():
    return audit_right_censored_ledger()


@pytest.fixture(scope="module")
def drift_persistence():
    return audit_multi_period_drift_persistence()


@pytest.fixture(scope="module")
def confirmatory_protocol():
    return design_confirmatory_evaluation_protocol()


@pytest.fixture(scope="module")
def extended_surveillance():
    return run_extended_shadow_surveillance()


# ---------------------------------------------------------------------
# TEST 1: RIGHT-CENSORED OUTCOMES ISOLATION
# ---------------------------------------------------------------------

def test_right_censored_ledger_isolation(censored_ledger):
    """Verifies that the 6 filing-cessation firms are kept strictly separate from binary metrics."""
    assert censored_ledger["right_censored_count"] == 6
    assert censored_ledger["total_prospective_cohort"] == 270
    assert censored_ledger["censored_proportion_pct"] == 2.22

    roster = censored_ledger["firm_roster"]
    assert len(roster) == 6
    censored_ids = sorted([f["company_id"] for f in roster])
    assert censored_ids == [68, 115, 159, 203, 276, 301]

    # Verify all exited at T=12 with negative distress index
    for firm in roster:
        assert firm["distress_threshold_breached_at_exit"] is True
        assert firm["last_financial_distress_value"] <= -0.5
        assert firm["governance_classification"] == "RIGHT_CENSORED_FILING_CESSATION"


# ---------------------------------------------------------------------
# TEST 2: NATIVE DECISION RECORD PRIORITIZATION
# ---------------------------------------------------------------------

def test_native_decision_record_prioritization(extended_surveillance):
    """Verifies that native decision records are used directly, resolving rounding discrepancies."""
    matured_perf = extended_surveillance["matured_confirmatory_performance"]
    assert "NATIVE_DECISION_RECORDS" in matured_perf["decision_source"]

    pb = matured_perf["production_baseline"]
    cand = matured_perf["candidate_model"]

    # Native baseline alert count is 29 (TP=5, FP=24)
    assert pb["tp"] == 5
    assert pb["fp"] == 24
    assert pb["fn"] == 3
    assert pb["tn"] == 232
    assert pb["alert_rate_pct"] == 10.98

    # Candidate alert count is 19 (TP=5, FP=14)
    assert cand["tp"] == 5
    assert cand["fp"] == 14
    assert cand["fn"] == 3
    assert cand["tn"] == 242
    assert cand["alert_rate_pct"] == 7.20

    # Recall parity maintained
    assert pb["recall"] == 0.6250
    assert cand["recall"] == 0.6250

    # Net workload relief is exactly -10 alerts on matured cohort
    assert matured_perf["paired_deltas"]["delta_alerts_matured"] == -10


# ---------------------------------------------------------------------
# TEST 3 & 4: MULTI-PERIOD DRIFT PERSISTENCE & RATIO STABILITY
# ---------------------------------------------------------------------

def test_multi_period_drift_persistence(drift_persistence):
    """Verifies x48 trajectory and confirms that transient drift does not trigger retraining."""
    summary = drift_persistence["feature_persistence_summary"]
    x48_data = summary["x48"]

    traj = x48_data["trajectory"]
    assert traj["T_12"]["psi"] == 0.4277
    assert traj["T_12"]["status"] == "SIGNIFICANT_DRIFT"
    assert traj["T_14"]["psi"] == 0.0670
    assert traj["T_14"]["status"] == "STABLE"

    assert x48_data["drift_pattern"] == "TRANSIENT_CYCLICAL"
    assert x48_data["escalation_recommended"] is False


def test_corroborating_ratio_stability(drift_persistence):
    """Verifies that core operating cash flow ratio (x12) remained stable across all periods."""
    summary = drift_persistence["feature_persistence_summary"]
    x12_data = summary["x12"]

    for t_key in ["T_10", "T_11", "T_12", "T_13"]:
        psi_val = x12_data["trajectory"][t_key]["psi"]
        assert psi_val < 0.10, f"Expected stable PSI < 0.10 for x12 at {t_key}, got {psi_val}"


# ---------------------------------------------------------------------
# TEST 5 & 6: CONFIRMATORY POWER SENSITIVITY & ALPHA-SPENDING PROTOCOL
# ---------------------------------------------------------------------

def test_confirmatory_power_sensitivity(confirmatory_protocol):
    """Verifies power curve calculations and confirms N >= 60 requirement for 80% power at delta=0.05."""
    reqs = confirmatory_protocol["sample_size_requirements"]
    assert reqs["margin_delta_0_05"]["power_80pct"] == 60
    assert reqs["margin_delta_0_10"]["power_80pct"] == 20

    power_matrix = confirmatory_protocol["power_sensitivity_matrix"]
    current_power_row = next(r for r in power_matrix if r["n_distress"] == 8)
    assert current_power_row["power_margin_5pct"] < 0.15 # ~11.55% power
    assert current_power_row["power_margin_10pct"] < 0.25 # ~22.65% power


def test_alpha_spending_sequential_schedule(confirmatory_protocol):
    """Verifies group sequential monitoring schedule and O'Brien-Fleming interim alpha spending."""
    looks = confirmatory_protocol["group_sequential_alpha_spending_schedule"]
    assert len(looks) == 3

    # Look 1 at N=20
    assert looks[0]["look_number"] == 1
    assert looks[0]["distress_cases_observed"] == 20
    assert looks[0]["nominal_alpha"] == 0.0015
    assert looks[0]["critical_z_score"] == 2.96

    # Look 2 at N=40
    assert looks[1]["look_number"] == 2
    assert looks[1]["distress_cases_observed"] == 40
    assert looks[1]["nominal_alpha"] == 0.0163

    # Final Look at N=60
    assert looks[2]["look_number"] == 3
    assert looks[2]["distress_cases_observed"] == 60
    assert looks[2]["nominal_alpha"] == 0.0440

    # Stopping rules check
    stopping = confirmatory_protocol["stopping_and_review_rules"]
    assert "early_stopping_for_futility_safety" in stopping
    assert "peeking_ban" in stopping


# ---------------------------------------------------------------------
# TEST 7: PROTECTED BASELINE SHA-256 INVARIANCE
# ---------------------------------------------------------------------

def test_phase6_2_protected_artifact_sha256_invariance():
    """Verifies all approved production artifacts remain 100% bitwise unmodified."""
    assert os.path.exists(MANIFEST_PATH)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for rel_path, expected_hash in manifest["pre_implementation_hashes"].items():
        assert os.path.exists(rel_path)
        actual_hash = compute_sha256(rel_path)
        assert actual_hash == expected_hash, f"Baseline integrity breach on {rel_path}!"
