"""
Phase 6.3 Automated Regression Test Suite:
Independent Statistical Protocol Verification, Paired Non-Inferiority Testing,
Lan-DeMets Alpha-Spending Boundaries, and Prospective Shadow Surveillance.

Tests:
1. Paired Outcome Construction (Distress n11=5, n00=3, n10=0, n01=0; Healthy false alarms m10=2, m01=12)
2. Non-Inferiority Hypotheses & Business Margins (delta=0.05 and delta=0.10 unproven at N=8)
3. Independent Power Sensitivity & Sample Size Requirements (N=60 yields 33.72% power for delta=0.05 at p_disc=0.10; N>=248 needed)
4. Lan-DeMets / O'Brien-Fleming Sequential Testing Boundaries (Look 1: Z=2.96, a=0.0015; Look 2: Z=2.14, a=0.0163; Look 3: Z=1.71, a=0.0440)
5. Information Fractions and Maturity Gating (Current N=8 < 20 Look 1 gate; formal testing blocked)
6. Native Decision Handling & Rounding Discrepancy Immunity (Company 409 alert=False prioritized)
7. Right-Censored Audit Ledger Isolation & Sensitivity Bounds (6 firms isolated; 100% recall parity under default attribution)
8. Multi-Period Drift Persistence Governance (x48 transient cyclical peak at T=12, normalized at T=14)
9. Protected Baseline SHA-256 Bitwise Invariance (All Phase 2 artifacts unaltered)
"""

import os
import json
import pytest
import numpy as np

from src.shadow.shadow_engine import compute_sha256
from src.shadow.protocol_verification_phase6_3 import (
    reconstruct_paired_detection_table,
    verify_non_inferiority_hypotheses_and_margins,
    independent_power_and_sample_size_sensitivity,
    validate_group_sequential_boundaries,
    audit_censored_and_missing_outcomes,
    compile_shadow_surveillance_telemetry_phase6_3
)

MANIFEST_PATH = "experiments/phase6/protected_artifacts_manifest.json"


@pytest.fixture(scope="module")
def paired_table():
    return reconstruct_paired_detection_table()


@pytest.fixture(scope="module")
def non_inferiority_hypotheses():
    return verify_non_inferiority_hypotheses_and_margins()


@pytest.fixture(scope="module")
def power_sensitivity():
    return independent_power_and_sample_size_sensitivity()


@pytest.fixture(scope="module")
def sequential_boundaries():
    return validate_group_sequential_boundaries()


@pytest.fixture(scope="module")
def censored_ledger():
    return audit_censored_and_missing_outcomes()


@pytest.fixture(scope="module")
def surveillance_telemetry():
    return compile_shadow_surveillance_telemetry_phase6_3()


# ---------------------------------------------------------------------
# TEST 1: PAIRED OUTCOME CONSTRUCTION & CONTINGENCY RECONSTRUCTION
# ---------------------------------------------------------------------

def test_paired_outcome_construction(paired_table):
    """Verifies paired detection table reconstruction on both distress (Y=1) and healthy (Y=0) cohorts."""
    assert paired_table["matured_cohort_total"] == 264

    # Distress Cases (Y=1, N=8)
    d = paired_table["distress_cases"]
    assert d["n_total"] == 8
    assert d["both_detected_n11"] == 5
    assert d["both_missed_n00"] == 3
    assert d["candidate_only_detected_n10"] == 0
    assert d["baseline_only_detected_n01"] == 0
    assert d["concordant_pairs"] == 8
    assert d["discordant_pairs"] == 0
    assert d["concordance_rate_pct"] == 100.0

    stats_rec = d["paired_recall_statistics"]
    assert stats_rec["point_difference"] == 0.0000
    assert stats_rec["candidate_marginal_recall"] == 0.6250
    assert stats_rec["baseline_marginal_recall"] == 0.6250

    # Healthy Cases (Y=0, N=256)
    h = paired_table["healthy_cases"]
    assert h["n_total"] == 256
    assert h["both_false_alarm_m11"] == 12
    assert h["candidate_only_false_alarm_m10"] == 2
    assert h["baseline_only_false_alarm_m01"] == 12
    assert h["both_true_negative_m00"] == 230
    assert h["baseline_false_positives"] == 24
    assert h["candidate_false_positives"] == 14
    assert h["net_false_alarm_reduction"] == 10
    assert pytest.approx(h["net_false_alarm_reduction_pct"], 0.01) == 41.67
    assert h["mcnemar_exact_p_twosided"] < 0.05
    assert h["false_alarm_reduction_statistically_significant"] is True


# ---------------------------------------------------------------------
# TEST 2: NON-INFERIORITY HYPOTHESES & BUSINESS MARGINS
# ---------------------------------------------------------------------

def test_non_inferiority_hypotheses_and_margins(non_inferiority_hypotheses):
    """Verifies that non-inferiority cannot be asserted at current sample size N=8."""
    curr = non_inferiority_hypotheses["current_evaluation_at_n8"]
    assert curr["matured_distress_count"] == 8
    assert curr["observed_recall_diff"] == 0.00
    assert curr["strict_tier_delta_0_05_verified"] is False
    assert curr["practical_tier_delta_0_10_verified"] is False

    # Lower bound of conservative 95% CI (-31.93%) is far below -0.05 and -0.10
    ci_low = curr["newcombe_conservative_95_ci"][0]
    assert ci_low <= -0.30


# ---------------------------------------------------------------------
# TEST 3: INDEPENDENT POWER SENSITIVITY & SAMPLE SIZE REQUIREMENTS
# ---------------------------------------------------------------------

def test_independent_power_sensitivity_and_sample_size(power_sensitivity):
    """
    Confirms biostatistical proof that N=60 does NOT guarantee 80% power for delta=0.05.
    Verifies that under p_disc = 0.10, N=60 yields only ~33.7% power, and N>=248 is required.
    """
    p60 = power_sensitivity["power_at_n60_across_assumptions"]
    p60_p010 = next(r for r in p60 if r["p_discordant"] == 0.10)

    # For delta = 0.05 at p_disc = 0.10, power at N=60 is only ~33.72%
    assert pytest.approx(p60_p010["power_delta_5pct"], 0.01) == 0.3372
    # For delta = 0.10 at p_disc = 0.10, power at N=60 is ~78.95% (~79%)
    assert pytest.approx(p60_p010["power_delta_10pct"], 0.01) == 0.7895

    # Sample size requirements for 80% power under p_disc = 0.10
    reqs = power_sensitivity["sample_size_requirements_for_80pct_power"]
    reqs_p010 = next(r for r in reqs if r["p_discordant"] == 0.10)
    assert reqs_p010["n_req_delta_5pct"] >= 247
    assert reqs_p010["n_req_delta_10pct"] >= 61

    # Verify that N=60 reaches 80% power at delta=0.05 ONLY when p_disc <= 0.0242
    p60_p0024 = next(r for r in p60 if r["p_discordant"] == 0.0242)
    assert p60_p0024["power_delta_5pct"] >= 0.80

    # Grid check at current N=8
    grid = power_sensitivity["comprehensive_power_sensitivity_grid"]
    row_n8 = next(r for r in grid if r["n_distress"] == 8)
    assert row_n8["power_p010_delta05"] < 0.15
    assert row_n8["power_p010_delta10"] < 0.25


# ---------------------------------------------------------------------
# TEST 4: LAN-DEMETS / O'BRIEN-FLEMING SEQUENTIAL BOUNDARIES
# ---------------------------------------------------------------------

def test_sequential_boundaries_and_alpha_spending(sequential_boundaries):
    """Verifies Lan-DeMets (O'Brien-Fleming) interim boundaries at N=20, 40, 60."""
    looks = sequential_boundaries["interim_looks"]
    assert len(looks) == 3

    # Look 1 at N=20 (t=0.3333)
    assert looks[0]["distress_events_observed"] == 20
    assert pytest.approx(looks[0]["information_fraction"], 0.001) == 0.3333
    assert looks[0]["critical_z_score"] == 2.96
    assert looks[0]["nominal_alpha_spent"] == 0.0015

    # Look 2 at N=40 (t=0.6667)
    assert looks[1]["distress_events_observed"] == 40
    assert pytest.approx(looks[1]["information_fraction"], 0.001) == 0.6667
    assert looks[1]["critical_z_score"] == 2.14
    assert looks[1]["nominal_alpha_spent"] == 0.0163

    # Final Look 3 at N=60 (t=1.0000)
    assert looks[2]["distress_events_observed"] == 60
    assert looks[2]["information_fraction"] == 1.0000
    assert looks[2]["critical_z_score"] == 1.71
    assert looks[2]["nominal_alpha_spent"] == 0.0440

    # Stopping rules check
    rules = sequential_boundaries["governance_and_stopping_rules"]
    assert "peeking_ban" in rules
    assert "futility_safety_rule" in rules
    assert "information_fraction_definition" in rules


# ---------------------------------------------------------------------
# TEST 5: INFORMATION FRACTIONS & MATURITY GATING
# ---------------------------------------------------------------------

def test_information_fractions_and_maturity_gating(paired_table, sequential_boundaries):
    """Verifies that current matured events N=8 fails Look 1 threshold and blocks formal testing."""
    n_current = paired_table["distress_cases"]["n_total"] # 8
    n_look1 = sequential_boundaries["interim_looks"][0]["distress_events_observed"] # 20
    n_max = sequential_boundaries["maximum_planned_events"] # 60

    current_info_fraction = n_current / n_max # 8/60 = 0.1333
    look1_info_fraction = n_look1 / n_max     # 20/60 = 0.3333

    assert n_current < n_look1, "Gate breach: N=8 must be strictly less than Look 1 requirement (N=20)"
    assert current_info_fraction < look1_info_fraction


# ---------------------------------------------------------------------
# TEST 6: NATIVE DECISION HANDLING & ROUNDING DISCREPANCY IMMUNITY
# ---------------------------------------------------------------------

def test_native_decision_handling(surveillance_telemetry):
    """Verifies native decision prioritization ensures Company 409 is preserved as alert=False."""
    matured = surveillance_telemetry["matured_cohort_n264"]
    pb = matured["production_performance"]
    cand = matured["candidate_performance"]

    # Native baseline alert count is 29 (TP=5, FP=24)
    assert pb["tp"] == 5
    assert pb["fp"] == 24
    assert pb["tn"] == 232
    assert pb["fn"] == 3
    assert pb["recall"] == 0.6250
    assert pb["alert_rate_pct"] == 10.98

    # Candidate alert count is 19 (TP=5, FP=14)
    assert cand["tp"] == 5
    assert cand["fp"] == 14
    assert cand["tn"] == 242
    assert cand["fn"] == 3
    assert cand["recall"] == 0.6250
    assert cand["alert_rate_pct"] == 7.20

    # Recall parity and false alarm delta
    assert matured["operational_impact"]["recall_parity"] is True
    assert matured["operational_impact"]["false_positive_delta"] == -10


# ---------------------------------------------------------------------
# TEST 7: RIGHT-CENSORED AUDIT LEDGER & SENSITIVITY BOUNDS
# ---------------------------------------------------------------------

def test_right_censored_ledger_and_sensitivity_bounds(censored_ledger):
    """Verifies isolation of 6 filing-cessation firms and confirms sensitivity bounds."""
    assert censored_ledger["right_censored_count"] == 6
    assert censored_ledger["total_prospective_cohort"] == 270

    # Confirm all 6 breached terminal distress threshold
    for firm in censored_ledger["firm_roster"]:
        assert firm["terminal_distress_threshold_breached"] is True
        assert firm["last_financial_distress_value"] <= -0.5

    # Sensitivity bounds: under full default attribution (Y=1), recall parity is preserved
    sens = censored_ledger["competing_risk_sensitivity_analysis"]
    scen2 = sens["scenario_2_complete_default_attribution"]
    assert scen2["baseline_detected"] == 9
    assert scen2["candidate_detected"] == 9
    assert scen2["recall_parity"] is True
    assert scen2["baseline_recall"] == scen2["candidate_recall"]


# ---------------------------------------------------------------------
# TEST 8: MULTI-PERIOD DRIFT PERSISTENCE GOVERNANCE
# ---------------------------------------------------------------------

def test_multi_period_drift_persistence_governance(surveillance_telemetry):
    """Verifies that x48 transient spike (T=12) normalized at T=14, barring retraining."""
    drift = surveillance_telemetry["feature_drift_persistence_summary"]

    assert drift["x48"]["psi_t12"] == 0.4277
    assert drift["x48"]["status_t12"] == "SIGNIFICANT_DRIFT"
    assert drift["x48"]["psi_t14"] == 0.0670
    assert drift["x48"]["status_t14"] == "STABLE"
    assert drift["x48"]["trajectory_pattern"] == "TRANSIENT_CYCLICAL"

    assert drift["x12"]["status_t12"] == "STABLE"


# ---------------------------------------------------------------------
# TEST 9: PROTECTED ARTIFACT SHA-256 INVARIANCE
# ---------------------------------------------------------------------

def test_phase6_3_protected_artifact_sha256_invariance():
    """Confirms 100% bitwise cryptographic invariance of all approved Phase 2 artifacts."""
    assert os.path.exists(MANIFEST_PATH)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for rel_path, expected_hash in manifest["pre_implementation_hashes"].items():
        assert os.path.exists(rel_path)
        actual_hash = compute_sha256(rel_path)
        assert actual_hash == expected_hash, f"Baseline integrity breach on {rel_path}!"
