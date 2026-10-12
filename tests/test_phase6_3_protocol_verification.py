"""
Phase 6.3 Automated Regression Test Suite:
Independent Statistical Protocol Verification, Paired Non-Inferiority Testing,
Lan-DeMets Alpha-Spending Boundaries, and Prospective Shadow Surveillance.

Updated to validate resolutions for:
- Issue 1: Prospective cohort alert capacity breach (8.52% > 8.0%) vs matured cohort compliance (7.20% <= 8.0%)
- Issue 2: Rigorous distinction between nominal boundary p-values and cumulative alpha spent
- Issue 3: Reconciled multi-period drift across all periods T in [10, 14], verifying x12 moderate drift at T=14
- Issue 4: Validation of right-censored source records (exit at T=12, unobserved at T=13, kept outside primary analysis)
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
# TEST 4: LAN-DEMETS SEQUENTIAL BOUNDARIES (NOMINAL P VS ALPHA SPENT)
# ---------------------------------------------------------------------

def test_sequential_boundaries_and_alpha_spending(sequential_boundaries):
    """
    Issue 2 Verification:
    Verifies that implementation rigorously distinguishes between:
      1) Nominal boundary p-values (p_k = 0.0015, 0.0162, 0.0436)
      2) Cumulative alpha spent under Lan-DeMets OBF (alpha*(t_k) = 0.0007, 0.0164, 0.0500)
    """
    looks = sequential_boundaries["interim_looks"]
    assert len(looks) == 3

    # Look 1 at N=20 (t=0.3333)
    assert looks[0]["distress_events_observed"] == 20
    assert pytest.approx(looks[0]["information_fraction"], abs=1e-3) == 0.3333
    assert looks[0]["critical_z_score"] == 2.96
    assert pytest.approx(looks[0]["nominal_boundary_p_value"], abs=1e-4) == 0.0015
    assert pytest.approx(looks[0]["cumulative_alpha_spent"], abs=1e-4) == 0.0007
    assert pytest.approx(looks[0]["incremental_alpha_spent"], abs=1e-4) == 0.0007

    # Look 2 at N=40 (t=0.6667)
    assert looks[1]["distress_events_observed"] == 40
    assert pytest.approx(looks[1]["information_fraction"], abs=1e-3) == 0.6667
    assert looks[1]["critical_z_score"] == 2.14
    assert pytest.approx(looks[1]["nominal_boundary_p_value"], abs=1e-4) == 0.0162
    assert pytest.approx(looks[1]["cumulative_alpha_spent"], abs=1e-4) == 0.0164
    assert pytest.approx(looks[1]["incremental_alpha_spent"], abs=1e-4) == 0.0157

    # Final Look 3 at N=60 (t=1.0000)
    assert looks[2]["distress_events_observed"] == 60
    assert looks[2]["information_fraction"] == 1.0000
    assert looks[2]["critical_z_score"] == 1.71
    assert pytest.approx(looks[2]["nominal_boundary_p_value"], abs=1e-3) == 0.0436
    assert pytest.approx(looks[2]["cumulative_alpha_spent"], abs=1e-4) == 0.0500
    assert pytest.approx(looks[2]["incremental_alpha_spent"], abs=1e-4) == 0.0336

    # Verify methodological clarification is documented
    verif = sequential_boundaries["statistical_design_verification"]
    assert "critical_distinction" in verif
    assert "total_type_I_error_guarantee" in verif


# ---------------------------------------------------------------------
# TEST 5: INFORMATION FRACTIONS & MATURITY GATING
# ---------------------------------------------------------------------

def test_information_fractions_and_maturity_gating(paired_table, sequential_boundaries):
    """Verifies that current matured events N=8 fails Look 1 threshold and blocks formal testing."""
    n_current = paired_table["distress_cases"]["n_total"]  # 8
    n_look1 = sequential_boundaries["interim_looks"][0]["distress_events_observed"]  # 20
    n_max = sequential_boundaries["maximum_planned_events"]  # 60

    current_info_fraction = n_current / n_max
    look1_info_fraction = n_look1 / n_max

    assert n_current < n_look1, "Gate breach: N=8 must be strictly less than Look 1 requirement (N=20)"
    assert current_info_fraction < look1_info_fraction


# ---------------------------------------------------------------------
# TEST 6: SEPARATE REPORTING OF PROSPECTIVE VS MATURED ALERT BUDGETS
# ---------------------------------------------------------------------

def test_separate_alert_budget_reporting(surveillance_telemetry):
    """
    Issue 1 Verification:
    Verifies that the prospective cohort (8.52%) is explicitly recognized as breaching
    the 8.0% capacity limit, while the matured subset (7.20%) complies with the limit.
    """
    prosp = surveillance_telemetry["prospective_cohort_n270"]
    mature = surveillance_telemetry["matured_cohort_n264"]

    # Prospective cohort: 23 alerts / 270 firms = 8.52% (BREACH of 8.0% limit)
    assert prosp["total_firms"] == 270
    assert prosp["candidate_alerts"] == 23
    assert prosp["candidate_alert_rate_pct"] == 8.52
    assert prosp["alert_capacity_limit_pct"] == 8.00
    assert prosp["budget_compliance_boolean"] is False
    assert "EXCEEDS_CAPACITY_LIMIT" in prosp["alert_budget_compliance"]

    # Matured cohort: 19 alerts / 264 firms = 7.20% (COMPLIANT with 8.0% limit)
    assert mature["matured_total"] == 264
    assert mature["candidate_alerts"] == 19
    assert mature["candidate_alert_rate_pct"] == 7.20
    assert mature["alert_capacity_limit_pct"] == 8.00
    assert mature["budget_compliance_boolean"] is True
    assert "COMPLIANT_WITHIN_LIMIT" in mature["alert_budget_compliance"]


# ---------------------------------------------------------------------
# TEST 7: RIGHT-CENSORED SOURCE RECORD VALIDATION & EXCLUSION
# ---------------------------------------------------------------------

def test_right_censored_ledger_and_source_validation(censored_ledger):
    """
    Issue 4 Verification:
    Verifies that the 6 filing-cessation firms exited at T=12 with contemporaneous
    Financial Distress <= -0.5, but have no forward filing at T=13 in source records.
    Confirms they are kept strictly OUTSIDE the primary matured analysis.
    """
    assert censored_ledger["right_censored_count"] == 6
    assert censored_ledger["total_prospective_cohort"] == 270

    for firm in censored_ledger["firm_roster"]:
        assert firm["last_period_observed_in_source"] == 12
        assert firm["exit_period_index_breached_distress_threshold"] is True
        assert firm["last_financial_distress_value_at_exit"] <= -0.5
        assert firm["forward_target_period_t13_record_exists"] is False
        assert firm["forward_t13_statutory_outcome"] == "UNOBSERVED_FILING_CESSATION"

    # Confirms primary matured analysis strictly excludes these 6 firms
    sens = censored_ledger["competing_risk_sensitivity_analysis"]
    prim = sens["primary_matured_analysis"]
    assert prim["status"] == "AUTHORITATIVE_OFFICIAL_LEDGER"
    assert prim["matured_distress_count"] == 8
    assert prim["matured_healthy_count"] == 256


# ---------------------------------------------------------------------
# TEST 8: RECONCILED MULTI-PERIOD DRIFT MONITORING ACROSS ALL PERIODS
# ---------------------------------------------------------------------

def test_multi_period_drift_reconciliation_all_periods(surveillance_telemetry):
    """
    Issue 3 Verification:
    Verifies that drift monitoring includes all periods T in [10, 14], and explicitly
    reconciles x12 trajectory: stable at T=10..13 (PSI < 0.10) but experiencing moderate
    drift at T=14 (PSI = 0.1785 > 0.10).
    """
    drift = surveillance_telemetry["feature_drift_reconciliation_all_periods"]
    x12 = drift["x12"]["trajectory"]

    # Verify x12 was stable through T=13
    assert x12["T_10"]["psi"] < 0.10
    assert x12["T_11"]["psi"] < 0.10
    assert x12["T_12"]["psi"] < 0.10
    assert x12["T_13"]["psi"] < 0.10

    # Verify x12 entered moderate drift at T=14
    assert x12["T_14"]["psi"] == 0.1785
    assert x12["T_14"]["status"] == "MODERATE_DRIFT"

    # Verify x48 transient peak and normalization
    x48 = drift["x48"]["trajectory"]
    assert x48["T_12"]["psi"] == 0.4277
    assert x48["T_14"]["psi"] == 0.0670
    assert x48["T_14"]["status"] == "STABLE"

    # Verify investigation verdict is documented
    verdict = surveillance_telemetry["x12_investigation_verdict"]
    assert "MODERATE_DRIFT_AT_T14" in verdict["overall_classification"]


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
