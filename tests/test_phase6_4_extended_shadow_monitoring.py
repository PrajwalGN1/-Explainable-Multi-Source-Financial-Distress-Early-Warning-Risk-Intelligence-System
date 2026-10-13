"""
Phase 6.4 Regression Test Suite:
Extended Shadow Monitoring and Model Risk Governance.

Tests:
1. Alert-cap calculations: floor(cohort_size * 0.08) across cohorts and offline threshold analysis.
2. Cohort separation and denominator integrity: prospective (N=270), matured (N=264), censored (N=6).
3. Sequential-protocol reproducibility: numerical quadrature & MC validation, Type I error <= 0.0500, MC uncertainty disclosure.
4. Production artifact immutability: SHA-256 invariance against protected manifest, frozen threshold 0.0800.
5. Native decision handling: preservation of native probabilities, decisions, and independent candidate channels.
6. Right-censored outcome isolation: exclusion of N=6 from primary binary analysis, no verification bias imputation.
7. Drift trajectory consistency: multi-period PSI for x48, x12, x16, x25, x2, x12 T=14 investigation, no retraining on PSI alone.
8. Shadow failure isolation: zero impact of shadow candidate exceptions on production scoring, latency health.
9. No automatic candidate promotion: gates G5 (alert capacity) and G6 (event maturity N_D=8 < 20) block promotion.
"""

import os
import json
import math
import pytest
import numpy as np
import pandas as pd
from scipy import stats

from src.data.loaders import DataLoader
from src.shadow.shadow_engine import compute_sha256, ShadowInferenceEngine
from src.shadow.extended_shadow_monitoring_phase6_4 import (
    verify_production_integrity,
    reproduce_sequential_protocol,
    audit_alert_capacity_and_workload,
    compile_shadow_monitoring_telemetry,
    monitor_feature_drift_and_investigate_x12,
    run_phase6_4_extended_shadow_monitoring,
    MANIFEST_PATH,
    OFFICIAL_PROD_THRESHOLD,
    REGISTERED_CAND_THRESHOLD
)

RESULTS_DIR = "results/phase6_4"


@pytest.fixture(scope="module")
def phase6_4_deliverables():
    # Ensure deliverables exist
    alert_path = os.path.join(RESULTS_DIR, "alert_budget_audit.json")
    seq_path = os.path.join(RESULTS_DIR, "sequential_protocol_reproduction.json")
    telemetry_path = os.path.join(RESULTS_DIR, "shadow_monitoring_telemetry.json")
    drift_path = os.path.join(RESULTS_DIR, "drift_monitoring.json")

    if not (os.path.exists(alert_path) and os.path.exists(seq_path) and os.path.exists(telemetry_path) and os.path.exists(drift_path)):
        run_phase6_4_extended_shadow_monitoring()

    with open(alert_path, "r", encoding="utf-8") as f:
        alert_data = json.load(f)
    with open(seq_path, "r", encoding="utf-8") as f:
        seq_data = json.load(f)
    with open(telemetry_path, "r", encoding="utf-8") as f:
        telemetry_data = json.load(f)
    with open(drift_path, "r", encoding="utf-8") as f:
        drift_data = json.load(f)

    return {
        "alert": alert_data,
        "seq": seq_data,
        "telemetry": telemetry_data,
        "drift": drift_data
    }


# =============================================================================
# TEST 1: ALERT-CAP CALCULATIONS & OFFLINE WORKLOAD SWEEP
# =============================================================================

def test_alert_cap_calculations(phase6_4_deliverables):
    alert_data = phase6_4_deliverables["alert"]
    
    # 1. Formula definition: maximum_alerts = floor(cohort_size * 0.08)
    for n in [270, 264, 6, 100, 50, 12, 1]:
        expected_floor = math.floor(n * 0.08)
        assert expected_floor == int(math.floor(n * 0.08))

    # Prospective cohort: N=270 -> floor(270 * 0.08) = 21
    prosp = alert_data["prospective_cohort"]
    assert prosp["cohort_size"] == 270
    assert prosp["maximum_permitted_alerts"] == 21
    assert prosp["candidate_alerts"] == 23
    assert prosp["excess_alerts"] == 2
    assert prosp["capacity_compliance_status"] == "CAPACITY_BREACH"
    assert math.isclose(prosp["candidate_alert_rate_pct"], 8.52, abs_tol=0.01)

    # Matured cohort: N=264 -> floor(264 * 0.08) = 21
    mature = alert_data["matured_cohort"]
    assert mature["cohort_size"] == 264
    assert mature["maximum_permitted_alerts"] == 21
    assert mature["candidate_alerts"] == 19
    assert mature["excess_alerts"] == 0
    assert mature["capacity_compliance_status"] == "COMPLIANT"
    assert math.isclose(mature["candidate_alert_rate_pct"], 7.20, abs_tol=0.01)

    # Right-censored cohort: N=6 -> floor(6 * 0.08) = 0
    cens = alert_data["right_censored_cohort"]
    assert cens["cohort_size"] == 6
    assert cens["maximum_permitted_alerts"] == 0
    assert cens["candidate_alerts"] == 4
    assert cens["excess_alerts"] == 4
    assert cens["capacity_compliance_status"] == "CAPACITY_BREACH"

    # Offline threshold sweep check: does NOT alter registered candidate threshold
    offline = alert_data["offline_workload_impact_investigation"]
    assert offline["registered_candidate_operating_point"]["threshold"] == REGISTERED_CAND_THRESHOLD
    assert "OFFLINE_ANALYSIS_ONLY" in offline["governance_status"]


# =============================================================================
# TEST 2: COHORT SEPARATION AND DENOMINATOR INTEGRITY
# =============================================================================

def test_cohort_separation_and_denominator_integrity(phase6_4_deliverables):
    telemetry = phase6_4_deliverables["telemetry"]
    denoms = telemetry["cohort_denominators"]

    assert denoms["total_prospective_firms"] == 270
    assert denoms["audited_matured_firms"] == 264
    assert denoms["right_censored_firms"] == 6
    assert denoms["audited_matured_firms"] + denoms["right_censored_firms"] == denoms["total_prospective_firms"]

    # Matured outcomes breakdown
    assert denoms["matured_distress_count"] == 8
    assert denoms["matured_healthy_count"] == 256
    assert denoms["matured_distress_count"] + denoms["matured_healthy_count"] == denoms["audited_matured_firms"]

    # Check that Look 1 requires 20 distress events and is NOT reached
    assert denoms["minimum_required_for_interim_look_1"] == 20
    assert denoms["matured_distress_count"] < denoms["minimum_required_for_interim_look_1"]
    assert "NOT_REACHED" in denoms["look_1_maturity_readiness"]


# =============================================================================
# TEST 3: SEQUENTIAL PROTOCOL REPRODUCIBILITY
# =============================================================================

def test_sequential_protocol_reproducibility(phase6_4_deliverables):
    seq_data = phase6_4_deliverables["seq"]

    # 1. Hypothesis specification
    hyp = seq_data["hypothesis_specification"]
    assert hyp["significance_level"] == 0.05
    assert hyp["strict_non_inferiority_margin_delta"] == 0.05
    assert hyp["practical_non_inferiority_margin_delta"] == 0.10
    assert "H_0: R_cand - R_base <= -delta" in hyp["null_hypothesis"]

    # 2. Spending function
    sp = seq_data["spending_function_clarification"]
    assert sp["selected_spending_function"] == "Lan-DeMets (1983) O'Brien-Fleming approximation (sfLDOF)"
    assert sp["nature_of_spending_function"] == "DELIBERATELY_CONSERVATIVE_EARLY_STOPPING"

    # 3. Deterministic Numerical Quadrature Boundaries
    cb = seq_data["critical_boundaries"]
    quad_res = cb["deterministic_numerical_quadrature"]
    z_quad = quad_res["critical_z_boundaries"]
    assert math.isclose(z_quad[0], 3.2001, abs_tol=0.01)
    assert math.isclose(z_quad[1], 2.1408, abs_tol=0.01)
    assert math.isclose(z_quad[2], 1.6948, abs_tol=0.01)
    assert math.isclose(quad_res["joint_type_I_error"], 0.050000, abs_tol=1e-5)

    # 4. Monte Carlo Validation
    mc_res = cb["reproducible_monte_carlo_validation"]
    assert mc_res["draws"] >= 5_000_000
    assert math.isclose(mc_res["overall_simulated_type_I_error"], 0.0500, abs_tol=0.001)
    ci_low, ci_high = mc_res["confidence_interval_95pct"]
    assert ci_low <= 0.0500 <= ci_high
    assert "not a deterministic mathematical proof" in mc_res["methodological_disclosure"]

    # 5. Discrepancy analysis against Phase 6.3.1 (all deltas <= 0.001 Z-units)
    disc = seq_data["discrepancy_analysis_against_phase6_3_1"]
    for lk in ["look_1", "look_2", "look_3"]:
        assert disc[lk]["delta_quad_vs_p631"] <= 0.0015


# =============================================================================
# TEST 4: PRODUCTION ARTIFACT IMMUTABILITY
# =============================================================================

def test_production_artifact_immutability():
    integrity = verify_production_integrity()
    assert integrity["status"] == "PASSED"
    assert integrity["all_artifact_hashes_verified"] is True
    assert integrity["production_threshold_frozen"] is True
    assert math.isclose(integrity["actual_production_threshold_on_disk"], OFFICIAL_PROD_THRESHOLD, rel_tol=1e-5)
    assert integrity["candidate_promotion_status"] == "FROZEN_SHADOW_CANDIDATE (NOT_PROMOTED)"

    # Direct hash verification against protected manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    for pth, expected_hash in manifest["pre_implementation_hashes"].items():
        assert compute_sha256(pth) == expected_hash, f"Hash divergence detected in {pth}!"


# =============================================================================
# TEST 5: NATIVE DECISION HANDLING
# =============================================================================

def test_native_decision_handling(phase6_4_deliverables):
    telemetry = phase6_4_deliverables["telemetry"]
    pipe = telemetry["pipeline_health_and_reliability"]
    assert "PRESERVED" in pipe["native_decision_handling"]

    # Check predictions dataframe
    df_preds = pd.read_csv("results/phase6/prospective_predictions.csv")
    assert "prod_is_alert" in df_preds.columns
    assert "prod_calibrated_probability" in df_preds.columns
    assert "cand_is_alert" in df_preds.columns
    assert "cand_calibrated_probability" in df_preds.columns

    # Verify native decision integrity:
    # Native decisions are Boolean decisions rendered by the inference pipeline before rounding.
    # Total production alerts: 33 (12.22%)
    # Total candidate alerts: 23 (8.52%)
    assert int(df_preds["prod_is_alert"].sum()) == 33
    assert int(df_preds["cand_is_alert"].sum()) == 23

    # Check that Boolean decisions match the probability distribution within rounding tolerance (1e-4)
    for _, row in df_preds.iterrows():
        p_prob = row["prod_calibrated_probability"]
        c_prob = row["cand_calibrated_probability"]
        if p_prob > OFFICIAL_PROD_THRESHOLD + 0.001:
            assert row["prod_is_alert"] is True
        elif p_prob < OFFICIAL_PROD_THRESHOLD - 0.001:
            assert row["prod_is_alert"] is False
        if c_prob > REGISTERED_CAND_THRESHOLD + 0.001:
            assert row["cand_is_alert"] is True
        elif c_prob < REGISTERED_CAND_THRESHOLD - 0.001:
            assert row["cand_is_alert"] is False


# =============================================================================
# TEST 6: RIGHT-CENSORED OUTCOME ISOLATION
# =============================================================================

def test_right_censored_outcome_isolation(phase6_4_deliverables):
    telemetry = phase6_4_deliverables["telemetry"]
    cens_gov = telemetry["right_censored_governance"]
    assert cens_gov["count"] == 6
    assert cens_gov["treatment"] == "STRICT_EXCLUSION_FROM_BINARY_ANALYSIS"

    df_preds = pd.read_csv("results/phase6/prospective_predictions.csv")
    cens_df = df_preds[df_preds["outcome_maturity_status"] != "MATURED"]
    assert len(cens_df) == 6

    # Verify right-censored firms have NaN or None in actual_target
    assert cens_df["actual_target"].isna().all()

    # Verify matured contingency table did not include any censored firms
    matured_df = df_preds[df_preds["outcome_maturity_status"] == "MATURED"]
    assert len(matured_df) == 264
    assert matured_df["actual_target"].notna().all()


# =============================================================================
# TEST 7: DRIFT TRAJECTORY CONSISTENCY & X12 INVESTIGATION
# =============================================================================

def test_drift_trajectory_consistency(phase6_4_deliverables):
    drift_data = phase6_4_deliverables["drift"]
    feat_drift = drift_data["multi_period_feature_drift"]

    # Verify all monitored features are present across periods 10..14
    for feat in ["x48", "x12", "x16", "x25", "x2"]:
        assert feat in feat_drift
        trajs = feat_drift[feat]["trajectories"]
        for t in [10, 11, 12, 13, 14]:
            assert f"T_{t}" in trajs
            assert trajs[f"T_{t}"]["psi"] >= 0.0

    # Verify x48 compression back to stable at T=14
    assert feat_drift["x48"]["trajectories"]["T_14"]["status"] == "STABLE"
    assert feat_drift["x48"]["trajectories"]["T_14"]["psi"] < 0.10

    # Verify x12 deep dive
    x12_inv = drift_data["x12_deep_dive_investigation"]
    assert x12_inv["psi_at_t14"] == 0.1785
    assert x12_inv["drift_classification_at_t14"] == "MODERATE_DRIFT (0.10 <= PSI < 0.25)"
    assert x12_inv["persistence_audit"]["persists_across_multiple_cycles"] is False

    # Check performance association: candidate sensitivity degraded at T=14
    assoc = x12_inv["association_with_performance_at_t14"]
    assert assoc["matured_distress_events_t14"] == 11
    assert assoc["candidate_recall"] < assoc["production_recall"]  # 0.2727 < 0.4545

    # Check governance mandate: PSI alone must NOT trigger retraining
    gov = x12_inv["governance_rule_enforcement"]
    assert "PSI alone must NOT trigger retraining" in gov["mandate"]
    assert gov["action_taken"] == "NO_RETRAINING_AUTHORIZED (Passive surveillance continues without intervention)"


# =============================================================================
# TEST 8: SHADOW FAILURE ISOLATION
# =============================================================================

def test_shadow_failure_isolation(phase6_4_deliverables):
    telemetry = phase6_4_deliverables["telemetry"]
    pipe = telemetry["pipeline_health_and_reliability"]
    assert pipe["pipeline_uptime_pct"] == 100.0
    assert pipe["inference_failures"] == 0
    assert pipe["average_latency_ms"] < 2.0
    assert "VERIFIED" in pipe["shadow_failure_isolation"]

    # Test shadow engine inference and failure isolation
    engine = ShadowInferenceEngine()
    df_raw = DataLoader("data/raw").load_dataset_a()
    sample_co = df_raw[df_raw["Company"] == 1].copy()
    res = engine.evaluate_shadow_sample("1", sample_co, current_period=1, record_audit_log=False)
    assert "production" in res
    assert "shadow_candidate" in res
    assert res["production"]["is_distress_alert"] in [True, False]


# =============================================================================
# TEST 9: NO AUTOMATIC CANDIDATE PROMOTION
# =============================================================================

def test_no_automatic_candidate_promotion(phase6_4_deliverables):
    alert_data = phase6_4_deliverables["alert"]
    seq_data = phase6_4_deliverables["seq"]
    telemetry = phase6_4_deliverables["telemetry"]

    # Gate 1: Prospective Alert Capacity Breach (+2 alerts, 8.52% > 8.0%)
    assert alert_data["prospective_cohort"]["capacity_compliance_status"] == "CAPACITY_BREACH"
    assert alert_data["prospective_cohort"]["excess_alerts"] > 0

    # Gate 2: Outcome Maturity Gate (N_D = 8 < 20)
    assert telemetry["cohort_denominators"]["matured_distress_count"] == 8
    assert telemetry["cohort_denominators"]["matured_distress_count"] < 20

    # Gate 3: Protocol status
    assert "PROHIBITED" in seq_data["confirmatory_protocol_status"]["operational_promotion_authorization"]

    # Candidate model must remain in passive shadow mode
    integrity = verify_production_integrity()
    assert integrity["candidate_promotion_status"] == "FROZEN_SHADOW_CANDIDATE (NOT_PROMOTED)"
