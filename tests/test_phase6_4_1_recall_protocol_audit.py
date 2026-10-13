"""
Phase 6.4.1 Regression Test Suite:
Recall Deterioration and Statistical Protocol Audit.

Tests:
1. Production baseline preservation: SHA-256 invariance, thresholds frozen (0.0800 production, 0.0894 candidate).
2. One-sided alpha-spending discrepancy: boundary verification across 3 designs, Type I error <= 0.0500, UNVERIFIED status.
3. Cohort reconciliation: T=12->T=13 vs T=13->T=14, zero distress overlap (0.0%), right-censored exclusion.
4. Case-level recall deterioration: 5/11 production vs 3/11 candidate, exact case identities (Cases 28 and 60).
5. Refutation of x12 causal drift claim: proves x12 was stable at prediction time T=13 (PSI = 0.0820).
6. Threshold-workload tradeoff: 23 alerts vs 21 cap (+2 excess), 0.0930 recall penalty (+1 miss, 12.5 pp drop), threshold frozen.
7. Candidate promotion lockout: candidate promotion remains strictly blocked.
"""

import os
import json
import math
import pytest
import numpy as np
import pandas as pd

from src.data.loaders import DataLoader
from src.shadow.shadow_engine import compute_sha256
from src.shadow.recall_and_protocol_audit_phase6_4_1 import (
    verify_production_baseline_frozen,
    audit_one_sided_sequential_designs,
    investigate_recall_deterioration_t13_t14,
    reconcile_evaluation_cohorts,
    audit_threshold_workload_tradeoff,
    run_phase6_4_1_recall_and_protocol_audit,
    MANIFEST_PATH,
    OFFICIAL_PROD_THRESHOLD,
    REGISTERED_CAND_THRESHOLD
)

RESULTS_DIR = "results/phase6_4_1"


@pytest.fixture(scope="module")
def phase6_4_1_deliverables():
    out_seq = os.path.join(RESULTS_DIR, "one_sided_sequential_validation.json")
    out_det = os.path.join(RESULTS_DIR, "recall_deterioration_audit.json")
    out_coh = os.path.join(RESULTS_DIR, "cohort_reconciliation.json")
    out_trd = os.path.join(RESULTS_DIR, "threshold_workload_tradeoff.json")

    if not (os.path.exists(out_seq) and os.path.exists(out_det) and os.path.exists(out_coh) and os.path.exists(out_trd)):
        run_phase6_4_1_recall_and_protocol_audit()

    with open(out_seq, "r", encoding="utf-8") as f:
        seq_data = json.load(f)
    with open(out_det, "r", encoding="utf-8") as f:
        det_data = json.load(f)
    with open(out_coh, "r", encoding="utf-8") as f:
        coh_data = json.load(f)
    with open(out_trd, "r", encoding="utf-8") as f:
        trd_data = json.load(f)

    return {
        "seq": seq_data,
        "det": det_data,
        "coh": coh_data,
        "trd": trd_data
    }


# =============================================================================
# TEST 1: PRODUCTION BASELINE PRESERVATION
# =============================================================================

def test_production_baseline_preservation():
    audit = verify_production_baseline_frozen()
    assert audit["production_integrity_status"] == "PASSED"
    assert audit["all_artifacts_unmodified"] is True
    assert audit["production_threshold_frozen"] is True
    assert math.isclose(audit["actual_threshold_on_disk"], OFFICIAL_PROD_THRESHOLD, rel_tol=1e-5)
    assert audit["candidate_promotion_state"] == "PASSIVE_SHADOW_CANDIDATE (FROZEN)"

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    for pth, exp_hash in manifest["pre_implementation_hashes"].items():
        assert compute_sha256(pth) == exp_hash


# =============================================================================
# TEST 2: ONE-SIDED ALPHA-SPENDING DISCREPANCY & DESIGNS
# =============================================================================

def test_one_sided_alpha_spending_discrepancy(phase6_4_1_deliverables):
    seq_data = phase6_4_1_deliverables["seq"]
    tf = seq_data["testing_framework"]
    assert tf["significance_level"] == 0.05
    assert tf["evaluated_margins"]["strict_tier"]["delta"] == 0.05
    assert tf["evaluated_margins"]["practical_tier"]["delta"] == 0.10

    dc = seq_data["discrepancy_analysis"]["design_comparison"]
    d1 = dc["design_1_lan_demets_sfldof_z0025"]
    d2 = dc["design_2_pure_one_sided_obf"]
    d3 = dc["design_3_classical_discrete_obf_1979"]

    # All designs must conserve total alpha = 0.0500
    assert math.isclose(d1["total_alpha_spent"], 0.0500, abs_tol=1e-4)
    assert math.isclose(d2["total_alpha_spent"], 0.0500, abs_tol=1e-4)
    assert math.isclose(d3["total_alpha_spent"], 0.0500, abs_tol=1e-4)

    # Design 1 (Z_0.025 symmetric constant)
    assert math.isclose(d1["critical_z_boundaries"][0], 3.2001, abs_tol=0.01)
    assert math.isclose(d1["critical_z_boundaries"][1], 2.1408, abs_tol=0.01)
    assert math.isclose(d1["critical_z_boundaries"][2], 1.6948, abs_tol=0.01)

    # Design 2 (Pure one-sided 1 - Phi)
    assert math.isclose(d2["critical_z_boundaries"][0], 2.8490, abs_tol=0.01)
    assert math.isclose(d2["critical_z_boundaries"][1], 2.0315, abs_tol=0.01)
    assert math.isclose(d2["critical_z_boundaries"][2], 1.7322, abs_tol=0.01)

    # Design 3 (Classical discrete OBF)
    assert math.isclose(d3["critical_z_boundaries"][0], 2.9611, abs_tol=0.01)
    assert math.isclose(d3["critical_z_boundaries"][1], 2.0938, abs_tol=0.01)
    assert math.isclose(d3["critical_z_boundaries"][2], 1.7096, abs_tol=0.01)

    # Monte Carlo simulation validation of Design 2
    mc_val = seq_data["simulation_validation_pure_one_sided_design"]
    assert mc_val["type_I_error_controlled"] is True
    assert mc_val["confidence_interval_95pct"][0] <= 0.0500 <= mc_val["confidence_interval_95pct"][1]

    # Protocol governance status must be UNVERIFIED_FOR_PROMOTION
    gov = seq_data["protocol_governance_verdict"]
    assert "UNVERIFIED_FOR_PROMOTION" in gov["status"]


# =============================================================================
# TEST 3: COHORT RECONCILIATION & ZERO DISTRESS OVERLAP
# =============================================================================

def test_cohort_reconciliation(phase6_4_1_deliverables):
    coh = phase6_4_1_deliverables["coh"]

    # Cohort 1: T=12 -> T=13 (Official)
    c1 = coh["registered_shadow_surveillance_cohort_t12_t13"]
    assert c1["total_prospective_firms"] == 270
    assert c1["matured_firms"] == 264
    assert c1["right_censored_firms"] == 6
    assert c1["matured_distress_count"] == 8
    assert c1["matured_healthy_count"] == 256
    assert c1["candidate_recall"] == 0.6250
    assert c1["production_recall"] == 0.6250

    # Cohort 2: T=13 -> T=14 (Diagnostic)
    c2 = coh["exploratory_diagnostic_cohort_t13_t14"]
    assert c2["total_prospective_firms"] == 274
    assert c2["matured_firms"] == 262
    assert c2["right_censored_firms"] == 12
    assert c2["matured_distress_count"] == 11
    assert c2["matured_healthy_count"] == 251

    # Overlap between T=13 and T=14 distress must be exactly 0
    comp_sum = coh["cohort_comparison_summary"]
    assert comp_sum["distress_overlap_count"] == 0


# =============================================================================
# TEST 4: CASE-LEVEL RECALL DETERIORATION AUDIT
# =============================================================================

def test_case_level_recall_deterioration(phase6_4_1_deliverables):
    det = phase6_4_1_deliverables["det"]
    perf = det["performance_reconstruction"]
    prod_rec = perf["production_baseline"]["recall"]
    cand_rec = perf["shadow_candidate"]["recall"]

    assert perf["production_baseline"]["detected_count"] == 5
    assert perf["shadow_candidate"]["detected_count"] == 3
    assert math.isclose(prod_rec, 5 / 11, abs_tol=1e-4)
    assert math.isclose(cand_rec, 3 / 11, abs_tol=1e-4)
    assert math.isclose(perf["performance_differential"]["recall_loss_percentage_points"], 18.18, abs_tol=0.01)

    # Paired breakdown
    paired = det["case_level_paired_detection_audit"]
    assert paired["total_distress_cases"] == 11
    assert paired["both_detected_count"] == 3
    assert paired["production_only_detected_count"] == 2
    assert paired["candidate_only_detected_count"] == 0
    assert paired["both_missed_count"] == 6

    # Verify the two production-only detected firms are Firm 28 and Firm 60
    prod_only_firms = [c["anonymized_firm_id"] for c in paired["production_only_detected_cases"]]
    assert set(prod_only_firms) == {28, 60}


# =============================================================================
# TEST 5: REFUTATION OF X12 CAUSAL DRIFT CLAIM
# =============================================================================

def test_refutation_of_x12_causal_hypothesis(phase6_4_1_deliverables):
    det = phase6_4_1_deliverables["det"]
    x12_ref = det["forensic_analysis_of_additional_misses"]["x12_causal_hypothesis_refutation"]

    # At prediction time T=13, x12 was STABLE
    assert x12_ref["temporal_audit"]["drift_status_at_t13"] == "STABLE (PSI = 0.0820 < 0.10)"
    assert math.isclose(x12_ref["temporal_audit"]["psi_x12_at_prediction_time_t13"], 0.0820, abs_tol=1e-4)

    # At realization time T=14, x12 showed moderate drift
    assert "MODERATE_DRIFT" in x12_ref["temporal_audit"]["drift_status_at_t14"]
    assert math.isclose(x12_ref["temporal_audit"]["psi_x12_at_target_realization_time_t14"], 0.1785, abs_tol=1e-4)

    # Causality conclusion must confirm refutation
    assert "REFUTED" in x12_ref["causal_conclusion"]

    # Verify missed cases probabilities: both below 0.0800
    for case in det["forensic_analysis_of_additional_misses"]["case_details"]:
        assert case["candidate_prob"] < OFFICIAL_PROD_THRESHOLD
        assert case["candidate_prob"] < REGISTERED_CAND_THRESHOLD


# =============================================================================
# TEST 6: THRESHOLD WORKLOAD TRADEOFF & INTEGER BUDGET AUDIT
# =============================================================================

def test_threshold_workload_tradeoff(phase6_4_1_deliverables):
    trd = phase6_4_1_deliverables["trd"]

    # Prospective cohort audit
    prosp = trd["prospective_cohort_audit"]
    assert prosp["cohort_size"] == 270
    assert prosp["integer_cap"] == 21
    assert prosp["candidate_alerts_at_registered_threshold"] == 23
    assert prosp["excess_alerts"] == 2
    assert prosp["compliance_status"] == "CAPACITY_BREACH"

    # Matured cohort audit
    mature = trd["matured_cohort_audit"]
    assert mature["cohort_size"] == 264
    assert mature["integer_cap"] == 21
    assert mature["candidate_alerts"] == 19
    assert mature["excess_alerts"] == 0
    assert mature["compliance_status"] == "COMPLIANT"

    # Policy assessment of 0.0930: recall drops from 0.625 to 0.500 (+1 missed event)
    alt = trd["policy_assessment_of_alternative_threshold_0_0930"]
    assert alt["candidate_threshold"] == 0.0930
    assert alt["distress_recall_impact"]["missed_distress_events_delta"] == 1
    assert math.isclose(alt["distress_recall_impact"]["recall_penalty_percentage_points"], 12.50, abs_tol=0.01)
    assert "REJECTED AS SILENT ADJUSTMENT" in alt["governance_ruling"]


# =============================================================================
# TEST 7: CANDIDATE PROMOTION LOCKOUT
# =============================================================================

def test_candidate_promotion_lockout(phase6_4_1_deliverables):
    # Gate 1: Prospective alert capacity breach
    trd = phase6_4_1_deliverables["trd"]
    assert trd["prospective_cohort_audit"]["excess_alerts"] > 0

    # Gate 2: Protocol unverified for promotion
    seq = phase6_4_1_deliverables["seq"]
    assert "UNVERIFIED" in seq["protocol_governance_verdict"]["status"]

    # Gate 3: Recall deterioration on T=14 stress cohort
    det = phase6_4_1_deliverables["det"]
    assert det["performance_reconstruction"]["shadow_candidate"]["recall"] < det["performance_reconstruction"]["production_baseline"]["recall"]

    # Promotion remains permanently blocked
    audit = verify_production_baseline_frozen()
    assert audit["candidate_promotion_state"] == "PASSIVE_SHADOW_CANDIDATE (FROZEN)"
