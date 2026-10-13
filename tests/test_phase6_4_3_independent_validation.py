"""
Phase 6.4.3 Regression Test Suite:
Independent Validation, Cohort Reconciliation & Research-Gate Review.

Tests:
1. Production baseline manifest immutability: SHA-256 digests verified, official threshold 0.0800 frozen.
2. Cohort reconciliation and denominator derivations: 270 vs 264 (T12->T13), 274 vs 262 (T13->T14),
   right-censored filing cessations, and 0.0% distress overlap.
3. Alert cap arithmetic across cohorts: integer floor logic floor(0.08 * N); reconciles 20, 23, and 19 alerts.
4. Native threshold decisions and metric recomputation: exact confusion matrices, unrounded scores, zero discrepancy.
5. Failure case reproduction for Firms 28 and 60: probabilities, thresholds, and ranking collapse below 0.0800.
6. Subgroup recall and sudden-transition generalization: 0.0% candidate recall on healthy firms vs 40.0% production;
   sudden defaults represent 34.0% (36/106) across all historical reporting periods.
7. Statistical protocol review and gate decision: UNVERIFIED status preserved; RESEARCH_EVIDENCE_REPRODUCIBLE_BUT_INSUFFICIENT.
"""

import os
import json
import math
import pytest
import numpy as np
import pandas as pd

from src.shadow.shadow_engine import compute_sha256
from src.shadow.independent_validation_phase6_4_3 import (
    verify_production_baseline_manifest,
    construct_cohort_reconciliation_ledger,
    recompute_phase6_4_2_metrics,
    recompute_failure_cases_diagnostics,
    verify_alert_budget_policies,
    audit_research_model_validity,
    review_statistical_protocol_status,
    MANIFEST_PATH,
    OFFICIAL_PROD_THRESHOLD,
    REGISTERED_CAND_THRESHOLD,
    RESULTS_DIR
)


@pytest.fixture(scope="module")
def phase6_4_3_deliverables():
    p_man = os.path.join(RESULTS_DIR, "production_integrity_manifest.json")
    c_rec = os.path.join(RESULTS_DIR, "cohort_reconciliation.json")
    m_rec = os.path.join(RESULTS_DIR, "metric_reconciliation.json")
    f_rep = os.path.join(RESULTS_DIR, "failure_case_reproduction.json")
    b_ver = os.path.join(RESULTS_DIR, "alert_budget_verification.json")
    r_val = os.path.join(RESULTS_DIR, "research_model_validity_audit.json")
    s_rev = os.path.join(RESULTS_DIR, "statistical_protocol_review.json")

    assert os.path.exists(p_man), f"Missing: {p_man}"
    assert os.path.exists(c_rec), f"Missing: {c_rec}"
    assert os.path.exists(m_rec), f"Missing: {m_rec}"
    assert os.path.exists(f_rep), f"Missing: {f_rep}"
    assert os.path.exists(b_ver), f"Missing: {b_ver}"
    assert os.path.exists(r_val), f"Missing: {r_val}"
    assert os.path.exists(s_rev), f"Missing: {s_rev}"

    with open(p_man, "r", encoding="utf-8") as f:
        p_data = json.load(f)
    with open(c_rec, "r", encoding="utf-8") as f:
        c_data = json.load(f)
    with open(m_rec, "r", encoding="utf-8") as f:
        m_data = json.load(f)
    with open(f_rep, "r", encoding="utf-8") as f:
        f_data = json.load(f)
    with open(b_ver, "r", encoding="utf-8") as f:
        b_data = json.load(f)
    with open(r_val, "r", encoding="utf-8") as f:
        r_data = json.load(f)
    with open(s_rev, "r", encoding="utf-8") as f:
        s_data = json.load(f)

    return {
        "p_man": p_data,
        "c_rec": c_data,
        "m_rec": m_data,
        "f_rep": f_data,
        "b_ver": b_data,
        "r_val": r_data,
        "s_rev": s_data
    }


# =============================================================================
# TEST 1: PRODUCTION BASELINE MANIFEST IMMUTABILITY
# =============================================================================

def test_production_baseline_manifest_immutability(phase6_4_3_deliverables):
    p_man = phase6_4_3_deliverables["p_man"]
    assert p_man["production_integrity_verdict"] == "PASSED"
    assert p_man["cryptographic_hash_invariance"] is True
    assert p_man["official_production_threshold_frozen"] is True
    assert math.isclose(p_man["threshold_on_disk"], OFFICIAL_PROD_THRESHOLD, rel_tol=1e-5)
    assert p_man["candidate_promotion_state"] == "PASSIVE_SHADOW_CANDIDATE (FROZEN)"

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    for pth, exp_hash in manifest["pre_implementation_hashes"].items():
        assert compute_sha256(pth) == exp_hash, f"Hash mismatch on {pth}"


# =============================================================================
# TEST 2: COHORT RECONCILIATION AND DENOMINATOR DERIVATIONS
# =============================================================================

def test_cohort_reconciliation_and_denominators(phase6_4_3_deliverables):
    c_rec = phase6_4_3_deliverables["c_rec"]
    cohorts = c_rec["cohorts"]

    # Cohort 1: T=12 Prospective (N=270, 6 right-censored pending)
    c12_p = cohorts["cohort_T12_prospective"]
    assert c12_p["eligible_firms"] == 270
    assert c12_p["pending_or_right_censored"] == 6
    assert c12_p["integer_8pct_cap"] == 21
    assert c12_p["candidate_alerts_at_0_0894"] == 23

    # Cohort 2: T=12 -> T=13 Matured (N=264, 8 distress events)
    c12_m = cohorts["cohort_T12_T13_matured"]
    assert c12_m["eligible_firms"] == 264
    assert c12_m["distress_events"] == 8
    assert c12_m["healthy_outcomes"] == 256
    assert c12_m["pending_or_right_censored"] == 6
    assert c12_m["integer_8pct_cap"] == 21
    assert c12_m["candidate_alerts_at_0_0894"] == 19
    assert c12_m["candidate_detections"] == 5

    # Cohort 3: T=13 Prospective (N=274, 12 right-censored pending)
    c13_p = cohorts["cohort_T13_prospective"]
    assert c13_p["eligible_firms"] == 274
    assert c13_p["pending_or_right_censored"] == 12

    # Cohort 4: T=13 -> T=14 Matured (N=262, 11 distress events)
    c13_m = cohorts["cohort_T13_T14_matured"]
    assert c13_m["eligible_firms"] == 262
    assert c13_m["distress_events"] == 11
    assert c13_m["healthy_outcomes"] == 251
    assert c13_m["pending_or_right_censored"] == 12
    assert c13_m["integer_8pct_cap"] == 20
    assert c13_m["production_alerts_at_0_0800"] == 37
    assert c13_m["candidate_alerts_at_0_0894"] == 20
    assert c13_m["production_detections"] == 5
    assert c13_m["candidate_detections"] == 3

    # Overlap audit: zero shared distress companies
    overlap = c_rec["independence_and_overlap_audit"]
    assert overlap["shared_distress_companies_count"] == 0
    assert overlap["shared_distress_company_ids"] == []


# =============================================================================
# TEST 3: ALERT CAP ARITHMETIC ACROSS COHORTS
# =============================================================================

def test_alert_cap_arithmetic_across_cohorts(phase6_4_3_deliverables):
    b_ver = phase6_4_3_deliverables["b_ver"]
    for row in b_ver["cohort_verifications"]:
        n = row["eligible"]
        cap = row["integer_cap"]
        assert cap == int(np.floor(0.08 * n)), f"Floor formula failure on N={n}"

    # Specific cohort checks
    assert int(np.floor(0.08 * 262)) == 20
    assert int(np.floor(0.08 * 270)) == 21
    assert int(np.floor(0.08 * 264)) == 21
    assert int(np.floor(0.08 * 274)) == 21

    # Reconciliation of reported alert counts
    reconciled = b_ver["reconciliation_of_reported_alert_counts"]
    assert "COMPLIANT" in reconciled["20_alerts_on_262_firms"]
    assert "BREACH" in reconciled["23_alerts_on_270_firms"]
    assert "COMPLIANT" in reconciled["19_alerts_on_264_firms"]
    assert "BREACH" in reconciled["37_alerts_on_262_firms"]


# =============================================================================
# TEST 4: NATIVE THRESHOLD DECISIONS AND METRIC RECOMPUTATION
# =============================================================================

def test_native_threshold_decisions_and_metrics(phase6_4_3_deliverables):
    m_rec = phase6_4_3_deliverables["m_rec"]
    assert m_rec["discrepancy_audit"]["material_discrepancies_found"] is False

    prod = m_rec["production_baseline_reproduction"]
    cand = m_rec["shadow_candidate_reproduction"]

    # Production confusion matrix & metrics
    cm_p = prod["confusion_matrix"]
    assert cm_p["true_positives"] == 5
    assert cm_p["false_positives"] == 32
    assert cm_p["false_negatives"] == 6
    assert cm_p["true_negatives"] == 219
    assert cm_p["total_alerts"] == 37
    assert math.isclose(prod["metrics"]["recall"], 0.4545, abs_tol=1e-3)
    assert math.isclose(prod["metrics"]["precision"], 0.1351, abs_tol=1e-3)
    assert math.isclose(prod["metrics"]["specificity"], 0.8725, abs_tol=1e-3)
    assert math.isclose(prod["metrics"]["roc_auc"], 0.7961, abs_tol=1e-3)
    assert math.isclose(prod["metrics"]["pr_auc"], 0.1816, abs_tol=1e-3)
    assert math.isclose(prod["metrics"]["brier_score"], 0.0382, abs_tol=1e-3)

    # Candidate confusion matrix & metrics
    cm_c = cand["confusion_matrix"]
    assert cm_c["true_positives"] == 3
    assert cm_c["false_positives"] == 17
    assert cm_c["false_negatives"] == 8
    assert cm_c["true_negatives"] == 234
    assert cm_c["total_alerts"] == 20
    assert math.isclose(cand["metrics"]["recall"], 0.2727, abs_tol=1e-3)
    assert math.isclose(cand["metrics"]["precision"], 0.1500, abs_tol=1e-3)
    assert math.isclose(cand["metrics"]["specificity"], 0.9323, abs_tol=1e-3)
    assert math.isclose(cand["metrics"]["roc_auc"], 0.7521, abs_tol=1e-3)
    assert math.isclose(cand["metrics"]["pr_auc"], 0.1410, abs_tol=1e-3)
    assert math.isclose(cand["metrics"]["brier_score"], 0.0389, abs_tol=1e-3)


# =============================================================================
# TEST 5: FAILURE CASE REPRODUCTION (FIRMS 28 & 60)
# =============================================================================

def test_failure_case_reproduction(phase6_4_3_deliverables):
    f_rep = phase6_4_3_deliverables["f_rep"]
    cases = f_rep["cases"]

    f28 = cases["firm_28"]
    f60 = cases["firm_60"]

    assert f28["antecedent_fd_t13"] > 0.0
    assert math.isclose(f28["antecedent_fd_t13"], 1.4430, abs_tol=1e-3)
    assert f28["production_alert_decision"] is True
    assert f28["candidate_alert_decision"] is False
    assert math.isclose(f28["production_calibrated_probability"], 0.0858, abs_tol=1e-3)
    assert math.isclose(f28["candidate_calibrated_probability"], 0.0496, abs_tol=1e-3)
    assert f28["below_production_threshold_0_0800"] is True

    assert f60["antecedent_fd_t13"] > 0.0
    assert math.isclose(f60["antecedent_fd_t13"], 0.1357, abs_tol=1e-3)
    assert f60["production_alert_decision"] is True
    assert f60["candidate_alert_decision"] is False
    assert math.isclose(f60["production_calibrated_probability"], 0.1135, abs_tol=1e-3)
    assert math.isclose(f60["candidate_calibrated_probability"], 0.0695, abs_tol=1e-3)
    assert f60["below_production_threshold_0_0800"] is True


# =============================================================================
# TEST 6: SUBGROUP RECALL AND SUDDEN TRANSITION GENERALIZATION
# =============================================================================

def test_subgroup_recall_and_generalization(phase6_4_3_deliverables):
    m_rec = phase6_4_3_deliverables["m_rec"]
    prod_sub = m_rec["production_baseline_reproduction"]["subgroup_recall"]
    cand_sub = m_rec["shadow_candidate_reproduction"]["subgroup_recall"]

    # Prior healthy
    assert prod_sub["prior_healthy_detected"] == 2
    assert math.isclose(prod_sub["prior_healthy_recall"], 0.4000, abs_tol=1e-3)
    assert cand_sub["prior_healthy_detected"] == 0
    assert math.isclose(cand_sub["prior_healthy_recall"], 0.0000, abs_tol=1e-3)

    # Prior distressed
    assert prod_sub["prior_distressed_detected"] == 3
    assert math.isclose(prod_sub["prior_distressed_recall"], 0.5000, abs_tol=1e-3)
    assert cand_sub["prior_distressed_detected"] == 3
    assert math.isclose(cand_sub["prior_distressed_recall"], 0.5000, abs_tol=1e-3)

    # Generalization audit across Dataset A
    r_val = phase6_4_3_deliverables["r_val"]["historical_transition_analysis"]
    assert r_val["total_distress_events"] == 106
    assert r_val["sudden_distress_transitions_count"] == 36
    assert r_val["chronic_distress_transitions_count"] == 70
    assert math.isclose(r_val["sudden_distress_share"], 0.3396, abs_tol=1e-3)


# =============================================================================
# TEST 7: STATISTICAL PROTOCOL REVIEW AND GATE DECISION
# =============================================================================

def test_statistical_protocol_and_gate_decision(phase6_4_3_deliverables):
    s_rev = phase6_4_3_deliverables["s_rev"]
    assert "UNVERIFIED_FOR_PROMOTION" in s_rev["protocol_governance_status"]
    assert s_rev["event_count_limitations"]["current_status"] == "STATISTICALLY_UNDERPOWERED_FOR_CONFIRMATORY_CLAIMS"

    p_man = phase6_4_3_deliverables["p_man"]
    assert p_man["candidate_promotion_state"] == "PASSIVE_SHADOW_CANDIDATE (FROZEN)"
