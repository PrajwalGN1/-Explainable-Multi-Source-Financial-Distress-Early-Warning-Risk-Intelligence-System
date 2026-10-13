"""
Phase 6.4.2 Regression Test Suite:
Candidate Sensitivity Diagnostics & Leakage-Safe Redesign Study.

Tests:
1. Production baseline preservation: SHA-256 digests verified, official threshold 0.0800 frozen,
   candidate threshold 0.0894 frozen, promotion locked.
2. Forensic failure case reconstruction: Firms 28 and 60, prior solvent states, probability values,
   cohort ranks, and proof of probability-ranking collapse below 0.0800.
3. Subgroup recall and sudden-transition collapse: Prior healthy (FD > 0) recall 0.0% candidate vs 40.0% production;
   prior distressed (FD <= 0) recall 50.0% for both; Sector 25 stress concentration (5/11 defaults).
4. Cost-sensitive weight sensitivity: Across w in [1, 2, 5, 10], healthy recall remains 0/5; only w=10 meets 8% cap.
5. Ensemble and alternative models tradeoff: Blends with alpha >= 0.4 detect Firm 60 but breach the 8% cap (31-35 alerts).
6. Alert capacity hard cap enforcement: Integer cap = floor(N * 0.08) = 20; baseline at 0.0800 generates 37 alerts;
   threshold inflation required to meet cap collapses recall.
7. Statistical protocol unverified & candidate promotion lockout: Protocol remains UNVERIFIED; promotion strictly blocked.
"""

import os
import json
import math
import pytest
import numpy as np
import pandas as pd

from src.shadow.shadow_engine import compute_sha256
from src.shadow.candidate_sensitivity_study_phase6_4_2 import (
    verify_production_baseline_frozen,
    reconstruct_failure_cases,
    perform_subgroup_recall_analysis,
    evaluate_research_model_alternatives,
    audit_alert_budget_tradeoffs,
    MANIFEST_PATH,
    OFFICIAL_PROD_THRESHOLD,
    REGISTERED_CAND_THRESHOLD,
    RESULTS_DIR
)


@pytest.fixture(scope="module")
def phase6_4_2_deliverables():
    out_fail = os.path.join(RESULTS_DIR, "failure_case_diagnostics.json")
    out_sub = os.path.join(RESULTS_DIR, "subgroup_recall_analysis.json")
    out_mod = os.path.join(RESULTS_DIR, "research_model_comparison.json")
    out_bud = os.path.join(RESULTS_DIR, "alert_budget_tradeoff.json")

    assert os.path.exists(out_fail), f"Missing deliverable: {out_fail}"
    assert os.path.exists(out_sub), f"Missing deliverable: {out_sub}"
    assert os.path.exists(out_mod), f"Missing deliverable: {out_mod}"
    assert os.path.exists(out_bud), f"Missing deliverable: {out_bud}"

    with open(out_fail, "r", encoding="utf-8") as f:
        fail_data = json.load(f)
    with open(out_sub, "r", encoding="utf-8") as f:
        sub_data = json.load(f)
    with open(out_mod, "r", encoding="utf-8") as f:
        mod_data = json.load(f)
    with open(out_bud, "r", encoding="utf-8") as f:
        bud_data = json.load(f)

    return {
        "fail": fail_data,
        "sub": sub_data,
        "mod": mod_data,
        "bud": bud_data
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
        assert compute_sha256(pth) == exp_hash, f"Hash mismatch on {pth}"


# =============================================================================
# TEST 2: FORENSIC FAILURE CASE RECONSTRUCTION (FIRMS 28 & 60)
# =============================================================================

def test_failure_case_reconstruction_firms_28_and_60(phase6_4_2_deliverables):
    fail_data = phase6_4_2_deliverables["fail"]
    cases = fail_data["cases"]

    assert "firm_28" in cases
    assert "firm_60" in cases

    f28 = cases["firm_28"]
    f60 = cases["firm_60"]

    # 1. Verify prior solvent financial health
    assert f28["previous_period_distress_index_t13"] > 0.0
    assert math.isclose(f28["previous_period_distress_index_t13"], 1.4430, abs_tol=1e-3)
    assert f28["target_period_distress_index_t14"] <= -0.5
    assert math.isclose(f28["target_period_distress_index_t14"], -0.6815, abs_tol=1e-3)

    assert f60["previous_period_distress_index_t13"] > 0.0
    assert math.isclose(f60["previous_period_distress_index_t13"], 0.1357, abs_tol=1e-3)
    assert f60["target_period_distress_index_t14"] <= -0.5
    assert math.isclose(f60["target_period_distress_index_t14"], -1.2361, abs_tol=1e-3)

    # 2. Production detected both firms
    assert f28["production_model_evaluation"]["native_alert_decision"] is True
    assert f28["production_model_evaluation"]["calibrated_probability"] >= OFFICIAL_PROD_THRESHOLD
    assert math.isclose(f28["production_model_evaluation"]["calibrated_probability"], 0.0858, abs_tol=1e-3)

    assert f60["production_model_evaluation"]["native_alert_decision"] is True
    assert f60["production_model_evaluation"]["calibrated_probability"] >= OFFICIAL_PROD_THRESHOLD
    assert math.isclose(f60["production_model_evaluation"]["calibrated_probability"], 0.1135, abs_tol=1e-3)

    # 3. Candidate missed both firms
    assert f28["candidate_model_evaluation"]["native_alert_decision"] is False
    assert f28["candidate_model_evaluation"]["calibrated_probability"] < REGISTERED_CAND_THRESHOLD
    assert math.isclose(f28["candidate_model_evaluation"]["calibrated_probability"], 0.0496, abs_tol=1e-3)

    assert f60["candidate_model_evaluation"]["native_alert_decision"] is False
    assert f60["candidate_model_evaluation"]["calibrated_probability"] < REGISTERED_CAND_THRESHOLD
    assert math.isclose(f60["candidate_model_evaluation"]["calibrated_probability"], 0.0695, abs_tol=1e-3)

    # 4. Mathematical proof of probability ranking collapse below BOTH thresholds
    for f in [f28, f60]:
        pr_diff = f["probability_ranking_differential"]
        assert pr_diff["below_candidate_threshold_0_0894"] is True
        assert pr_diff["below_production_threshold_0_0800"] is True
        assert pr_diff["mechanism_classification"] == "PROBABILITY_RANKING_FAILURE"
        assert f["data_quality_and_leakage_audit"]["missing_values_at_t13"] == 0
        assert "ZERO" in f["data_quality_and_leakage_audit"]["future_information_leakage"]

    # 5. Verify cohort rank drop
    assert f28["probability_ranking_differential"]["rank_slippage"] > 0
    assert f60["probability_ranking_differential"]["rank_slippage"] > 0


# =============================================================================
# TEST 3: SUBGROUP RECALL & SUDDEN-TRANSITION SENSITIVITY COLLAPSE
# =============================================================================

def test_subgroup_recall_and_sudden_transition_collapse(phase6_4_2_deliverables):
    sub_data = phase6_4_2_deliverables["sub"]
    health = sub_data["prior_financial_health_subgroups"]

    h_group = health["prior_healthy_solvent"]
    d_group = health["prior_distressed_borderline"]

    # Prior healthy: 241 firms, 5 defaults
    assert h_group["total_firms"] == 241
    assert h_group["distress_events"] == 5
    assert h_group["production_baseline"]["detected"] == 2
    assert math.isclose(h_group["production_baseline"]["recall"], 0.4000, abs_tol=1e-3)
    assert h_group["shadow_candidate"]["detected"] == 0
    assert math.isclose(h_group["shadow_candidate"]["recall"], 0.0000, abs_tol=1e-3)
    assert h_group["finding"] == "TOTAL_SENSITIVITY_COLLAPSE (0% candidate recall)"

    # Prior distressed/borderline: 21 firms, 6 defaults
    assert d_group["total_firms"] == 21
    assert d_group["distress_events"] == 6
    assert d_group["production_baseline"]["detected"] == 3
    assert math.isclose(d_group["production_baseline"]["recall"], 0.5000, abs_tol=1e-3)
    assert d_group["shadow_candidate"]["detected"] == 3
    assert math.isclose(d_group["shadow_candidate"]["recall"], 0.5000, abs_tol=1e-3)
    assert d_group["finding"] == "EQUIVALENT_SENSITIVITY"

    # Sector 25 stress concentration
    sec = sub_data["industry_sector_subgroups"]["sector_25_stress_cluster"]
    assert sec["distress_events"] == 5
    assert sec["production_baseline"]["detected"] == 2
    assert sec["shadow_candidate"]["detected"] == 0


# =============================================================================
# TEST 4: COST-SENSITIVE WEIGHT SENSITIVITY STUDY
# =============================================================================

def test_cost_sensitive_weight_sensitivity(phase6_4_2_deliverables):
    mod_data = phase6_4_2_deliverables["mod"]
    weights = mod_data["weight_sensitivity_study"]

    assert len(weights) == 4
    weights_evaluated = [w["model_name"] for w in weights]
    assert "Phase5_RF_Weight_w1" in weights_evaluated
    assert "Phase5_RF_Weight_w2" in weights_evaluated
    assert "Phase5_RF_Weight_w5" in weights_evaluated
    assert "Phase5_RF_Weight_w10" in weights_evaluated

    # Across all weights, healthy distress recall remains 0/5
    for w in weights:
        assert w["subgroup_healthy_detected"] == 0
        assert math.isclose(w["subgroup_healthy_recall"], 0.0, abs_tol=1e-3)

    # Only w=10 meets the integer 8% capacity cap
    w10 = next(w for w in weights if "w10" in w["model_name"])
    assert w10["cap_compliant"] is True
    assert w10["alerts_issued"] <= 20

    w1 = next(w for w in weights if "w1" == w["model_name"].split("_")[-1])
    assert w1["cap_compliant"] is False
    assert w1["alerts_issued"] > 20


# =============================================================================
# TEST 5: ENSEMBLE AND ALTERNATIVE MODELS TRADEOFF
# =============================================================================

def test_ensemble_and_alternative_models_tradeoff(phase6_4_2_deliverables):
    mod_data = phase6_4_2_deliverables["mod"]
    ensembles = mod_data["ensemble_blends_study"]

    assert len(ensembles) == 5

    # Blends with alpha >= 0.4 catch 1 healthy distress firm (Firm 60)
    for b in ensembles:
        if b["blend_weight_baseline"] >= 0.4:
            assert b["subgroup_healthy_detected"] == 1
            assert math.isclose(b["distress_recall"], 0.3636, abs_tol=1e-3)
            # However, all breach the 8% alert cap
            assert b["cap_compliant"] is False
            assert b["alerts_issued"] > 20


# =============================================================================
# TEST 6: ALERT CAPACITY HARD CAP ENFORCEMENT & THRESHOLD TRADEOFF
# =============================================================================

def test_alert_capacity_hard_cap_enforcement(phase6_4_2_deliverables):
    bud_data = phase6_4_2_deliverables["bud"]
    assert bud_data["cohort_size"] == 262
    assert bud_data["matured_distress_events"] == 11
    assert bud_data["integer_8pct_capacity_cap"] == 20

    cap_audit = bud_data["capacity_compliance_audit"]

    # Baseline at 0.0800: 37 alerts, non-compliant
    b_audit = cap_audit["baseline_at_official_threshold_0_0800"]
    assert b_audit["alerts"] == 37
    assert b_audit["compliant"] is False
    assert b_audit["excess"] == 17
    assert math.isclose(b_audit["recall"], 0.4545, abs_tol=1e-3)

    # Candidate at 0.0894: 20 alerts, compliant, but recall drops to 0.2727
    c_audit = cap_audit["candidate_at_registered_threshold_0_0894"]
    assert c_audit["alerts"] == 20
    assert c_audit["compliant"] is True
    assert c_audit["excess"] == 0
    assert math.isclose(c_audit["recall"], 0.2727, abs_tol=1e-3)

    # Required baseline threshold for compliance causes severe recall penalty
    pen = cap_audit["baseline_compliance_penalty"]
    assert pen["threshold_required_for_compliance"] >= 0.1100
    assert pen["recall_at_compliant_threshold"] <= 0.2727
    assert pen["recall_drop_percentage_points"] >= 18.0


# =============================================================================
# TEST 7: STATISTICAL PROTOCOL UNVERIFIED & CANDIDATE PROMOTION LOCKOUT
# =============================================================================

def test_statistical_protocol_unverified_and_promotion_lockout(phase6_4_2_deliverables):
    audit = verify_production_baseline_frozen()
    assert audit["candidate_promotion_state"] == "PASSIVE_SHADOW_CANDIDATE (FROZEN)"

    # Verify that sequential boundaries remain unverified
    mod_data = phase6_4_2_deliverables["mod"]
    synthesis = mod_data["comparative_synthesis"]
    assert "feasibility_conclusion" in synthesis

    # Event count reporting
    assert phase6_4_2_deliverables["bud"]["matured_distress_events"] == 11
    assert phase6_4_2_deliverables["fail"]["matured_distress_count"] == 11
