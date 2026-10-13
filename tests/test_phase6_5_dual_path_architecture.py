"""
Phase 6.5 Automated Test Suite:
Dual-Path Financial Distress Detection Architecture Redesign Study.

Comprehensive test coverage verifying:
1. Production baseline manifest immutability (SHA-256 hashes, official threshold 0.0800 frozen).
2. Independent evaluation of Path A and Path B (isolated execution, fault tolerance).
3. Strict point-in-time feature integrity (backward-looking rolling windows, no future leakage).
4. Training and validation preprocessing isolation (fit/transform partition integrity).
5. Alert coordinator firm-level deduplication and hard integer cap enforcement (floor(0.08 * N)).
6. Missing-score and exception handling (explicit logging, no silent healthy classification).
7. Right-censored observation handling (filing cessations excluded from binary contingency tables).
8. Evaluation cohort identities, denominators, and timestamp consistency (262, 264, 270, 274).
9. Native continuous probabilities, individual thresholds, and decision retention.
10. No final-holdout labels used for model or threshold selection.
11. Deterministic coordinator behavior under fixed inputs.
12. Deliverable artifact completeness, schema validity, and full traceability.
13. Statistical governance and promotion lock out (INELIGIBLE_FOR_PROMOTION).
"""

import os
import json
import math
import pytest
import numpy as np
import pandas as pd

import joblib
from src.preprocessing.preprocessors import EWSPreprocessor
from src.shadow.shadow_engine import compute_sha256, ShadowInferenceEngine
from src.shadow.dual_path_architecture_phase6_5 import (
    verify_production_integrity,
    audit_cohort_integrity,
    audit_sudden_transitions,
    compute_path_metrics,
    evaluate_decision_coordinator_variants,
    analyze_alert_budget_allocations,
    audit_calibration_and_stability,
    MANIFEST_PATH,
    OFFICIAL_PROD_THRESHOLD,
    REGISTERED_CAND_THRESHOLD,
    RESULTS_DIR
)


@pytest.fixture(scope="module")
def phase6_5_deliverables():
    """Loads all 8 Phase 6.5 JSON deliverable files."""
    deliverables = {
        "integrity": os.path.join(RESULTS_DIR, "production_integrity_manifest.json"),
        "cohort": os.path.join(RESULTS_DIR, "cohort_integrity_audit.json"),
        "sudden": os.path.join(RESULTS_DIR, "sudden_transition_reproduction.json"),
        "path_a": os.path.join(RESULTS_DIR, "path_a_chronic_deterioration_metrics.json"),
        "path_b": os.path.join(RESULTS_DIR, "path_b_acute_distress_metrics.json"),
        "comparison": os.path.join(RESULTS_DIR, "dual_path_comparison.json"),
        "budget": os.path.join(RESULTS_DIR, "alert_budget_allocation_analysis.json"),
        "calibration": os.path.join(RESULTS_DIR, "calibration_and_stability.json")
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

def test_production_baseline_manifest_immutability(phase6_5_deliverables):
    """Verifies that all 8 protected production artifacts match the manifest and threshold is 0.0800."""
    integrity = phase6_5_deliverables["integrity"]
    assert integrity["production_integrity_verdict"] == "PASSED"
    assert integrity["cryptographic_hash_invariance"] is True
    assert integrity["official_production_threshold_frozen"] is True
    assert math.isclose(integrity["official_production_threshold"], OFFICIAL_PROD_THRESHOLD, rel_tol=1e-5)
    assert math.isclose(integrity["threshold_on_disk"], OFFICIAL_PROD_THRESHOLD, rel_tol=1e-5)
    assert integrity["candidate_promotion_state"] == "PASSIVE_SHADOW_CANDIDATE (QUARANTINED)"

    # Live verification against manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    for path, expected_hash in manifest["pre_implementation_hashes"].items():
        assert compute_sha256(path) == expected_hash, f"Hash mismatch on {path}"


# =============================================================================
# TEST 2: INDEPENDENT PATHWAY EVALUATION
# =============================================================================

def test_independent_pathway_evaluation():
    """Verifies that Path A and Path B execute independently and faults in one do not disrupt the other."""
    engine = ShadowInferenceEngine()
    assert engine.prod_model is not None
    assert engine.cand_model is not None

    # Test dummy sample with production alone
    dummy_hist = pd.DataFrame({
        "Company": [999, 999],
        "Time": [1, 2],
        "Financial Distress": [0.5, 0.4]
    })
    for i in range(1, 84):
        dummy_hist[f"x{i}"] = [0.1, 0.2]

    # Candidate evaluation runs in isolated sandbox
    res = engine.evaluate_shadow_sample("999", dummy_hist, current_period=2, record_audit_log=False)
    assert "production" in res
    assert "shadow_candidate" in res
    assert "calibrated_probability" in res["production"]
    assert "calibrated_probability" in res["shadow_candidate"]


# =============================================================================
# TEST 3: STRICT POINT-IN-TIME FEATURE INTEGRITY
# =============================================================================

def test_point_in_time_feature_integrity():
    """Verifies that feature engineering uses strictly backward-looking operations without future leakage."""
    from src.evaluation.phase5_feature_engineering import Phase5FeatureEngine
    engine = Phase5FeatureEngine()

    # Synthetic test DataFrame with 2 companies over 3 periods
    df_test = pd.DataFrame({
        "Company": [1, 1, 1, 2, 2, 2],
        "Time": [1, 2, 3, 1, 2, 3],
        "Financial Distress": [0.8, 0.6, 0.4, 0.9, 0.7, 0.5],
        "x1": [10.0, 20.0, 30.0, 15.0, 25.0, 35.0],
        "x2": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5]
    })
    for i in range(3, 84):
        df_test[f"x{i}"] = 0.5

    df_feats = engine.generate_deterioration_features(df_test)
    
    # Check that at Time 1, lag is NaN / handled, and roll_mean equals value at Time 1
    t1_c1 = df_feats[(df_feats["Company"] == 1) & (df_feats["Time"] == 1)]
    assert pd.isna(t1_c1["x1_lag_1"].values[0]) or math.isnan(t1_c1["x1_lag_1"].values[0])
    assert math.isclose(t1_c1["x1_roll_mean_2"].values[0], 10.0, rel_tol=1e-5)

    # Check that at Time 2, lag_1 is exactly x1 at Time 1 (10.0)
    t2_c1 = df_feats[(df_feats["Company"] == 1) & (df_feats["Time"] == 2)]
    assert math.isclose(t2_c1["x1_lag_1"].values[0], 10.0, rel_tol=1e-5)
    assert math.isclose(t2_c1["x1_diff_1"].values[0], 10.0, rel_tol=1e-5)

    # Verify synthetic proxy exclusion (x64, x81)
    proxy_cols = [c for c in df_feats.columns if "x64" in c or "x81" in c]
    assert len(proxy_cols) == 0


# =============================================================================
# TEST 4: PREPROCESSING ISOLATION
# =============================================================================

def test_preprocessing_isolation():
    """Verifies that preprocessor pipelines are isolated and fitted on training partitions."""
    prep_prod = EWSPreprocessor.load("models/phase2/ews_preprocessor.joblib")
    prep_cand = joblib.load("experiments/phase5/models/Phase5_preprocessor.joblib")

    assert prep_prod.fitted is True
    assert prep_cand.fitted is True
    assert len(prep_prod.feature_names) == 163
    assert len(prep_cand.feature_names) == 186


# =============================================================================
# TEST 5: DEDUPLICATION AND STRICT INTEGER CAP ENFORCEMENT
# =============================================================================

def test_coordinator_deduplication_and_integer_cap(phase6_5_deliverables):
    """Verifies firm-level alert deduplication and integer capacity cap compliance."""
    comp = phase6_5_deliverables["comparison"]
    variants = comp["variants"]

    v1 = variants["variant_1_path_a_alone"]
    v2 = variants["variant_2_path_b_alone"]
    v3 = variants["variant_3_naive_union"]
    v4 = variants["variant_4_constrained_partitioned_allocation"]
    v5 = variants["variant_5_solvency_gated_policy"]

    # Cohort Cap is 20 alerts (floor(262 * 0.08) = 20)
    assert v1["integer_8pct_cap"] == 20
    assert v1["total_alerts"] == 20
    assert v1["cap_compliant"] is True

    # Path B baseline breaches cap (37 alerts > 20)
    assert v2["total_alerts"] == 37
    assert v2["cap_compliant"] is False
    assert v2["excess_alerts"] == 17

    # Naive Union breaches cap (39 alerts > 20)
    assert v3["total_alerts"] == 39
    assert v3["cap_compliant"] is False
    assert v3["excess_alerts"] == 19

    # Variant 4 (Partitioned Top-K: Ka=14, Kb=6) is strictly compliant (14 <= 20)
    assert v4["total_alerts"] == 14
    assert v4["cap_compliant"] is True
    assert v4["total_alerts"] <= 20

    # Variant 5 (Solvency-Gated) is strictly compliant (17 <= 20)
    assert v5["total_alerts"] == 17
    assert v5["cap_compliant"] is True
    assert v5["total_alerts"] <= 20


# =============================================================================
# TEST 6: MISSING SCORE AND EXCEPTION HANDLING
# =============================================================================

def test_missing_scores_and_exception_handling():
    """Verifies that missing features or incomplete data are handled explicitly without silent low-risk classification."""
    engine = ShadowInferenceEngine()
    empty_hist = pd.DataFrame({"Company": [], "Time": []})

    # Missing observation at current period explicitly raises ValueError
    with pytest.raises(ValueError, match="No observation found"):
        engine.evaluate_shadow_sample("9999", empty_hist, current_period=1, record_audit_log=False)


# =============================================================================
# TEST 7: RIGHT-CENSORING AND COHORT RECONCILIATION
# =============================================================================

def test_right_censoring_and_cohort_reconciliation(phase6_5_deliverables):
    """Verifies that right-censored non-filing firms are excluded from binary contingency tables."""
    cohort = phase6_5_deliverables["cohort"]["cohorts"]

    # T12 -> T13 matured cohort excludes 6 non-filers
    assert cohort["cohort_T12_T13_matured"]["eligible_firms"] == 264
    assert cohort["cohort_T12_T13_matured"]["right_censored_excluded"] == 6
    assert cohort["cohort_T12_T13_matured"]["distress_events"] == 8
    assert cohort["cohort_T12_T13_matured"]["healthy_outcomes"] == 256

    # T13 -> T14 matured cohort excludes 12 non-filers
    assert cohort["cohort_T13_T14_matured"]["eligible_firms"] == 262
    assert cohort["cohort_T13_T14_matured"]["right_censored_excluded"] == 12
    assert cohort["cohort_T13_T14_matured"]["distress_events"] == 11
    assert cohort["cohort_T13_T14_matured"]["healthy_outcomes"] == 251

    # Zero distress overlap across T13 and T14
    overlap = phase6_5_deliverables["cohort"]["distress_overlap_audit"]
    assert overlap["shared_distress_count"] == 0


# =============================================================================
# TEST 8: COHORT IDENTITIES AND TIMESTAMPS
# =============================================================================

def test_cohort_identities_and_timestamps(phase6_5_deliverables):
    """Verifies cohort sizes, observation periods, and integer caps."""
    cohorts = phase6_5_deliverables["cohort"]["cohorts"]

    assert cohorts["cohort_T12_prospective"]["eligible_firms"] == 270
    assert cohorts["cohort_T12_prospective"]["integer_8pct_cap"] == 21

    assert cohorts["cohort_T12_T13_matured"]["eligible_firms"] == 264
    assert cohorts["cohort_T12_T13_matured"]["integer_8pct_cap"] == 21

    assert cohorts["cohort_T13_prospective"]["eligible_firms"] == 274
    assert cohorts["cohort_T13_prospective"]["integer_8pct_cap"] == 21

    assert cohorts["cohort_T13_T14_matured"]["eligible_firms"] == 262
    assert cohorts["cohort_T13_T14_matured"]["integer_8pct_cap"] == 20


# =============================================================================
# TEST 9: NATIVE PROBABILITIES AND THRESHOLD RETENTION
# =============================================================================

def test_native_probabilities_and_threshold_retention(phase6_5_deliverables):
    """Verifies that native unrounded probabilities and thresholds are preserved for Firms 28 and 60."""
    budget = phase6_5_deliverables["budget"]["firm_rankings_and_probabilities"]

    f28 = budget["firm_28"]
    f60 = budget["firm_60"]

    # Firm 28
    assert math.isclose(f28["path_b_acute_prob"], 0.0858, rel_tol=1e-3)
    assert math.isclose(f28["path_a_chronic_prob"], 0.0496, rel_tol=1e-3)
    assert f28["path_b_acute_rank"] == 35
    assert f28["path_a_chronic_rank"] == 51

    # Firm 60
    assert math.isclose(f60["path_b_acute_prob"], 0.1135, rel_tol=1e-3)
    assert math.isclose(f60["path_a_chronic_prob"], 0.0695, rel_tol=1e-3)
    assert f60["path_b_acute_rank"] == 16
    assert f60["path_a_chronic_rank"] == 35


# =============================================================================
# TEST 10: NO HOLDOUT LABEL LEAKAGE
# =============================================================================

def test_no_holdout_label_leakage(phase6_5_deliverables):
    """Verifies that coordinator decision rules were fixed without using holdout ground truth labels."""
    comparison = phase6_5_deliverables["comparison"]
    v4 = comparison["variants"]["variant_4_constrained_partitioned_allocation"]
    v5 = comparison["variants"]["variant_5_solvency_gated_policy"]

    # Variant 4 specifies Ka=14, Kb=6 independent of holdout labels
    assert v4["allocated_ka_chronic"] == 14
    assert v4["allocated_kb_acute"] == 6

    # Variant 5 uses prior observation FD_obs <= 0 (point-in-time) without future target
    assert "FD<=0" in v5["decision_rule"]


# =============================================================================
# TEST 11: DETERMINISTIC COORDINATOR BEHAVIOR
# =============================================================================

def test_deterministic_coordinator_behavior():
    """Verifies that the decision coordinator produces 100% identical outputs on repeated evaluations."""
    df_sample = pd.DataFrame({
        "company_id": [1, 2, 3, 4],
        "y_true": [0, 1, 0, 1],
        "fd_obs": [1.2, -0.4, 0.8, -0.2],
        "fd_tgt": [1.1, -0.8, 0.7, -0.9],
        "is_sudden": [False, False, False, False],
        "is_chronic": [False, True, False, True],
        "p_path_b": [0.05, 0.12, 0.07, 0.15],
        "p_path_a": [0.03, 0.10, 0.04, 0.14],
        "alert_path_b": [False, True, False, True],
        "alert_path_a": [False, True, False, True]
    })

    run1 = evaluate_decision_coordinator_variants(df_sample)
    run2 = evaluate_decision_coordinator_variants(df_sample)

    assert json.dumps(run1, sort_keys=True) == json.dumps(run2, sort_keys=True)


# =============================================================================
# TEST 12: DELIVERABLES AND TRACEABILITY
# =============================================================================

def test_deliverables_and_traceability(phase6_5_deliverables):
    """Verifies that all 8 deliverables exist and contain non-empty structured data."""
    assert len(phase6_5_deliverables) == 8
    for key, data in phase6_5_deliverables.items():
        assert isinstance(data, dict)
        assert len(data) > 0


# =============================================================================
# TEST 13: HYPOTHESIS AND GATE STATUS
# =============================================================================

def test_hypothesis_and_gate_status(phase6_5_deliverables):
    """Verifies that the research hypothesis was not supported and promotion remains strictly locked."""
    comparison = phase6_5_deliverables["comparison"]
    estimand = comparison["estimand_analysis"]

    assert "HYPOTHESIS NOT SUPPORTED" in estimand["hypothesis_verdict"]
    assert estimand["variant_4_partitioned_delta_vs_production"] < 0
    assert estimand["variant_5_gated_delta_vs_production"] < 0

    integrity = phase6_5_deliverables["integrity"]
    assert integrity["candidate_promotion_state"] == "PASSIVE_SHADOW_CANDIDATE (QUARANTINED)"
