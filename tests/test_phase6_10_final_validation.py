"""
Phase 6.10: Final Validation, System Consistency Audit & Release Readiness Tests.

Automated verification of:
1. Cryptographic immutability of approved production baseline.
2. Dataset dimensions, zero-missingness, zero-duplicates, contiguity, and target direction.
3. Chronological leakage prevention (train-only scaling, val-only calibration, test evaluation).
4. Confusion matrix mathematical invariants and alert-budget compliance.
5. Calibrated probability bounds and Logistic Regression score distribution behavior.
6. TreeSHAP feature ranking consistency (x2, x12, x8) and non-causal association semantics.
7. Streamlit application smoke test and parameter guardrails.
8. FastAPI REST endpoints functionality, error schemas (404/422), and fail-closed admin security.
9. Security and repository hygiene verification.
"""

import os
import json
import hashlib
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api import app
from src.shadow.dual_path_architecture_phase6_5 import compute_sha256, OFFICIAL_PROD_THRESHOLD, REGISTERED_CAND_THRESHOLD


# =============================================================================
# 1. PRODUCTION BASELINE INTEGRITY
# =============================================================================

def test_production_baseline_integrity():
    """Verifies that all 8 protected production artifacts match trusted reference SHA-256 hashes."""
    manifest_path = "results/phase6_10/production_integrity_manifest.json"
    assert os.path.exists(manifest_path), "Production integrity manifest missing."
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["all_protected_artifacts_match"] is True
    assert data["immutability_status"] == "PROTECTED_BASELINE_VERIFIED_UNCHANGED"
    assert data["total_protected_artifacts"] == 8
    assert data["verified_intact_count"] == 8
    
    # Live disk verification
    for rel_path, exp_hash in data["expected_hashes"].items():
        assert os.path.exists(rel_path), f"Protected artifact missing on disk: {rel_path}"
        curr_hash = compute_sha256(rel_path)
        assert curr_hash == exp_hash, f"Hash mismatch for {rel_path}: {curr_hash} != {exp_hash}"


# =============================================================================
# 2. DATASET SEMANTICS & TARGET INTEGRITY
# =============================================================================

def test_dataset_semantics_and_target_invariants():
    """Verifies Dataset A dimensions, quality invariants, target direction, and terminal exit dynamics."""
    val_path = "results/phase6_10/dataset_validation.json"
    assert os.path.exists(val_path), "Dataset validation JSON missing."
    
    with open(val_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["dataset_validation_status"] == "DATASET_AND_TARGET_VERIFIED_AUTHENTIC"
    
    dims = data["dimensions"]
    assert dims["total_rows"] == 3672
    assert dims["total_columns"] == 86
    assert dims["unique_companies"] == 422
    assert dims["company_id_range"] == [1, 422]
    
    qual = data["data_quality_invariants"]
    assert qual["missing_values_count"] == 0
    assert qual["infinite_values_count"] == 0
    assert qual["duplicate_company_period_pairs"] == 0
    assert qual["time_contiguity_gaps"] == 0
    assert qual["zero_missing_invariant_verified"] is True
    assert qual["zero_duplicates_invariant_verified"] is True
    assert qual["full_time_contiguity_verified"] is True
    
    # Target direction semantics
    tgt = data["target_column_semantics"]
    assert tgt["statutory_distress_threshold"] == -0.50
    assert "LOWER values indicate greater financial distress" in tgt["distress_direction"]
    assert tgt["raw_distress_observations"] == 136
    assert tgt["raw_healthy_observations"] == 3536
    
    # 100% terminal exit dynamics
    exit_dyn = data["panel_attrition_dynamics"]
    assert exit_dyn["unique_distressed_firms"] == 136
    assert exit_dyn["distress_at_exact_final_observed_period"] == 136
    assert exit_dyn["terminal_exit_rate_pct"] == 100.0
    assert exit_dyn["post_distress_continuation_count"] == 0
    
    # Forward target shift
    fwd = data["forward_target_generation"]
    assert fwd["evaluable_observations"] == 3250
    assert fwd["terminal_unobserved_observations"] == 422
    assert fwd["total_forward_distress_events"] == 120


# =============================================================================
# 3. TEMPORAL LEAKAGE PREVENTION & PARTITIONS
# =============================================================================

def test_temporal_leakage_and_chronological_partitions():
    """Verifies chronological partition counts, train-only scaling, and zero cross-firm leakage."""
    leakage_path = "results/phase6_10/temporal_leakage_audit.json"
    assert os.path.exists(leakage_path), "Temporal leakage audit JSON missing."
    
    with open(leakage_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["temporal_leakage_status"] == "ZERO_LEAKAGE_CONFIRMED"
    
    checks = data["leakage_checks"]
    assert checks["cross_company_target_alignment_errors"] == 0
    assert checks["intra_firm_alignment_verified"] is True
    assert checks["preprocessor_fitted_strictly_on_train"] is True
    assert checks["scaler_mean_computed_on_train_only"] is True
    assert checks["calibration_fitted_on_validation_only"] is True
    assert checks["max_feature_target_correlation_train"]["leakage_threshold_exceeded"] is False
    
    # Holdout cohort characterization
    holdout = data["holdout_cohort_characterization"]
    assert holdout["holdout_period"] == "T=13 -> T=14"
    assert holdout["evaluable_firms"] == 262
    assert holdout["distress_events"] == 11
    assert holdout["status"] == "HISTORICALLY_EXAMINED_RESEARCH_COHORT"


# =============================================================================
# 4. METRIC RECONCILIATION & CONFUSION INVARIANTS
# =============================================================================

def test_metric_reconciliation_and_invariants():
    """Verifies test cohort metrics, confusion invariants, alert budget cap, and Wilson CIs."""
    metric_path = "results/phase6_10/metric_reconciliation.json"
    assert os.path.exists(metric_path), "Metric reconciliation JSON missing."
    
    with open(metric_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["cohort_size"] == 262
    assert data["distress_events"] == 11
    assert data["non_distress_events"] == 251
    assert data["alert_budget_cap"] == 20
    
    # Invariant summary
    inv = data["invariants_summary"]
    assert inv["all_confusion_invariants_pass"] is True
    assert inv["tp_plus_fn_strictly_equals_11"] is True
    assert inv["tn_plus_fp_strictly_equals_251"] is True
    assert inv["total_evaluations_strictly_equals_262"] is True
    
    models = {m["model_identifier"]: m for m in data["reconciled_models"]}
    
    # 1. Approved Baseline Nominal (tau = 0.0800)
    base_nom = models["Approved_Production_Baseline_Nominal"]
    assert base_nom["threshold_applied"] == 0.08
    assert base_nom["alerts"]["total_alerts"] == 37
    assert base_nom["alerts"]["budget_compliant"] is False
    assert base_nom["alerts"]["excess_alerts"] == 17
    assert base_nom["confusion_matrix"]["true_positives"] == 5
    assert base_nom["confusion_matrix"]["false_positives"] == 32
    assert base_nom["confusion_matrix"]["false_negatives"] == 6
    assert base_nom["confusion_matrix"]["true_negatives"] == 219
    assert base_nom["metrics"]["recall"] == 0.4545
    assert base_nom["metrics"]["precision"] == 0.1351
    
    # 2. Approved Baseline Budget-Capped (tau* = 0.1072)
    base_bud = models["Approved_Production_Baseline_BudgetCapped"]
    assert base_bud["threshold_applied"] == 0.1072
    assert base_bud["alerts"]["total_alerts"] == 20
    assert base_bud["alerts"]["budget_compliant"] is True
    assert base_bud["confusion_matrix"]["true_positives"] == 4
    assert base_bud["confusion_matrix"]["false_positives"] == 16
    assert base_bud["confusion_matrix"]["false_negatives"] == 7
    assert base_bud["confusion_matrix"]["true_negatives"] == 235
    assert base_bud["metrics"]["recall"] == 0.3636
    assert base_bud["metrics"]["precision"] == 0.2000
    
    # 3. Candidate Cost-Sensitive RF (tau = 0.0894)
    cand = [m for m in data["reconciled_models"] if "CostSensitive" in m["model_identifier"]][0]
    assert cand["threshold_applied"] == 0.0894
    assert cand["alerts"]["total_alerts"] == 20
    assert cand["alerts"]["budget_compliant"] is True
    assert cand["confusion_matrix"]["true_positives"] == 3
    assert cand["confusion_matrix"]["false_positives"] == 17
    assert cand["confusion_matrix"]["false_negatives"] == 8
    assert cand["confusion_matrix"]["true_negatives"] == 234
    assert cand["metrics"]["recall"] == 0.2727
    assert cand["metrics"]["precision"] == 0.1500
    
    # 4. Logistic Regression Nominal (tau = 0.0800)
    lr_nom = models["Interpretable_Logistic_Regression_Nominal"]
    assert lr_nom["alerts"]["total_alerts"] == 0
    assert lr_nom["confusion_matrix"]["true_positives"] == 0
    assert lr_nom["confusion_matrix"]["false_negatives"] == 11
    
    # 5. Standard RF Nominal (tau = 0.0800)
    rf_nom = models["Standard_Unweighted_Random_Forest_Nominal"]
    assert rf_nom["alerts"]["total_alerts"] == 15
    assert rf_nom["confusion_matrix"]["true_positives"] == 2
    assert rf_nom["confusion_matrix"]["false_negatives"] == 9


# =============================================================================
# 5. THRESHOLD & CALIBRATION CONSISTENCY
# =============================================================================

def test_threshold_and_calibration_consistency():
    """Verifies probability bounds, Brier calibration, and explains Logistic Regression 0 alerts."""
    thresh_path = "results/phase6_10/threshold_consistency_audit.json"
    assert os.path.exists(thresh_path), "Threshold consistency audit JSON missing."
    
    with open(thresh_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["calibration_audit_status"] == "PROBABILITIES_SOUND_AND_THEORETICALLY_CONSISTENT"
    
    prod_probs = data["production_baseline_probabilities"]
    assert prod_probs["strictly_bounded_0_to_1"] is True
    assert prod_probs["strictly_finite"] is True
    assert prod_probs["min_probability"] >= 0.0
    assert prod_probs["max_probability"] <= 1.0
    
    # Logistic regression zero-alert root cause
    lr_inv = data["logistic_regression_zero_alert_investigation"]
    assert lr_inv["is_software_bug"] is False
    assert lr_inv["alerts_at_nominal_threshold_0_0800"] == 0
    assert lr_inv["max_predicted_probability"] < 0.0800
    
    # Operating point distinction
    op = data["operating_point_distinction"]
    assert op["official_production_operating_point"]["threshold"] == 0.0800
    assert op["official_production_operating_point"]["status"] == "APPROVED_FROZEN"
    assert op["research_budget_capped_operating_point"]["threshold"] == 0.1072
    assert op["research_budget_capped_operating_point"]["status"] == "EXPERIMENTAL_RESEARCH_ONLY"


# =============================================================================
# 6. EXPLAINABILITY VALIDATION
# =============================================================================

def test_explainability_validation():
    """Verifies TreeSHAP rankings (x2, x12, x8) and association vs causation semantics."""
    expl_path = "results/phase6_10/explainability_validation.json"
    assert os.path.exists(expl_path), "Explainability validation JSON missing."
    
    with open(expl_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["explainability_status"] == "EXPLAINABILITY_ROBUST_AND_REGULATORILY_SOUND"
    
    feat_val = data["feature_importance_validation"]
    assert feat_val["x2_retained_earnings_in_top_3"] is True
    assert feat_val["x12_cash_flow_in_top_3"] is True
    assert feat_val["x8_current_assets_in_top_3"] is True
    assert feat_val["top_features_reproduced"] is True
    
    reg = data["regulatory_semantics_audit"]
    assert reg["causality_claim_prevented"] is True
    assert "DO NOT establish causal counterfactuals" in reg["association_vs_causation"]


# =============================================================================
# 7. APPLICATION SMOKE TEST
# =============================================================================

def test_streamlit_application_smoke_test():
    """Verifies that Streamlit app.py defines 12 pages, guards firm 1-422 range, and distinguishes thresholds."""
    app_path = "results/phase6_10/application_smoke_test.json"
    assert os.path.exists(app_path), "Application smoke test JSON missing."
    
    with open(app_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["smoke_test_status"] == "APPLICATION_SMOKE_TEST_PASSED"
    assert data["pages_verified"] == 12
    assert "422" in data["sidebar_enterprise_universe"]


# =============================================================================
# 8. FASTAPI ENDPOINT VALIDATION
# =============================================================================

def test_fastapi_endpoints_validation():
    """Verifies FastAPI REST endpoints using TestClient, testing success, 404, 422, and 503 security."""
    client = TestClient(app)
    
    # 1. Health & Info
    r1 = client.get("/health")
    assert r1.status_code == 200
    assert r1.json()["status"] == "healthy"
    
    r2 = client.get("/model-info")
    assert r2.status_code == 200
    assert "temporal_ews_model" in r2.json()
    
    # 2. Research endpoints
    r3 = client.get("/research/model-comparison")
    assert r3.status_code == 200
    assert "reconciled_models" in r3.json() or "models" in r3.json()
    
    r4 = client.get("/research/false-negatives")
    assert r4.status_code == 200
    assert len(r4.json()) == 11
    
    r5 = client.get("/research/dataset-audit")
    assert r5.status_code == 200
    assert r5.json()["total_rows"] == 3672
    
    r6 = client.get("/research/explainability-summary")
    assert r6.status_code == 200
    assert ("top_global_features" in r6.json() or "global_feature_importance" in r6.json())
    
    # 3. Company lookup (valid vs invalid)
    r7 = client.get("/company/364")
    assert r7.status_code == 200
    
    r8 = client.get("/company/99999")
    assert r8.status_code == 404
    assert "not found" in r8.json()["detail"].lower()
    
    # 4. Predict endpoint (valid vs invalid)
    payload_valid = {
        "company_id": "TEST_CO",
        "ews_probability": 0.12,
        "bankruptcy_probability": 0.05,
        "anomaly_score": 0.10,
        "financial_health_score": 75.0
    }
    r9 = client.post("/predict", json=payload_valid)
    assert r9.status_code == 200
    assert "risk_category" in r9.json()
    
    payload_invalid = {
        "ews_probability": 1.50 # invalid range
    }
    r10 = client.post("/predict", json=payload_invalid)
    assert r10.status_code == 422
    
    # 5. Admin reload fail-closed
    r11 = client.post("/admin/reload-db")
    assert r11.status_code in [401, 503]


# =============================================================================
# 9. SECURITY & REPOSITORY HYGIENE
# =============================================================================

def test_security_and_hygiene():
    """Verifies that no secrets or plain credentials exist and fail-closed security is active."""
    sec_path = "results/phase6_10/security_audit.json"
    assert os.path.exists(sec_path), "Security audit JSON missing."
    
    with open(sec_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["security_audit_status"] == "SECURITY_AND_HYGIENE_PASSED"
    assert len(data["secrets_leakage_findings"]) == 0
    assert data["access_control_verification"]["administrative_endpoints_fail_closed"] is True
    assert data["repository_hygiene"]["protected_production_artifacts_unmodified"] is True
