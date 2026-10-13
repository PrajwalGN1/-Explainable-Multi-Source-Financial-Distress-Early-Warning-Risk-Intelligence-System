"""
Tests for Phase 6.9: Existing-Data Model Improvement and Project Completion.

Verifies:
1. Production baseline immutability (cryptographic hash invariance).
2. Dataset audit integrity and target semantics (FD <= -0.50 is distress).
3. Leakage-free chronological splitting (Train: T <= 10, Val: T in [11, 12], Test: T = 13).
4. Feature pipeline and zero-leakage guarantees.
5. Model benchmarking under 8% alert constraint (Cap = 20 alerts on test).
6. False-negative diagnostic report completeness (all 11 test distress cases audited).
7. TreeSHAP explainability synthesis and association vs causation disclaimer.
8. API endpoints for Phase 6.9 research deliverables and error handling for missing entities.
"""

import os
import json
import hashlib
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from api import app
from src.features.temporal_features import TemporalFeatureEngine
from src.preprocessing.preprocessors import EWSPreprocessor


# =============================================================================
# 1. PRODUCTION BASELINE IMMUTABILITY
# =============================================================================

def test_production_baseline_immutability():
    """Verifies that all 8 protected production artifacts are unaltered."""
    manifest_path = "results/phase6_9/production_integrity_manifest.json"
    assert os.path.exists(manifest_path), "Production integrity manifest missing."
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    assert manifest.get("baseline_verified") is True
    assert manifest.get("immutability_status") == "PROTECTED_BASELINE_VERIFIED_UNCHANGED"
    
    # Verify current file hashes on disk match expected
    expected = manifest.get("expected_hashes", {})
    for path, exp_hash in expected.items():
        assert os.path.exists(path), f"Protected artifact missing: {path}"
        with open(path, "rb") as f:
            curr_hash = hashlib.sha256(f.read()).hexdigest()
        assert curr_hash == exp_hash, f"Hash mismatch for {path}: {curr_hash} != {exp_hash}"


# =============================================================================
# 2. DATASET AUDIT & TARGET DIRECTION
# =============================================================================

def test_dataset_audit_completeness():
    """Verifies empirical properties of raw CSV and target semantics."""
    audit_path = "results/phase6_9/dataset_audit.json"
    assert os.path.exists(audit_path), "Dataset audit JSON missing."
    
    with open(audit_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["total_rows"] == 3672
    assert data["total_columns"] == 86
    assert data["unique_companies"] == 422
    assert data["missing_values_count"] == 0
    assert data["duplicate_company_period_records"] == 0
    assert data["time_contiguity"]["has_time_gaps"] is False
    assert data["time_contiguity"]["total_time_gaps"] == 0
    
    # Target semantics: continuous index where <= -0.5 is distress
    tgt = data["target_semantics"]
    assert tgt["statutory_threshold"] == -0.50
    assert tgt["raw_distress_records"] == 136
    assert tgt["raw_healthy_records"] == 3536
    assert "LOWER values indicate greater financial distress" in tgt["direction_interpretation"]
    
    # Terminal exit dynamics
    exit_dyn = data["panel_attrition_and_terminal_dynamics"]
    assert exit_dyn["unique_distressed_firms"] == 136
    assert exit_dyn["distress_at_exact_final_observed_period"] == 136
    assert exit_dyn["terminal_exit_rate_pct"] == 100.0


# =============================================================================
# 3. LEAKAGE-FREE CHRONOLOGICAL SPLITS & FEATURE AUDIT
# =============================================================================

def test_chronological_splits_and_leakage_safety():
    """Verifies chronological partition counts and leakage prevention."""
    audit_path = "results/phase6_9/dataset_audit.json"
    with open(audit_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    parts = data["chronological_partitions"]
    assert parts["training_set"]["sample_count"] == 2471
    assert parts["training_set"]["distress_events"] == 95
    assert parts["validation_set"]["sample_count"] == 517
    assert parts["validation_set"]["distress_events"] == 14
    assert parts["test_holdout_cohort"]["sample_count"] == 262
    assert parts["test_holdout_cohort"]["distress_events"] == 11


def test_feature_audit_and_zero_leakage():
    """Verifies feature pipeline properties and correlation bounds."""
    feat_path = "results/phase6_9/feature_audit.json"
    assert os.path.exists(feat_path), "Feature audit JSON missing."
    
    with open(feat_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["total_model_features"] == 163
    assert data["raw_financial_ratios_count"] == 83
    assert data["temporal_engineered_features_count"] == 80
    assert data["core_ratio_temporal_derivations"]["derivations_per_ratio"] == 10
    
    leakage = data["leakage_verification"]
    assert leakage["temporal_features_leakage_status"] == "VERIFIED_LEAK_FREE"
    assert leakage["max_feature_correlation_with_forward_target"] < 0.25
    assert leakage["max_feature_correlation_with_continuous_fd"] < 0.50


# =============================================================================
# 4. MODEL COMPARISON & ALERT BUDGET ARITHMETIC
# =============================================================================

def test_model_comparison_and_budget_constraints():
    """Verifies empirical benchmark results and 8% alert budget compliance."""
    comp_path = "results/phase6_9/model_comparison.json"
    assert os.path.exists(comp_path), "Model comparison JSON missing."
    
    with open(comp_path, "r", encoding="utf-8") as f:
        comp = json.load(f)
        
    assert comp["cohort_size"] == 262
    assert comp["distress_events"] == 11
    assert comp["integer_8pct_alert_cap"] == 20
    
    models = {m["model_identifier"]: m for m in comp["models"]}
    
    # 1. Approved Baseline Nominal
    prod_nom = models["Approved_Production_Baseline_Nominal"]
    assert prod_nom["threshold_applied"] == 0.0800
    assert prod_nom["confusion_matrix"]["true_positives"] == 5
    assert prod_nom["confusion_matrix"]["false_positives"] == 32
    assert prod_nom["alerts"]["total_alerts"] == 37
    assert prod_nom["alerts"]["budget_compliant"] is False
    assert prod_nom["alerts"]["excess_alerts"] == 17
    
    # 2. Approved Baseline Budget-Capped
    prod_cap = models["Approved_Production_Baseline_BudgetCapped"]
    assert prod_cap["alerts"]["total_alerts"] == 20
    assert prod_cap["alerts"]["budget_compliant"] is True
    assert prod_cap["confusion_matrix"]["true_positives"] == 4
    assert prod_cap["confusion_matrix"]["false_positives"] == 16
    assert prod_cap["metrics"]["recall"] == 0.3636
    assert prod_cap["metrics"]["precision"] == 0.2000
    
    # 3. Candidate Nominal
    cand_nom = models["Shadow_Candidate_RF_CostSensitive_Nominal"]
    assert cand_nom["threshold_applied"] == 0.0894
    assert cand_nom["confusion_matrix"]["true_positives"] == 3
    assert cand_nom["confusion_matrix"]["false_positives"] == 17
    assert cand_nom["alerts"]["total_alerts"] == 20
    assert cand_nom["alerts"]["budget_compliant"] is True
    assert cand_nom["metrics"]["recall"] == 0.2727
    assert cand_nom["metrics"]["precision"] == 0.1500
    
    # 4. Logistic Regression Budget-Capped
    lr_cap = models["Interpretable_Logistic_Regression_BudgetCapped"]
    assert lr_cap["alerts"]["total_alerts"] == 20
    assert lr_cap["alerts"]["budget_compliant"] is True
    assert lr_cap["confusion_matrix"]["true_positives"] == 3
    assert lr_cap["metrics"]["recall"] == 0.2727


# =============================================================================
# 5. FALSE NEGATIVE DIAGNOSTIC AUDIT
# =============================================================================

def test_false_negative_audit_completeness():
    """Verifies that all 11 distress firms are comprehensively diagnosed."""
    fn_path = "results/phase6_9/false_negative_audit.csv"
    assert os.path.exists(fn_path), "False negative audit CSV missing."
    
    df_fn = pd.read_csv(fn_path)
    assert len(df_fn) == 11, f"Expected 11 distress firms, got {len(df_fn)}"
    
    # Verify expected columns
    expected_cols = [
        "company_id", "prediction_period", "target_period", "actual_target",
        "antecedent_fd_t13", "realized_fd_t14", "transition_dynamic",
        "prod_baseline_prob", "prod_alert_nominal_tau0800", "prod_alert_budget_capped",
        "candidate_prob", "candidate_alert_nominal_tau0894", "candidate_alert_budget_capped",
        "failure_taxonomy", "model_discrepancy_explanation",
        "x1_working_capital", "x2_retained_earnings", "x5_cash_ratio",
        "x10_ebit_margin", "x25_liquidity", "x48_total_debt", "x64_leverage", "x81_operating_cashflow"
    ]
    for col in expected_cols:
        assert col in df_fn.columns, f"Missing column in false negative audit: {col}"
        
    # Check that known acute crash cases are present
    acute_firms = df_fn[df_fn["transition_dynamic"] == "SUDDEN_CRASH"]["company_id"].tolist()
    assert set(acute_firms) == {28, 60, 84, 225, 376}
    
    # Verify Firm 28 and Firm 60 detection by baseline
    f28 = df_fn[df_fn["company_id"] == 28].iloc[0]
    assert bool(f28["prod_alert_nominal_tau0800"]) is True
    assert bool(f28["candidate_alert_nominal_tau0894"]) is False


# =============================================================================
# 6. EXPLAINABILITY SUMMARY & REGULATORY DISCLAIMER
# =============================================================================

def test_explainability_summary_and_causal_distinction():
    """Verifies TreeSHAP summary features and supervisory causal distinction."""
    expl_path = "results/phase6_9/explainability_summary.json"
    assert os.path.exists(expl_path), "Explainability summary JSON missing."
    
    with open(expl_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert "TreeSHAP" in data["methodology"]
    assert len(data["top_global_features"]) == 10
    
    # Check rank 1 feature
    r1 = data["top_global_features"][0]
    assert r1["rank"] == 1
    assert "x2" in r1["feature"]
    
    # Verify mandatory causal disclaimer text
    disclaimer = data["causal_vs_predictive_governance_disclaimer"]["critical_distinction"]
    assert "STATISTICAL PREDICTIVE ASSOCIATIONS" in disclaimer
    assert "NOT CAUSAL MECHANISMS" in disclaimer


# =============================================================================
# 7. FASTAPI RESEARCH ENDPOINTS & ERROR HANDLING
# =============================================================================

def test_api_research_endpoints_and_error_handling():
    """Tests FastAPI client endpoints for Phase 6.9 research and input validation."""
    client = TestClient(app)
    
    # 1. Model comparison endpoint
    resp_comp = client.get("/research/model-comparison")
    assert resp_comp.status_code == 200
    comp_json = resp_comp.json()
    assert comp_json["cohort_size"] == 262
    
    # 2. False negatives endpoint
    resp_fn = client.get("/research/false-negatives")
    assert resp_fn.status_code == 200
    fn_list = resp_fn.json()
    assert len(fn_list) == 11
    
    # 3. Dataset audit endpoint
    resp_data = client.get("/research/dataset-audit")
    assert resp_data.status_code == 200
    assert resp_data.json()["total_rows"] == 3672
    
    # 4. Explainability endpoint
    resp_expl = client.get("/research/explainability-summary")
    assert resp_expl.status_code == 200
    assert len(resp_expl.json()["top_global_features"]) == 10
    
    # 5. Invalid company ID produces clear 404 error rather than fabricated prediction
    resp_invalid = client.get("/company/INVALID_9999")
    assert resp_invalid.status_code == 404
    assert "not found in database" in resp_invalid.json()["detail"].lower()
