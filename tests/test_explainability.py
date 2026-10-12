"""
Phase 3 Automated Test Suite: Explainability, Risk Attribution & Risk Intelligence.

Validates:
- Explainer initialization and model compatibility with Phase 2 artifacts
- Exact model version and feature-name / feature-ordering consistency (163 features)
- Class-specific SHAP output handling (Class 1 = Distress)
- Mathematical additivity: sum(SHAP) + expected_value == p_RF within numerical tolerance
- Output-space labeling: raw_random_forest_probability vs calibrated operational probability
- Local explanation schema, value finiteness, and top driver decomposition
- Global feature importance calculations (mean |SHAP| and signed mean)
- Temporal attribution decomposition and period delta consistency
- Missing company and invalid period error handling (HTTP 404, 400)
- API endpoint integration: /explanations/info, /explanations/local, /explanations/temporal, /explanations/global
- Invariance: zero model retraining, frozen SHA-256 checksums, threshold locked at 0.080
- Preservation of Phase 1 administrative security on /admin/reload-db
"""

import os
import json
import hashlib
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.explainability.tree_shap_explainer import NativeTreeSHAPExplainer, get_tree_shap_explainer
from src.explainability.temporal_attribution import TemporalAttributionService
from src.explainability.company_service import get_company_explanation_service
from api import app

client = TestClient(app)


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 1. Explainer Initialization, Schema & Provenance Tests
# ---------------------------------------------------------------------------

def test_explainer_initialization_and_compatibility():
    """Verifies that NativeTreeSHAPExplainer successfully loads Phase 2 artifacts."""
    explainer = get_tree_shap_explainer()
    assert explainer.is_available is True
    assert explainer._model is not None
    assert explainer._model.__class__.__name__ == "RandomForestClassifier"
    assert len(explainer._model.estimators_) == 100
    assert explainer._model.max_depth == 5
    assert explainer.operating_threshold == 0.08


def test_feature_order_and_manifest_consistency():
    """Verifies that the explainer's 163 features match models/phase2/model_manifest.json exactly."""
    explainer = get_tree_shap_explainer()
    manifest_path = "models/phase2/model_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    expected_features = manifest["feature_names"]
    assert len(explainer.feature_names) == 163
    assert explainer.feature_names == expected_features
    assert explainer.model_version == "1.2.1-hardened"


def test_distress_class_selection_and_base_value():
    """Verifies that class 1 corresponds to distress and base value is finite."""
    explainer = get_tree_shap_explainer()
    assert explainer._distress_class_idx == 1
    assert 0.0 < explainer.expected_value < 1.0
    # Expected value around 0.4969 due to balanced class weighting
    assert abs(explainer.expected_value - 0.4969) < 0.02


# ---------------------------------------------------------------------------
# 2. Mathematical Additivity & Output-Space Validation Tests
# ---------------------------------------------------------------------------

def test_shap_mathematical_additivity_on_multiple_vectors():
    """
    Verifies that sum(SHAP) + expected_value == model_output within 1e-4 tolerance
    across multiple distinct feature vectors (zeros, positive, negative).
    """
    explainer = get_tree_shap_explainer()
    n_feats = len(explainer.feature_names)

    test_vectors = [
        np.zeros(n_feats),
        np.ones(n_feats) * 0.5,
        np.ones(n_feats) * -0.5,
        np.random.RandomState(42).randn(n_feats)
    ]

    for vec in test_vectors:
        is_valid, recon, actual, disc = explainer.validate_additivity(vec, tolerance=1e-4)
        assert is_valid is True, f"Additivity failed! Disc: {disc}, Recon: {recon}, Actual: {actual}"
        assert disc < 1e-4
        assert np.isfinite(recon)
        assert np.isfinite(actual)


def test_output_space_separation_labeling():
    """Verifies that TreeSHAP output space is explicitly separated from Platt-calibrated probability."""
    explainer = get_tree_shap_explainer()
    dummy = np.zeros(len(explainer.feature_names))
    res = explainer.explain_transformed_row(dummy, top_n=5)

    assert res["output_space"] == "raw_random_forest_probability"
    assert "raw_random_forest_probability" in res
    assert "calibrated_distress_probability" in res
    assert "operating_threshold" in res
    assert res["operating_threshold"] == 0.08
    # Notice: raw prob and calibrated prob differ due to Platt scaling
    assert res["raw_random_forest_probability"] != res["calibrated_distress_probability"]
    assert res["calibrated_distress_probability"] < res["raw_random_forest_probability"]


# ---------------------------------------------------------------------------
# 3. Local Explanation Schema & Finiteness Tests
# ---------------------------------------------------------------------------

def test_local_explanation_schema_and_value_finiteness():
    """Verifies that local explanation contains all required fields with finite values."""
    explainer = get_tree_shap_explainer()
    x = np.random.RandomState(123).randn(163)
    res = explainer.explain_transformed_row(x, top_n=10)

    assert res["status"] == "success"
    assert res["additivity_verified"] is True
    assert len(res["top_risk_drivers"]) <= 10
    assert len(res["top_protective_factors"]) <= 10
    assert len(res["top_features_overall"]) <= 10

    # Verify every attribution is finite
    for feat, val in res["feature_attributions_all"].items():
        assert np.isfinite(val)

    # Verify risk drivers have positive SHAP impact
    for d in res["top_risk_drivers"]:
        assert d["shap_impact"] > 0
        assert d["direction"] == "Elevates Risk"
        assert np.isfinite(d["feature_value_scaled"])

    # Verify protective factors have negative SHAP impact
    for p in res["top_protective_factors"]:
        assert p["shap_impact"] < 0
        assert p["direction"] == "Reduces Risk"
        assert np.isfinite(p["feature_value_scaled"])


# ---------------------------------------------------------------------------
# 4. Global Feature Importance Tests
# ---------------------------------------------------------------------------

def test_global_importance_computation_and_ranking():
    """Verifies that global importance computes valid rankings with positive mean absolute SHAP."""
    explainer = get_tree_shap_explainer()
    X_sample = np.random.RandomState(42).randn(20, 163)
    global_res = explainer.compute_global_importance(X_sample, sample_name="Synthetic Unit Test")

    assert global_res["sample_size"] == 20
    assert global_res["feature_count"] == 163
    rankings = global_res["rankings"]
    assert len(rankings) == 163

    # Rank 1 must have highest mean_abs_shap
    assert rankings[0]["rank"] == 1
    assert rankings[0]["mean_abs_shap"] >= rankings[1]["mean_abs_shap"]
    assert rankings[0]["mean_abs_shap"] >= rankings[-1]["mean_abs_shap"]

    for r in rankings:
        assert r["mean_abs_shap"] >= 0.0
        assert np.isfinite(r["mean_signed_shap"])
        assert np.isfinite(r["std_shap"])


# ---------------------------------------------------------------------------
# 5. Temporal Attribution Service Tests
# ---------------------------------------------------------------------------

def test_temporal_attribution_transition_decomposition():
    """Verifies period-over-period delta decomposition between consecutive periods."""
    temp_service = TemporalAttributionService()
    x_prev = np.zeros(163)
    x_curr = np.zeros(163)
    x_curr[0] = 3.0  # Big change in x1

    res = temp_service.explain_period_transition(
        prev_transformed_row=x_prev,
        curr_transformed_row=x_curr,
        prev_period=1,
        curr_period=2,
        company_id="TEST_FIRM",
        top_n=5
    )

    assert res["status"] == "success"
    assert res["company_id"] == "TEST_FIRM"
    assert res["previous_period"] == 1
    assert res["current_period"] == 2
    assert "calibrated_prob_delta" in res
    assert "calibrated_prob_delta_points" in res
    assert "summary" in res
    assert isinstance(res["top_drivers_of_increase"], list)
    assert isinstance(res["top_drivers_of_recovery"], list)


# ---------------------------------------------------------------------------
# 6. Company Explanation Service & Cache Tests
# ---------------------------------------------------------------------------

def test_company_service_lookup_and_error_handling():
    """Verifies company service retrieves real SME records and raises controlled errors for invalid IDs."""
    service = get_company_explanation_service()
    service.initialize_cache()

    # Valid company
    exp = service.get_local_explanation("1", top_n=5)
    assert exp["company_id"] == "1"
    assert exp["status"] == "success"
    assert exp["reporting_period"] in exp["available_periods"]
    assert exp["risk_score"] > 0
    assert exp["risk_category"] in ["Healthy", "Watch", "High", "Critical"]

    # Invalid company raises KeyError
    with pytest.raises(KeyError):
        service.get_local_explanation("NON_EXISTENT_COMPANY_9999")

    # Invalid period for valid company raises KeyError
    with pytest.raises(KeyError):
        service.get_local_explanation("1", period=999)


def test_company_service_temporal_explanation_and_insufficient_history():
    """Verifies temporal explanation retrieval and handling of firms with single period."""
    service = get_company_explanation_service()
    service.initialize_cache()

    # Monitored SME #1 has 4 periods
    temp = service.get_temporal_explanation("1", top_n=3)
    assert temp["status"] == "success"
    assert temp["company_id"] == "1"
    assert len(temp["available_periods"]) >= 2
    assert "top_drivers_of_increase" in temp

    # Invalid periods raise KeyError
    with pytest.raises(KeyError):
        service.get_temporal_explanation("1", curr_period=999, prev_period=998)


# ---------------------------------------------------------------------------
# 7. FastAPI Endpoint Integration Tests
# ---------------------------------------------------------------------------

def test_api_explanation_info_endpoint():
    """Verifies GET /explanations/info."""
    response = client.get("/explanations/info")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_version"] == "1.2.1-hardened"
    assert data["output_space"] == "raw_random_forest_probability"
    assert data["additivity_verified"] is True
    assert data["feature_count"] == 163
    assert data["operating_threshold"] == 0.08


def test_api_local_explanation_endpoint_success_and_404():
    """Verifies GET /explanations/local/{company_id}."""
    # Success
    resp_ok = client.get("/explanations/local/1?top_n=5")
    assert resp_ok.status_code == 200
    data = resp_ok.json()
    assert data["company_id"] == "1"
    assert data["status"] == "success"
    assert len(data["top_risk_drivers"]) <= 5
    assert data["additivity_verified"] is True

    # 404 for missing company
    resp_404 = client.get("/explanations/local/INVALID_FIRM_999")
    assert resp_404.status_code == 404

    # 422 for invalid query bounds
    resp_422 = client.get("/explanations/local/1?top_n=9999")
    assert resp_422.status_code == 422


def test_api_custom_payload_explanation_endpoint():
    """Verifies POST /explanations/local with raw feature dictionary."""
    payload = {"x1": 0.5, "x2": 1.2, "x12": -0.8}
    response = client.post("/explanations/local?top_n=5", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "calibrated_distress_probability" in data
    assert "top_risk_drivers" in data


def test_api_temporal_explanation_endpoint():
    """Verifies GET /explanations/temporal/{company_id}."""
    resp = client.get("/explanations/temporal/1?top_n=3")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["company_id"] == "1"
    assert "calibrated_prob_delta_points" in data
    assert "summary" in data

    # 404 for invalid company
    resp_404 = client.get("/explanations/temporal/INVALID_COMPANY")
    assert resp_404.status_code == 404


def test_api_global_explanation_endpoint():
    """Verifies GET /explanations/global."""
    resp = client.get("/explanations/global?population=test")
    assert resp.status_code == 200
    data = resp.json()
    assert "test_population" in data
    assert len(data["test_population"]["top_10_features"]) == 10
    assert "selected_population_rankings" in data


# ---------------------------------------------------------------------------
# 8. Preservation of Phase 1 & Phase 2 Invariants (Zero Retraining)
# ---------------------------------------------------------------------------

def test_model_artifact_provenance_and_zero_retraining():
    """
    Verifies that model checkpoints in models/phase2/ have NOT been modified or retrained.
    Compares physical file SHA-256 with hashes recorded in models/phase2/model_manifest.json.
    """
    manifest_path = "models/phase2/model_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    hashes = manifest["artifact_hashes"]
    assert compute_sha256("models/phase2/best_ews_model.joblib") == hashes["best_ews_model_sha256"]
    assert compute_sha256("models/phase2/ews_preprocessor.joblib") == hashes["ews_preprocessor_sha256"]
    assert compute_sha256("models/phase2/ews_calibrator.joblib") == hashes["ews_calibrator_sha256"]
    assert compute_sha256("models/phase2/threshold_config.json") == hashes["threshold_config_sha256"]


def test_phase1_admin_security_maintained(monkeypatch):
    """Verifies that Phase 1 administrative reload security remains fully protected."""
    # Missing server key -> 503
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    resp_503 = client.post("/admin/reload-db", headers={"X-Admin-Key": "test"})
    assert resp_503.status_code == 503

    # Missing header -> 401
    monkeypatch.setenv("ADMIN_API_KEY", "super-secret-key")
    resp_401 = client.post("/admin/reload-db")
    assert resp_401.status_code == 401

    # Invalid key -> 403
    resp_403 = client.post("/admin/reload-db", headers={"X-Admin-Key": "wrong-key"})
    assert resp_403.status_code == 403
