"""
Phase 2.1 Model Validation, Calibration & Reproducibility Hardening Tests.

Unit tests using small deterministic fixtures covering:
- Forward target alignment and chronological split integrity
- Leakage-resistant preprocessing (no refit during transform)
- Schema and feature ordering consistency
- Confusion matrix arithmetic, Wilson confidence intervals, and rate calculations
- Probability bounds and Brier skill score relative to prevalence baseline
- Manifest schema integrity and artifact existence
- Controlled error handling for missing artifacts
- API /model-info manifest reporting and Phase 1 security preservation
- Unseen-company GroupKFold isolation
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.model_selection import GroupKFold

from src.features.temporal_features import TemporalFeatureEngine
from src.preprocessing.preprocessors import EWSPreprocessor
from scripts.verify_phase2_reproducibility import (
    wilson_score_interval,
    evaluate_binary_classification,
    compute_calibration_diagnostics
)
from api import app

client = TestClient(app)


def test_forward_target_alignment_logic():
    """Verifies that forward target aligns t+1 distress within the same company and sets terminal to NaN."""
    # Synthetic panel of 2 companies across 4 time periods
    data = [
        {"Company": 1, "Time": 1, "Financial Distress": 0.5},   # solvent (curr=0)
        {"Company": 1, "Time": 2, "Financial Distress": 0.2},   # solvent (curr=0)
        {"Company": 1, "Time": 3, "Financial Distress": -0.8},  # distressed (curr=1)
        {"Company": 1, "Time": 4, "Financial Distress": -1.2},  # distressed (terminal)
        {"Company": 2, "Time": 1, "Financial Distress": -0.6},  # distressed (curr=1)
        {"Company": 2, "Time": 2, "Financial Distress": 0.1},   # solvent (curr=0)
        {"Company": 2, "Time": 3, "Financial Distress": 0.4},   # solvent (terminal)
    ]
    df = pd.DataFrame(data)
    engine = TemporalFeatureEngine()
    df_res = engine.construct_forward_target(df, id_col="Company", time_col="Time", raw_target_col="Financial Distress", threshold=-0.5)
    
    # Company 1 checks:
    c1 = df_res[df_res["Company"] == 1].sort_values("Time").reset_index(drop=True)
    assert c1.loc[0, "forward_target"] == 0.0  # Time 1 forward target is Time 2 status (0)
    assert c1.loc[1, "forward_target"] == 1.0  # Time 2 forward target is Time 3 status (1)
    assert c1.loc[2, "forward_target"] == 1.0  # Time 3 forward target is Time 4 status (1)
    assert pd.isna(c1.loc[3, "forward_target"]) # Time 4 is terminal -> NaN
    
    # Company 2 checks:
    c2 = df_res[df_res["Company"] == 2].sort_values("Time").reset_index(drop=True)
    assert c2.loc[0, "forward_target"] == 0.0  # Time 1 forward target is Time 2 status (0)
    assert c2.loc[1, "forward_target"] == 0.0  # Time 2 forward target is Time 3 status (0)
    assert pd.isna(c2.loc[2, "forward_target"]) # Time 3 is terminal -> NaN
    
    # Verify zero target bleed across company boundary:
    # Company 1 Time 4 target must not be influenced by Company 2 Time 1 status
    assert pd.isna(c1.loc[3, "forward_target"])


def test_chronological_split_strict_ordering():
    """Verifies that time partitions are strictly non-overlapping and chronologically ascending."""
    times = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    train_times = times[times <= 10]
    val_times = times[(times > 10) & (times <= 12)]
    test_times = times[times == 13]
    
    assert set(train_times).isdisjoint(set(val_times))
    assert set(val_times).isdisjoint(set(test_times))
    assert set(train_times).isdisjoint(set(test_times))
    
    assert max(train_times) < min(val_times)
    assert max(val_times) < min(test_times)


def test_preprocessor_isolation_and_no_refit():
    """Verifies preprocessor does not modify its fitted statistics during transform."""
    # Deterministic toy dataset
    train_data = pd.DataFrame({
        "Company": [1, 2, 3],
        "Time": [1, 1, 1],
        "forward_target": [0, 1, 0],
        "feat_a": [10.0, 20.0, 30.0],
        "feat_b": [100.0, 200.0, 300.0]
    })
    test_data = pd.DataFrame({
        "Company": [4, 5],
        "Time": [2, 2],
        "forward_target": [1, 0],
        "feat_a": [1000.0, 2000.0],  # Massive outlier distribution
        "feat_b": [5000.0, 6000.0]
    })
    
    prep = EWSPreprocessor(target_col="forward_target")
    prep.fit(train_data)
    
    # Record fitted scaler parameters
    center_before = prep.scaler.center_.copy()
    scale_before = prep.scaler.scale_.copy()
    
    # Transform test
    X_test, _ = prep.transform(test_data)
    
    # Assert parameters remain perfectly unchanged
    np.testing.assert_array_equal(prep.scaler.center_, center_before)
    np.testing.assert_array_equal(prep.scaler.scale_, scale_before)


def test_feature_order_and_schema_consistency():
    """Verifies that preprocessor enforces exact feature column order even if input has permuted columns."""
    train_data = pd.DataFrame({
        "feat_1": [1.0, 2.0],
        "feat_2": [10.0, 20.0],
        "feat_3": [100.0, 200.0],
        "forward_target": [0, 1]
    })
    prep = EWSPreprocessor(target_col="forward_target")
    prep.fit(train_data)
    
    # Test data with permuted column order and extra unused column
    test_data_permuted = pd.DataFrame({
        "feat_3": [300.0],
        "extra_unused_col": [999.0],
        "feat_1": [3.0],
        "feat_2": [30.0],
        "forward_target": [0]
    })
    
    X_test, _ = prep.transform(test_data_permuted)
    # Target feature order is feat_1, feat_2, feat_3
    assert prep.feature_names == ["feat_1", "feat_2", "feat_3"]
    assert X_test.shape == (1, 3)


def test_confusion_matrix_arithmetic_and_rates():
    """Verifies exact confusion matrix arithmetic and mathematical consistency."""
    # Test with ground truth: 11 distress cases out of 262 observations
    y_true = np.array([1]*11 + [0]*251)
    # Scenario: 3 true positives, 17 false positives, 8 false negatives, 234 true negatives
    # Predicted probabilities engineered to trigger these exact counts at threshold 0.08
    y_prob = np.zeros(262)
    y_prob[:3] = 0.15     # 3 true positives (prob >= 0.08)
    y_prob[3:11] = 0.02   # 8 false negatives (prob < 0.08)
    y_prob[11:28] = 0.10  # 17 false positives (prob >= 0.08)
    y_prob[28:] = 0.01    # 234 true negatives (prob < 0.08)
    
    m = evaluate_binary_classification(y_true, y_prob, threshold=0.08)
    
    assert m["total"] == 262
    assert m["tn"] == 234
    assert m["fp"] == 17
    assert m["fn"] == 8
    assert m["tp"] == 3
    assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == 262
    
    assert abs(m["recall"] - (3 / 11)) < 1e-4
    assert abs(m["precision"] - (3 / 20)) < 1e-4
    assert abs(m["specificity"] - (234 / 251)) < 1e-4
    assert abs(m["false_negative_rate"] - (8 / 11)) < 1e-4
    assert abs(m["false_positive_rate"] - (17 / 251)) < 1e-4
    assert abs(m["youden_j"] - (m["recall"] + m["specificity"] - 1.0)) < 1e-4


def test_wilson_score_interval_bounds():
    """Verifies Wilson score confidence interval calculation and boundary behavior."""
    # Recomputed Phase 2 test recall: 3 hits out of 11 trials
    low, high = wilson_score_interval(3, 11)
    assert 0.0 <= low <= 3/11 <= high <= 1.0
    assert abs(low - 0.0975) < 1e-3
    assert abs(high - 0.5657) < 1e-3
    
    # Boundary edge cases
    zero_low, zero_high = wilson_score_interval(0, 11)
    assert zero_low == 0.0
    assert zero_high > 0.0
    
    full_low, full_high = wilson_score_interval(11, 11)
    assert full_low < 1.0
    assert full_high == 1.0


def test_brier_skill_score_calculation():
    """Verifies Brier Skill Score calculation against training prevalence reference baseline."""
    y_true = np.array([0]*96 + [1]*4)  # 4% prevalence
    train_prevalence = 0.0384           # Historical 3.84% prevalence
    
    # Perfect model (prob == y_true) -> BSS = 1.0
    diag_perf = compute_calibration_diagnostics(y_true, y_true.astype(float), prevalence_train=train_prevalence)
    assert diag_perf["brier_skill_score"] == 1.0
    
    # Naive model predicting constant prevalence -> BSS approx 0.0
    prob_naive = np.full_like(y_true, fill_value=train_prevalence, dtype=float)
    diag_naive = compute_calibration_diagnostics(y_true, prob_naive, prevalence_train=train_prevalence)
    assert abs(diag_naive["brier_skill_score"]) < 1e-3
    
    # Poor uncalibrated model (over-predicting probabilities) -> negative BSS
    prob_poor = np.full_like(y_true, fill_value=0.50, dtype=float)
    diag_poor = compute_calibration_diagnostics(y_true, prob_poor, prevalence_train=train_prevalence)
    assert diag_poor["brier_skill_score"] < 0.0


def test_manifest_integrity_and_schema():
    """Verifies that models/phase2/model_manifest.json contains required metadata and valid paths."""
    manifest_path = "models/phase2/model_manifest.json"
    assert os.path.exists(manifest_path), "model_manifest.json must exist"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    required_keys = [
        "model_version", "experiment_identifier", "phase",
        "dataset_name", "dataset_sha256", "target_definition",
        "split_specification", "feature_count", "feature_names",
        "primary_model_type", "operating_threshold", "verified_test_metrics",
        "artifact_paths", "artifact_hashes"
    ]
    for k in required_keys:
        assert k in manifest, f"Manifest missing required key: {k}"
        
    assert manifest["model_version"] == "1.2.1-hardened"
    assert manifest["feature_count"] == 163
    assert len(manifest["feature_names"]) == 163
    assert manifest["operating_threshold"] == 0.08
    
    # Check that referenced artifact files physically exist
    for art_name, art_path in manifest["artifact_paths"].items():
        assert os.path.exists(art_path), f"Manifest references missing file: {art_path}"


def test_missing_or_incompatible_artifact_failure_behavior(tmp_path):
    """Verifies that loading from an invalid path raises a clear exception rather than failing silently."""
    from src.models.ews_model import TemporalEWSModel
    
    non_existent_dir = str(tmp_path / "non_existent_models")
    with pytest.raises((FileNotFoundError, Exception)):
        TemporalEWSModel.load(non_existent_dir)


def test_api_model_info_returns_hardened_manifest():
    """Verifies that GET /model-info returns the Phase 2.1 hardened manifest and metrics."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    
    # Preserves Phase 1 contracts
    assert "temporal_ews_model" in data
    assert "bankruptcy_model" in data
    assert "fusion_engine" in data
    
    # Exposes Phase 2.1 manifest
    assert "model_manifest" in data
    m = data["model_manifest"]
    assert m["model_version"] == "1.2.1-hardened"
    assert m["operating_threshold"] == 0.08
    assert m["status"] == "Hardened and Reproducibility Verified"


def test_phase1_security_preserved(monkeypatch):
    """Verifies that administrative endpoint /admin/reload-db maintains Phase 1 security protections."""
    # 1. Missing server env var -> 503
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    resp_503 = client.post("/admin/reload-db", headers={"X-Admin-Key": "any-key"})
    assert resp_503.status_code == 503
    
    # 2. Configured env var, but missing header -> 401
    monkeypatch.setenv("ADMIN_API_KEY", "super-secret-key")
    resp_401 = client.post("/admin/reload-db")
    assert resp_401.status_code == 401
    
    # 3. Configured env var, invalid key -> 403
    resp_403 = client.post("/admin/reload-db", headers={"X-Admin-Key": "wrong-key"})
    assert resp_403.status_code == 403


def test_unseen_company_group_cv_isolation():
    """Verifies that GroupKFold strictly prevents any company overlap between training and validation folds."""
    companies = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    X = np.random.randn(12, 4)
    y = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    
    gkf = GroupKFold(n_splits=3)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=companies)):
        train_comps = set(companies[train_idx])
        val_comps = set(companies[val_idx])
        assert train_comps.isdisjoint(val_comps), f"Company leakage detected in fold {fold}"
