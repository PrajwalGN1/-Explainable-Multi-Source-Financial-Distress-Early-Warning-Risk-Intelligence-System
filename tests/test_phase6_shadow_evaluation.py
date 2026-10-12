"""
Phase 6 Automated Test Suite:
Prospective Shadow Evaluation & Operational Validation Safeguards.

Tests:
1. Baseline Artifact Immutability (Bitwise SHA-256 verification against pre-implementation manifest)
2. Production Threshold Immutability (Frozen strictly at tau* = 0.0800)
3. Shadow Candidate Failure Isolation (Candidate exception cannot disrupt production decision)
4. Non-Interference Guarantee (Production prediction identical whether candidate succeeds or fails)
5. Timestamp & Audit Provenance Integrity (ISO 8601 UTC, pseudonymous ID, target period t+1)
6. Missing Feature Robustness in Sandbox
7. Pre-Registered Outcome Maturity Policy & Uncertainty Safeguards
8. Population Stability Index (PSI) Drift Computation
9. FastAPI Shadow Endpoints (/shadow/status, /shadow/telemetry, /shadow/maturity-evaluation, /shadow/predict)
"""

import os
import json
import hashlib
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from fastapi.testclient import TestClient

from api import app
from src.shadow.shadow_engine import ShadowInferenceEngine, compute_sha256
from src.shadow.maturity_and_drift import OutcomeMaturityPolicy, ShadowDriftMonitor, calculate_psi


MANIFEST_PATH = "experiments/phase6/protected_artifacts_manifest.json"


@pytest.fixture(scope="module")
def api_client():
    return TestClient(app)


@pytest.fixture(scope="module")
def shadow_engine():
    return ShadowInferenceEngine()


# ---------------------------------------------------------------------
# TEST 1 & 2: BASELINE IMMUTABILITY & OFFICIAL THRESHOLD PROTECTION
# ---------------------------------------------------------------------

def test_baseline_sha256_immutability():
    """Verifies that all Phase 2 approved production artifacts match pre-implementation SHA-256 hashes."""
    assert os.path.exists(MANIFEST_PATH), f"Manifest missing at {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for rel_path, expected_hash in manifest["pre_implementation_hashes"].items():
        assert os.path.exists(rel_path), f"Protected artifact missing: {rel_path}"
        actual_hash = compute_sha256(rel_path)
        assert actual_hash == expected_hash, (
            f"BASELINE INTEGRITY VIOLATION for {rel_path}! "
            f"Expected {expected_hash}, got {actual_hash}. Baseline must remain untouched."
        )


def test_production_threshold_immutability(shadow_engine):
    """Verifies that the approved production threshold remains frozen at 0.0800."""
    assert shadow_engine.prod_threshold == 0.0800
    assert shadow_engine.OFFICIAL_PROD_THRESHOLD == 0.0800
    with open("models/phase2/threshold_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["operating_threshold"] == 0.0800


# ---------------------------------------------------------------------
# TEST 3 & 4: FAILURE ISOLATION & NON-INTERFERENCE
# ---------------------------------------------------------------------

def test_candidate_failure_isolation():
    """
    Simulates a total failure or exception in the candidate model.
    Verifies that production prediction executes completely intact and unaffected.
    """
    engine = ShadowInferenceEngine()
    
    # Load sample company history
    from src.data.loaders import DataLoader
    df_raw = DataLoader("data/raw").load_dataset_a()
    df_co = df_raw[df_raw["Company"] == 112].copy()

    # Normal execution baseline
    normal_res = engine.evaluate_shadow_sample("112", df_co, current_period=12, record_audit_log=False)
    assert normal_res["production"]["decision_authority"] == "OFFICIAL_PRODUCTION_DECISION"
    assert normal_res["shadow_candidate"]["sandbox_status"] == "SUCCESS_EXECUTED"

    # Injected candidate failure via monkeypatching cand_model.predict_proba
    with patch.object(engine.cand_model, "predict_proba", side_effect=RuntimeError("Simulated CUDA/Memory Crash in Candidate")):
        isolated_res = engine.evaluate_shadow_sample("112", df_co, current_period=12, record_audit_log=False)
        
        # Production must be 100% unaffected
        assert isolated_res["production"]["calibrated_probability"] == normal_res["production"]["calibrated_probability"]
        assert isolated_res["production"]["risk_score"] == normal_res["production"]["risk_score"]
        assert isolated_res["production"]["is_distress_alert"] == normal_res["production"]["is_distress_alert"]
        assert isolated_res["production"]["decision_authority"] == "OFFICIAL_PRODUCTION_DECISION"

        # Shadow status must capture error gracefully without crashing
        assert "SANDBOX_ERROR" in isolated_res["shadow_candidate"]["sandbox_status"]
        assert isolated_res["shadow_candidate"]["is_shadow_alert"] is False
        assert isolated_res["shadow_candidate"]["agreement_with_production"] is None


def test_non_interference_when_candidate_missing():
    """Verifies production inference continues seamlessly if candidate model artifacts are missing."""
    engine = ShadowInferenceEngine()
    engine.cand_model = None
    engine.cand_cal = None

    from src.data.loaders import DataLoader
    df_raw = DataLoader("data/raw").load_dataset_a()
    df_co = df_raw[df_raw["Company"] == 112].copy()

    res = engine.evaluate_shadow_sample("112", df_co, current_period=12, record_audit_log=False)
    assert res["production"]["is_distress_alert"] is True
    assert res["shadow_candidate"]["sandbox_status"] == "CANDIDATE_UNAVAILABLE"


# ---------------------------------------------------------------------
# TEST 5 & 6: AUDIT PROVENANCE, TIMESTAMPS & MISSING FEATURES
# ---------------------------------------------------------------------

def test_telemetry_and_audit_provenance(shadow_engine):
    """Verifies that shadow evaluation generates structured, compliant telemetry records."""
    from src.data.loaders import DataLoader
    df_raw = DataLoader("data/raw").load_dataset_a()
    df_co = df_raw[df_raw["Company"] == 112].copy()

    rec = shadow_engine.evaluate_shadow_sample("112", df_co, current_period=12, record_audit_log=False)

    # Required fields check
    assert "timestamp_utc" in rec
    assert rec["pseudonymous_company_id"] == "112"
    assert rec["observation_period"] == 12
    assert rec["distress_target_period"] == 13
    assert rec["outcome_maturity_status"] == "PENDING_T_PLUS_1_AUDIT"
    assert "execution_latency_ms" in rec
    assert rec["shadow_candidate"]["label"] == "EXPERIMENTAL_RESEARCH_CANDIDATE_ONLY"
    assert rec["shadow_candidate"]["research_threshold"] == 0.0894


def test_missing_features_robustness(shadow_engine):
    """Verifies that missing features or partial records in candidate inference are handled without crashing."""
    # Synthetic observation with missing columns
    synth_row = pd.DataFrame([{
        "Company": 9999,
        "Time": 12,
        "x1": 0.5,
        "x2": np.nan,
        "x12": 0.1
    }])
    
    # Must run without unhandled exception
    res = shadow_engine.evaluate_shadow_sample("9999", synth_row, current_period=12, record_audit_log=False)
    assert "production" in res
    assert "shadow_candidate" in res


# ---------------------------------------------------------------------
# TEST 7 & 8: OUTCOME MATURITY & POPULATION DRIFT (PSI)
# ---------------------------------------------------------------------

def test_outcome_maturity_policy_classification():
    """Verifies that maturity policy accurately separates matured from pending predictions."""
    # t=12 predicting t=13
    # If latest available is 12, target at 13 is unobservable -> PENDING
    assert OutcomeMaturityPolicy.classify_maturity(observation_period=12, current_available_period=12) == "PENDING_MATURITY"
    # If latest available is 13, target at 13 is observable -> MATURED
    assert OutcomeMaturityPolicy.classify_maturity(observation_period=12, current_available_period=13) == "MATURED"


def test_outcome_maturity_uncertainty_disclosure():
    """Verifies that sample-size thresholds trigger uncertainty flags when distress events < 10."""
    df_small = pd.DataFrame([
        {
            "outcome_maturity_status": "MATURED",
            "actual_target": 1,
            "prod_calibrated_probability": 0.15,
            "cand_calibrated_probability": 0.18
        },
        {
            "outcome_maturity_status": "MATURED",
            "actual_target": 0,
            "prod_calibrated_probability": 0.02,
            "cand_calibrated_probability": 0.03
        }
    ])
    summary = OutcomeMaturityPolicy.evaluate_matured_predictions(df_small)
    assert summary["meets_preregistered_sample_protocol"] is False
    assert summary["comparison"]["statistically_verified"] is False


def test_population_stability_index_drift():
    """Verifies that calculate_psi detects stable vs drifting feature distributions."""
    rng = np.random.RandomState(42)
    baseline = rng.normal(loc=0.0, scale=1.0, size=500)
    
    # Identical distribution -> PSI should be very close to 0 (< 0.05)
    identical = rng.normal(loc=0.0, scale=1.0, size=500)
    psi_stable = calculate_psi(baseline, identical)
    assert psi_stable < 0.10, f"Expected stable PSI < 0.10, got {psi_stable}"

    # Heavily shifted distribution -> PSI should indicate significant drift (>= 0.25)
    shifted = rng.normal(loc=2.0, scale=1.5, size=500)
    psi_drift = calculate_psi(baseline, shifted)
    assert psi_drift >= 0.25, f"Expected drifting PSI >= 0.25, got {psi_drift}"


# ---------------------------------------------------------------------
# TEST 9: FASTAPI SHADOW ENDPOINTS
# ---------------------------------------------------------------------

def test_api_shadow_status(api_client):
    """Tests GET /shadow/status endpoint returns valid operational state and model paths."""
    resp = api_client.get("/shadow/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "OPERATIONAL"
    assert data["shadow_mode"] == "PARALLEL_NON_INTERFERING"
    assert data["production_baseline"]["official_threshold"] == 0.0800
    assert data["shadow_candidate"]["research_threshold"] == 0.0894
    assert data["shadow_candidate"]["decision_authority"] == "EXPERIMENTAL_RESEARCH_CANDIDATE_ONLY_NO_PRODUCTION_EFFECT"


def test_api_shadow_telemetry(api_client):
    """Tests GET /shadow/telemetry returns prospective metrics and drift status."""
    resp = api_client.get("/shadow/telemetry")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_prospective_observations" in data
    assert "production_alert_rate_pct" in data
    assert "shadow_candidate_alert_rate_pct" in data
    assert "feature_drift_psi" in data


def test_api_shadow_maturity_evaluation(api_client):
    """Tests GET /shadow/maturity-evaluation returns matured outcomes with uncertainty disclosures."""
    resp = api_client.get("/shadow/maturity-evaluation")
    assert resp.status_code == 200
    data = resp.json()
    assert data["evaluation_cohort"] == "MATURED_ONLY"
    assert "production_baseline" in data
    assert "shadow_candidate" in data
    assert "comparison" in data


def test_api_shadow_predict(api_client):
    """Tests POST /shadow/predict returns experimental header and non-interfering parallel prediction."""
    resp = api_client.post("/shadow/predict", json={"company_id": "112", "current_period": 12})
    assert resp.status_code == 200
    assert resp.headers.get("x-shadow-evaluation-mode") == "NON_AUTHORITATIVE_EXPERIMENTAL"
    data = resp.json()
    assert "production" in data
    assert "shadow_candidate" in data
    assert data["production"]["decision_authority"] == "OFFICIAL_PRODUCTION_DECISION"
    assert data["shadow_candidate"]["label"] == "EXPERIMENTAL_RESEARCH_CANDIDATE_ONLY"
