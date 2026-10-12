"""
Phase 6.1 Automated Regression Test Suite:
Shadow Evaluation Audit, Metric Reconciliation, Feature Drift & Statistical Readiness.

Tests:
1. Metric Reconciliation Denominators (270 prospective, 264 matured, 6 pending right-censored firms)
2. Company 409 Rounding Boundary Verification (Unrounded 0.079967 vs 0.0800)
3. Recomputed Matured Metrics Consistency (Confusion matrices, recall, precision, specificity, FPR, Brier)
4. x48 Feature Drift Bins and Temporal Reversion (Peak at T=12, Reversion at T=14)
5. Related Ratios Stability (x12 Operating Cash Flow PSI < 0.10)
6. Statistical Non-Inferiority Readiness & Underpowered Sample Gating
7. Protected Baseline SHA-256 Invariance (Zero modification to approved Phase 2 artifacts)
"""

import os
import json
import numpy as np
import pandas as pd
import pytest

from src.shadow.shadow_engine import ShadowInferenceEngine, compute_sha256
from src.shadow.maturity_and_drift import calculate_psi
from src.shadow.audit_phase6_1 import (
    run_metric_reconciliation,
    run_feature_drift_x48_investigation,
    run_statistical_readiness_power_analysis,
    wilson_ci
)


MANIFEST_PATH = "experiments/phase6/protected_artifacts_manifest.json"


@pytest.fixture(scope="module")
def reconciliation_data():
    return run_metric_reconciliation()


@pytest.fixture(scope="module")
def drift_data():
    return run_feature_drift_x48_investigation()


@pytest.fixture(scope="module")
def readiness_data():
    return run_statistical_readiness_power_analysis()


# ---------------------------------------------------------------------
# TEST 1: DENOMINATORS & PENDING COHORT RECONCILIATION
# ---------------------------------------------------------------------

def test_metric_reconciliation_exact_denominators(reconciliation_data):
    """Verifies prospective (270), matured (264), and pending (6) counts and reasons."""
    denoms = reconciliation_data["denominators"]
    assert denoms["prospective_surveillance_cohort"] == 270
    assert denoms["matured_cohort"] == 264
    assert denoms["pending_cohort"] == 6
    assert denoms["matured_distress_cases"] == 8
    assert denoms["matured_non_distress_cases"] == 256

    pending_firms = reconciliation_data["pending_cohort_analysis"]["pending_firms"]
    assert len(pending_firms) == 6
    pending_ids = sorted([f["company_id"] for f in pending_firms])
    assert pending_ids == [68, 115, 159, 203, 276, 301]

    # All 6 ceased filing due to terminal distress at T=12
    for firm in pending_firms:
        assert firm["was_distressed_at_exit"] is True
        assert firm["last_financial_distress_value"] <= -0.5


# ---------------------------------------------------------------------
# TEST 2: COMPANY 409 ROUNDING BOUNDARY VERIFICATION
# ---------------------------------------------------------------------

def test_company_409_rounding_boundary(reconciliation_data):
    """Verifies that Company 409 unrounded probability is strictly < 0.0800, explaining alert count discrepancy."""
    audit_409 = reconciliation_data["rounding_boundary_audit_company_409"]
    assert audit_409["unrounded_calibrated_probability"] < 0.0800
    assert audit_409["engine_boolean_prod_is_alert"] is False
    assert audit_409["rounded_calibrated_probability"] == 0.0800


# ---------------------------------------------------------------------
# TEST 3: RECOMPUTED MATURED METRICS CONSISTENCY
# ---------------------------------------------------------------------

def test_recomputed_matured_metrics_consistency(reconciliation_data):
    """Verifies recomputed recall, precision, specificity, FPR, and Brier score."""
    metrics = reconciliation_data["reconciled_metrics"]
    base_nat = metrics["approved_baseline_native_boolean"]
    cand = metrics["shadow_candidate"]

    # Both models achieve exact recall parity
    assert base_nat["recall"] == 0.6250
    assert cand["recall"] == 0.6250
    assert base_nat["tp"] == 5
    assert cand["tp"] == 5
    assert base_nat["fn"] == 3
    assert cand["fn"] == 3

    # Candidate precision is strictly superior
    assert cand["precision"] > base_nat["precision"]
    assert cand["precision"] == 0.2632 # 5 / 19
    assert base_nat["precision"] == 0.1724 # 5 / 29

    # Candidate false positives are significantly lower
    assert cand["fp"] == 14
    assert base_nat["fp"] == 24
    assert cand["specificity"] == 0.9453 # 242 / 256
    assert base_nat["specificity"] == 0.9062 # 232 / 256

    # Brier scores
    assert cand["brier_score"] <= base_nat["brier_score"]


# ---------------------------------------------------------------------
# TEST 4 & 5: FEATURE DRIFT x48 BINS, TRAJECTORY & RELATED FEATURES
# ---------------------------------------------------------------------

def test_feature_drift_x48_investigation(drift_data):
    """Verifies x48 PSI, decile bin skew, and subsequent temporal normalization."""
    assert drift_data["reported_psi_at_t12"] == 0.4277
    assert drift_data["verification_status"] == "VERIFIED_ACCURATE"

    # Decile bin 10 skew
    bins = drift_data["decile_bin_breakdown"]
    bin_10 = bins[-1]
    assert bin_10["bin_index"] == 10
    assert bin_10["actual_pct"] > 30.0 # 31.11% in top decile
    assert bin_10["psi_contribution"] > 0.20

    # Temporal trajectory: T=12 peak, T=14 normalization
    traj = drift_data["temporal_trajectory"]
    assert traj["Time_12"]["psi_vs_baseline"] == 0.4277
    assert traj["Time_14"]["psi_vs_baseline"] < 0.10 # Reversion to stability
    assert traj["Time_14"]["drift_classification"] == "STABLE"

    # Related features: x12 operating cash flow remains stable
    related = drift_data["related_features_drift"]
    assert related["x12"]["status"] == "STABLE"
    assert related["x12"]["psi_t12"] < 0.10


# ---------------------------------------------------------------------
# TEST 6: STATISTICAL PROMOTION READINESS & POWER GATES
# ---------------------------------------------------------------------

def test_statistical_promotion_readiness(readiness_data):
    """Verifies that statistical promotion is strictly blocked due to underpowered sample."""
    sample = readiness_data["current_sample"]
    assert sample["matured_distress_count"] == 8
    assert sample["pre_registered_minimum_gate"] == 10
    assert sample["gate_satisfied"] is False

    # Paired detection: 0 discordant pairs among distress cases
    paired = readiness_data["paired_distress_detection_table"]
    assert paired["both_detected_tp"] == 5
    assert paired["both_missed_fn"] == 3
    assert paired["candidate_only_tp"] == 0
    assert paired["baseline_only_tp"] == 0

    # Confidence interval for recall difference includes substantial negative values
    ci_diff = readiness_data["paired_recall_estimation"]["newcombe_95_ci_difference"]
    assert ci_diff[0] < -0.25 # Lower bound is -31.9%

    # Power analysis verifies low power at N=8
    power_05 = readiness_data["non_inferiority_power_curves"]["power_curve_delta_0_05"][0]
    assert power_05["n_distress"] == 8
    assert power_05["statistical_power"] < 0.20 # ~11.5% power

    # Promotion recommendation must be DO NOT PROMOTE
    assert "DO NOT PROMOTE" in readiness_data["promotion_decision"]["recommendation"]


# ---------------------------------------------------------------------
# TEST 7: PROTECTED ARTIFACT SHA-256 INTEGRITY
# ---------------------------------------------------------------------

def test_protected_artifact_sha256_invariance():
    """Verifies that all Phase 2 approved production artifacts match exact SHA-256 digests."""
    assert os.path.exists(MANIFEST_PATH)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for rel_path, expected_hash in manifest["pre_implementation_hashes"].items():
        assert os.path.exists(rel_path)
        actual_hash = compute_sha256(rel_path)
        assert actual_hash == expected_hash, f"Cryptographic integrity violation for {rel_path}!"
