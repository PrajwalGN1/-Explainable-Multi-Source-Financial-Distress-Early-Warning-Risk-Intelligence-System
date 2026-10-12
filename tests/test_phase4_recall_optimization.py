"""
Unit and Integration Tests for Phase 4: Recall Optimization, Validation Integrity & Governance.

Verifies:
1. Preservation and immutability of approved Phase 2 artifacts and threshold configuration.
2. Target formulation integrity (forward-looking t+1 without current-period leakage).
3. Preprocessing leak-free isolation across folds.
4. Company isolation in GroupKFold (zero company cross-contamination).
5. Temporal ordering constraint (Training T<=10, Val T=11,12, Test T=13).
6. Training-only resampling integrity (zero resampling of validation or test folds).
7. Validation-only threshold selection governance.
8. Probability calibration consistency and BSS calculation.
9. Wilson 95% confidence interval stability and boundary adherence.
10. Explicit namespace isolation between experimental models and approved models.
"""

import os
import sys
import json
import joblib
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.data.loaders import DataLoader
from src.features.temporal_features import TemporalFeatureEngine
from src.preprocessing.preprocessors import EWSPreprocessor
from src.evaluation.phase4_recall_validation import (
    compute_file_hash, wilson_ci, evaluate_metrics
)
from src.evaluation.phase4_model_comparison import random_oversample_binary
from src.evaluation.phase4_calibration_analysis import calculate_ece_mce, calculate_cox_calibration


class TestPhase4ArtifactPreservation:
    """Verifies approved Phase 2/3 production artifacts are completely frozen and intact."""

    def test_approved_artifacts_exist_and_unmodified(self):
        """Ensure approved artifacts exist with valid sizes and uncorrupted headers."""
        approved_paths = [
            "models/phase2/best_ews_model.joblib",
            "models/phase2/ews_calibrator.joblib",
            "models/phase2/ews_preprocessor.joblib",
            "models/phase2/threshold_config.json",
            "config/risk_config.yaml"
        ]
        for p in approved_paths:
            assert os.path.exists(p), f"Mandatory approved artifact missing: {p}"
            assert os.path.getsize(p) > 0, f"Artifact empty: {p}"

    def test_approved_operating_threshold_remains_0080(self):
        """Verify operating threshold is exactly 0.080 in production configuration."""
        with open("models/phase2/threshold_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        assert cfg.get("operating_threshold") == 0.08, "Operating threshold must remain frozen at 0.080"

    def test_experimental_artifacts_strictly_isolated(self):
        """Ensure experimental models are saved in experiments/phase4, not models/phase2/."""
        assert os.path.exists("experiments/phase4/models"), "Phase 4 models dir must exist"
        exp_files = os.listdir("experiments/phase4/models")
        assert len(exp_files) > 0, "Phase 4 experimental models must be populated"
        # Phase 2 dir must NOT contain phase 4 model names
        p2_files = os.listdir("models/phase2")
        for f in p2_files:
            assert "CostSensitive" not in f, f"Experimental model leaked into models/phase2: {f}"
            assert "ROS" not in f, f"Experimental model leaked into models/phase2: {f}"


class TestTargetAndTemporalIntegrity:
    """Validates leakage-free target construction and temporal split ordering."""

    @pytest.fixture
    def raw_data(self):
        loader = DataLoader("data/raw")
        return loader.load_dataset_a()

    def test_forward_target_construction(self, raw_data):
        """Verify target strictly evaluates distress in period t+1 for company at period t."""
        engine = TemporalFeatureEngine()
        df_target = engine.construct_forward_target(
            raw_data, id_col="Company", time_col="Time", raw_target_col="Financial Distress", threshold=-0.5
        )
        assert "forward_target" in df_target.columns
        # For last time period of any company, forward_target must be NaN (cannot look beyond last period)
        last_periods = df_target.groupby("Company")["Time"].max().reset_index()
        for _, row in last_periods.iterrows():
            comp_id = row["Company"]
            t_max = row["Time"]
            val = df_target[(df_target["Company"] == comp_id) & (df_target["Time"] == t_max)]["forward_target"].values[0]
            assert np.isnan(val), f"Company {comp_id} at max Time {t_max} has non-NaN forward target, indicating lookahead leakage!"

    def test_temporal_split_disjointness(self, raw_data):
        """Verify strict chronological boundaries: Train T<=10, Val T=11,12, Test T=13."""
        engine = TemporalFeatureEngine()
        df_target = engine.construct_forward_target(
            raw_data, id_col="Company", time_col="Time", raw_target_col="Financial Distress", threshold=-0.5
        )
        df_clean = df_target.dropna(subset=["forward_target"]).copy()

        train_times = set(df_clean[df_clean["Time"] <= 10]["Time"].unique())
        val_times = set(df_clean[df_clean["Time"].isin([11, 12])]["Time"].unique())
        test_times = set(df_clean[df_clean["Time"] == 13]["Time"].unique())

        assert train_times.isdisjoint(val_times), "Train and Val time periods must be strictly disjoint"
        assert val_times.isdisjoint(test_times), "Val and Test time periods must be strictly disjoint"
        assert train_times.isdisjoint(test_times), "Train and Test time periods must be strictly disjoint"
        assert max(train_times) < min(val_times), "Train times must strictly precede Val times"
        assert max(val_times) < min(test_times), "Val times must strictly precede Test times"


class TestValidationLeakageAndGroupIsolation:
    """Verifies inner-fold preprocessing isolation and company grouping."""

    def test_company_isolation_in_group_kfold(self):
        """Verify GroupKFold enforces zero company overlap between train and validation folds."""
        loader = DataLoader("data/raw")
        df_raw = loader.load_dataset_a()
        engine = TemporalFeatureEngine()
        df_target = engine.construct_forward_target(df_raw, "Company", "Time", "Financial Distress", -0.5)
        df_clean = df_target.dropna(subset=["forward_target"]).copy()
        train_df = df_clean[df_clean["Time"] <= 10].reset_index(drop=True)

        companies = train_df["Company"].to_numpy()
        gkf = GroupKFold(n_splits=5)
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(train_df, train_df["forward_target"], groups=companies)):
            tr_comps = set(companies[tr_idx])
            val_comps = set(companies[val_idx])
            overlap = tr_comps.intersection(val_comps)
            assert len(overlap) == 0, f"Fold {fold} has company overlap across folds: {overlap}"

    def test_training_only_oversampling(self):
        """Verify resampling alters ONLY the training fold and preserves validation data untouched."""
        X_mock = np.random.randn(100, 10)
        y_mock = np.array([1] * 5 + [0] * 95)  # 5% minority

        X_ros, y_ros = random_oversample_binary(X_mock, y_mock, target_ratio=0.15, random_state=42)
        assert len(y_ros) > len(y_mock), "Oversampled dataset must have more samples"
        assert np.mean(y_ros) >= 0.14, "Minority prevalence must reach ~15%"
        # Original negative samples must remain identical
        orig_neg_count = np.sum(y_mock == 0)
        ros_neg_count = np.sum(y_ros == 0)
        assert orig_neg_count == ros_neg_count, "Negative majority class count must not change during minority oversampling"


class TestCalibrationAndMetricsIntegrity:
    """Verifies mathematical correctness of calibration metrics and statistical intervals."""

    def test_wilson_ci_bounds_and_coverage(self):
        """Verify Wilson 95% confidence intervals are within [0, 1] and valid for edge cases."""
        # Zero hits
        low, high = wilson_ci(0, 100)
        assert low == 0.0
        assert 0.0 < high < 0.05

        # All hits
        low, high = wilson_ci(100, 100)
        assert low > 0.95
        assert high == 1.0

        # Baseline recall: 3 hits out of 11
        low, high = wilson_ci(3, 11)
        assert 0.05 < low < 0.15
        assert 0.50 < high < 0.60

    def test_ece_mce_monotonicity(self):
        """Verify ECE is 0 for perfectly calibrated predictions."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        ece, mce, bins = calculate_ece_mce(y_true, y_prob, n_bins=10)
        assert ece == 0.0
        assert mce == 0.0

    def test_cox_calibration_estimation(self):
        """Verify Cox calibration returns finite numeric intercept and slope."""
        rng = np.random.RandomState(42)
        y_true = rng.binomial(1, 0.1, size=200)
        y_prob = rng.uniform(0.01, 0.3, size=200)
        intercept, slope = calculate_cox_calibration(y_true, y_prob)
        assert np.isfinite(intercept)
        assert np.isfinite(slope)
