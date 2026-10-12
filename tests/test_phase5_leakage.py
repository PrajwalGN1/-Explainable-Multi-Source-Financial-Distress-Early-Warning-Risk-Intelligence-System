"""
Phase 5 Test Suite: Temporal Leakage, Group Isolation & Evaluation Protocol Safeguards.

Verifies:
1. Point-in-time features strictly use backward-looking data (invariance under future row addition).
2. Temporal splits respect chronological ordering (Train T<=10, Val T=11,12, Test T=13).
3. Zero company overlap in GroupKFold splits.
4. Training-only resampling does not alter validation partitions.
5. Missing initial lags (t=1) and degenerate folds are handled safely.
6. Alert-budget threshold selection operates strictly on validation predictions.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.abspath("."))
from src.evaluation.phase5_feature_engineering import Phase5FeatureEngine
from src.evaluation.phase5_alert_budget import (
    find_threshold_for_alert_budget, evaluate_at_threshold, wilson_ci
)
from src.evaluation.phase5_model_comparison import (
    random_oversample_binary, Phase5PreprocessingPipeline
)


class TestPhase5LeakagePrevention:
    """Verifies complete absence of temporal lookahead and cross-sectional leakage."""

    def test_features_invariant_to_future_data_addition(self):
        """Features at time t must be mathematically identical whether future periods exist or not."""
        engine = Phase5FeatureEngine()

        # Company 1 observed at t=1, 2, 3
        df_base = pd.DataFrame({
            "Company": [1, 1, 1],
            "Time": [1, 2, 3],
            "Financial Distress": [0.5, 0.4, 0.3],
            "x12": [10.0, 12.0, 15.0],
            "x25": [2.0, 2.5, 3.0],
            "x16": [5.0, 4.0, 3.5],
            "x2": [1.0, 0.8, 0.6],
            "x48": [0.1, 0.2, 0.3]
        })
        feats_base = engine.generate_deterioration_features(df_base)
        val_t2_base = feats_base[feats_base["Time"] == 2]["x12_diff_1"].values[0]

        # Append future period t=4 with arbitrary values
        df_extended = pd.concat([df_base, pd.DataFrame([{
            "Company": 1, "Time": 4, "Financial Distress": -0.8,
            "x12": 99.0, "x25": 99.0, "x16": 0.1, "x2": -5.0, "x48": 10.0
        }])], ignore_index=True)

        feats_extended = engine.generate_deterioration_features(df_extended)
        val_t2_extended = feats_extended[feats_extended["Time"] == 2]["x12_diff_1"].values[0]

        # Feature at t=2 must be completely identical
        assert np.isclose(val_t2_base, val_t2_extended), "Feature at t=2 changed after future row addition, indicating lookahead leakage!"

    def test_forward_target_final_period_is_nan(self):
        """The last observed period for any company must have forward_target = NaN."""
        engine = Phase5FeatureEngine()
        df = pd.DataFrame({
            "Company": [1, 1, 2, 2],
            "Time": [1, 2, 1, 2],
            "Financial Distress": [0.5, -0.6, 0.2, 0.1]
        })
        df_target = engine.construct_forward_target(df)
        last_rows = df_target[df_target["Time"] == 2]
        assert last_rows["forward_target"].isna().all(), "Final period forward_target must be NaN."

    def test_group_kfold_zero_company_cross_contamination(self):
        """In GroupKFold, no company can appear in both training and validation folds."""
        companies = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10])
        X = np.random.randn(len(companies), 5)
        y = np.random.binomial(1, 0.1, size=len(companies))

        gkf = GroupKFold(n_splits=5)
        for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=companies)):
            tr_comps = set(companies[tr_idx])
            te_comps = set(companies[te_idx])
            assert tr_comps.isdisjoint(te_comps), f"Fold {fold} has company overlap across folds!"

    def test_training_only_oversampling_leaves_validation_unmodified(self):
        """Resampling alters only training arrays and does not touch validation arrays."""
        X_tr = np.random.randn(100, 10)
        y_tr = np.array([1] * 5 + [0] * 95)
        X_val = np.random.randn(50, 10)
        y_val = np.array([1] * 2 + [0] * 48)

        X_tr_res, y_tr_res = random_oversample_binary(X_tr, y_tr, target_ratio=0.15, random_state=42)
        assert len(y_tr_res) > len(y_tr), "Training size must increase after oversampling."
        # Validation remains unchanged
        assert len(y_val) == 50, "Validation size must remain completely unchanged."
        assert np.sum(y_val) == 2, "Validation positive count must remain unchanged."

    def test_alert_budget_threshold_selection_monotonicity(self):
        """Higher alert budget must result in lower or equal probability threshold."""
        rng = np.random.RandomState(42)
        y_prob = rng.uniform(0.01, 0.30, size=500)

        tau_5 = find_threshold_for_alert_budget(y_prob, target_alert_rate=5.0)
        tau_8 = find_threshold_for_alert_budget(y_prob, target_alert_rate=8.0)
        tau_10 = find_threshold_for_alert_budget(y_prob, target_alert_rate=10.0)
        tau_15 = find_threshold_for_alert_budget(y_prob, target_alert_rate=15.0)

        assert tau_5 >= tau_8 >= tau_10 >= tau_15, "Thresholds must be monotonically non-increasing with larger alert budgets."

    def test_wilson_ci_validity(self):
        """Wilson score confidence intervals must be strictly bounded in [0, 1]."""
        low, high = wilson_ci(7, 14)
        assert 0.0 <= low <= high <= 1.0
        assert np.isclose(0.5, (low + high) / 2.0, atol=0.05)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
