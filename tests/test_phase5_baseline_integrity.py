"""
Phase 5 Test Suite: Baseline Artifact & Configuration Integrity Safeguards.

Verifies:
1. All protected Phase 2 model artifacts, preprocessor, and calibration files exist and match SHA-256 manifest.
2. Official production operating threshold remains frozen at exactly tau = 0.080.
3. Production configuration files (risk_config.yaml, config.yaml) are unaltered.
4. Phase 5 candidate models are strictly isolated in experiments/phase5/ and do not contaminate models/phase2/.
5. Regression compatibility with approved prediction interface.
"""

import os
import sys
import json
import hashlib
import pytest
import joblib
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from src.preprocessing.preprocessors import EWSPreprocessor


def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class TestPhase5BaselineIntegrity:
    """Safeguards ensuring approved Phase 2/3 baseline is immutable."""

    MANIFEST_PATH = "experiments/phase5/protected_artifacts_manifest.json"

    def test_protected_artifacts_manifest_exists(self):
        assert os.path.exists(self.MANIFEST_PATH), "Protected manifest must exist."

    def test_approved_artifacts_match_sha256_manifest(self):
        """Every protected artifact must match the recorded SHA-256 hash."""
        with open(self.MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for filepath, expected_hash in manifest.get("hashes", {}).items():
            assert os.path.exists(filepath), f"Protected file missing: {filepath}"
            actual_hash = compute_sha256(filepath)
            assert actual_hash == expected_hash, f"Hash mismatch on protected file {filepath}: expected {expected_hash}, got {actual_hash}"

    def test_official_operating_threshold_remains_0080(self):
        """Official production threshold must remain 0.080."""
        with open("models/phase2/threshold_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        assert cfg.get("operating_threshold") == 0.08, "Operating threshold must remain frozen at 0.080"

    def test_phase5_artifacts_strictly_isolated(self):
        """Verify experimental Phase 5 artifacts do not leak into models/phase2/."""
        assert os.path.exists("experiments/phase5/models"), "Phase 5 models dir must exist."
        p2_files = os.listdir("models/phase2")
        for f in p2_files:
            assert "Phase5" not in f, f"Phase 5 model leaked into models/phase2: {f}"
            assert "Deterioration" not in f, f"Deterioration model leaked into models/phase2: {f}"

    def test_approved_baseline_model_inference_regression(self):
        """Ensure approved baseline model loads and executes prediction correctly."""
        model = joblib.load("models/phase2/best_ews_model.joblib")
        calibrator = joblib.load("models/phase2/ews_calibrator.joblib")
        preprocessor = EWSPreprocessor.load("models/phase2/ews_preprocessor.joblib")

        # Mock vector matching feature names
        dummy_df = pd.DataFrame([{col: 0.0 for col in preprocessor.feature_names}])
        dummy_df["Company"] = 999
        dummy_df["Time"] = 1
        dummy_df["Financial Distress"] = 0.5
        dummy_df["forward_target"] = 0

        X_mock, _ = preprocessor.transform(dummy_df)
        raw_p = model.predict_proba(X_mock)[:, 1]
        cal_p = calibrator.predict_proba(X_mock)[:, 1]

        assert 0.0 <= raw_p[0] <= 1.0
        assert 0.0 <= cal_p[0] <= 1.0


if __name__ == "__main__":
    import pandas as pd
    pytest.main(["-v", __file__])
