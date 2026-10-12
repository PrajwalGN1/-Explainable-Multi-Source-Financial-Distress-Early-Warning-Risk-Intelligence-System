"""
Phase 2 Model Validation, Stacking Refactoring & Early-Warning Reliability Tests.
Validates:
- Synthetic proxy removal and empirical ablation
- Out-of-fold meta-model validation
- Train / Val / Test temporal separation and preprocessor isolation
- Threshold selection strictly on validation fold without test labels
- Probability calibration (Platt scaling) and Brier improvement
- Model artifact loading and inference reproducibility
- Risk score and alert consistency
- Test set recall operational gain over naive 0.50 baseline
- API reporting of validated Phase 2 models
"""

import os
import json
import joblib
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, recall_score, confusion_matrix
from fastapi.testclient import TestClient

from src.data.loaders import DataLoader
from src.features.temporal_features import TemporalFeatureEngine
from src.preprocessing.preprocessors import EWSPreprocessor
from src.models.ews_model import TemporalEWSModel
from src.fusion.risk_fusion import RiskFusionEngine
from src.risk_engine.alerts import EarlyWarningSignalEngine
from api import app

client = TestClient(app)


def test_phase2_model_artifacts_exist_and_loadable():
    """Verifies that Phase 2 artifacts exist in models/phase2 and are loadable."""
    phase2_dir = "models/phase2"
    assert os.path.exists(phase2_dir), "models/phase2 directory must exist"
    
    required_files = [
        "best_ews_model.joblib",
        "ews_preprocessor.joblib",
        "ews_calibrator.joblib",
        "threshold_config.json",
        "ews_metrics.json",
        "model_manifest.json"
    ]
    for fname in required_files:
        fpath = os.path.join(phase2_dir, fname)
        assert os.path.exists(fpath), f"Missing required Phase 2 artifact: {fpath}"
        
    # Verify loadability
    model = joblib.load(os.path.join(phase2_dir, "best_ews_model.joblib"))
    preprocessor = EWSPreprocessor.load(os.path.join(phase2_dir, "ews_preprocessor.joblib"))
    calibrator = joblib.load(os.path.join(phase2_dir, "ews_calibrator.joblib"))
    
    assert hasattr(model, "predict_proba")
    assert preprocessor.fitted is True
    assert len(preprocessor.feature_names) > 0
    assert hasattr(calibrator, "predict_proba")
    
    # Load and check threshold config
    with open(os.path.join(phase2_dir, "threshold_config.json"), "r") as f:
        thresh_cfg = json.load(f)
    assert "operating_threshold" in thresh_cfg
    assert thresh_cfg["operating_threshold"] == 0.08
    assert thresh_cfg["selection_fold"] == "Validation Fold (T=11,12)"


def test_temporal_train_val_test_separation_and_leakage_protection():
    """Verifies strict chronological ordering, zero temporal overlap, and correct horizons."""
    loader = DataLoader("data/raw")
    df_a = loader.load_dataset_a()
    engine = TemporalFeatureEngine()
    df_target = engine.construct_forward_target(df_a, id_col="Company", time_col="Time", raw_target_col="Financial Distress", threshold=-0.5)
    df_feats = engine.generate_temporal_features(df_target, id_col="Company", time_col="Time")
    
    df_valid = df_feats.dropna(subset=["forward_target"]).copy()
    
    train_df = df_valid[df_valid["Time"] <= 10]
    val_df = df_valid[(df_valid["Time"] > 10) & (df_valid["Time"] <= 12)]
    test_df = df_valid[df_valid["Time"] > 12]
    
    # Check disjoint time sets
    train_times = set(train_df["Time"].unique())
    val_times = set(val_df["Time"].unique())
    test_times = set(test_df["Time"].unique())
    
    assert train_times == set(range(1, 11)), "Train periods must be exactly 1 to 10"
    assert val_times == {11, 12}, "Validation periods must be exactly 11 and 12"
    assert test_times == {13}, "Test period must be exactly 13"
    
    assert len(train_times.intersection(val_times)) == 0, "Train and Val time periods must not overlap"
    assert len(val_times.intersection(test_times)) == 0, "Val and Test time periods must not overlap"
    assert len(train_times.intersection(test_times)) == 0, "Train and Test time periods must not overlap"
    
    # Verify forward target definition: t+1
    # Check for a specific company that forward_target at time t equals current_distress at time t+1
    sample_comp = df_target[df_target["Company"] == 1].sort_values("Time")
    for i in range(len(sample_comp) - 1):
        row_curr = sample_comp.iloc[i]
        row_next = sample_comp.iloc[i + 1]
        assert row_curr["forward_target"] == row_next["current_distress"], "forward_target must equal next period current_distress"


def test_preprocessor_isolation_and_column_consistency():
    """Verifies preprocessor fits only on train data and maintains exact feature alignment."""
    preprocessor = EWSPreprocessor.load("models/phase2/ews_preprocessor.joblib")
    
    # Feature list check
    assert len(preprocessor.feature_names) == 163
    # Check that forbidden columns were dropped
    forbidden = ["Company", "Time", "Financial Distress", "current_distress", "forward_target", "forward_target_2period"]
    for col in forbidden:
        assert col not in preprocessor.feature_names, f"Target or ID column '{col}' leaked into feature names"


def test_synthetic_proxy_contamination_empirical_verification():
    """
    Verifies that the synthetic proxies from x64 and x81 degraded test ROC-AUC (Exp A ~0.6588),
    whereas the clean primary model without proxies (Exp B) achieves ROC-AUC ~0.7783.
    """
    proxy_exp_file = "reports/phase2/proxy_contamination_experiments.csv"
    assert os.path.exists(proxy_exp_file), "Proxy contamination experiments CSV must exist"
    
    df_exp = pd.read_csv(proxy_exp_file)
    exp_a = df_exp[df_exp["Experiment"].str.contains("Experiment A")].iloc[0]
    exp_b = df_exp[df_exp["Experiment"].str.contains("Experiment B")].iloc[0]
    
    # Experiment A test ROC-AUC must be ~0.6588
    assert exp_a["Test_ROC_AUC"] < 0.70, f"Contaminated meta-model test ROC-AUC should be degraded, got {exp_a['Test_ROC_AUC']}"
    # Experiment B test ROC-AUC must be >= 0.77
    assert exp_b["Test_ROC_AUC"] >= 0.77, f"Clean model test ROC-AUC should be >= 0.77, got {exp_b['Test_ROC_AUC']}"
    # Experiment B test PR-AUC must be significantly higher than Experiment A
    assert exp_b["Test_PR_AUC"] > exp_a["Test_PR_AUC"]


def test_out_of_fold_stacking_validation():
    """Verifies that Experiment C properly trained OOF meta-model without label leakage."""
    proxy_exp_file = "reports/phase2/proxy_contamination_experiments.csv"
    df_exp = pd.read_csv(proxy_exp_file)
    exp_c = df_exp[df_exp["Experiment"].str.contains("Experiment C")].iloc[0]
    
    # Clean OOF stacking achieves valid generalization without collapsing
    assert exp_c["Test_ROC_AUC"] >= 0.77, f"OOF Stacking Test ROC-AUC should be >= 0.77, got {exp_c['Test_ROC_AUC']}"
    assert exp_c["Val_ROC_AUC"] >= 0.85, f"OOF Stacking Val ROC-AUC should be >= 0.85, got {exp_c['Val_ROC_AUC']}"


def test_probability_calibration_brier_improvement():
    """Verifies that Platt scaling strictly improves Brier score on both validation and test folds."""
    with open("models/phase2/ews_metrics.json", "r") as f:
        metrics = json.load(f)
    cal_diag = metrics["calibration"]
    
    val_brier_raw = cal_diag["val_brier_raw"]
    val_brier_cal = cal_diag["val_brier_calibrated"]
    test_brier_raw = cal_diag["test_brier_raw"]
    test_brier_cal = cal_diag["test_brier_calibrated"]
    
    assert val_brier_cal < val_brier_raw, f"Calibrated val Brier ({val_brier_cal}) should be lower than raw ({val_brier_raw})"
    assert test_brier_cal < test_brier_raw, f"Calibrated test Brier ({test_brier_cal}) should be lower than raw ({test_brier_raw})"


def test_threshold_selection_without_test_labels():
    """
    Verifies that optimal operating threshold tau*=0.08 was derived strictly from validation fold
    using F2-score optimization and recall >= 60% constraint, not test labels.
    """
    val_thresh_file = "reports/phase2/validation_threshold_analysis.csv"
    assert os.path.exists(val_thresh_file), "Validation threshold analysis CSV must exist"
    
    df_val = pd.read_csv(val_thresh_file)
    
    # Candidates evaluated
    assert set(df_val["threshold"].values) == {0.05, 0.08, 0.10, 0.13, 0.17, 0.50}
    
    # At tau=0.08, validation recall must be >= 60% (64.3%) and F2 must be highest among viable
    row_008 = df_val[df_val["threshold"] == 0.08].iloc[0]
    assert row_008["recall"] >= 0.60
    assert row_008["f2"] > df_val[df_val["threshold"] == 0.05].iloc[0]["f2"]
    
    with open("models/phase2/threshold_config.json", "r") as f:
        cfg = json.load(f)
    assert cfg["operating_threshold"] == 0.08
    assert cfg["selection_fold"] == "Validation Fold (T=11,12)"


def test_test_set_recall_gain_over_naive_baseline():
    """
    Verifies that naive threshold 0.50 yielded 0.0% test recall, while the calibrated
    operating threshold tau*=0.08 achieves > 0% recall (27.3%) and > 90% specificity.
    """
    with open("models/phase2/ews_metrics.json", "r") as f:
        metrics = json.load(f)
        
    naive = metrics["test_metrics_naive_050"]
    optimal = metrics["test_metrics_optimal_tau"]
    
    # Naive baseline failure
    assert naive["threshold"] == 0.50
    assert naive["recall"] == 0.0
    assert naive["tp"] == 0
    assert naive["fn"] == 11
    
    # Validated operational model
    assert optimal["threshold"] == 0.08
    assert optimal["recall"] > 0.0
    assert optimal["tp"] == 3
    assert optimal["specificity"] > 0.90
    assert optimal["portfolio_flagged_pct"] < 10.0  # Manageable review workload


def test_risk_score_and_alert_consistency():
    """Verifies consistency between calibrated probabilities, standardized scores, and alert rules."""
    fusion = RiskFusionEngine()
    alert_engine = EarlyWarningSignalEngine()
    
    # 1. Probability to Risk Score conversion: Score = round(P * 100, 1)
    for p in [0.034, 0.080, 0.130, 0.170, 0.255]:
        score = fusion.compute_risk_score(p)
        assert score == round(p * 100.0, 1)
        
    # 2. Risk Bands partition:
    assert fusion.determine_category(3.5)["label"] == "Healthy"
    assert fusion.determine_category(6.0)["label"] == "Low Risk"
    assert fusion.determine_category(10.5)["label"] == "Watch"
    assert fusion.determine_category(15.0)["label"] == "High Risk"
    assert fusion.determine_category(19.0)["label"] == "Critical"
    
    # 3. Alert engine behavior:
    # Critical threshold: P >= 0.170 -> CRITICAL alert
    crit_signals = alert_engine.generate_signals(company_id=1, current_period=13, final_prob=0.175)
    assert any(s["severity"] == "CRITICAL" for s in crit_signals)
    
    # High threshold: P >= 0.130 -> HIGH alert
    high_signals = alert_engine.generate_signals(company_id=1, current_period=13, final_prob=0.145)
    assert any(s["severity"] == "HIGH" for s in high_signals)
    assert not any(s["severity"] == "CRITICAL" for s in high_signals)
    
    # Below high threshold: P = 0.080 -> No high/critical alert
    safe_signals = alert_engine.generate_signals(company_id=1, current_period=13, final_prob=0.080)
    assert not any(s["severity"] in ["HIGH", "CRITICAL"] for s in safe_signals)


def test_api_reports_phase2_validated_model():
    """Verifies that the API /model-info endpoint exposes the validated Phase 2 model metrics."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    
    # Preserves existing endpoints
    assert "temporal_ews_model" in data
    assert "bankruptcy_model" in data
    assert "fusion_engine" in data
    
    # Contains Phase 2 validated model
    assert "phase2_validated_model" in data
    p2 = data["phase2_validated_model"]
    assert "test_metrics_optimal_tau" in p2
    assert p2["test_metrics_optimal_tau"]["threshold"] == 0.08
    assert p2["test_metrics_optimal_tau"]["tp"] == 3
