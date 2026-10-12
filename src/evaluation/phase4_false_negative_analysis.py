"""
Phase 4: Detailed False-Negative Analysis & Test Distress Audit.

Audits the 8 test-set false negatives and compares them with the 3 true positives
on the held-out test cohort (T=13, N=262, N_pos=11):
- Company identifiers and observation/target periods
- Raw RF probabilities, Platt-calibrated probabilities, risk scores, and categories
- Ranks among all 262 test instances
- SHAP feature attribution breakdown (top drivers vs top protective factors)
- Minimum alerting threshold required per case
- Empirical associations (threshold restrictiveness, weak separation, temporal noise)
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.data.loaders import DataLoader
from src.features.temporal_features import TemporalFeatureEngine
from src.preprocessing.preprocessors import EWSPreprocessor
from src.fusion.risk_fusion import RiskFusionEngine
from src.explainability.tree_shap_explainer import NativeTreeSHAPExplainer
from src.utils.logger import setup_logger

logger = setup_logger("phase4_fn_analysis")


def run_false_negative_analysis() -> Dict[str, Any]:
    os.makedirs("results/phase4", exist_ok=True)
    os.makedirs("experiments/phase4", exist_ok=True)

    # 1. Load Data
    logger.info("Loading Dataset A via DataLoader...")
    loader = DataLoader("data/raw")
    df_a = loader.load_dataset_a()

    # 2. Forward target and temporal features
    engine = TemporalFeatureEngine()
    df_target = engine.construct_forward_target(
        df_a, id_col="Company", time_col="Time", raw_target_col="Financial Distress", threshold=-0.5
    )
    df_feats = engine.generate_temporal_features(df_target, id_col="Company", time_col="Time")
    df_valid = df_feats.dropna(subset=["forward_target"]).copy()
    df_valid["forward_target"] = df_valid["forward_target"].astype(int)

    test_df = df_valid[df_valid["Time"] > 12].copy()  # T = 13
    logger.info(f"Test cohort: {len(test_df)} observations, {test_df['forward_target'].sum()} positive events")

    # 3. Load Frozen Phase 2 Approved Model Artifacts (ZERO Retraining)
    preprocessor = EWSPreprocessor.load("models/phase2/ews_preprocessor.joblib")
    model = joblib.load("models/phase2/best_ews_model.joblib")
    calibrator = joblib.load("models/phase2/ews_calibrator.joblib")
    fusion = RiskFusionEngine()
    explainer = NativeTreeSHAPExplainer()

    with open("models/phase2/threshold_config.json", "r", encoding="utf-8") as f:
        thresh_cfg = json.load(f)
    operating_threshold = float(thresh_cfg.get("operating_threshold", 0.08))

    # 4. Transform Test Set and Predict
    X_test, y_test = preprocessor.transform(test_df)
    raw_rf_probs = model.predict_proba(X_test)[:, 1]
    cal_probs = calibrator.predict_proba(X_test)[:, 1]

    test_df["raw_rf_prob"] = raw_rf_probs
    test_df["calibrated_prob"] = cal_probs
    test_df["risk_score"] = [fusion.compute_risk_score(p) for p in cal_probs]
    test_df["risk_category"] = [fusion.determine_category(s)["label"] for s in test_df["risk_score"]]
    test_df["alert_triggered"] = test_df["calibrated_prob"] >= operating_threshold
    test_df["rank_in_test"] = test_df["calibrated_prob"].rank(ascending=False, method="min").astype(int)

    # Classify cases
    test_df["outcome_class"] = "TN"
    test_df.loc[(test_df["forward_target"] == 1) & (test_df["alert_triggered"]), "outcome_class"] = "TP"
    test_df.loc[(test_df["forward_target"] == 1) & (~test_df["alert_triggered"]), "outcome_class"] = "FN"
    test_df.loc[(test_df["forward_target"] == 0) & (test_df["alert_triggered"]), "outcome_class"] = "FP"

    tp_df = test_df[test_df["outcome_class"] == "TP"].sort_values("calibrated_prob", ascending=False)
    fn_df = test_df[test_df["outcome_class"] == "FN"].sort_values("calibrated_prob", ascending=False)
    fp_df = test_df[test_df["outcome_class"] == "FP"].sort_values("calibrated_prob", ascending=False)

    logger.info(f"Test Classifications: TP={len(tp_df)}, FN={len(fn_df)}, FP={len(fp_df)}, TN={(test_df['outcome_class'] == 'TN').sum()}")

    # 5. Detailed Case Audit of all 11 positive events (3 TP + 8 FN)
    cases_audit = []
    feature_names = preprocessor.feature_names

    for idx, row in test_df[test_df["forward_target"] == 1].sort_values("calibrated_prob", ascending=False).iterrows():
        cid = int(row["Company"])
        obs_time = int(row["Time"])
        target_time = obs_time + 1
        raw_p = float(row["raw_rf_prob"])
        cal_p = float(row["calibrated_prob"])
        score = float(row["risk_score"])
        cat = str(row["risk_category"])
        alert = bool(row["alert_triggered"])
        rank = int(row["rank_in_test"])
        outcome = str(row["outcome_class"])

        # Row position in X_test
        row_pos = test_df.index.get_loc(idx)
        x_row = X_test[row_pos]

        # Compute SHAP explanation
        exp = explainer.explain_transformed_row(x_row, top_n=5)
        top_drivers = exp.get("top_risk_drivers", [])
        top_protective = exp.get("top_protective_factors", [])

        driver_summary = "; ".join([f"{d['feature']} (+{d['shap_impact']:.3f})" for d in top_drivers[:3]])
        protective_summary = "; ".join([f"{p['feature']} ({p['shap_impact']:.3f})" for p in top_protective[:3]])

        # Key underlying financial ratios
        x1_val = float(row.get("x1", np.nan))
        x2_val = float(row.get("x2", np.nan))
        x8_val = float(row.get("x8", np.nan))
        x12_val = float(row.get("x12", np.nan))

        # Diagnostic classification for the miss
        if outcome == "FN":
            if cal_p >= 0.050:
                miss_mechanism = "Near-Miss: Probability elevated (5.0%-7.9%) but suppressed below 0.080 threshold"
            elif cal_p >= 0.020:
                miss_mechanism = "Moderate Suppression: Strong protective factors masked distress signal"
            else:
                miss_mechanism = "Severe Attenuation: Model saw company as healthy; zero early warning detected"
        else:
            miss_mechanism = "True Detection: Risk drivers successfully overcame protective baseline"

        cases_audit.append({
            "company_id": cid,
            "observation_period": obs_time,
            "target_period": target_time,
            "outcome_class": outcome,
            "actual_label": 1,
            "raw_rf_prob": round(raw_p, 4),
            "calibrated_prob": round(cal_p, 4),
            "risk_score": score,
            "risk_category": cat,
            "operating_threshold": operating_threshold,
            "alert_triggered": alert,
            "rank_in_test": rank,
            "total_test_observations": len(test_df),
            "percentile_rank": round((1.0 - (rank / len(test_df))) * 100.0, 1),
            "min_threshold_to_alert": round(cal_p, 4),
            "top_risk_drivers": driver_summary,
            "top_protective_factors": protective_summary,
            "miss_mechanism": miss_mechanism,
            "x1_val": round(x1_val, 4) if np.isfinite(x1_val) else None,
            "x2_val": round(x2_val, 4) if np.isfinite(x2_val) else None,
            "x8_val": round(x8_val, 4) if np.isfinite(x8_val) else None,
            "x12_val": round(x12_val, 4) if np.isfinite(x12_val) else None
        })

    df_cases_audit = pd.DataFrame(cases_audit)
    df_cases_audit.to_csv("results/phase4/test_distress_cases_comparison.csv", index=False)

    df_fn_only = df_cases_audit[df_cases_audit["outcome_class"] == "FN"].copy()
    df_fn_only.to_csv("results/phase4/false_negatives_audit.csv", index=False)

    # 6. Comparative Statistical Summary: TP vs FN
    tp_cal_probs = [c["calibrated_prob"] for c in cases_audit if c["outcome_class"] == "TP"]
    fn_cal_probs = [c["calibrated_prob"] for c in cases_audit if c["outcome_class"] == "FN"]
    tn_cal_probs = test_df[test_df["outcome_class"] == "TN"]["calibrated_prob"].tolist()
    fp_cal_probs = test_df[test_df["outcome_class"] == "FP"]["calibrated_prob"].tolist()

    comparison_summary = {
        "true_positives": {
            "count": len(tp_cal_probs),
            "mean_cal_prob": round(float(np.mean(tp_cal_probs)), 4),
            "min_cal_prob": round(float(np.min(tp_cal_probs)), 4),
            "max_cal_prob": round(float(np.max(tp_cal_probs)), 4),
            "company_ids": [c["company_id"] for c in cases_audit if c["outcome_class"] == "TP"]
        },
        "false_negatives": {
            "count": len(fn_cal_probs),
            "mean_cal_prob": round(float(np.mean(fn_cal_probs)), 4),
            "median_cal_prob": round(float(np.median(fn_cal_probs)), 4),
            "min_cal_prob": round(float(np.min(fn_cal_probs)), 4),
            "max_cal_prob": round(float(np.max(fn_cal_probs)), 4),
            "company_ids": [c["company_id"] for c in cases_audit if c["outcome_class"] == "FN"],
            "near_misses_005_to_008": int(sum(1 for p in fn_cal_probs if p >= 0.05)),
            "moderate_misses_002_to_005": int(sum(1 for p in fn_cal_probs if 0.02 <= p < 0.05)),
            "severe_misses_under_002": int(sum(1 for p in fn_cal_probs if p < 0.02))
        },
        "false_positives": {
            "count": len(fp_cal_probs),
            "mean_cal_prob": round(float(np.mean(fp_cal_probs)), 4),
            "median_cal_prob": round(float(np.median(fp_cal_probs)), 4)
        },
        "true_negatives": {
            "count": len(tn_cal_probs),
            "mean_cal_prob": round(float(np.mean(tn_cal_probs)), 4),
            "median_cal_prob": round(float(np.median(tn_cal_probs)), 4)
        }
    }

    with open("results/phase4/false_negatives_summary.json", "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)

    logger.info("Saved false negative audit tables to results/phase4/")
    return {
        "cases_audit": cases_audit,
        "summary": comparison_summary
    }


if __name__ == "__main__":
    run_false_negative_analysis()
