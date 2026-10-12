"""
Phase 3: Global Feature Attribution Generator using Native TreeSHAP.

Computes global SHAP rankings across the holdout test set (T=13, N=262) and validation set (T=11,12, N=517):
- Mean absolute SHAP values (overall feature importance)
- Signed mean contributions (average direction of push)
- Feature standard deviation
- Strict isolation: ZERO model retraining, read-only artifacts
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.data.loaders import DataLoader
from src.features.temporal_features import TemporalFeatureEngine
from src.preprocessing.preprocessors import EWSPreprocessor
from src.explainability.tree_shap_explainer import NativeTreeSHAPExplainer
from src.utils.logger import setup_logger

logger = setup_logger("generate_global_explanations")


def generate_global_explanations() -> Dict[str, Any]:
    os.makedirs("reports/phase3/global_explanations", exist_ok=True)

    # 1. Load Data
    logger.info("Loading Dataset A via DataLoader...")
    loader = DataLoader()
    df_raw = loader.load_dataset_a()

    # 2. Construct Temporal Features
    engine = TemporalFeatureEngine()
    df_target = engine.construct_forward_target(df_raw, id_col="Company", time_col="Time", raw_target_col="Financial Distress", threshold=-0.5)
    df_feats = engine.generate_temporal_features(df_target, id_col="Company", time_col="Time")

    df_valid = df_feats.dropna(subset=["forward_target"]).copy()
    val_df = df_valid[(df_valid["Time"] > 10) & (df_valid["Time"] <= 12)].copy()
    test_df = df_valid[df_valid["Time"] > 12].copy()

    # 3. Load Explainer & Preprocessor (ZERO Retraining)
    explainer = NativeTreeSHAPExplainer()

    # Transform without refit
    X_val, _ = explainer._preprocessor.transform(val_df)
    X_test, _ = explainer._preprocessor.transform(test_df)

    logger.info(f"Computing TreeSHAP global attributions for Holdout Test (T=13, N={len(test_df)})...")
    test_global = explainer.compute_global_importance(X_test, sample_name="Holdout Test (T=13)")

    logger.info(f"Computing TreeSHAP global attributions for Validation Set (T=11,12, N={len(val_df)})...")
    val_global = explainer.compute_global_importance(X_val, sample_name="Validation Fold (T=11,12)")

    # Save CSVs
    df_test_rankings = pd.DataFrame(test_global["rankings"])
    df_test_rankings.to_csv("reports/phase3/global_explanations/global_importance_test.csv", index=False)

    df_val_rankings = pd.DataFrame(val_global["rankings"])
    df_val_rankings.to_csv("reports/phase3/global_explanations/global_importance_val.csv", index=False)

    # Save combined summary JSON
    summary = {
        "model_version": explainer.model_version,
        "explanation_method": explainer.explanation_method,
        "output_space": explainer.output_space,
        "feature_count": len(explainer.feature_names),
        "test_population": {
            "name": "Holdout Test (T=13)",
            "sample_size": len(test_df),
            "top_10_features": test_global["top_10"]
        },
        "validation_population": {
            "name": "Validation Fold (T=11,12)",
            "sample_size": len(val_df),
            "top_10_features": val_global["top_10"]
        },
        "anonymized_feature_policy": (
            "Features x1 through x83 are masked financial indicators from Dataset A. "
            "Temporal features (_lag_, _diff_, _growth_, _roll_mean_, etc.) represent mathematically verified "
            "temporal constructions of their base indicators. No unverified accounting narratives are assumed."
        )
    }

    with open("reports/phase3/global_explanations/global_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved global explanations to reports/phase3/global_explanations/")
    return summary


if __name__ == "__main__":
    generate_global_explanations()
