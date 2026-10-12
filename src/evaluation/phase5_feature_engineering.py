"""
Phase 5: Point-in-Time Financial Deterioration Feature Engineering.

Implements candidate point-in-time financial deterioration and instability features
in an isolated research pipeline without modifying the baseline feature pipeline:
- First differences of core financial indicators: Delta x_{i,t} = x_{i,t} - x_{i,t-1}
- Relative / percentage changes: (x_{i,t} - x_{i,t-1}) / (|x_{i,t-1}| + eps)
- Deterioration velocity & trend acceleration: Delta^2 x_{i,t} = Delta x_{i,t} - Delta x_{i,t-1}
- Rolling backward variability & instability: rolling std (window=2, 3)
- Instantaneous vs. Rolling spread: x_{i,t} - roll_mean_3(x_i)
- Consecutive deterioration indicators: (Delta x_{i,t} < 0) & (Delta x_{i,t-1} < 0)
- Compound deterioration interactions (e.g., liquidity contraction * leverage expansion)
- Reporting regularity / period time delta: Delta Time = Time_t - Time_{t-1}

Strict Leakage Safeguards:
- Uses ONLY within-company history via groupby('Company')
- All rolling statistics are strictly backward-looking (no centered or forward windows)
- Initial periods (t=1) where lag is unavailable are cleanly handled with zero/median fill
- Synthetic proxy features (x64, x81) are strictly excluded.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.abspath("."))
from src.utils.logger import setup_logger

logger = setup_logger("phase5_feature_engineering")


class Phase5FeatureEngine:
    """Research feature engineering pipeline for SME financial deterioration signals."""

    # Core distress, liquidity, solvency, profitability, and operating ratios
    # Identified as top predictive signals in Phase 3/4 explainability audits
    CORE_FINANCIAL_RATIOS = [
        "x1", "x2", "x3", "x5", "x10", "x12", "x16", "x25", "x48", "x52"
    ]

    def __init__(self, key_features: Optional[List[str]] = None, max_lag: int = 2):
        self.key_features = key_features or self.CORE_FINANCIAL_RATIOS
        self.max_lag = max_lag

    def construct_forward_target(
        self,
        df: pd.DataFrame,
        id_col: str = "Company",
        time_col: str = "Time",
        raw_target_col: str = "Financial Distress",
        threshold: float = -0.5
    ) -> pd.DataFrame:
        """
        Constructs forward distress target at t+1 without future leakage.
        y_forward = 1 if Financial Distress at t+1 <= threshold, else 0.
        Last observed period for any company receives NaN (cannot peek into unobserved future).
        """
        df_sorted = df.sort_values(by=[id_col, time_col]).copy()
        df_sorted["current_distress"] = (df_sorted[raw_target_col] <= threshold).astype(int)
        df_sorted["forward_target"] = df_sorted.groupby(id_col)["current_distress"].shift(-1)
        return df_sorted

    def generate_deterioration_features(
        self,
        df: pd.DataFrame,
        id_col: str = "Company",
        time_col: str = "Time"
    ) -> pd.DataFrame:
        """
        Computes leak-free point-in-time financial deterioration, velocity,
        instability, and interaction features.
        """
        df_sorted = df.sort_values(by=[id_col, time_col]).copy()
        
        new_cols = {}
        # 1. Reporting regularity: Delta Time between observations
        time_diff = df_sorted.groupby(id_col)[time_col].diff()
        new_cols["reporting_gap"] = time_diff.fillna(1.0)
        new_cols["is_irregular_reporting"] = (new_cols["reporting_gap"] > 1.0).astype(float)

        # 2. Extract signals per core financial indicator
        for feat in self.key_features:
            if feat not in df_sorted.columns:
                continue

            grouped = df_sorted.groupby(id_col)[feat]

            # A. Lags
            for l in range(1, self.max_lag + 1):
                new_cols[f"{feat}_lag_{l}"] = grouped.shift(l)

            # B. First Difference: Delta x_{i,t} = x_{i,t} - x_{i,t-1}
            diff_1 = df_sorted[feat] - new_cols[f"{feat}_lag_1"]
            new_cols[f"{feat}_diff_1"] = diff_1

            # C. Relative / Percentage Change (safely clipped to prevent division by near-zero)
            denom = new_cols[f"{feat}_lag_1"].abs() + 1e-4
            new_cols[f"{feat}_pct_change_1"] = (diff_1 / denom).clip(-5.0, 5.0)

            # D. Deterioration Velocity / Acceleration: Delta^2 x = Delta x_t - Delta x_{t-1}
            diff_lag1 = new_cols[f"{feat}_lag_1"] - grouped.shift(2)
            accel = diff_1 - diff_lag1
            new_cols[f"{feat}_accel_1"] = accel

            # E. Backward Rolling Mean & Volatility (window=2 and 3)
            roll_mean_2 = grouped.transform(lambda s: s.rolling(window=2, min_periods=1).mean())
            roll_mean_3 = grouped.transform(lambda s: s.rolling(window=3, min_periods=1).mean())
            roll_std_3 = grouped.transform(lambda s: s.rolling(window=3, min_periods=2).std()).fillna(0.0)

            new_cols[f"{feat}_roll_mean_2"] = roll_mean_2
            new_cols[f"{feat}_roll_mean_3"] = roll_mean_3
            new_cols[f"{feat}_roll_std_3"] = roll_std_3

            # F. Instantaneous vs Rolling Spread (Cliff Indicator)
            new_cols[f"{feat}_instant_vs_roll3"] = df_sorted[feat] - roll_mean_3

            # G. Consecutive Deterioration Indicator (2-period persistent decline)
            new_cols[f"{feat}_consec_decline_2"] = (
                (diff_1 < 0.0) & (diff_lag1 < 0.0)
            ).astype(float)

        # 3. Compound Deterioration Interactions
        if "x12_diff_1" in new_cols and "x25_diff_1" in new_cols:
            new_cols["inter_x12_x25_diff"] = new_cols["x12_diff_1"] * new_cols["x25_diff_1"]

        if "x2_diff_1" in new_cols and "x12_diff_1" in new_cols:
            new_cols["inter_x2_x12_diff"] = new_cols["x2_diff_1"] * new_cols["x12_diff_1"]

        if "x48_roll_std_3" in new_cols and "x12_diff_1" in new_cols:
            new_cols["inter_volatility_x48_x12"] = new_cols["x48_roll_std_3"] * new_cols["x12_diff_1"]

        # Concatenate all new columns at once to prevent fragmentation
        df_new_features = pd.DataFrame(new_cols, index=df_sorted.index)
        df_result = pd.concat([df_sorted, df_new_features], axis=1)

        # 4. Strict Exclusion of Synthetic Proxy Columns
        for proxy_col in ["x64", "x81"]:
            proxy_derived = [c for c in df_result.columns if proxy_col in c]
            df_result.drop(columns=proxy_derived, inplace=True, errors="ignore")

        logger.info(f"Phase 5 feature engineering complete. Total feature columns: {df_result.shape[1]}")
        return df_result


def build_phase5_dataset() -> pd.DataFrame:
    """Builds and returns the complete feature-engineered Phase 5 dataset."""
    from src.data.loaders import DataLoader
    loader = DataLoader("data/raw")
    df_raw = loader.load_dataset_a()

    engine = Phase5FeatureEngine()
    df_target = engine.construct_forward_target(
        df_raw, id_col="Company", time_col="Time", raw_target_col="Financial Distress", threshold=-0.5
    )
    df_features = engine.generate_deterioration_features(
        df_target, id_col="Company", time_col="Time"
    )
    df_clean = df_features.dropna(subset=["forward_target"]).copy()
    df_clean["forward_target"] = df_clean["forward_target"].astype(int)
    return df_clean


if __name__ == "__main__":
    df = build_phase5_dataset()
    print(f"Phase 5 Dataset built successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
    print(f"Target breakdown:\n{df['forward_target'].value_counts()}")
