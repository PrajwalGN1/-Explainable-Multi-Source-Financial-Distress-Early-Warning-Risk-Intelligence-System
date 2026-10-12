import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from src.utils.logger import setup_logger

logger = setup_logger("temporal_features")

class TemporalFeatureEngine:
    """Builds leak-free backward-looking temporal and trajectory features for panel data."""
    
    def __init__(self, key_features: List[str] = None, max_lag: int = 2):
        self.key_features = key_features or ["x1", "x2", "x5", "x10", "x25", "x48", "x64", "x81"]
        self.max_lag = max_lag

    def construct_forward_target(self, df: pd.DataFrame, id_col: str = "Company", time_col: str = "Time", 
                                 raw_target_col: str = "Financial Distress", threshold: float = -0.5) -> pd.DataFrame:
        """
        Constructs Forward Distress Target (t+1) without future information leakage.
        y_forward = 1 if Financial Distress at t+1 <= threshold, else 0.
        Rows where t+1 is not available (e.g. final period of surviving firms or terminal distress period) 
        are marked with forward_target = NaN.
        """
        df_sorted = df.sort_values(by=[id_col, time_col]).copy()
        
        # Binary indicator at current time
        df_sorted["current_distress"] = (df_sorted[raw_target_col] <= threshold).astype(int)
        
        # Forward shift target within company
        df_sorted["forward_target"] = df_sorted.groupby(id_col)["current_distress"].shift(-1)
        
        # Also compute 2-period forward target (distress within t+1 or t+2)
        shift_2 = df_sorted.groupby(id_col)["current_distress"].shift(-2)
        df_sorted["forward_target_2period"] = np.where(
            df_sorted["forward_target"].isna(),
            np.nan,
            np.where((df_sorted["forward_target"] == 1) | (shift_2 == 1), 1, 0)
        )
        
        return df_sorted

    def generate_temporal_features(self, df: pd.DataFrame, id_col: str = "Company", time_col: str = "Time") -> pd.DataFrame:
        """
        Extracts lags, deltas, momentum, rolling backward statistics, and deterioration streaks.
        Guarantees that all features at time t use ONLY observations at or before time t.
        """
        df_sorted = df.sort_values(by=[id_col, time_col]).copy()
        engineered = []
        
        for feat in self.key_features:
            if feat not in df_sorted.columns:
                continue
                
            grouped = df_sorted.groupby(id_col)[feat]
            
            # 1. Lags
            for l in range(1, self.max_lag + 1):
                df_sorted[f"{feat}_lag_{l}"] = grouped.shift(l)
                
            # 2. Difference (current - lag1)
            diff_col = f"{feat}_diff_1"
            df_sorted[diff_col] = df_sorted[feat] - df_sorted[f"{feat}_lag_1"]
            
            # 3. Growth rate (safe division, clipped)
            denom = df_sorted[f"{feat}_lag_1"].abs() + 1e-4
            growth_col = f"{feat}_growth_1"
            df_sorted[growth_col] = (df_sorted[diff_col] / denom).clip(-5.0, 5.0)
            
            # 4. Acceleration (diff_1 - diff_lag1)
            diff_lag1 = df_sorted[f"{feat}_lag_1"] - grouped.shift(2)
            df_sorted[f"{feat}_acceleration"] = df_sorted[diff_col] - diff_lag1
            
            # 5. Backward Rolling Mean & Volatility (window=2 and window=3)
            # Using rolling on past values:
            roll_mean_2 = grouped.transform(lambda s: s.rolling(window=2, min_periods=1).mean())
            roll_mean_3 = grouped.transform(lambda s: s.rolling(window=3, min_periods=1).mean())
            roll_std_3 = grouped.transform(lambda s: s.rolling(window=3, min_periods=2).std()).fillna(0.0)
            
            df_sorted[f"{feat}_roll_mean_2"] = roll_mean_2
            df_sorted[f"{feat}_roll_mean_3"] = roll_mean_3
            df_sorted[f"{feat}_roll_std_3"] = roll_std_3
            
            # 6. Consecutive Deterioration Indicator
            # For features where negative diff is bad (e.g. declining x81, x48)
            # Consecutive 2-period drop: (diff_1 < 0) and (diff_lag1 < 0)
            df_sorted[f"{feat}_consec_decline_2"] = (
                (df_sorted[diff_col] < 0) & (diff_lag1 < 0)
            ).astype(int)
            
            # 7. Baseline deviation: current value minus expanding historical mean of past periods
            past_expanding_mean = grouped.transform(lambda s: s.shift(1).expanding().mean())
            df_sorted[f"{feat}_dev_from_baseline"] = df_sorted[feat] - past_expanding_mean
            
        logger.info(f"Temporal features generated. Total columns now: {df_sorted.shape[1]}")
        return df_sorted
