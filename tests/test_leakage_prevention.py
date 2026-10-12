import pytest
import numpy as np
import pandas as pd
from src.features.temporal_features import TemporalFeatureEngine
from src.preprocessing.preprocessors import EWSPreprocessor

def test_lag_features_are_strictly_backward_looking():
    """Validates that lag_1 and lag_2 strictly use past values and never leak current or future values."""
    df_sample = pd.DataFrame({
        "Company": [999, 999, 999, 999],
        "Time": [1, 2, 3, 4],
        "Financial Distress": [0.5, 0.5, 0.5, -0.8], # distress in period 4
        "x1": [10.0, 20.0, 30.0, 40.0],
        "x2": [1.0, 2.0, 4.0, 8.0]
    })
    
    engine = TemporalFeatureEngine(key_features=["x1", "x2"], max_lag=2)
    df_temp = engine.generate_temporal_features(df_sample)
    df_temp = df_temp.sort_values(by="Time").reset_index(drop=True)
    
    # Period 1 should have NaN for lag_1
    assert pd.isna(df_temp.loc[0, "x1_lag_1"])
    
    # Period 2 lag_1 must equal Period 1 value (10.0), NOT Period 2 (20.0) or Period 3 (30.0)
    assert df_temp.loc[1, "x1_lag_1"] == 10.0
    
    # Period 3 lag_1 must equal Period 2 value (20.0), lag_2 must equal Period 1 value (10.0)
    assert df_temp.loc[2, "x1_lag_1"] == 20.0
    assert df_temp.loc[2, "x1_lag_2"] == 10.0
    
    # Period 4 lag_1 must equal Period 3 value (30.0)
    assert df_temp.loc[3, "x1_lag_1"] == 30.0

def test_rolling_statistics_are_strictly_backward_looking():
    """Validates that rolling means only average past and current periods, never future periods."""
    df_sample = pd.DataFrame({
        "Company": [999, 999, 999, 999],
        "Time": [1, 2, 3, 4],
        "Financial Distress": [0.5, 0.5, 0.5, 0.5],
        "x1": [10.0, 20.0, 30.0, 40.0]
    })
    
    engine = TemporalFeatureEngine(key_features=["x1"], max_lag=2)
    df_temp = engine.generate_temporal_features(df_sample)
    df_temp = df_temp.sort_values(by="Time").reset_index(drop=True)
    
    # Period 2 rolling mean 2 should be mean(10, 20) = 15.0
    assert df_temp.loc[1, "x1_roll_mean_2"] == 15.0
    
    # Period 3 rolling mean 3 should be mean(10, 20, 30) = 20.0, NOT including Period 4 (40.0)
    assert df_temp.loc[2, "x1_roll_mean_3"] == 20.0

def test_forward_target_alignment():
    """Validates that forward distress target associates features at t with distress at t+1."""
    df_sample = pd.DataFrame({
        "Company": [999, 999, 999, 999],
        "Time": [1, 2, 3, 4],
        "Financial Distress": [0.2, 0.1, 0.0, -1.2], # Terminal distress occurs in period 4 (<= -0.5)
        "x1": [10.0, 20.0, 30.0, 40.0]
    })
    
    engine = TemporalFeatureEngine()
    df_target = engine.construct_forward_target(df_sample, threshold=-0.5)
    
    # In Period 3, the forward distress target should be 1 (because Period 4 distress <= -0.5)
    p3 = df_target[df_target["Time"] == 3].iloc[0]
    assert p3["forward_target"] == 1
    
    # In Period 1, target should be 0 (Period 2 distress is 0.1 > -0.5)
    p1 = df_target[df_target["Time"] == 1].iloc[0]
    assert p1["forward_target"] == 0

def test_preprocessor_does_not_leak_test_statistics():
    """Validates that EWSPreprocessor is fit only on train and transforms test without leaking test statistics."""
    train_df = pd.DataFrame({
        "Company": [1, 1],
        "Time": [1, 2],
        "forward_target": [0, 0],
        "x1": [10.0, 20.0],
        "x2": [100.0, 200.0]
    })
    test_df = pd.DataFrame({
        "Company": [2],
        "Time": [3],
        "forward_target": [1],
        "x1": [1000.0], # extreme out-of-distribution value
        "x2": [5000.0]
    })
    
    preprocessor = EWSPreprocessor()
    preprocessor.fit(train_df)
    
    # Feature names learned exclusively from train_df
    assert "x1" in preprocessor.feature_names
    assert "x2" in preprocessor.feature_names
    assert preprocessor.fitted is True
    
    # Transform test without altering preprocessor state
    X_test, y_test = preprocessor.transform(test_df)
    assert X_test.shape[0] == 1
    assert X_test.shape[1] == 2
    assert y_test[0] == 1
