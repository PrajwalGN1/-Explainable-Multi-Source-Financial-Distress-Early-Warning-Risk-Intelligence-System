import pytest
import numpy as np
import pandas as pd
from src.features.feature_registry import FeatureRegistry
from src.features.financial_features import FinancialHealthEngine

def test_feature_registry_lookup():
    registry = FeatureRegistry()
    assert len(registry.registry) > 0
    
    # Test mapped feature from Dataset A
    meta = registry.get("x25")
    assert meta["category"] == "Liquidity/Working Capital"
    assert meta["direction_of_risk"] == -1
    
    # Test mapped feature from Dataset B
    meta_b = registry.get("Current Ratio")
    assert meta_b["category"] == "Liquidity"
    assert meta_b["direction_of_risk"] == -1
    
    # Test unmapped feature fallback
    meta_unmapped = registry.get("x999")
    assert meta_unmapped["category"] == "Statistical/Unmapped Feature"
    assert "x999" in meta_unmapped["interpretation"]

def test_financial_health_score_bounds():
    engine = FinancialHealthEngine()
    
    # Normal case
    scores = engine.compute_health_from_ratios(
        roa=0.12,
        operating_margin=0.20,
        current_ratio=1.8,
        quick_ratio=1.2,
        debt_ratio=0.35,
        interest_coverage=6.0,
        working_capital_ratio=0.25
    )
    assert 0.0 <= scores["financial_health_score"] <= 100.0
    assert 0.0 <= scores["profitability_score"] <= 100.0
    assert 0.0 <= scores["liquidity_score"] <= 100.0
    assert 0.0 <= scores["solvency_score"] <= 100.0
    assert 0.0 <= scores["leverage_score"] <= 100.0
    assert 0.0 <= scores["working_capital_score"] <= 100.0

def test_financial_health_edge_cases_and_clipping():
    engine = FinancialHealthEngine()
    
    # Extreme positive values (should not overflow)
    scores_high = engine.compute_health_from_ratios(
        roa=1.0, operating_margin=1.0, current_ratio=10.0,
        quick_ratio=10.0, debt_ratio=0.01, interest_coverage=50.0, working_capital_ratio=1.0
    )
    assert 0.0 <= scores_high["financial_health_score"] <= 100.0
    
    # Extreme negative/distressed values (should not underflow or crash)
    scores_low = engine.compute_health_from_ratios(
        roa=-0.5, operating_margin=-0.5, current_ratio=0.1,
        quick_ratio=0.05, debt_ratio=1.5, interest_coverage=0.1, working_capital_ratio=-0.5
    )
    assert 0.0 <= scores_low["financial_health_score"] <= 100.0
    
    # Severe distress inputs should yield lower health score than healthy inputs
    assert scores_low["financial_health_score"] < scores_high["financial_health_score"]

def test_financial_health_dataset_a_row():
    engine = FinancialHealthEngine()
    sample_row = pd.Series({"x81": 1.2, "x48": 0.8, "x25": 0.5, "x64": -0.2})
    scores = engine.compute_for_dataset_a_row(sample_row)
    assert 0.0 <= scores["financial_health_score"] <= 100.0
    assert 0.0 <= scores["financial_health_risk"] <= 1.0
