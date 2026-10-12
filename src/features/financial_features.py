import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from src.utils.logger import setup_logger

logger = setup_logger("financial_features")

class FinancialHealthEngine:
    """Calculates interpretable domain financial health scores (0-100) across balance sheet dimensions."""
    
    def __init__(self):
        # Benchmark reference bounds (5th and 95th percentiles from healthy firms in Dataset B / standard finance)
        self.bounds = {
            "roa": (0.0, 0.25),                # higher is better
            "operating_margin": (0.0, 0.40),   # higher is better
            "current_ratio": (0.8, 3.5),       # higher is better
            "quick_ratio": (0.5, 2.5),         # higher is better
            "debt_ratio": (0.1, 0.75),         # lower is better
            "interest_coverage": (1.0, 15.0),  # higher is better
            "working_capital_ratio": (0.0, 0.5)# higher is better
        }

    def _normalize_metric(self, value: float, low_ref: float, high_ref: float, higher_is_better: bool = True) -> float:
        """Normalizes a metric to 0-100 scale using continuous piecewise linear clamping."""
        if pd.isna(value) or np.isinf(value):
            return 50.0
            
        clipped = np.clip(value, low_ref, high_ref)
        pct = (clipped - low_ref) / (high_ref - low_ref + 1e-8)
        score = pct * 100.0 if higher_is_better else (1.0 - pct) * 100.0
        return float(np.clip(score, 0.0, 100.0))

    def compute_health_from_ratios(self, roa: float, operating_margin: float, current_ratio: float, 
                                  quick_ratio: float, debt_ratio: float, interest_coverage: float = 5.0,
                                  working_capital_ratio: float = 0.2) -> Dict[str, float]:
        """Calculates dimensional financial scores from canonical ratios."""
        p_roa = self._normalize_metric(roa, self.bounds["roa"][0], self.bounds["roa"][1], higher_is_better=True)
        p_margin = self._normalize_metric(operating_margin, self.bounds["operating_margin"][0], self.bounds["operating_margin"][1], higher_is_better=True)
        profitability_score = 0.6 * p_roa + 0.4 * p_margin
        
        l_curr = self._normalize_metric(current_ratio, self.bounds["current_ratio"][0], self.bounds["current_ratio"][1], higher_is_better=True)
        l_quick = self._normalize_metric(quick_ratio, self.bounds["quick_ratio"][0], self.bounds["quick_ratio"][1], higher_is_better=True)
        liquidity_score = 0.5 * l_curr + 0.5 * l_quick
        
        # Leverage: higher debt is riskier, so lower score
        leverage_score = self._normalize_metric(debt_ratio, self.bounds["debt_ratio"][0], self.bounds["debt_ratio"][1], higher_is_better=False)
        
        # Solvency: interest coverage
        solvency_score = self._normalize_metric(interest_coverage, self.bounds["interest_coverage"][0], self.bounds["interest_coverage"][1], higher_is_better=True)
        
        # Working capital
        wc_score = self._normalize_metric(working_capital_ratio, self.bounds["working_capital_ratio"][0], self.bounds["working_capital_ratio"][1], higher_is_better=True)
        
        # Composite Financial Health Score (0 = extreme distress, 100 = pristine health)
        financial_health_score = (
            0.30 * profitability_score +
            0.25 * liquidity_score +
            0.20 * leverage_score +
            0.15 * solvency_score +
            0.10 * wc_score
        )
        
        return {
            "profitability_score": round(profitability_score, 1),
            "liquidity_score": round(liquidity_score, 1),
            "leverage_score": round(leverage_score, 1),
            "solvency_score": round(solvency_score, 1),
            "working_capital_score": round(wc_score, 1),
            "financial_health_score": round(financial_health_score, 1),
            "financial_health_risk": round(1.0 - (financial_health_score / 100.0), 3)
        }

    def compute_for_dataset_a_row(self, row: pd.Series) -> Dict[str, float]:
        """
        Calculates financial health proxy for Dataset A observations where features are x1..x83.
        Uses key normalized discriminating factors (x81, x48, x25, x64).
        """
        # x81 and x48 have positive correlation with healthy distress index (protective)
        # x64 has negative correlation (elevates risk)
        val_x81 = row.get("x81", 0.0)
        val_x48 = row.get("x48", 0.0)
        val_x25 = row.get("x25", 0.0)
        val_x64 = row.get("x64", 0.0)
        
        # Sigmoid-normalized score proxies with safe clipping to prevent overflow
        p_score = 100.0 / (1.0 + np.exp(np.clip(-1.5 * val_x48, -50.0, 50.0)))
        l_score = 100.0 / (1.0 + np.exp(np.clip(-1.5 * val_x25, -50.0, 50.0)))
        lev_score = 100.0 / (1.0 + np.exp(np.clip(1.5 * val_x64, -50.0, 50.0))) # higher x64 reduces leverage score
        solv_score = 100.0 / (1.0 + np.exp(np.clip(-1.5 * val_x81, -50.0, 50.0)))
        wc_score = (l_score + p_score) / 2.0
        
        composite = 0.35 * solv_score + 0.25 * p_score + 0.20 * lev_score + 0.20 * l_score
        composite = float(np.clip(composite, 0.0, 100.0))
        
        return {
            "profitability_score": round(float(np.clip(p_score, 0, 100)), 1),
            "liquidity_score": round(float(np.clip(l_score, 0, 100)), 1),
            "leverage_score": round(float(np.clip(lev_score, 0, 100)), 1),
            "solvency_score": round(float(np.clip(solv_score, 0, 100)), 1),
            "working_capital_score": round(float(np.clip(wc_score, 0, 100)), 1),
            "financial_health_score": round(composite, 1),
            "financial_health_risk": round(1.0 - (composite / 100.0), 3)
        }
