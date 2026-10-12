import pandas as pd
from typing import Dict, List, Optional

class FeatureRegistry:
    """Central registry documenting metadata, provenance, categories, and risk direction."""
    
    def __init__(self):
        self.registry: Dict[str, Dict[str, any]] = {}
        self._initialize_core_registry()
        
    def _initialize_core_registry(self):
        # Key ratios from Dataset B
        b_definitions = {
            "ROA(C) before interest and depreciation before interest": {
                "source": "Dataset B (Taiwan)",
                "category": "Profitability",
                "formula": "Operating Income / Total Assets",
                "interpretation": "Asset earning power prior to financing charges",
                "direction_of_risk": -1 # Lower ROA = higher risk
            },
            "ROA(A) before interest and % after tax": {
                "source": "Dataset B (Taiwan)",
                "category": "Profitability",
                "formula": "Net Income / Total Assets",
                "interpretation": "Overall after-tax return on assets",
                "direction_of_risk": -1
            },
            "Operating Gross Margin": {
                "source": "Dataset B (Taiwan)",
                "category": "Profitability",
                "formula": "Gross Profit / Net Revenue",
                "interpretation": "Core product pricing power and cost absorption",
                "direction_of_risk": -1
            },
            "Current Ratio": {
                "source": "Dataset B (Taiwan)",
                "category": "Liquidity",
                "formula": "Current Assets / Current Liabilities",
                "interpretation": "Short-term debt coverage capability",
                "direction_of_risk": -1 # Lower ratio = higher risk
            },
            "Quick Ratio": {
                "source": "Dataset B (Taiwan)",
                "category": "Liquidity",
                "formula": "(Cash + Marketable Sec + AR) / Current Liabilities",
                "interpretation": "Immediate cash-equivalent liquidity",
                "direction_of_risk": -1
            },
            "Debt ratio %": {
                "source": "Dataset B (Taiwan)",
                "category": "Leverage",
                "formula": "Total Debt / Total Assets",
                "interpretation": "Financial leverage and structural insolvency risk",
                "direction_of_risk": 1 # Higher debt = higher risk
            },
            "Net worth/Assets": {
                "source": "Dataset B (Taiwan)",
                "category": "Solvency",
                "formula": "Stockholders Equity / Total Assets",
                "interpretation": "Equity cushion absorbing losses",
                "direction_of_risk": -1
            },
            "Borrowing dependency": {
                "source": "Dataset B (Taiwan)",
                "category": "Leverage",
                "formula": "Interest Bearing Debt / Total Assets",
                "interpretation": "Reliance on external interest-bearing credit",
                "direction_of_risk": 1
            },
            "Working Capital to Total Assets": {
                "source": "Dataset B (Taiwan)",
                "category": "Working Capital",
                "formula": "(Current Assets - Current Liabilities) / Total Assets",
                "interpretation": "Operating buffer relative to asset scale",
                "direction_of_risk": -1
            },
            "Cash flow rate": {
                "source": "Dataset B (Taiwan)",
                "category": "Cash Flow",
                "formula": "Operating Cash Flow / Total Liabilities",
                "interpretation": "Ability of cash operations to service debt",
                "direction_of_risk": -1
            },
            "Persistent EPS in the Last Four Seasons": {
                "source": "Dataset B (Taiwan)",
                "category": "Profitability",
                "formula": "Cumulative 4-Quarter EPS",
                "interpretation": "Earnings persistence and stability",
                "direction_of_risk": -1
            },
            "Interest Coverage Ratio (Interest expense to EBIT)": {
                "source": "Dataset B (Taiwan)",
                "category": "Solvency",
                "formula": "EBIT / Interest Expense",
                "interpretation": "Earnings headroom for debt interest obligations",
                "direction_of_risk": -1
            }
        }
        for k, v in b_definitions.items():
            self.register(k, **v)
            
        # Top correlated variables from Dataset A (Tehran Stock Exchange panel)
        a_definitions = {
            "x81": {
                "source": "Dataset A (Tehran Panel)",
                "category": "Financial/Economic Health",
                "formula": "Reported Financial Indicator x81",
                "interpretation": "High positive correlation with distress index (protective factor)",
                "direction_of_risk": -1
            },
            "x48": {
                "source": "Dataset A (Tehran Panel)",
                "category": "Profitability/Efficiency",
                "formula": "Reported Financial Ratio x48",
                "interpretation": "Positive correlation with solvency score",
                "direction_of_risk": -1
            },
            "x25": {
                "source": "Dataset A (Tehran Panel)",
                "category": "Liquidity/Working Capital",
                "formula": "Reported Financial Ratio x25",
                "interpretation": "Cash/operating stability indicator",
                "direction_of_risk": -1
            },
            "x64": {
                "source": "Dataset A (Tehran Panel)",
                "category": "Financial Vulnerability/Cost",
                "formula": "Reported Financial/Macro Indicator x64",
                "interpretation": "Negative correlation with distress score (elevates distress risk)",
                "direction_of_risk": 1
            },
            "x80": {
                "source": "Dataset A (Tehran Panel)",
                "category": "Industry Sector",
                "formula": "Discrete Industry Code (37 sectors)",
                "interpretation": "Industry sector categorization",
                "direction_of_risk": 0
            }
        }
        for k, v in a_definitions.items():
            self.register(k, **v)

    def register(self, feature_name: str, source: str, category: str, formula: str, interpretation: str, direction_of_risk: int):
        self.registry[feature_name] = {
            "feature_name": feature_name,
            "source": source,
            "category": category,
            "formula": formula,
            "interpretation": interpretation,
            "direction_of_risk": direction_of_risk # +1: higher is riskier, -1: lower is riskier, 0: neutral
        }

    def get(self, feature_name: str) -> Dict[str, any]:
        if feature_name in self.registry:
            return self.registry[feature_name]
        return {
            "feature_name": feature_name,
            "source": "Statistical/Model Feature",
            "category": "Statistical/Unmapped Feature",
            "formula": "Derived or Unmapped Observation",
            "interpretation": f"Statistical predictor {feature_name}",
            "direction_of_risk": 0
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.registry.values()))
