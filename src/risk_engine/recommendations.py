from typing import List, Dict, Any
from src.config.settings import get_risk_config

class RecommendationEngine:
    """Produces prioritized, context-aware risk surveillance recommendations."""
    
    def __init__(self, risk_config: Dict[str, Any] = None):
        self.risk_config = risk_config or get_risk_config()
        self.rec_rules = self.risk_config.get("recommendations_rules", {})

    def generate_recommendations(self, risk_category: str, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        cat_key = risk_category.lower().replace(" ", "_")
        base_recs = self.rec_rules.get(cat_key, [
            "Maintain standard portfolio monitoring protocol.",
            "Verify next financial disclosure upon release."
        ])
        
        specific_actions = []
        for sig in signals:
            stype = sig.get("signal_type", "")
            if "Liquidity" in stype:
                specific_actions.append("Prioritize independent cash-burn and working capital liquidity audit.")
            elif "Trajectory" in stype or "Persistent" in stype:
                specific_actions.append("Convene relationship management review to assess turnaround initiatives.")
            elif "Debt" in stype or "Leverage" in stype:
                specific_actions.append("Audit existing debt covenants, collateral liens, and upcoming refinancing dates.")
                
        # Deduplicate while preserving order
        all_actions = []
        for a in base_recs + specific_actions:
            if a not in all_actions:
                all_actions.append(a)
                
        return {
            "monitoring_tier": risk_category.upper(),
            "recommended_actions": all_actions[:6],
            "disclaimer": "Model-generated surveillance recommendations — for academic/research risk intelligence only. Not automated credit underwriting or banking advice."
        }
