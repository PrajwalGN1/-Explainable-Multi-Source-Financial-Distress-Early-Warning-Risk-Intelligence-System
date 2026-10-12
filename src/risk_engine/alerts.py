import pandas as pd
import numpy as np
from typing import Dict, List, Any

class EarlyWarningSignalEngine:
    """Generates explainable, deterministic early warning signal alerts with clear severity and evidence."""
    
    @staticmethod
    def generate_signals(company_id: any, current_period: int, final_prob: float, 
                         trajectory_info: Dict[str, Any], health_info: Dict[str, Any],
                         key_feature_shifts: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        signals = []
        key_feature_shifts = key_feature_shifts or []
        
        # Rule 1: Critical Absolute Distress Threshold
        if final_prob >= 0.70:
            signals.append({
                "signal_type": "Critical Distress Risk Level",
                "severity": "CRITICAL",
                "evidence": f"Estimated future distress probability reached {final_prob*100:.1f}%.",
                "period": current_period,
                "feature": "Final Risk Model",
                "change": f"{final_prob*100:.1f}%",
                "interpretation": "Firm is exhibiting severe distress signals approaching default/failure boundaries."
            })
        elif final_prob >= 0.50:
            signals.append({
                "signal_type": "Elevated Distress Warning",
                "severity": "HIGH",
                "evidence": f"Estimated future distress probability reached {final_prob*100:.1f}%.",
                "period": current_period,
                "feature": "Final Risk Model",
                "change": f"{final_prob*100:.1f}%",
                "interpretation": "Elevated risk profile warrants inclusion on institutional surveillance watchlist."
            })
            
        # Rule 2: Accelerating or Persistent Deterioration Trajectory
        pattern = trajectory_info.get("pattern", "")
        if pattern == "Accelerating Deterioration":
            signals.append({
                "signal_type": "Accelerating Risk Trajectory",
                "severity": "CRITICAL",
                "evidence": f"Distress risk accelerated with {trajectory_info.get('consecutive_increases', 0)} consecutive increases and +{trajectory_info.get('risk_change_pct_points', 0)} pts jump.",
                "period": current_period,
                "feature": "Risk Trajectory",
                "change": f"+{trajectory_info.get('risk_change_pct_points', 0)}% pts",
                "interpretation": "Deterioration momentum is compounding, signaling compounding financial stress."
            })
        elif pattern == "Persistent Deterioration":
            signals.append({
                "signal_type": "Persistent Deterioration Streak",
                "severity": "HIGH",
                "evidence": f"Distress probability has increased for {trajectory_info.get('consecutive_increases', 0)} consecutive reporting periods.",
                "period": current_period,
                "feature": "Risk Trajectory",
                "change": f"+{trajectory_info.get('risk_change_pct_points', 0)}% pts",
                "interpretation": "Persistent negative trend over multiple cycles indicates structural decline."
            })
            
        # Rule 3: Weak Financial Health Score
        health_score = health_info.get("financial_health_score", 100.0)
        if health_score < 30.0:
            signals.append({
                "signal_type": "Impaired Financial Health Composite",
                "severity": "HIGH",
                "evidence": f"Financial Health Composite index fell to {health_score}/100.",
                "period": current_period,
                "feature": "Financial Health Index",
                "change": f"{health_score}/100",
                "interpretation": "Substantial erosion of underlying solvency and liquidity buffers."
            })
            
        # Rule 4: Severe Liquidity / Solvency Subscore Degradation
        if health_info.get("liquidity_score", 100.0) < 25.0:
            signals.append({
                "signal_type": "Severe Liquidity Compression",
                "severity": "HIGH",
                "evidence": f"Liquidity subscore compressed to {health_info.get('liquidity_score')}/100.",
                "period": current_period,
                "feature": "Liquidity Buffer",
                "change": "Depleted",
                "interpretation": "Short-term debt obligations significantly outpace liquid reserves."
            })
            
        if health_info.get("leverage_score", 100.0) < 25.0:
            signals.append({
                "signal_type": "Excessive Debt Burden",
                "severity": "MEDIUM",
                "evidence": f"Leverage subscore dropped to {health_info.get('leverage_score')}/100 due to high debt reliance.",
                "period": current_period,
                "feature": "Leverage Structure",
                "change": "Overleveraged",
                "interpretation": "Capital structure is overly levered, making the firm vulnerable to cash shocks."
            })
            
        # Rule 5: Specific Detected Feature Shifts
        for shift in key_feature_shifts[:3]:
            if shift.get("shap_delta", 0) > 0.10: # Significant driver of risk jump
                signals.append({
                    "signal_type": f"Significant Deterioration in {shift.get('category', 'Financial Feature')}",
                    "severity": "MEDIUM",
                    "evidence": f"{shift.get('feature')} changed from {shift.get('prev_value')} to {shift.get('curr_value')}.",
                    "period": current_period,
                    "feature": shift.get("feature"),
                    "change": str(shift.get("value_change")),
                    "interpretation": shift.get("interpretation", "Material negative shift observed.")
                })
                
        return signals
