import numpy as np
import pandas as pd
from typing import Dict, List, Any

class RiskTrajectoryEngine:
    """Analyzes longitudinal risk trajectory, trend slope, momentum, and acceleration."""
    
    @staticmethod
    def analyze_trajectory(time_periods: List[int], risk_probabilities: List[float]) -> Dict[str, Any]:
        """
        Evaluates risk trajectory dynamics across historical reporting periods.
        """
        if not risk_probabilities:
            return {
                "trend": "Insufficient Data",
                "risk_change": 0.0,
                "risk_acceleration": 0.0,
                "consecutive_increases": 0,
                "pattern": "Stable",
                "trajectory_history": []
            }
            
        n = len(risk_probabilities)
        trajectory_history = [
            {"period": int(t), "risk_pct": round(p * 100.0, 1)}
            for t, p in zip(time_periods, risk_probabilities)
        ]
        
        if n == 1:
            return {
                "trend": "Baseline Initial Observation",
                "risk_change": 0.0,
                "risk_acceleration": 0.0,
                "consecutive_increases": 0,
                "pattern": "Single Period",
                "trajectory_history": trajectory_history
            }
            
        current_p = risk_probabilities[-1]
        prev_p = risk_probabilities[-2]
        delta_p = current_p - prev_p
        
        # Acceleration (change in delta) if at least 3 periods
        if n >= 3:
            prev_delta = risk_probabilities[-2] - risk_probabilities[-3]
            acceleration = delta_p - prev_delta
        else:
            acceleration = 0.0
            
        # Count consecutive increases
        consecutive_inc = 0
        for i in range(len(risk_probabilities) - 1, 0, -1):
            if risk_probabilities[i] > risk_probabilities[i - 1]:
                consecutive_inc += 1
            else:
                break
                
        # Classify trend
        if delta_p > 0.05:
            trend = "Increasing"
        elif delta_p < -0.05:
            trend = "Decreasing"
        else:
            trend = "Stable"
            
        # Pattern detection
        if consecutive_inc >= 2 and delta_p > 0.15:
            pattern = "Accelerating Deterioration"
        elif consecutive_inc >= 2:
            pattern = "Persistent Deterioration"
        elif delta_p > 0.20:
            pattern = "Sudden Spike"
        elif delta_p < -0.15:
            pattern = "Rapid Recovery"
        else:
            pattern = "Stable Fluctuations"
            
        return {
            "trend": trend,
            "risk_change_pct_points": round(delta_p * 100.0, 1),
            "risk_acceleration": round(acceleration * 100.0, 1),
            "consecutive_increases": consecutive_inc,
            "pattern": pattern,
            "trajectory_history": trajectory_history
        }
