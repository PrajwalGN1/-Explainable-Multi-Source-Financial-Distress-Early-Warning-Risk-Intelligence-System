import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from src.config.settings import get_risk_config

def _hex_to_rgba(hex_code: str, alpha: float = 0.25) -> str:
    """Helper to convert hex color code into rgba string."""
    hex_code = hex_code.lstrip("#")
    if len(hex_code) == 6:
        r = int(hex_code[0:2], 16)
        g = int(hex_code[2:4], 16)
        b = int(hex_code[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    return f"rgba(239, 68, 68, {alpha})"

def get_canonical_risk_bands(risk_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Returns canonical risk score bands from risk_config.yaml.
    Guarantees contiguous thresholds matching the risk scoring engine.
    """
    if risk_config is None:
        try:
            risk_config = get_risk_config()
        except Exception:
            risk_config = {}
    cats = risk_config.get("risk_categories", {})
    defaults = [
        ("healthy", 0.0, 4.0, "Healthy", "#10b981"),
        ("low", 4.0, 8.0, "Low Risk", "#3b82f6"),
        ("watch", 8.0, 13.0, "Watch", "#f59e0b"),
        ("high", 13.0, 17.0, "High Risk", "#f97316"),
        ("critical", 17.0, 100.0, "Critical", "#ef4444"),
    ]
    bands = []
    for key, d_min, d_max, d_label, d_color in defaults:
        conf = cats.get(key, {})
        b_min = float(conf.get("min", d_min))
        b_max = float(conf.get("max", d_max))
        b_label = str(conf.get("label", d_label))
        b_color = str(conf.get("color", d_color))
        bands.append({
            "key": key,
            "min": b_min,
            "max": b_max,
            "label": b_label,
            "color": b_color,
        })
    return bands

def create_gauge_chart(score: float, category: str, color: str = "#ef4444", risk_config: Optional[Dict[str, Any]] = None) -> go.Figure:
    """Builds an interactive radial gauge chart with bands synchronized to canonical risk config."""
    bands = get_canonical_risk_bands(risk_config)
    steps = [
        {'range': [b["min"], b["max"]], 'color': _hex_to_rgba(b["color"], 0.25)}
        for b in bands
    ]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>Risk Score: {category.upper()}</b>", 'font': {'size': 18, 'color': '#f8fafc'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8", 'tickfont': {'color': '#94a3b8'}},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "rgba(30, 41, 59, 0.5)",
            'borderwidth': 1,
            'bordercolor': "rgba(255, 255, 255, 0.1)",
            'steps': steps,
            'threshold': {
                'line': {'color': "#ffffff", 'width': 3},
                'thickness': 0.8,
                'value': score
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': '#f8fafc', 'family': 'Inter, sans-serif'},
        height=260,
        margin=dict(l=25, r=25, t=40, b=15)
    )
    return fig

def create_timeline_chart(trajectory_history: List[Dict[str, Any]], signals: List[Dict[str, Any]] = None, risk_config: Optional[Dict[str, Any]] = None) -> go.Figure:
    """Builds multi-period interactive timeline chart overlayed with warning signals and canonical risk bands."""
    df_traj = pd.DataFrame(trajectory_history)
    fig = go.Figure()
    
    # Background threshold bands synchronized with canonical risk config
    bands = get_canonical_risk_bands(risk_config)
    for b in bands:
        fig.add_hrect(
            y0=b["min"],
            y1=b["max"],
            fillcolor=b["color"],
            opacity=0.08,
            line_width=0,
            annotation_text=b["label"],
            annotation_position="top left",
            annotation_font_color=b["color"]
        )
    
    # Trajectory line
    fig.add_trace(go.Scatter(
        x=df_traj["period"],
        y=df_traj["risk_pct"],
        mode="lines+markers",
        name="Risk Probability (%)",
        line=dict(color="#38bdf8", width=3, shape="spline"),
        marker=dict(size=8, color="#38bdf8", symbol="circle")
    ))
    
    # Overlay alert markers if present on latest period
    if signals and len(df_traj) > 0:
        latest_period = df_traj["period"].iloc[-1]
        latest_risk = df_traj["risk_pct"].iloc[-1]
        fig.add_trace(go.Scatter(
            x=[latest_period],
            y=[latest_risk],
            mode="markers",
            name="Warning Triggered",
            marker=dict(size=14, color="#ef4444", symbol="star")
        ))
        
    fig.update_layout(
        title="<b>Multi-Period Risk Trajectory & Surveillance Timeline</b>",
        xaxis=dict(title="Reporting Period (Time)", gridcolor="rgba(255,255,255,0.08)", tickmode="linear"),
        yaxis=dict(title="Distress Risk Probability (%)", range=[0, 100], gridcolor="rgba(255,255,255,0.08)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        font=dict(color="#f8fafc", family="Inter, sans-serif"),
        height=380,
        margin=dict(l=40, r=30, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_health_radar_chart(health_breakdown: Dict[str, float]) -> go.Figure:
    """Builds a radar/spider chart of dimensional financial health scores vs benchmark."""
    categories = ['Profitability', 'Liquidity', 'Solvency', 'Leverage', 'Working Capital']
    company_scores = [
        health_breakdown.get('profitability_score', 50.0),
        health_breakdown.get('liquidity_score', 50.0),
        health_breakdown.get('solvency_score', 50.0),
        health_breakdown.get('leverage_score', 50.0),
        health_breakdown.get('working_capital_score', 50.0)
    ]
    benchmark_scores = [70, 75, 70, 75, 65]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=benchmark_scores + [benchmark_scores[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor="rgba(148, 163, 184, 0.15)",
        name='Benchmark Median',
        line=dict(color='#94a3b8', dash='dot')
    ))
    fig.add_trace(go.Scatterpolar(
        r=company_scores + [company_scores[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor="rgba(56, 189, 248, 0.3)",
        name='Company Profile',
        line=dict(color='#38bdf8', width=2)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.1)", tickfont=dict(color="#94a3b8")),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickfont=dict(color="#f8fafc"))
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        height=330,
        margin=dict(l=35, r=35, t=30, b=30),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig

def create_waterfall_chart(drivers: List[Dict[str, Any]], protective: List[Dict[str, Any]]) -> go.Figure:
    """Builds a waterfall chart of local SHAP feature contributions pushing risk up or down."""
    x_labels = []
    y_values = []
    measures = []
    
    for d in drivers[:4]:
        x_labels.append(d["feature"])
        y_values.append(d["shap_impact"])
        measures.append("relative")
        
    for p in protective[:4]:
        x_labels.append(p["feature"])
        y_values.append(p["shap_impact"])
        measures.append("relative")
        
    fig = go.Figure(go.Waterfall(
        name="SHAP Contributions",
        orientation="v",
        measure=measures,
        x=x_labels,
        y=y_values,
        connector={"line": {"color": "rgba(255,255,255,0.2)"}},
        decreasing={"marker": {"color": "#10b981"}},
        increasing={"marker": {"color": "#ef4444"}}
    ))
    
    fig.update_layout(
        title="<b>Local SHAP Feature Contributions to Distress Risk</b>",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        font=dict(color="#f8fafc"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickangle=-30),
        yaxis=dict(title="SHAP Impact (Log-Odds)", gridcolor="rgba(255,255,255,0.08)"),
        height=360,
        margin=dict(l=40, r=30, t=50, b=60)
    )
    return fig
