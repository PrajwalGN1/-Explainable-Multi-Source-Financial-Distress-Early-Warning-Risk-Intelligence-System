import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Dict, Any

def create_gauge_chart(score: float, category: str, color: str = "#ef4444") -> go.Figure:
    """Builds an interactive radial gauge chart for the 0-100 Risk Score."""
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
            'steps': [
                {'range': [0, 20], 'color': "rgba(16, 185, 129, 0.25)"},
                {'range': [20, 40], 'color': "rgba(59, 130, 246, 0.25)"},
                {'range': [40, 60], 'color': "rgba(245, 158, 11, 0.25)"},
                {'range': [60, 80], 'color': "rgba(249, 115, 22, 0.25)"},
                {'range': [80, 100], 'color': "rgba(239, 68, 68, 0.25)"}
            ],
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

def create_timeline_chart(trajectory_history: List[Dict[str, Any]], signals: List[Dict[str, Any]] = None) -> go.Figure:
    """Builds multi-period interactive timeline chart overlayed with warning signal events."""
    df_traj = pd.DataFrame(trajectory_history)
    fig = go.Figure()
    
    # Background threshold bands
    fig.add_hrect(y0=0, y1=20, fillcolor="#10b981", opacity=0.08, line_width=0, annotation_text="Healthy", annotation_position="top left", annotation_font_color="#10b981")
    fig.add_hrect(y0=20, y1=40, fillcolor="#3b82f6", opacity=0.08, line_width=0, annotation_text="Low Risk", annotation_position="top left", annotation_font_color="#3b82f6")
    fig.add_hrect(y0=40, y1=60, fillcolor="#f59e0b", opacity=0.08, line_width=0, annotation_text="Watch", annotation_position="top left", annotation_font_color="#f59e0b")
    fig.add_hrect(y0=60, y1=80, fillcolor="#f97316", opacity=0.08, line_width=0, annotation_text="High Risk", annotation_position="top left", annotation_font_color="#f97316")
    fig.add_hrect(y0=80, y1=100, fillcolor="#ef4444", opacity=0.08, line_width=0, annotation_text="Critical", annotation_position="top left", annotation_font_color="#ef4444")
    
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
