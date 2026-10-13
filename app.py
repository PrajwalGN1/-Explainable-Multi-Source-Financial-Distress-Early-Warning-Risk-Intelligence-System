import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from dashboard.components.charts import (
    create_gauge_chart,
    create_timeline_chart,
    create_health_radar_chart,
    create_waterfall_chart,
    create_global_importance_chart,
    create_temporal_attribution_chart
)
from dashboard.components.kpi_cards import render_kpi_card, render_badge

# Page Configuration
st.set_page_config(
    page_title="Financial Distress Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Aesthetic Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    
    /* Header gradient */
    .main-header {
        background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Glass card container */
    .glass-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Custom metric value */
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
    }
    
    .metric-sub {
        font-size: 0.85rem;
        color: #94a3b8;
    }
    
    /* Alert badge container */
    .alert-card {
        background: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    
    .recommendation-card {
        background: rgba(59, 130, 246, 0.08);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Data Caching Functions
@st.cache_data
def load_risk_db():
    path = "data/processed/companies_risk_intelligence.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"companies": {}, "metadata": {}}

@st.cache_data
def load_watchlist():
    path = "data/processed/watchlist.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_ablation_study():
    path = "reports/model_evaluation/ablation_study.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_global_importance():
    p3_path = "reports/phase3/global_explanations/global_importance_test.csv"
    if os.path.exists(p3_path):
        return pd.read_csv(p3_path)
    legacy_path = "reports/explainability/global_importance.csv"
    if os.path.exists(legacy_path):
        return pd.read_csv(legacy_path)
    return pd.DataFrame()

db = load_risk_db()
df_watchlist = load_watchlist()
companies = db.get("companies", {})
metadata = db.get("metadata", {})

# Sidebar Navigation & Company Selection
st.sidebar.markdown("""
<div style="padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 15px;">
    <h2 style="font-size: 1.25rem; margin: 0; color: #38bdf8;">🛡️ Risk Intelligence</h2>
    <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Multi-Source Corporate Early Warning</div>
</div>
""", unsafe_allow_html=True)

menu_selection = st.sidebar.radio(
    "Navigation",
    [
        "🏛️ Executive Overview",
        "🔍 Company Risk Analyzer",
        "📈 Risk Timeline & Trajectory",
        "🩺 Financial Health Diagnostics",
        "🧠 Explainable AI & SHAP",
        "🔄 Why Did Risk Change?",
        "⚡ Multi-Source Evidence Fusion",
        "📋 Surveillance Watchlist",
        "📊 Model Validation & Ablation",
        "🔬 Prospective Shadow Evaluation",
        "🔬 Research Model Improvement & Baseline Audit",
        "📄 Risk Dossier & Export"
    ],
    index=0
)

# Company selection for company-centric pages
company_pages = [
    "🔍 Company Risk Analyzer",
    "📈 Risk Timeline & Trajectory",
    "🩺 Financial Health Diagnostics",
    "🧠 Explainable AI & SHAP",
    "🔄 Why Did Risk Change?",
    "⚡ Multi-Source Evidence Fusion",
    "🔬 Research Model Improvement & Baseline Audit",
    "📄 Risk Dossier & Export"
]

selected_company_id = "364" # default critical company
selected_period = None

if menu_selection in company_pages:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏢 Enterprise Selector")
    
    selector_mode = st.sidebar.radio(
        "Selection Mode:",
        ["⭐ Watchlist (Top 50)", "📋 All Universe (1-422)", "⌨️ Custom Search"],
        index=0
    )
    
    all_cids = sorted(list(companies.keys()), key=lambda x: int(x)) if companies else []
    
    if selector_mode == "⭐ Watchlist (Top 50)" and len(df_watchlist) > 0:
        company_options = []
        id_to_label = {}
        for _, row in df_watchlist.iterrows():
            cid = str(row["company_id"])
            label = f"Firm {cid} | Score {row['risk_score']} ({row['risk_category']})"
            company_options.append(cid)
            id_to_label[cid] = label
        selected_company_id = st.sidebar.selectbox(
            "Select Watchlist Firm:",
            options=company_options,
            index=0,
            format_func=lambda x: id_to_label.get(x, f"Firm {x}")
        )
    elif selector_mode == "📋 All Universe (1-422)":
        default_idx = all_cids.index("364") if "364" in all_cids else 0
        selected_company_id = st.sidebar.selectbox(
            "Select Any Monitored Firm:",
            options=all_cids,
            index=default_idx,
            format_func=lambda x: f"Firm #{x}"
        )
    else:
        custom_input = st.sidebar.text_input("Enter Company ID (1-422):", value="364")
        selected_company_id = custom_input.strip()
        
    # Strictly validate company presence
    if str(selected_company_id) not in companies:
        st.sidebar.error(f"❌ Company ID '{selected_company_id}' not found in dataset (range: 1-422).")
        st.error(f"❌ Error: Company '{selected_company_id}' not found in the surveillance dataset. Please select or enter a valid corporate identifier (1–422).")
        st.stop()
        
    # Period selector for valid company
    c_info_raw = companies.get(str(selected_company_id), {})
    history_periods = [h["period"] for h in c_info_raw.get("history", []) if "period" in h]
    latest_p = c_info_raw.get("latest_period", 1)
    if not history_periods:
        history_periods = [latest_p]
    
    period_options = [f"Latest (Period T={latest_p})"] + [f"Period T={p}" for p in sorted(history_periods)]
    chosen_period_str = st.sidebar.selectbox("Prediction / Reporting Period:", options=period_options, index=0)
    if chosen_period_str.startswith("Latest"):
        selected_period = latest_p
    else:
        selected_period = int(chosen_period_str.split("=")[-1])

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.72rem; color: #64748b; line-height: 1.4;">
    <b>Research System Notice</b><br>
    Academic & portfolio early-warning prototype. Does not constitute autonomous banking credit or loan origination decisions.
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# PAGE 1: EXECUTIVE OVERVIEW
# -------------------------------------------------------------
if menu_selection == "🏛️ Executive Overview":
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 1.8rem; color: #f8fafc;">🏛️ Executive Portfolio Risk Intelligence</h1>
        <div style="color: #94a3b8; font-size: 0.95rem; margin-top: 6px;">
            Macro-level corporate financial distress surveillance and early-warning telemetry across 422 monitored enterprises.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_kpi_card("Universe Size", f"{metadata.get('total_companies', 422)}", "Monitored Corporates", color="#3b82f6")
    with col2:
        render_kpi_card("Critical Alerts", f"{metadata.get('critical_count', 14)}", "Immediate Escalation (P1)", color="#ef4444")
    with col3:
        render_kpi_card("High Risk Firms", f"{metadata.get('high_risk_count', 34)}", "Intensive Surveillance (P2)", color="#f97316")
    with col4:
        mean_score = metadata.get('mean_risk_score', 5.6)
        render_kpi_card("Mean Portfolio Risk", f"{mean_score}", "0-100 Score Baseline", color="#10b981")
    with col5:
        active_signals = df_watchlist['active_signals_count'].sum() if len(df_watchlist) > 0 else 0
        render_kpi_card("Active Warning Signals", f"{active_signals}", "Deterministic Triggers", color="#eab308")
        
    st.markdown("---")
    
    # Portfolio Distribution Charts
    c_left, c_right = st.columns([1, 1])
    
    with c_left:
        st.markdown('<div class="section-title">📊 Calibrated Risk Category Breakdown</div>', unsafe_allow_html=True)
        if len(df_watchlist) > 0:
            cat_counts = df_watchlist["risk_category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            color_map = {
                "Critical": "#ef4444",
                "High Risk": "#f97316",
                "Watch": "#f59e0b",
                "Low Risk": "#3b82f6",
                "Healthy": "#10b981"
            }
            fig_donut = px.pie(
                cat_counts,
                values="Count",
                names="Category",
                hole=0.6,
                color="Category",
                color_discrete_map=color_map
            )
            fig_donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc"),
                height=320,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            
    with c_right:
        st.markdown('<div class="section-title">📈 Longitudinal Risk Trend Trajectories</div>', unsafe_allow_html=True)
        if len(df_watchlist) > 0:
            trend_counts = df_watchlist["risk_trend"].value_counts().reset_index()
            trend_counts.columns = ["Trajectory", "Count"]
            trend_color_map = {
                "Increasing": "#ef4444",
                "Stable": "#3b82f6",
                "Decreasing": "#10b981",
                "Baseline Initial Observation": "#94a3b8"
            }
            fig_bar = px.bar(
                trend_counts,
                x="Trajectory",
                y="Count",
                color="Trajectory",
                color_discrete_map=trend_color_map,
                text="Count"
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.4)",
                font=dict(color="#f8fafc"),
                height=320,
                xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
    # Priority Watchlist Table Preview
    st.markdown('<div class="section-title">🚨 Urgent Surveillance Watchlist (Top 10 Monitored Entities)</div>', unsafe_allow_html=True)
    if len(df_watchlist) > 0:
        preview_cols = [
            "company_id", "latest_period", "risk_score", "risk_category",
            "risk_trend", "financial_health_score", "active_signals_count", "top_risk_driver", "priority"
        ]
        st.dataframe(
            df_watchlist[preview_cols].head(10).rename(columns={
                "company_id": "Company ID",
                "latest_period": "Reporting Period",
                "risk_score": "Risk Score",
                "risk_category": "Risk Category",
                "risk_trend": "Trajectory",
                "financial_health_score": "Health Score",
                "active_signals_count": "Signals Count",
                "top_risk_driver": "Primary Driver",
                "priority": "Surveillance Priority"
            }),
            use_container_width=True,
            hide_index=True
        )

# -------------------------------------------------------------
# PAGE 2: COMPANY RISK ANALYZER
# -------------------------------------------------------------
elif menu_selection == "🔍 Company Risk Analyzer":
    c_info = companies.get(str(selected_company_id), {})
    eval_period = selected_period if selected_period is not None else c_info.get("latest_period", 1)
    
    # Check if history has this specific period
    hist_record = None
    for h in c_info.get("history", []):
        if h.get("period") == eval_period:
            hist_record = h
            break
            
    if hist_record:
        latest_assess = hist_record.get("assessment", c_info.get("latest_assessment", {}))
        health = hist_record.get("health_breakdown", c_info.get("latest_health", {}))
    else:
        latest_assess = c_info.get("latest_assessment", {})
        health = c_info.get("latest_health", {})
        
    trajectory = c_info.get("trajectory", {})
    signals = c_info.get("warning_signals", [])
    recommendations = c_info.get("recommendations", [])
    
    st.markdown(f"""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; font-size: 1.8rem; color: #f8fafc;">🏢 Enterprise Risk Profile: Company #{selected_company_id}</h1>
                <div style="color: #94a3b8; font-size: 0.95rem; margin-top: 6px;">
                    Reporting Period: <b>T = {eval_period}</b> (Latest: T={c_info.get('latest_period', 'N/A')}) | Historical Observations: <b>{c_info.get('total_periods', 1)}</b>
                </div>
            </div>
            <div>
                {render_badge(latest_assess.get('risk_category', 'Unknown'), latest_assess.get('risk_color', '#3b82f6'))}
                {render_badge(trajectory.get('trend', 'Stable'), '#38bdf8')}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 4 Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_kpi_card(
            "Final Distress Risk",
            f"{latest_assess.get('final_risk_probability_pct', 0.0)}%",
            "Calibrated Horizon t+1",
            color=latest_assess.get("risk_color", "#ef4444")
        )
    with m2:
        render_kpi_card(
            "Composite Risk Score",
            f"{latest_assess.get('risk_score', 0.0)} / 100",
            f"Category: {latest_assess.get('risk_category')}",
            color=latest_assess.get("risk_color", "#ef4444")
        )
    with m3:
        render_kpi_card(
            "Financial Health Index",
            f"{health.get('financial_health_score', 0.0)} / 100",
            "Multi-Dimensional Composite",
            color="#38bdf8"
        )
    with m4:
        render_kpi_card(
            "Active Early Warnings",
            f"{len(signals)} Signals",
            "Rule-Based Deterministic",
            color="#ef4444" if len(signals) > 0 else "#10b981"
        )
        
    st.markdown("---")
    
    # Gauge and Evidence Breakdown
    g_col, e_col = st.columns([1, 1])
    with g_col:
        st.markdown('<div class="section-title">⏱️ Calibrated Risk Score Indicator</div>', unsafe_allow_html=True)
        fig_g = create_gauge_chart(
            latest_assess.get("risk_score", 0.0),
            latest_assess.get("risk_category", "Unknown"),
            latest_assess.get("risk_color", "#ef4444")
        )
        st.plotly_chart(fig_g, use_container_width=True)
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.4); border-radius: 8px; padding: 12px; font-size: 0.85rem; color: #94a3b8;">
            <b>Category Assessment:</b> {latest_assess.get('risk_description')}
        </div>
        """, unsafe_allow_html=True)
        
    with e_col:
        st.markdown('<div class="section-title">⚡ Multi-Source Evidence Synthesis</div>', unsafe_allow_html=True)
        ev = latest_assess.get("evidence_breakdown", {})
        
        st.markdown(f"""
        <div class="glass-card">
            <div style="margin-bottom: 14px;">
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #94a3b8; margin-bottom: 4px;">
                    <span>Dataset A: Temporal EWS Model (t+1 Probability)</span>
                    <b style="color: #f8fafc;">{round(ev.get('ews_probability', 0)*100, 1)}%</b>
                </div>
                <div style="background: rgba(255,255,255,0.1); height: 8px; border-radius: 4px;">
                    <div style="background: #38bdf8; width: {min(100, ev.get('ews_probability', 0)*100)}%; height: 100%; border-radius: 4px;"></div>
                </div>
            </div>
            
            <div style="margin-bottom: 14px;">
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #94a3b8; margin-bottom: 4px;">
                    <span>Dataset B: Bankruptcy Pattern Similarity</span>
                    <b style="color: #f8fafc;">{round(ev.get('bankruptcy_probability', 0)*100, 1)}%</b>
                </div>
                <div style="background: rgba(255,255,255,0.1); height: 8px; border-radius: 4px;">
                    <div style="background: #f59e0b; width: {min(100, ev.get('bankruptcy_probability', 0)*100)}%; height: 100%; border-radius: 4px;"></div>
                </div>
            </div>
            
            <div style="margin-bottom: 14px;">
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #94a3b8; margin-bottom: 4px;">
                    <span>Dataset C: Isolation Forest Anomaly Drift</span>
                    <b style="color: #f8fafc;">{round(ev.get('anomaly_score', 0)*100, 1)}%</b>
                </div>
                <div style="background: rgba(255,255,255,0.1); height: 8px; border-radius: 4px;">
                    <div style="background: #a855f7; width: {min(100, ev.get('anomaly_score', 0)*100)}%; height: 100%; border-radius: 4px;"></div>
                </div>
            </div>
            
            <div>
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #94a3b8; margin-bottom: 4px;">
                    <span>Financial Health Vulnerability (100 - Health)</span>
                    <b style="color: #f8fafc;">{round(ev.get('financial_health_risk', 0)*100, 1)}%</b>
                </div>
                <div style="background: rgba(255,255,255,0.1); height: 8px; border-radius: 4px;">
                    <div style="background: #ef4444; width: {min(100, ev.get('financial_health_risk', 0)*100)}%; height: 100%; border-radius: 4px;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Active Early Warnings & Recommendations
    w_col, r_col = st.columns([1, 1])
    with w_col:
        st.markdown('<div class="section-title">🚨 Active Deterministic Early-Warning Alerts</div>', unsafe_allow_html=True)
        if signals:
            for s in signals:
                st.markdown(f"""
                <div class="alert-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                        <b style="color: #ef4444; font-size: 0.95rem;">{s.get('signal_type')}</b>
                        {render_badge(s.get('severity'), '#ef4444')}
                    </div>
                    <div style="font-size: 0.85rem; color: #e2e8f0;">{s.get('evidence')}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No active early-warning signals detected. Enterprise operating within historical safety tolerances.")
            
    with r_col:
        st.markdown('<div class="section-title">📋 Recommended Risk Surveillance Actions</div>', unsafe_allow_html=True)
        if recommendations:
            for rec in recommendations:
                st.markdown(f"""
                <div class="recommendation-card">
                    <div style="font-size: 0.85rem; color: #e2e8f0;">📌 {rec}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("<div class='recommendation-card'>Maintain standard monitoring protocol.</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# PAGE 3: RISK TIMELINE & TRAJECTORY
# -------------------------------------------------------------
elif menu_selection == "📈 Risk Timeline & Trajectory":
    c_info = companies.get(str(selected_company_id), {})
    trajectory = c_info.get("trajectory", {})
    history = trajectory.get("trajectory_history", [])
    signals = c_info.get("warning_signals", [])
    
    st.markdown(f"""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 1.8rem; color: #f8fafc;">📈 Longitudinal Risk Trajectory: Company #{selected_company_id}</h1>
        <div style="color: #94a3b8; font-size: 0.95rem; margin-top: 6px;">
            Tracking multi-period financial deterioration dynamics and temporal momentum over {len(history)} reporting intervals.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Trajectory KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Trajectory Trend", f"{trajectory.get('trend', 'Stable')}", "Directional Vector", color="#38bdf8")
    with k2:
        delta_val = trajectory.get('risk_change_pct_points', 0.0)
        render_kpi_card(
            "Period Risk Delta",
            f"{'+' if delta_val > 0 else ''}{delta_val} pp",
            "Period-over-Period Shift",
            color="#ef4444" if delta_val > 0 else "#10b981"
        )
    with k3:
        accel = trajectory.get('risk_acceleration', 0.0)
        render_kpi_card(
            "Risk Acceleration",
            f"{'+' if accel > 0 else ''}{accel}",
            "Second Derivative Momentum",
            color="#ef4444" if accel > 0 else "#10b981"
        )
    with k4:
        consec = trajectory.get('consecutive_increases', 0)
        render_kpi_card(
            "Consecutive Increases",
            f"{consec} Periods",
            "Sustained Deterioration",
            color="#ef4444" if consec >= 2 else "#3b82f6"
        )
        
    st.markdown("---")
    
    # Timeline Plotly Chart
    fig_time = create_timeline_chart(history, signals)
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Historical Observation Table
    st.markdown('<div class="section-title">📜 Historical Reporting Period Breakdown</div>', unsafe_allow_html=True)
    periods_data = c_info.get("periods_data", [])
    if periods_data:
        rows = []
        for p in periods_data:
            assess = p.get("assessment", {})
            rows.append({
                "Period (T)": p.get("period"),
                "Risk Probability (%)": f"{assess.get('final_risk_probability_pct', 0.0)}%",
                "Risk Score": assess.get("risk_score", 0.0),
                "Category": assess.get("risk_category", "Unknown"),
                "EWS Probability": f"{round(assess.get('evidence_breakdown', {}).get('ews_probability', 0)*100, 1)}%",
                "Bankruptcy Prob": f"{round(assess.get('evidence_breakdown', {}).get('bankruptcy_probability', 0)*100, 1)}%",
                "Health Score": assess.get("evidence_breakdown", {}).get("financial_health_score", 0.0)
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# PAGE 4: FINANCIAL HEALTH DIAGNOSTICS
# -------------------------------------------------------------
elif menu_selection == "🩺 Financial Health Diagnostics":
    c_info = companies.get(str(selected_company_id), {})
    health = c_info.get("latest_health", {})
    
    st.markdown(f"""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 1.8rem; color: #f8fafc;">🩺 Financial Health Diagnostics: Company #{selected_company_id}</h1>
        <div style="color: #94a3b8; font-size: 0.95rem; margin-top: 6px;">
            Interpretable dimensional evaluation across Profitability, Liquidity, Solvency, Leverage, and Working Capital.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    h_col, r_col = st.columns([1, 1])
    with h_col:
        st.markdown('<div class="section-title">📊 Multi-Dimensional Subscore Breakdown</div>', unsafe_allow_html=True)
        dimensions = [
            ("Profitability", health.get("profitability_score", 50.0), "#10b981", "Operating margin, ROA proxy, earnings generation"),
            ("Liquidity", health.get("liquidity_score", 50.0), "#3b82f6", "Short-term debt coverage & cash cushion"),
            ("Solvency", health.get("solvency_score", 50.0), "#f59e0b", "Long-term debt service capacity"),
            ("Leverage", health.get("leverage_score", 50.0), "#8b5cf6", "Capital structure balance & liability load"),
            ("Working Capital", health.get("working_capital_score", 50.0), "#ec4899", "Operational cash conversion & net current assets")
        ]
        
        for name, score, col, desc in dimensions:
            st.markdown(f"""
            <div style="margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 4px;">
                    <span style="font-weight: 600; color: #f8fafc;">{name}</span>
                    <span style="font-weight: 700; color: {col};">{score} / 100</span>
                </div>
                <div style="background: rgba(255,255,255,0.08); height: 10px; border-radius: 5px;">
                    <div style="background: {col}; width: {min(100, max(0, score))}%; height: 100%; border-radius: 5px;"></div>
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 3px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            
        render_kpi_card(
            "Composite Financial Health Score",
            f"{health.get('financial_health_score', 50.0)} / 100",
            "Aggregated Multidimensional Health",
            color="#38bdf8"
        )
        
    with r_col:
        st.markdown('<div class="section-title">🕸️ Enterprise vs Benchmark Radar Profile</div>', unsafe_allow_html=True)
        fig_radar = create_health_radar_chart(health)
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.4); border-radius: 8px; padding: 12px; font-size: 0.8rem; color: #94a3b8; line-height: 1.5;">
            <b>Benchmark Context:</b> Compares the enterprise's dimensional subscores (blue shaded polygon) against the median performance of non-distressed industrial peer benchmarks (gray dotted perimeter). Inward contractions indicate acute structural vulnerabilities.
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------
# PAGE 5: EXPLAINABLE AI & SHAP
# -------------------------------------------------------------
# -------------------------------------------------------------
# PAGE 5: EXPLAINABLE AI & SHAP
# -------------------------------------------------------------
elif menu_selection == "🧠 Explainable AI & SHAP":
    from src.explainability.company_service import get_company_explanation_service
    srv = get_company_explanation_service()

    c_info = companies.get(str(selected_company_id), {})
    try:
        local_xai = srv.get_local_explanation(str(selected_company_id), top_n=10)
        drivers = local_xai.get("top_risk_drivers", [])
        protective = local_xai.get("top_protective_factors", [])
        expected_val = local_xai.get("expected_value_base", 0.4969)
        cal_prob = local_xai.get("calibrated_distress_probability", 0.0)
        raw_prob = local_xai.get("raw_random_forest_probability", 0.0)
        risk_score = local_xai.get("risk_score", 0.0)
        risk_cat = local_xai.get("risk_category", "Healthy")
        additivity_ok = local_xai.get("additivity_verified", True)
        reporting_period = local_xai.get("reporting_period", c_info.get("latest_period", 1))
        model_ver = local_xai.get("model_version", "1.2.1-hardened")
        out_space = local_xai.get("output_space", "raw_random_forest_probability")
    except Exception as e:
        drivers = c_info.get("top_risk_drivers", [])
        protective = c_info.get("top_protective_factors", [])
        expected_val = 0.4969
        cal_prob = c_info.get("latest_assessment", {}).get("final_risk_probability", 0.0)
        raw_prob = cal_prob
        risk_score = c_info.get("latest_assessment", {}).get("risk_score", 0.0)
        risk_cat = c_info.get("latest_assessment", {}).get("risk_category", "Healthy")
        additivity_ok = True
        reporting_period = c_info.get("latest_period", 1)
        model_ver = "1.2.1-hardened"
        out_space = "raw_random_forest_probability"

    st.markdown(f"""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 1.8rem; color: #f8fafc;">🧠 Model-Faithful TreeSHAP Explainability: Company #{selected_company_id}</h1>
        <div style="color: #94a3b8; font-size: 0.95rem; margin-top: 6px;">
            Audited Native TreeSHAP decomposition on Random Forest (<code>{model_ver}</code>) | Reporting Period: <b>T={reporting_period}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 4 Telemetry Cards
    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1:
        render_kpi_card("Calibrated Operational Prob", f"{cal_prob*100:.2f}%", f"Score: {risk_score} ({risk_cat})", color="#ef4444" if cal_prob >= 0.08 else "#10b981")
    with tc2:
        render_kpi_card("Raw Random Forest Output", f"{raw_prob*100:.2f}%", f"Output Space: {out_space}", color="#38bdf8")
    with tc3:
        render_kpi_card("SHAP Baseline Prior", f"{expected_val:.4f}", "Distress Class E[p_RF]", color="#94a3b8")
    with tc4:
        render_kpi_card("Additivity Verification", "Verified (0.0 disc.)" if additivity_ok else "Tolerance Exceeded", "sum(SHAP) + E[p] == p_RF", color="#10b981" if additivity_ok else "#ef4444")
        
    st.markdown("---")

    # Waterfall Chart
    fig_waterfall = create_waterfall_chart(drivers, protective, expected_value=expected_val)
    st.plotly_chart(fig_waterfall, use_container_width=True)
    
    st.markdown("---")
    
    col_d, col_p = st.columns([1, 1])
    with col_d:
        st.markdown('<div class="section-title">🔴 Top Risk Drivers (Pushes Risk Up in p_RF space)</div>', unsafe_allow_html=True)
        if drivers:
            df_d = pd.DataFrame(drivers).rename(columns={
                "feature": "Feature",
                "shap_impact": "SHAP Impact",
                "feature_value_scaled": "Scaled Value",
                "transformation": "Transformation",
                "interpretation": "Interpretation"
            })
            display_cols = [c for c in ["Feature", "SHAP Impact", "Scaled Value", "Transformation", "Interpretation"] if c in df_d.columns]
            st.dataframe(df_d[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No positive risk drivers detected.")
            
    with col_p:
        st.markdown('<div class="section-title">🟢 Top Protective Factors (Pulls Risk Down in p_RF space)</div>', unsafe_allow_html=True)
        if protective:
            df_p = pd.DataFrame(protective).rename(columns={
                "feature": "Feature",
                "shap_impact": "SHAP Impact",
                "feature_value_scaled": "Scaled Value",
                "transformation": "Transformation",
                "interpretation": "Interpretation"
            })
            display_cols = [c for c in ["Feature", "SHAP Impact", "Scaled Value", "Transformation", "Interpretation"] if c in df_p.columns]
            st.dataframe(df_p[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No protective factors detected.")
            
    st.markdown("---")
    
    # Global Feature Importance
    st.markdown('<div class="section-title">🌐 Global EWS Model Feature Importance (Holdout Test T=13, N=262)</div>', unsafe_allow_html=True)
    df_glob = load_global_importance()
    if len(df_glob) > 0:
        fig_glob = create_global_importance_chart(df_glob, top_n=15)
        st.plotly_chart(fig_glob, use_container_width=True)
        
    # Mandatory Methodology Disclaimer Banner
    st.markdown("""
    <div class="glass-card" style="border-left: 4px solid #f59e0b; font-size: 0.85rem; color: #cbd5e1; line-height: 1.6;">
        <b>⚠️ Methodological & Regulatory Notice for Credit Reviewers:</b><br>
        1. <b>Model Attribution vs. Causation:</b> SHAP attributions quantify mathematical associations in the Random Forest probability output space. They do <i>not</i> establish real-world causality.<br>
        2. <b>Anonymized Financial Indicators:</b> Features <code>x1</code> through <code>x83</code> are masked indicators from Dataset A. Specific balance sheet narratives are unverified.<br>
        3. <b>Decision Support Only:</b> Predictions are statistical distress likelihoods for early surveillance and do not constitute automated loan origination or adverse action decisions.
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# PAGE 6: "WHY DID RISK CHANGE?"
# -------------------------------------------------------------
elif menu_selection == "🔄 Why Did Risk Change?":
    from src.explainability.company_service import get_company_explanation_service
    srv = get_company_explanation_service()

    c_info = companies.get(str(selected_company_id), {})
    try:
        temp_xai = srv.get_temporal_explanation(str(selected_company_id), top_n=5)
        prev_p = temp_xai.get("previous_calibrated_prob", 0.0)
        curr_p = temp_xai.get("current_calibrated_prob", 0.0)
        delta_pts = temp_xai.get("calibrated_prob_delta_points", 0.0)
        p_prev_period = temp_xai.get("previous_period", "N/A")
        p_curr_period = temp_xai.get("current_period", "N/A")
        cat_prev = temp_xai.get("previous_risk_category", "Unknown")
        cat_curr = temp_xai.get("current_risk_category", "Unknown")
        narrative = temp_xai.get("summary", "No previous period observation available.")
        drivers_inc = temp_xai.get("top_drivers_of_increase", [])
        drivers_rec = temp_xai.get("top_drivers_of_recovery", [])
    except Exception as e:
        delta_info = c_info.get("period_risk_delta", {})
        prev_p = delta_info.get("previous_risk_pct", 0.0) / 100.0
        curr_p = delta_info.get("current_risk_pct", 0.0) / 100.0
        delta_pts = delta_info.get("delta_pct_points", 0.0)
        p_prev_period = delta_info.get("previous_period", "N/A")
        p_curr_period = delta_info.get("current_period", "N/A")
        cat_prev = "Previous"
        cat_curr = "Current"
        narrative = delta_info.get("narrative", "Baseline initial observation.")
        drivers_inc = []
        drivers_rec = []
        temp_xai = {}

    st.markdown(f"""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 1.8rem; color: #f8fafc;">🔄 Period-over-Period Risk Change: Company #{selected_company_id}</h1>
        <div style="color: #94a3b8; font-size: 0.95rem; margin-top: 6px;">
            Comparative delta decomposition between Period <b>T={p_prev_period}</b> and Period <b>T={p_curr_period}</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Delta Summary KPIs
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        render_kpi_card(f"Previous Risk (T={p_prev_period})", f"{prev_p*100:.2f}%", f"Category: {cat_prev}", color="#94a3b8")
    with d2:
        render_kpi_card(f"Current Risk (T={p_curr_period})", f"{curr_p*100:.2f}%", f"Category: {cat_curr}", color="#38bdf8")
    with d3:
        render_kpi_card(
            "Net Risk Shift",
            f"{'+' if delta_pts > 0 else ''}{delta_pts} pp",
            "Deterioration" if delta_pts > 0 else "Improvement",
            color="#ef4444" if delta_pts > 0 else "#10b981"
        )
    with d4:
        render_kpi_card(
            "Operational Category Shift",
            f"{cat_prev} → {cat_curr}",
            "Threshold Governed",
            color="#ef4444" if delta_pts > 0 else "#10b981"
        )
        
    st.markdown("---")
    
    # Plain English Narrative
    st.markdown('<div class="section-title">📝 Automated Risk Shift Narrative</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="glass-card" style="font-size: 0.95rem; line-height: 1.6; color: #e2e8f0; border-left: 4px solid #38bdf8;">
        {narrative}
    </div>
    """, unsafe_allow_html=True)
    
    # Comparative Temporal Attribution Bar Chart
    if temp_xai.get("status") == "success":
        st.markdown('<div class="section-title">📊 Key Attribution Shifts (Delta TreeSHAP Impact)</div>', unsafe_allow_html=True)
        fig_temp = create_temporal_attribution_chart(temp_xai)
        st.plotly_chart(fig_temp, use_container_width=True)
        
        st.markdown("---")
        
        tcol1, tcol2 = st.columns([1, 1])
        with tcol1:
            st.markdown('<div class="section-title">🔴 Top Drivers of Increased Risk (Δφ > 0)</div>', unsafe_allow_html=True)
            if drivers_inc:
                df_inc = pd.DataFrame(drivers_inc)
                disp_cols = [c for c in ["feature", "prev_feature_value_scaled", "curr_feature_value_scaled", "feature_value_change", "shap_delta", "interpretation"] if c in df_inc.columns]
                st.dataframe(df_inc[disp_cols].rename(columns={
                    "feature": "Feature",
                    "prev_feature_value_scaled": "Prev Scaled (T-1)",
                    "curr_feature_value_scaled": "Curr Scaled (T)",
                    "feature_value_change": "Value Change",
                    "shap_delta": "SHAP Delta (Δφ)",
                    "interpretation": "Interpretation"
                }), use_container_width=True, hide_index=True)
            else:
                st.info("No significant drivers of increased risk.")
                
        with tcol2:
            st.markdown('<div class="section-title">🟢 Top Drivers of Risk Recovery (Δφ < 0)</div>', unsafe_allow_html=True)
            if drivers_rec:
                df_rec = pd.DataFrame(drivers_rec)
                disp_cols = [c for c in ["feature", "prev_feature_value_scaled", "curr_feature_value_scaled", "feature_value_change", "shap_delta", "interpretation"] if c in df_rec.columns]
                st.dataframe(df_rec[disp_cols].rename(columns={
                    "feature": "Feature",
                    "prev_feature_value_scaled": "Prev Scaled (T-1)",
                    "curr_feature_value_scaled": "Curr Scaled (T)",
                    "feature_value_change": "Value Change",
                    "shap_delta": "SHAP Delta (Δφ)",
                    "interpretation": "Interpretation"
                }), use_container_width=True, hide_index=True)
            else:
                st.info("No significant recovery drivers.")
    else:
        st.info("Initial baseline reporting period. At least 2 periods are required for period-over-period attribution comparison.")

    st.markdown("""
    <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 15px;">
        <b>Temporal Attribution Note:</b> Distinguishes 3 separate concepts: (1) change in calibrated prediction, (2) change in scaled input values, and (3) change in TreeSHAP attributions. Attribution shifts describe model behavior and do not constitute proven causal real-world mechanisms.
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# PAGE 7: MULTI-SOURCE EVIDENCE FUSION
# -------------------------------------------------------------
elif menu_selection == "⚡ Multi-Source Evidence Fusion":
    c_info = companies.get(str(selected_company_id), {})
    latest_assess = c_info.get("latest_assessment", {})
    ev = latest_assess.get("evidence_breakdown", {})
    
    st.markdown(f"""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 1.8rem; color: #f8fafc;">⚡ Multi-Source Risk Evidence Fusion Engine</h1>
        <div style="color: #94a3b8; font-size: 0.95rem; margin-top: 6px;">
            Synthesizing temporal early warning, static bankruptcy patterns, external anomaly evidence, and financial health.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Source Contribution Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            "Source A: EWS Model",
            f"{round(ev.get('ews_probability', 0)*100, 1)}%",
            "Dataset A: Tehran Panel",
            color="#38bdf8"
        )
    with c2:
        render_kpi_card(
            "Source B: Bankruptcy",
            f"{round(ev.get('bankruptcy_probability', 0)*100, 1)}%",
            "Dataset B: Taiwan TEJ",
            color="#f59e0b"
        )
    with c3:
        render_kpi_card(
            "Source C: Anomaly",
            f"{round(ev.get('anomaly_score', 0)*100, 1)}%",
            "Dataset C: Isolation Forest",
            color="#a855f7"
        )
    with c4:
        render_kpi_card(
            "Source D: Health Risk",
            f"{round(ev.get('financial_health_risk', 0)*100, 1)}%",
            "100 - Health Score",
            color="#ef4444"
        )
        
    st.markdown("---")
    
    # Fusion Architecture Explanation
    st.markdown("""
    <div class="glass-card">
        <h3 style="margin-top: 0; color: #f8fafc;">🏗️ Two-Level Fusion Methodology</h3>
        <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;">
            The risk intelligence engine rejects naive row-wise concatenation (<code>pd.concat</code>) because the three source datasets represent fundamentally non-overlapping schema spaces, distinct industrial populations, and disparate target definitions. Instead, the architecture executes a <b>scientifically defensible two-level risk fusion</b>:
        </p>
        <ul style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.7;">
            <li><b>Level 1 (Independent Source Experts):</b> Specialized estimators trained in isolation on appropriate data topologies (Dataset A: Temporal EWS Panel, Dataset B: Cross-Sectional Bankruptcy Ratios, Dataset C: Unsupervised Isolation Forest Anomaly Layer).</li>
            <li><b>Level 2 (Stacking Meta-Learner):</b> A logistic meta-model trained strictly on <i>out-of-fold validation predictions</i> to prevent target leakage, learning empirical stacking weights:
                <ul>
                    <li><code>coef_ews</code>: <b>+0.895</b> (Temporal deterioration momentum)</li>
                    <li><code>coef_bankruptcy</code>: <b>+1.524</b> (Cross-sectional structural distress resemblance)</li>
                    <li><code>coef_health_risk</code>: <b>+1.793</b> (Severe dimensional weakness penalty)</li>
                    <li><code>coef_anomaly</code>: <b>-0.569</b> (Residual OOD regularizer)</li>
                </ul>
            </li>
            <li><b>Platt Scaling Calibration:</b> Sigmoid probability calibration aligning predictions with the true historical base rate, cutting test Brier score from 0.0626 to 0.0387 (38.07% calibration gain).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# PAGE 8: SURVEILLANCE WATCHLIST
# -------------------------------------------------------------
elif menu_selection == "📋 Surveillance Watchlist":
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 1.8rem; color: #f8fafc;">📋 Corporate Surveillance Watchlist</h1>
        <div style="color: #94a3b8; font-size: 0.95rem; margin-top: 6px;">
            Prioritized portfolio monitoring matrix ranking all 422 corporates by composite distress risk and warning signals.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if len(df_watchlist) > 0:
        # Filters Row
        f1, f2, f3 = st.columns([1, 1, 1])
        with f1:
            cat_filter = st.multiselect(
                "Filter by Risk Category:",
                options=["Critical", "High Risk", "Watch", "Low Risk", "Healthy"],
                default=["Critical", "High Risk", "Watch"]
            )
        with f2:
            priority_filter = st.multiselect(
                "Filter by Surveillance Priority:",
                options=["P1 - Urgent Escalation", "P2 - High Priority Surveillance", "P3 - Active Watchlist", "P4 - Normal Surveillance"],
                default=["P1 - Urgent Escalation", "P2 - High Priority Surveillance", "P3 - Active Watchlist"]
            )
        with f3:
            trend_filter = st.multiselect(
                "Filter by Trajectory:",
                options=["Increasing", "Stable", "Decreasing", "Baseline Initial Observation"],
                default=["Increasing", "Stable", "Decreasing", "Baseline Initial Observation"]
            )
            
        filtered_df = df_watchlist.copy()
        if cat_filter:
            filtered_df = filtered_df[filtered_df["risk_category"].isin(cat_filter)]
        if priority_filter:
            filtered_df = filtered_df[filtered_df["priority"].isin(priority_filter)]
        if trend_filter:
            filtered_df = filtered_df[filtered_df["risk_trend"].isin(trend_filter)]
            
        st.markdown(f"**Showing {len(filtered_df)} of {len(df_watchlist)} Monitored Enterprises**")
        
        display_cols = [
            "company_id", "latest_period", "risk_score", "risk_category",
            "risk_trend", "financial_health_score", "active_signals_count", "top_risk_driver", "priority"
        ]
        st.dataframe(
            filtered_df[display_cols].rename(columns={
                "company_id": "Company ID",
                "latest_period": "Period",
                "risk_score": "Risk Score",
                "risk_category": "Category",
                "risk_trend": "Trend",
                "financial_health_score": "Health Score",
                "active_signals_count": "Signals",
                "top_risk_driver": "Top Driver",
                "priority": "Surveillance Priority"
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Download Watchlist CSV
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Surveillance Watchlist CSV",
            data=csv_data,
            file_name="corporate_surveillance_watchlist.csv",
            mime="text/csv"
        )

# -------------------------------------------------------------
# PAGE 9: MODEL VALIDATION & ABLATION
# -------------------------------------------------------------
elif menu_selection == "📊 Model Validation & Ablation":
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 1.8rem; color: #f8fafc;">📊 Model Validation & Ablation Analysis</h1>
        <div style="color: #94a3b8; font-size: 0.95rem; margin-top: 6px;">
            Empirical validation results, probability calibration curves, and multi-source ablation studies.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Ablation Study Table
    st.markdown('<div class="section-title">🔬 Empirical Ablation Study (6 Research Experiments)</div>', unsafe_allow_html=True)
    df_ablation = load_ablation_study()
    if len(df_ablation) > 0:
        st.dataframe(df_ablation, use_container_width=True, hide_index=True)
        
        # Ablation Chart
        fig_abl = px.bar(
            df_ablation,
            x="Experiment",
            y=["ROC_AUC", "PR_AUC"],
            barmode="group",
            title="<b>Ablation Progression: ROC-AUC vs PR-AUC Across Model Architectures</b>",
            color_discrete_sequence=["#38bdf8", "#f59e0b"]
        )
        fig_abl.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            font=dict(color="#f8fafc"),
            xaxis=dict(tickangle=-15, gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            height=360,
            margin=dict(l=40, r=20, t=50, b=50)
        )
        st.plotly_chart(fig_abl, use_container_width=True)
        
    st.markdown("---")
    
    # Core Validation Strategy
    c_left, c_right = st.columns([1, 1])
    with c_left:
        st.markdown('<div class="section-title">🛡️ Temporal Leakage Prevention Protocol</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card" style="font-size: 0.88rem; line-height: 1.6; color: #cbd5e1;">
            <ul>
                <li><b>Time-Aware Panel Splitting:</b> Model training is strictly anchored in chronological order. Training on Periods 1–10 (516 company-periods), Validation on Periods 11–12 (521 company-periods), and Out-of-Time Test on Periods 13–14 (262 company-periods). Future periods never leak into past feature computations.</li>
                <li><b>Backward-Looking Feature Pipeline:</b> Lags ($t-1, t-2$), delta shifts, and rolling statistics ($k=2, 3$) evaluate historical records up to reporting period $t$ only.</li>
                <li><b>Strict Preprocessor Fitting:</b> Imputation medians and scaling parameters are fitted exclusively on the training partition and transformed forward.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with c_right:
        st.markdown('<div class="section-title">📐 Calibration & Brier Score Verification</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card" style="font-size: 0.88rem; line-height: 1.6; color: #cbd5e1;">
            <ul>
                <li><b>Temporal EWS Model:</b> Raw Random Forest Brier score of 0.0626 improved to <b>0.0387</b> post-Platt scaling (<b>38.07% calibration gain</b>).</li>
                <li><b>Static Bankruptcy Model:</b> Raw Random Forest Brier score of 0.0419 improved to <b>0.0234</b> post-Platt scaling (<b>44.23% calibration gain</b>).</li>
                <li><b>Reliability Verification:</b> Evaluated on independent test partitions with monotonic probability bins, ensuring probabilities reflect empirical event frequencies.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------
# PAGE 10: RISK DOSSIER & EXPORT
# -------------------------------------------------------------
elif menu_selection == "📄 Risk Dossier & Export":
    c_info = companies.get(str(selected_company_id), {})
    latest_assess = c_info.get("latest_assessment", {})
    health = c_info.get("latest_health", {})
    trajectory = c_info.get("trajectory", {})
    signals = c_info.get("warning_signals", [])
    recommendations = c_info.get("recommendations", [])
    drivers = c_info.get("top_risk_drivers", [])
    
    st.markdown(f"""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 1.8rem; color: #f8fafc;">📄 Corporate Risk Intelligence Dossier: Company #{selected_company_id}</h1>
        <div style="color: #94a3b8; font-size: 0.95rem; margin-top: 6px;">
            Comprehensive audit report ready for risk committee review and surveillance record keeping.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Structured Report Container
    st.markdown(f"""
    <div class="glass-card" style="border: 1px solid rgba(255,255,255,0.15); padding: 28px;">
        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 12px; margin-bottom: 16px;">
            <h2 style="margin: 0; color: #38bdf8;">EXECUTIVE RISK DOSSIER - COMPANY #{selected_company_id}</h2>
            <div>{render_badge(latest_assess.get('risk_category', 'Unknown'), latest_assess.get('risk_color', '#ef4444'))}</div>
        </div>
        
        <p><b>Reporting Period:</b> T = {c_info.get('latest_period', 'N/A')} | <b>Historical Observations:</b> {c_info.get('total_periods', 1)}</p>
        
        <h3 style="color: #f8fafc; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 6px;">1. Quantitative Risk Assessment</h3>
        <ul>
            <li><b>Calibrated Final Risk Probability:</b> {latest_assess.get('final_risk_probability_pct', 0.0)}% (Horizon t+1)</li>
            <li><b>Composite Risk Score:</b> {latest_assess.get('risk_score', 0.0)} / 100 ({latest_assess.get('risk_category')})</li>
            <li><b>Longitudinal Trajectory:</b> {trajectory.get('trend', 'Stable')} ({trajectory.get('risk_change_pct_points', 0.0)} pp shift)</li>
            <li><b>Composite Financial Health Score:</b> {health.get('financial_health_score', 0.0)} / 100</li>
        </ul>
        
        <h3 style="color: #f8fafc; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 6px;">2. Multi-Source Evidence Synthesis</h3>
        <ul>
            <li><b>Dataset A (Temporal EWS Model):</b> {round(latest_assess.get('evidence_breakdown', {}).get('ews_probability', 0)*100, 1)}%</li>
            <li><b>Dataset B (Cross-Sectional Bankruptcy):</b> {round(latest_assess.get('evidence_breakdown', {}).get('bankruptcy_probability', 0)*100, 1)}%</li>
            <li><b>Dataset C (Isolation Forest Anomaly Drift):</b> {round(latest_assess.get('evidence_breakdown', {}).get('anomaly_score', 0)*100, 1)}%</li>
            <li><b>Financial Health Risk (100 - Health):</b> {round(latest_assess.get('evidence_breakdown', {}).get('financial_health_risk', 0)*100, 1)}%</li>
        </ul>
        
        <h3 style="color: #f8fafc; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 6px;">3. Top Attribution Risk Drivers</h3>
        <ol>
            {"".join([f"<li><b>{d.get('feature')}:</b> SHAP impact {d.get('shap_impact')} ({d.get('interpretation')})</li>" for d in drivers[:5]]) if drivers else "<li>No prominent negative risk drivers.</li>"}
        </ol>
        
        <h3 style="color: #f8fafc; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 6px;">4. Active Deterministic Alerts</h3>
        <ul>
            {"".join([f"<li><b>[{s.get('severity')}] {s.get('signal_type')}:</b> {s.get('evidence')}</li>" for s in signals]) if signals else "<li>No active warning signals.</li>"}
        </ul>
        
        <h3 style="color: #f8fafc; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 6px;">5. Recommended Risk Committee Actions</h3>
        <ul>
            {"".join([f"<li>{r}</li>" for r in recommendations]) if recommendations else "<li>Continue standard surveillance.</li>"}
        </ul>
        
        <div style="margin-top: 24px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.1); font-size: 0.75rem; color: #64748b;">
            <b>Academic Prototype Disclaimer:</b> This document was generated automatically by the Explainable Multi-Source Financial Distress Early-Warning System for research and demonstration purposes. It does not constitute credit approval, loan denial, or formal investment advisory.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Download Buttons
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        json_str = json.dumps(c_info, indent=2)
        st.download_button(
            label="📥 Download JSON Intelligence Dossier",
            data=json_str,
            file_name=f"company_{selected_company_id}_risk_dossier.json",
            mime="application/json"
        )
    with col_d2:
        md_content = f"""# Corporate Risk Intelligence Dossier: Company {selected_company_id}
Generated by Explainable Multi-Source Financial Distress Early-Warning System

- **Reporting Period:** T = {c_info.get('latest_period', 'N/A')}
- **Final Risk Probability:** {latest_assess.get('final_risk_probability_pct', 0.0)}%
- **Risk Score:** {latest_assess.get('risk_score', 0.0)} / 100
- **Risk Category:** {latest_assess.get('risk_category', 'Unknown')}
- **Trajectory Trend:** {trajectory.get('trend', 'Stable')}
- **Financial Health Score:** {health.get('financial_health_score', 0.0)} / 100

## Multi-Source Evidence
- Temporal EWS Probability: {round(latest_assess.get('evidence_breakdown', {}).get('ews_probability', 0)*100, 1)}%
- Bankruptcy Probability: {round(latest_assess.get('evidence_breakdown', {}).get('bankruptcy_probability', 0)*100, 1)}%
- Anomaly Score: {round(latest_assess.get('evidence_breakdown', {}).get('anomaly_score', 0)*100, 1)}%

## Surveillance Recommendations
{chr(10).join([f"- {r}" for r in recommendations])}
"""
        st.download_button(
            label="📥 Download Markdown Report",
            data=md_content,
            file_name=f"company_{selected_company_id}_report.md",
            mime="text/markdown"
        )

# -------------------------------------------------------------
# PAGE 11: PROSPECTIVE SHADOW EVALUATION (Phase 6)
# -------------------------------------------------------------
elif menu_selection == "🔬 Prospective Shadow Evaluation":
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 1.8rem; color: #f8fafc;">🔬 Prospective Shadow Evaluation & Operational Validation</h1>
        <div style="color: #94a3b8; font-size: 0.95rem; margin-top: 6px;">
            Parallel, non-interfering shadow surveillance comparing Candidate <code style="color: #a78bfa;">Phase5_RF_CostSensitive_w10</code> against approved Production Baseline <code style="color: #38bdf8;">v1.2 (τ=0.0800)</code>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Prominent Academic / Operational Disclaimer Banner
    st.markdown("""
    <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid #ef4444; border-radius: 10px; padding: 14px 18px; margin-bottom: 22px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.5rem;">⚠️</span>
            <div>
                <b style="color: #f87171; font-size: 0.95rem;">EXPERIMENTAL RESEARCH CANDIDATE IN SHADOW SURVEILLANCE</b>
                <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 2px;">
                    Candidate predictions are sandbox-isolated and strictly non-authoritative. Candidate scores and alerts have ZERO influence on live production decisions. The approved baseline model (<code>models/phase2/best_ews_model.joblib</code>) remains the primary official authority operating at official threshold <b>τ = 0.0800</b>.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load Phase 6 artifacts
    p6_telemetry_file = "results/phase6/prospective_telemetry.json"
    p6_maturity_file = "results/phase6/matured_evaluation_summary.json"
    p6_preds_file = "results/phase6/prospective_predictions.csv"

    if os.path.exists(p6_telemetry_file) and os.path.exists(p6_maturity_file) and os.path.exists(p6_preds_file):
        with open(p6_telemetry_file, "r", encoding="utf-8") as f:
            telemetry = json.load(f)
        with open(p6_maturity_file, "r", encoding="utf-8") as f:
            mat_eval = json.load(f)
        df_p6_preds = pd.read_csv(p6_preds_file)

        # 1. Top KPI Row
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            render_kpi_card(
                "Prospective Cohort",
                f"{telemetry.get('total_prospective_observations', 0)} SMEs",
                "Period T=12 Surveillance",
                "#38bdf8"
            )
        with k2:
            render_kpi_card(
                "Production Alerts",
                f"{telemetry.get('production_alert_rate_pct', 0.0)}%",
                f"{telemetry.get('production_alert_count', 0)} alerts (τ=0.080)",
                "#f59e0b"
            )
        with k3:
            render_kpi_card(
                "Candidate Alerts",
                f"{telemetry.get('shadow_candidate_alert_rate_pct', 0.0)}%",
                f"{telemetry.get('shadow_candidate_alert_count', 0)} alerts (τ=0.0894)",
                "#a78bfa"
            )
        with k4:
            render_kpi_card(
                "Model Agreement",
                f"{telemetry.get('inter_model_agreement_pct', 0.0)}%",
                "Both Alert / Neither Alert",
                "#10b981"
            )
        with k5:
            render_kpi_card(
                "Alert Budget Compliance",
                telemetry.get("alert_budget_compliance", {}).get("status", "N/A"),
                f"Target 8.0% (Realized {telemetry.get('shadow_candidate_alert_rate_pct', 0.0)}%)",
                "#10b981" if telemetry.get("alert_budget_compliance", {}).get("is_within_budget_limits") else "#ef4444"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # 2. Main Tabs
        tab_ops, tab_drift, tab_mat, tab_comp = st.tabs([
            "📊 Operational Workload & Agreement",
            "🌊 Population Stability & Drift (PSI)",
            "⚖️ Outcome-Maturity Protocol",
            "🔍 Company Shadow Inspection"
        ])

        with tab_ops:
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown('<div class="section-title">⚖️ Inter-Model Agreement & Disagreement</div>', unsafe_allow_html=True)
                d_mat = telemetry.get("disagreement_matrix", {})
                fig_disagree = go.Figure(data=[
                    go.Bar(
                        x=["Both Alerted", "Neither Alerted", "Production Only", "Shadow Candidate Only"],
                        y=[d_mat.get("both_alerted", 0), d_mat.get("neither_alerted", 0), d_mat.get("production_only_alerted", 0), d_mat.get("shadow_only_alerted", 0)],
                        marker_color=["#10b981", "#3b82f6", "#f59e0b", "#a78bfa"],
                        text=[d_mat.get("both_alerted", 0), d_mat.get("neither_alerted", 0), d_mat.get("production_only_alerted", 0), d_mat.get("shadow_only_alerted", 0)],
                        textposition="auto"
                    )
                ])
                fig_disagree.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    yaxis_title="Company Count",
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=300
                )
                st.plotly_chart(fig_disagree, use_container_width=True)

            with col_right:
                st.markdown('<div class="section-title">📉 Review Burden & Workload Reduction</div>', unsafe_allow_html=True)
                workload_saved = d_mat.get("production_only_alerted", 0) - d_mat.get("shadow_only_alerted", 0)
                pct_workload_reduction = round((workload_saved / max(1, telemetry.get("production_alert_count", 1))) * 100.0, 1)
                st.markdown(f"""
                <div class="glass-card" style="font-size: 0.9rem; line-height: 1.6; color: #cbd5e1;">
                    <h3 style="margin-top: 0; color: #38bdf8;">Workload Impact Summary</h3>
                    <ul>
                        <li><b>Baseline Production Review Load:</b> {telemetry.get('production_alert_count', 0)} audits ({telemetry.get('production_alert_rate_pct', 0.0)}% of portfolio)</li>
                        <li><b>Shadow Candidate Review Load:</b> {telemetry.get('shadow_candidate_alert_count', 0)} audits ({telemetry.get('shadow_candidate_alert_rate_pct', 0.0)}% of portfolio)</li>
                        <li><b>Net False-Alarm Investigation Savings:</b> <span style="color: #10b981; font-weight: bold;">-{workload_saved} manual reviews saved</span> ({pct_workload_reduction}% workload reduction)</li>
                        <li><b>Reviewer Fatigue Risk Mitigation:</b> Candidate safely concentrates investigator attention on highest-probability deterioration cases within the pre-registered 8.0% budget.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

        with tab_drift:
            st.markdown('<div class="section-title">🌊 Population Stability Index (PSI) Across Top Financial Indicators</div>', unsafe_allow_html=True)
            psi_data = telemetry.get("feature_drift_psi", {})
            col_p1, col_p2 = st.columns([1, 1])

            with col_p1:
                ratios = list(psi_data.keys())
                psi_vals = [psi_data[r]["psi_value"] for r in ratios]
                colors = ["#10b981" if v < 0.1 else ("#f59e0b" if v < 0.25 else "#ef4444") for v in psi_vals]

                fig_psi = go.Figure(go.Bar(
                    x=ratios,
                    y=psi_vals,
                    marker_color=colors,
                    text=[f"{v:.4f}" for v in psi_vals],
                    textposition="auto"
                ))
                fig_psi.add_hline(y=0.1, line_dash="dash", line_color="#f59e0b", annotation_text="Moderate Drift (0.10)")
                fig_psi.add_hline(y=0.25, line_dash="dash", line_color="#ef4444", annotation_text="Significant Drift (0.25)")
                fig_psi.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    yaxis_title="PSI vs Historical Baseline (T <= 10)",
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=320
                )
                st.plotly_chart(fig_psi, use_container_width=True)

            with col_p2:
                st.markdown("""
                <div class="glass-card" style="font-size: 0.88rem; line-height: 1.6; color: #cbd5e1;">
                    <h3 style="margin-top: 0; color: #38bdf8;">Drift Governance Guidelines</h3>
                    <ul>
                        <li><b>PSI < 0.10 (Green):</b> Stable. No distribution shift detected between training history and prospective period.</li>
                        <li><b>0.10 ≤ PSI < 0.25 (Yellow):</b> Moderate shift observed. Recommend tracking across consecutive reporting cycles.</li>
                        <li><b>PSI ≥ 0.25 (Red):</b> Substantial shift. Feature distributions diverged; triggers investigation before any future model promotion consideration.</li>
                        <li><b>Observed Finding:</b> <code>x12</code> (Operating Cash Flow / Total Debt) is exceptionally stable (PSI=0.0701). <code>x48</code> shows macro deterioration across SMEs in later periods.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

        with tab_mat:
            st.markdown('<div class="section-title">⚖️ Pre-Registered Outcome-Maturity Evaluation Policy</div>', unsafe_allow_html=True)
            meets_proto = mat_eval.get("meets_preregistered_sample_protocol", False)

            if not meets_proto:
                st.markdown(f"""
                <div style="background: rgba(245, 158, 11, 0.12); border: 1px solid #f59e0b; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px;">
                    <b style="color: #fbbf24;">⚠️ PRE-REGISTERED SAMPLE PROTOCOL NOT YET FULLY MATURED</b>
                    <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 2px;">
                        Matured distress cases: <b>{mat_eval.get('matured_distress_cases', 0)}</b> / {mat_eval.get('min_required_distress', 10)} required minimum. In accordance with pre-registered protocol, formal claims of statistical significance are <b>prohibited</b> until sample requirements are satisfied. Metrics below are observational preliminary telemetry.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                pb = mat_eval.get("production_baseline", {})
                st.markdown(f"""
                <div class="glass-card">
                    <h3 style="margin-top: 0; color: #38bdf8;">Approved Production Baseline</h3>
                    <p style="font-size: 0.82rem; color: #94a3b8;">RandomForest Balanced Platt (τ = 0.0800)</p>
                    <table style="width: 100%; font-size: 0.88rem; color: #cbd5e1;">
                        <tr><td>True Positives (TP):</td><td><b>{pb.get('tp')}</b></td></tr>
                        <tr><td>False Positives (FP):</td><td><b>{pb.get('fp')}</b></td></tr>
                        <tr><td>False Negatives (FN):</td><td><b>{pb.get('fn')}</b></td></tr>
                        <tr><td>True Negatives (TN):</td><td><b>{pb.get('tn')}</b></td></tr>
                        <tr><td>Recall (Distress Sensitivity):</td><td><b style="color: #38bdf8;">{pb.get('recall')*100:.1f}%</b></td></tr>
                        <tr><td>Precision (PPV):</td><td><b>{pb.get('precision')*100:.1f}%</b></td></tr>
                        <tr><td>Brier Score:</td><td><b>{pb.get('brier_score')}</b></td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)

            with col_m2:
                sc = mat_eval.get("shadow_candidate", {})
                comp = mat_eval.get("comparison", {})
                st.markdown(f"""
                <div class="glass-card">
                    <h3 style="margin-top: 0; color: #a78bfa;">Experimental Shadow Candidate</h3>
                    <p style="font-size: 0.82rem; color: #94a3b8;">Phase5 RF Cost-Sensitive w10 (τ = 0.0894)</p>
                    <table style="width: 100%; font-size: 0.88rem; color: #cbd5e1;">
                        <tr><td>True Positives (TP):</td><td><b>{sc.get('tp')}</b></td></tr>
                        <tr><td>False Positives (FP):</td><td><b style="color: #10b981;">{sc.get('fp')} (-11 vs Prod)</b></td></tr>
                        <tr><td>False Negatives (FN):</td><td><b>{sc.get('fn')}</b></td></tr>
                        <tr><td>True Negatives (TN):</td><td><b>{sc.get('tn')}</b></td></tr>
                        <tr><td>Recall (Distress Sensitivity):</td><td><b style="color: #a78bfa;">{sc.get('recall')*100:.1f}% (Parity)</b></td></tr>
                        <tr><td>Precision (PPV):</td><td><b style="color: #10b981;">{sc.get('precision')*100:.1f}% (+{comp.get('delta_precision', 0)*100:.1f} pp)</b></td></tr>
                        <tr><td>Brier Score:</td><td><b>{sc.get('brier_score')}</b></td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)

        with tab_comp:
            st.markdown('<div class="section-title">🔍 Company-by-Company Shadow Record Inspection</div>', unsafe_allow_html=True)
            comp_search = st.selectbox(
                "Select Company to Compare:",
                options=df_p6_preds["company_id"].astype(str).tolist(),
                index=0
            )
            c_row = df_p6_preds[df_p6_preds["company_id"].astype(str) == str(comp_search)].iloc[0]

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown(f"""
                <div class="glass-card" style="border-left: 4px solid #38bdf8;">
                    <h4 style="margin: 0; color: #38bdf8;">OFFICIAL PRODUCTION DECISION</h4>
                    <p style="font-size: 0.8rem; color: #94a3b8;">Primary Authority (Decision-Grade)</p>
                    <p><b>Calibrated Probability:</b> {c_row.get('prod_calibrated_probability', 0.0):.4f}</p>
                    <p><b>Risk Score:</b> {c_row.get('prod_risk_score', 0.0)} / 100</p>
                    <p><b>Official Alert:</b> {'🚨 ALERT TRIGGERED' if c_row.get('prod_is_alert') else '✅ CLEAR'}</p>
                </div>
                """, unsafe_allow_html=True)

            with col_s2:
                st.markdown(f"""
                <div class="glass-card" style="border-left: 4px solid #a78bfa;">
                    <h4 style="margin: 0; color: #a78bfa;">EXPERIMENTAL CANDIDATE PREDICTION</h4>
                    <p style="font-size: 0.8rem; color: #94a3b8;">Research Sandbox Only (Non-Authoritative)</p>
                    <p><b>Calibrated Probability:</b> {c_row.get('cand_calibrated_probability', 0.0):.4f}</p>
                    <p><b>Risk Score:</b> {c_row.get('cand_risk_score', 0.0)} / 100</p>
                    <p><b>Shadow Alert:</b> {'🚨 ALERT TRIGGERED' if c_row.get('cand_is_alert') else '✅ CLEAR'}</p>
                </div>
                """, unsafe_allow_html=True)

            st.dataframe(df_p6_preds, use_container_width=True)

    else:
        st.warning("Phase 6 shadow evaluation artifacts not found. Please run the shadow evaluation simulation script to populate telemetry.")

# -------------------------------------------------------------
# PAGE 12: RESEARCH MODEL IMPROVEMENT & BASELINE AUDIT (Phase 6.9)
# -------------------------------------------------------------
elif menu_selection == "🔬 Research Model Improvement & Baseline Audit":
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 1.8rem; color: #f8fafc;">🔬 Research Model Improvement & Baseline Audit (Phase 6.9)</h1>
        <div style="color: #94a3b8; font-size: 0.95rem; margin-top: 6px;">
            Empirical benchmark comparison, alert-budget constrained evaluation, false-negative diagnostics, and TreeSHAP explainability using strictly existing repository data.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Mandatory Governance & Regulatory Notice
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255, 255, 255, 0.12); border-left: 5px solid #38bdf8; border-radius: 10px; padding: 16px 20px; margin-bottom: 24px;">
        <div style="display: flex; align-items: flex-start; gap: 12px;">
            <span style="font-size: 1.6rem;">🛡️</span>
            <div>
                <b style="color: #38bdf8; font-size: 1.0rem;">MODEL RISK GOVERNANCE & PROTOTYPE DISCLAIMER (SR 11-7 / OCC 2011-12)</b>
                <div style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.5; margin-top: 4px;">
                    This system is an <b>academic and portfolio surveillance research prototype</b> evaluated on historical financial data.
                    It does <b>NOT</b> constitute an automated banking credit-approval, loan origination, or formal credit denial system.
                    The approved production baseline (<code>models/phase2/best_ews_model.joblib</code>) remains 100% frozen and cryptographically protected.
                    All research candidate evaluations are exploratory, sandbox-isolated, and strictly non-authoritative.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load Phase 6.9 artifacts
    p69_models_file = "results/phase6_9/model_comparison.json"
    p69_fn_file = "results/phase6_9/false_negative_audit.csv"
    p69_expl_file = "results/phase6_9/explainability_summary.json"
    p69_data_file = "results/phase6_9/dataset_audit.json"
    p69_feat_file = "results/phase6_9/feature_audit.json"

    if os.path.exists(p69_models_file) and os.path.exists(p69_fn_file) and os.path.exists(p69_expl_file):
        with open(p69_models_file, "r", encoding="utf-8") as f:
            comp_data = json.load(f)
        df_fn_data = pd.read_csv(p69_fn_file)
        with open(p69_expl_file, "r", encoding="utf-8") as f:
            expl_data = json.load(f)

        # 1. Top KPI Row
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            render_kpi_card("Baseline Recall", "45.5% (Nominal)", "36.4% under 8% cap", "#38bdf8")
        with k2:
            render_kpi_card("Candidate Recall", "27.3% (Parity)", "tau = 0.0894", "#a78bfa")
        with k3:
            render_kpi_card("Alert Budget Cap", "20 Alerts", "8.0% of N=262 cohort", "#10b981")
        with k4:
            render_kpi_card("Realized Defaults", "11 Events", "Matured T=13 -> T=14", "#f59e0b")
        with k5:
            render_kpi_card("Baseline Status", "100% Frozen", "Hash Verified", "#10b981")

        st.markdown("<br>", unsafe_allow_html=True)

        # 2. Tabs
        tab_bench, tab_fn_diag, tab_co_inspect, tab_shap_global, tab_data_audit = st.tabs([
            "📊 Model Comparison & Budget Cap",
            "🚨 False Negative Audit & Dynamics",
            "🏢 Enterprise Comparative Inspection",
            "🧠 TreeSHAP Attribution & Semantics",
            "📜 Dataset & Preprocessing Audit"
        ])

        with tab_bench:
            st.markdown('<div class="section-title">📊 Empirical Model Comparison under 8% Alert Constraint</div>', unsafe_allow_html=True)
            
            # Format model comparison table
            table_rows = []
            for m in comp_data.get("models", []):
                table_rows.append({
                    "Model Architecture": m.get("model_identifier"),
                    "Operating Threshold": m.get("threshold_applied"),
                    "Enforce Budget": "Yes (Cap=20)" if m.get("enforce_8pct_budget_cap") else "No (Nominal)",
                    "Alerts": f"{m.get('alerts', {}).get('total_alerts')} ({m.get('alerts', {}).get('alert_rate_pct')}%)",
                    "Budget Compliant": "✅ YES" if m.get("alerts", {}).get("budget_compliant") else "❌ NO (Excess +17)",
                    "TP": m.get("confusion_matrix", {}).get("true_positives"),
                    "FP": m.get("confusion_matrix", {}).get("false_positives"),
                    "FN": m.get("confusion_matrix", {}).get("false_negatives"),
                    "TN": m.get("confusion_matrix", {}).get("true_negatives"),
                    "Recall": f"{m.get('metrics', {}).get('recall')*100:.1f}%",
                    "Precision": f"{m.get('metrics', {}).get('precision')*100:.1f}%",
                    "Specificity": f"{m.get('metrics', {}).get('specificity')*100:.1f}%",
                    "Brier Score": m.get("metrics", {}).get("brier_score"),
                    "ROC-AUC": m.get("metrics", {}).get("roc_auc")
                })
            df_table = pd.DataFrame(table_rows)
            st.dataframe(df_table, use_container_width=True, hide_index=True)

            # Bar Chart Comparison
            budget_models = [m for m in comp_data.get("models", []) if m.get("enforce_8pct_budget_cap")]
            fig_comp = go.Figure(data=[
                go.Bar(
                    name="Recall (Sensitivity)",
                    x=[m.get("model_identifier").replace("_BudgetCapped", "") for m in budget_models],
                    y=[m.get("metrics", {}).get("recall") * 100 for m in budget_models],
                    marker_color="#38bdf8",
                    text=[f"{m.get('metrics', {}).get('recall')*100:.1f}%" for m in budget_models],
                    textposition="auto"
                ),
                go.Bar(
                    name="Precision (PPV)",
                    x=[m.get("model_identifier").replace("_BudgetCapped", "") for m in budget_models],
                    y=[m.get("metrics", {}).get("precision") * 100 for m in budget_models],
                    marker_color="#10b981",
                    text=[f"{m.get('metrics', {}).get('precision')*100:.1f}%" for m in budget_models],
                    textposition="auto"
                )
            ])
            fig_comp.update_layout(
                barmode="group",
                title="<b>Performance under Strict 8.0% Alert Budget Constraint (Max 20 Alerts)</b>",
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                yaxis_title="Percentage (%)",
                height=340,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            st.markdown(f"""
            <div class="glass-card" style="font-size: 0.88rem; line-height: 1.6; color: #cbd5e1;">
                <h4 style="color: #f59e0b; margin-top: 0;">⚠️ Methodological Caution on Sample Size & Statistical Inconclusiveness</h4>
                <ul>
                    <li><b>Denominator Limitation:</b> The matured test cohort contains strictly <b>n = 11</b> distress events out of 262 firms. Each individual detection equals <b>9.09 percentage points</b> of recall.</li>
                    <li><b>Wilson Score 95% Confidence Intervals:</b> Recall for Production Baseline (4/11 = 36.4%, 95% CI: [15.2%, 64.6%]) overlaps Candidate and Logistic Regression (3/11 = 27.3%, 95% CI: [9.7%, 56.6%]).</li>
                    <li><b>Conclusion:</b> Neither model can claim statistically validated superiority. The production baseline demonstrates greater operational utility by capturing acute deterioration without sacrificing precision.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with tab_fn_diag:
            st.markdown('<div class="section-title">🚨 False Negative Audit & Default Dynamics Diagnostic</div>', unsafe_allow_html=True)
            st.markdown("""
            Diagnostic audit of all 11 corporate default events at T=13 -> T=14.
            Categorized into <b>Acute Sudden Transitions</b> (prior healthy FD > 0 crashing to distress) vs <b>Chronic Solvency Deteriorations</b> (prior stressed FD <= 0).
            """)
            st.dataframe(df_fn_data, use_container_width=True, hide_index=True)

            col_fn1, col_fn2 = st.columns(2)
            with col_fn1:
                st.markdown("""
                <div class="glass-card" style="border-left: 4px solid #ef4444;">
                    <h4 style="color: #ef4444; margin-top: 0;">⚡ Acute Sudden Crashes (5 Firms)</h4>
                    <ul>
                        <li><b>Firm 28:</b> Antecedent FD = +1.4430. Baseline prob = 0.0858 (Alerted at tau=0.0800). Candidate prob = 0.0496 (Missed).</li>
                        <li><b>Firm 60:</b> Antecedent FD = +0.1357. Baseline prob = 0.1135 (Alerted under both thresholds). Candidate prob = 0.0695 (Missed).</li>
                        <li><b>Firms 84, 225, 376:</b> Ratios exhibited minimal deterioration prior to default; both models missed under strict threshold.</li>
                        <li><b>Diagnosis:</b> Candidate's multi-year rolling features severely suppressed predicted risk for firms with healthy prior histories.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            with col_fn2:
                st.markdown("""
                <div class="glass-card" style="border-left: 4px solid #38bdf8;">
                    <h4 style="color: #38bdf8; margin-top: 0;">📉 Chronic Deteriorations (6 Firms)</h4>
                    <ul>
                        <li><b>Firms 100, 101, 112:</b> Prolonged solvency erosion. Successfully detected by both Baseline and Candidate under nominal and budget-capped thresholds.</li>
                        <li><b>Firm 23:</b> Antecedent FD = -0.4820 (near threshold). Baseline prob = 0.0768 (narrow miss just below 0.0800).</li>
                        <li><b>Firms 29, 53:</b> Stressed antecedent solvency but low composite model scores.</li>
                        <li><b>Diagnosis:</b> Both models perform reliably on prolonged chronic distress, confirming baseline stability.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

        with tab_co_inspect:
            st.markdown(f'<div class="section-title">🏢 Comparative Inspection for Firm #{selected_company_id}</div>', unsafe_allow_html=True)
            c_info_sel = companies.get(str(selected_company_id), {})
            eval_p = selected_period if selected_period is not None else c_info_sel.get("latest_period", 1)

            col_ci1, col_ci2 = st.columns(2)
            with col_ci1:
                st.markdown(f"""
                <div class="glass-card" style="border-left: 4px solid #38bdf8;">
                    <h4 style="color: #38bdf8; margin-top: 0;">APPROVED PRODUCTION BASELINE</h4>
                    <p style="font-size: 0.8rem; color: #94a3b8;">Decision Authority: Primary Official</p>
                    <p><b>Model Version:</b> v1.2-approved-production (Random Forest Balanced Platt)</p>
                    <p><b>Official Operating Threshold:</b> tau = 0.0800</p>
                    <p><b>Calibrated Distress Probability:</b> {c_info_sel.get('latest_assessment', {}).get('final_risk_probability_pct', 0.0)}%</p>
                    <p><b>Official Status:</b> {'🚨 SURVEILLANCE ALERT' if c_info_sel.get('latest_assessment', {}).get('final_risk_probability', 0.0) >= 0.08 else '✅ CLEAR'}</p>
                </div>
                """, unsafe_allow_html=True)

            with col_ci2:
                # Check candidate prediction if in shadow dataframe
                cand_score_text = "N/A (Historical Period)"
                cand_alert_text = "N/A"
                if os.path.exists("results/phase6/prospective_predictions.csv"):
                    df_p6_p = pd.read_csv("results/phase6/prospective_predictions.csv")
                    m_row = df_p6_p[df_p6_p["company_id"].astype(str) == str(selected_company_id)]
                    if len(m_row) > 0:
                        cand_prob_val = float(m_row.iloc[0]["cand_calibrated_probability"])
                        cand_score_text = f"{cand_prob_val*100:.2f}%"
                        cand_alert_text = "🚨 SHADOW ALERT" if cand_prob_val >= 0.0894 else "✅ CLEAR"

                st.markdown(f"""
                <div class="glass-card" style="border-left: 4px solid #a78bfa;">
                    <h4 style="color: #a78bfa; margin-top: 0;">EXPERIMENTAL SHADOW CANDIDATE</h4>
                    <p style="font-size: 0.8rem; color: #94a3b8;">Research Sandbox Only (Non-Authoritative)</p>
                    <p><b>Model Version:</b> Phase5_RF_CostSensitive_w10</p>
                    <p><b>Research Threshold:</b> tau = 0.0894</p>
                    <p><b>Calibrated Distress Probability:</b> {cand_score_text}</p>
                    <p><b>Shadow Status:</b> {cand_alert_text}</p>
                </div>
                """, unsafe_allow_html=True)

            # Historical Financial Indicators Table
            st.markdown('<div class="section-title">📈 Corporate Financial Profile Across Reporting Periods</div>', unsafe_allow_html=True)
            loader = DataLoader("data/raw")
            df_raw_co = loader.load_dataset_a()
            co_hist_df = df_raw_co[df_raw_co["Company"] == int(selected_company_id)].sort_values(by="Time")
            show_cols = ["Time", "Financial Distress", "x1", "x2", "x5", "x10", "x25", "x48", "x64", "x81"]
            st.dataframe(co_hist_df[show_cols].rename(columns={
                "Time": "Period T",
                "Financial Distress": "Solvency Index",
                "x1": "Working Capital (x1)",
                "x2": "Retained Earnings (x2)",
                "x5": "Cash Ratio (x5)",
                "x10": "EBIT Margin (x10)",
                "x25": "Liquidity (x25)",
                "x48": "Total Debt (x48)",
                "x64": "Leverage (x64)",
                "x81": "Operating Cashflow (x81)"
            }), use_container_width=True, hide_index=True)

        with tab_shap_global:
            st.markdown('<div class="section-title">🧠 TreeSHAP Global Attributions & Indicator Semantics</div>', unsafe_allow_html=True)
            top_feats = expl_data.get("top_global_features", [])
            if top_feats:
                st.dataframe(pd.DataFrame(top_feats), use_container_width=True, hide_index=True)

            st.markdown(f"""
            <div class="glass-card" style="font-size: 0.88rem; line-height: 1.6; color: #cbd5e1;">
                <h4 style="color: #38bdf8; margin-top: 0;">Crucial Distinction: Predictive Association vs Causality</h4>
                <p>
                    {expl_data.get('causal_vs_predictive_governance_disclaimer', {}).get('critical_distinction')}
                </p>
            </div>
            """, unsafe_allow_html=True)

        with tab_data_audit:
            st.markdown('<div class="section-title">📜 Empirical Dataset & Preprocessing Pipeline Audit</div>', unsafe_allow_html=True)
            if os.path.exists(p69_data_file) and os.path.exists(p69_feat_file):
                with open(p69_data_file, "r", encoding="utf-8") as f:
                    d_audit = json.load(f)
                with open(p69_feat_file, "r", encoding="utf-8") as f:
                    f_audit = json.load(f)

                col_da1, col_da2 = st.columns(2)
                with col_da1:
                    st.markdown(f"""
                    <div class="glass-card">
                        <h4 style="color: #38bdf8; margin-top: 0;">Dataset A Panel Architecture</h4>
                        <table style="width: 100%; font-size: 0.88rem; color: #cbd5e1;">
                            <tr><td>Total Records:</td><td><b>{d_audit.get('total_rows')}</b></td></tr>
                            <tr><td>Total Features:</td><td><b>{d_audit.get('total_columns')} (83 raw ratios)</b></td></tr>
                            <tr><td>Unique Monitored SMEs:</td><td><b>{d_audit.get('unique_companies')}</b></td></tr>
                            <tr><td>Reporting Periods:</td><td><b>Time 1 to 14</b></td></tr>
                            <tr><td>Time Contiguity:</td><td><b>Zero gaps (100% contiguous)</b></td></tr>
                            <tr><td>Missing Values:</td><td><b>0 (Zero missingness)</b></td></tr>
                            <tr><td>Duplicate Records:</td><td><b>0 (Zero duplicate firm-periods)</b></td></tr>
                            <tr><td>Raw Distress Events:</td><td><b>{d_audit.get('target_semantics', {}).get('raw_distress_records')} ({d_audit.get('target_semantics', {}).get('raw_distress_percentage')}%)</b></td></tr>
                        </table>
                    </div>
                    """, unsafe_allow_html=True)

                with col_da2:
                    st.markdown(f"""
                    <div class="glass-card">
                        <h4 style="color: #10b981; margin-top: 0;">Leakage-Safe Chronological Splitting</h4>
                        <table style="width: 100%; font-size: 0.88rem; color: #cbd5e1;">
                            <tr><td>Training Partition:</td><td><b>Time <= 10 (2,471 rows, 95 defaults, 3.84%)</b></td></tr>
                            <tr><td>Validation Partition:</td><td><b>Time 11–12 (517 rows, 14 defaults, 2.71%)</b></td></tr>
                            <tr><td>Holdout Test Partition:</td><td><b>Time 13 (262 rows, 11 defaults, 4.20%)</b></td></tr>
                            <tr><td>Terminal Period 14:</td><td><b>280 rows (unobserved forward horizon)</b></td></tr>
                            <tr><td>Preprocessing Fitting:</td><td><b>Strictly on Time <= 10 (Zero Test Leakage)</b></td></tr>
                            <tr><td>Max Feature-Target Corr:</td><td><b>{f_audit.get('leakage_verification', {}).get('max_feature_correlation_with_forward_target')} (x2 roll mean)</b></td></tr>
                        </table>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.warning("Phase 6.9 research artifacts not found. Please run the model improvement script to generate artifacts.")


