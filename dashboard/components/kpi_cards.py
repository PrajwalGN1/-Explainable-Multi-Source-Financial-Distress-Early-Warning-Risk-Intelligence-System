import streamlit as st

def render_kpi_card(title: str, value: str, subtitle: str = "", delta: str = None, color: str = "#3b82f6"):
    """Renders a styled KPI metric card with rich aesthetics."""
    delta_html = f"<div style='font-size: 0.82rem; color: {'#10b981' if '+' in str(delta) else '#ef4444'}; margin-top: 4px;'>{delta}</div>" if delta else ""
    subtitle_html = f"<div style='font-size: 0.78rem; color: #94a3b8; margin-top: 2px;'>{subtitle}</div>" if subtitle else ""
    
    html = f"""
    <div style="
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 4px solid {color};
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(10px);
        margin-bottom: 12px;
    ">
        <div style="font-size: 0.80rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8;">{title}</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: #f8fafc; margin: 4px 0;">{value}</div>
        {delta_html}
        {subtitle_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_badge(text: str, color: str = "#3b82f6", bg_opacity: float = 0.15):
    """Renders a sleek pill badge."""
    return f"""
    <span style="
        background-color: {color}26;
        color: {color};
        border: 1px solid {color}4D;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    ">{text}</span>
    """
