import streamlit as st

def render_kpi_card(title: str, value: str, subtitle: str = "", delta: str = None, color: str = None):
    """
    Renders a styled KPI metric card with strictly uniform dimensions across all grid columns.
    Uses flexbox vertical distribution and adaptive font sizing so every card in a row has identical size.
    """
    # Gracefully handle callers passing color positionally as 4th argument
    if delta and (str(delta).startswith("#") or str(delta).startswith("rgba")):
        color = delta
        delta = None
    if color is None:
        color = "#3b82f6"

    delta_html = f"<div style='font-size: 0.78rem; font-weight: 500; color: {'#10b981' if '+' in str(delta) else '#ef4444'}; margin-top: 2px;'>{delta}</div>" if delta else ""
    subtitle_html = f"<div style='font-size: 0.76rem; color: #94a3b8; margin-top: 2px; line-height: 1.2;'>{subtitle}</div>" if subtitle else ""
    
    # Auto-adjust font size for longer values (e.g., 'Critical → Critical') so text fits cleanly
    val_str = str(value)
    if len(val_str) > 14:
        val_size = "1.25rem"
    elif len(val_str) > 9:
        val_size = "1.50rem"
    else:
        val_size = "1.80rem"
    
    html = (
        f'<div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8)); '
        f'border: 1px solid rgba(255, 255, 255, 0.08); border-left: 4px solid {color}; border-radius: 10px; '
        f'padding: 14px 18px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25); backdrop-filter: blur(10px); '
        f'margin-bottom: 12px; height: 136px; box-sizing: border-box; display: flex; flex-direction: column; '
        f'justify-content: space-between;">'
        f'<div>'
        f'<div style="font-size: 0.76rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; '
        f'color: #94a3b8; line-height: 1.25; min-height: 28px; display: flex; align-items: flex-start;">{title}</div>'
        f'<div style="font-size: {val_size}; font-weight: 700; color: #f8fafc; margin: 2px 0; line-height: 1.2;">{value}</div>'
        f'</div>'
        f'<div style="margin-top: auto;">'
        f'{delta_html}'
        f'{subtitle_html}'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def render_badge(text: str, color: str = "#3b82f6", bg_opacity: float = 0.15):
    """Renders a sleek pill badge."""
    return (
        f'<span style="background-color: {color}26; color: {color}; border: 1px solid {color}4D; '
        f'padding: 4px 12px; border-radius: 9999px; font-size: 0.85rem; font-weight: 600; display: inline-block;">'
        f'{text}</span>'
    )
