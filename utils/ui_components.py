"""
UI Components Module
=====================

Purpose:
    Provides reusable Streamlit UI elements and custom CSS styling for
    a premium enterprise dashboard appearance.

Design Decisions:
    - Glassmorphism-style KPI cards with backdrop blur for a modern,
      premium aesthetic.
    - Google Fonts (Inter) for clean typography.
    - Custom CSS is injected via st.markdown with unsafe_allow_html=True,
      which is the standard approach for Streamlit styling.
    - Color palette: Indigo primary (#6C63FF), Teal accent (#00D4AA),
      Coral warning (#FF6B6B).

Assumptions:
    - The app uses the dark theme configured in .streamlit/config.toml.
    - Streamlit >= 1.28.0 is used.
"""

import streamlit as st


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

def apply_custom_css():
    """
    Inject global custom CSS for the enterprise dashboard look.

    This should be called once at the top of each page to ensure
    consistent styling across the application.
    """
    st.markdown("""
    <style>
        /* ---- Google Font Import ---- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* ---- Global Styles ---- */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* ---- Main Container ---- */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        /* ---- KPI Card ---- */
        .kpi-card {
            background: linear-gradient(135deg, rgba(108, 99, 255, 0.1), rgba(0, 212, 170, 0.05));
            border: 1px solid rgba(108, 99, 255, 0.2);
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            transition: all 0.3s ease;
            min-height: 140px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .kpi-card:hover {
            transform: translateY(-4px);
            border-color: rgba(108, 99, 255, 0.5);
            box-shadow: 0 8px 32px rgba(108, 99, 255, 0.15);
        }

        .kpi-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }

        .kpi-value {
            font-size: 2rem;
            font-weight: 700;
            color: #FAFAFA;
            line-height: 1.2;
        }

        .kpi-label {
            font-size: 0.85rem;
            font-weight: 400;
            color: rgba(250, 250, 250, 0.6);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 0.3rem;
        }

        .kpi-delta {
            font-size: 0.8rem;
            font-weight: 500;
            margin-top: 0.25rem;
        }

        .kpi-delta-positive {
            color: #00D4AA;
        }

        .kpi-delta-negative {
            color: #FF6B6B;
        }

        /* ---- Section Header ---- */
        .section-header {
            font-size: 1.8rem;
            font-weight: 700;
            color: #FAFAFA;
            margin-bottom: 0.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid rgba(108, 99, 255, 0.3);
        }

        .section-subtitle {
            font-size: 1rem;
            color: rgba(250, 250, 250, 0.5);
            margin-bottom: 1.5rem;
        }

        /* ---- Info Card ---- */
        .info-card {
            background: rgba(26, 31, 46, 0.8);
            border: 1px solid rgba(108, 99, 255, 0.15);
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
        }

        /* ---- Risk Badges ---- */
        .risk-badge {
            display: inline-block;
            padding: 0.4rem 1.2rem;
            border-radius: 50px;
            font-weight: 600;
            font-size: 0.9rem;
            letter-spacing: 0.05em;
        }

        .risk-high {
            background: rgba(255, 107, 107, 0.2);
            color: #FF6B6B;
            border: 1px solid rgba(255, 107, 107, 0.4);
        }

        .risk-medium {
            background: rgba(255, 217, 61, 0.2);
            color: #FFD93D;
            border: 1px solid rgba(255, 217, 61, 0.4);
        }

        .risk-low {
            background: rgba(0, 212, 170, 0.2);
            color: #00D4AA;
            border: 1px solid rgba(0, 212, 170, 0.4);
        }

        /* ---- Feature Cards ---- */
        .feature-card {
            background: linear-gradient(135deg, rgba(26, 31, 46, 0.9), rgba(26, 31, 46, 0.6));
            border: 1px solid rgba(108, 99, 255, 0.15);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
            height: 100%;
        }

        .feature-card:hover {
            transform: translateY(-2px);
            border-color: rgba(108, 99, 255, 0.4);
        }

        .feature-card h3 {
            color: #6C63FF;
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
        }

        .feature-card p {
            color: rgba(250, 250, 250, 0.6);
            font-size: 0.85rem;
            line-height: 1.4;
        }

        /* ---- Prediction Result ---- */
        .prediction-result {
            background: linear-gradient(135deg, rgba(108, 99, 255, 0.1), rgba(0, 0, 0, 0));
            border: 2px solid rgba(108, 99, 255, 0.3);
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
        }

        .prediction-stay {
            border-color: rgba(0, 212, 170, 0.5);
            background: linear-gradient(135deg, rgba(0, 212, 170, 0.08), rgba(0, 0, 0, 0));
        }

        .prediction-leave {
            border-color: rgba(255, 107, 107, 0.5);
            background: linear-gradient(135deg, rgba(255, 107, 107, 0.08), rgba(0, 0, 0, 0));
        }

        /* ---- Tabs Styling ---- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0px 0px;
            padding: 10px 20px;
        }

        /* ---- Sidebar ---- */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0E1117 0%, #1A1F2E 100%);
        }

        /* ---- Expander ---- */
        .streamlit-expanderHeader {
            font-weight: 600;
        }

        /* ---- Metric Cards ---- */
        [data-testid="stMetric"] {
            background: rgba(26, 31, 46, 0.6);
            border: 1px solid rgba(108, 99, 255, 0.15);
            border-radius: 12px;
            padding: 1rem;
        }

        /* ---- Scrollbar ---- */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #0E1117;
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(108, 99, 255, 0.3);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: rgba(108, 99, 255, 0.5);
        }

        /* ---- Divider ---- */
        .styled-divider {
            height: 2px;
            background: linear-gradient(90deg, rgba(108, 99, 255, 0.5), rgba(0, 212, 170, 0.5), rgba(108, 99, 255, 0.05));
            border: none;
            margin: 2rem 0;
            border-radius: 1px;
        }

        /* ---- Hero Section ---- */
        .hero-section {
            text-align: center;
            padding: 3rem 1rem;
        }

        .hero-title {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #6C63FF, #00D4AA);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 1rem;
            line-height: 1.2;
        }

        .hero-subtitle {
            font-size: 1.15rem;
            color: rgba(250, 250, 250, 0.6);
            max-width: 700px;
            margin: 0 auto;
            line-height: 1.6;
        }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Reusable Components
# ---------------------------------------------------------------------------

def render_kpi_card(icon: str, value: str, label: str, delta: str = "", delta_positive: bool = True):
    """
    Render a glassmorphism-style KPI card.

    Args:
        icon: Emoji icon for the card.
        value: Main metric value (formatted string).
        label: Metric label text.
        delta: Optional delta/change value.
        delta_positive: Whether the delta is positive (green) or negative (red).
    """
    delta_html = ""
    if delta:
        delta_class = "kpi-delta-positive" if delta_positive else "kpi-delta-negative"
        delta_html = f'<div class="kpi-delta {delta_class}">{delta}</div>'

    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_header(title: str, subtitle: str = ""):
    """
    Render a styled section header with optional subtitle.

    Args:
        title: Header title text.
        subtitle: Optional subtitle text.
    """
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def render_risk_badge(risk_level: str):
    """
    Render a color-coded risk level badge.

    Args:
        risk_level: One of "High Risk", "Medium Risk", or "Low Risk".
    """
    risk_class_map = {
        "High Risk": "risk-high",
        "Medium Risk": "risk-medium",
        "Low Risk": "risk-low",
    }
    css_class = risk_class_map.get(risk_level, "risk-low")
    st.markdown(
        f'<span class="risk-badge {css_class}">{risk_level}</span>',
        unsafe_allow_html=True,
    )


def render_feature_card(icon: str, title: str, description: str):
    """
    Render a feature card for the home page.

    Args:
        icon: Emoji icon.
        title: Feature title.
        description: Feature description.
    """
    st.markdown(f"""
    <div class="feature-card">
        <div style="font-size: 2.5rem; margin-bottom: 0.75rem;">{icon}</div>
        <h3>{title}</h3>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)


def render_divider():
    """Render a styled gradient divider."""
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)


def render_info_card(content: str):
    """
    Render an info card with styled background.

    Args:
        content: Markdown content to display inside the card.
    """
    st.markdown(f"""
    <div class="info-card">
        {content}
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_info():
    """Render sidebar branding and system information."""
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">👥</div>
        <div style="font-size: 1.1rem; font-weight: 700;
            background: linear-gradient(135deg, #6C63FF, #00D4AA);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;">
            Attrition Predictor
        </div>
        <div style="font-size: 0.75rem; color: rgba(250,250,250,0.4); margin-top: 0.25rem;">
            ML Analytics Platform v1.0
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("---")
