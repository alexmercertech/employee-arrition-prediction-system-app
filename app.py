"""
Employee Attrition Prediction System — Main Application
=========================================================

This is the main entry point for the Streamlit multi-page application.
It serves as the Home/Landing page with an overview of the system's
capabilities and navigation to individual analysis pages.

Usage:
    streamlit run app.py
"""

import streamlit as st
from utils.ui_components import apply_custom_css, render_feature_card, render_divider, render_sidebar_info


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Employee Attrition Prediction System",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "# Employee Attrition Prediction System\nAn ML-powered HR analytics platform.",
    },
)

# Apply custom styling
apply_custom_css()
render_sidebar_info()


# ---------------------------------------------------------------------------
# Hero Section
# ---------------------------------------------------------------------------

st.markdown("""
<div class="hero-section">
    <div class="hero-title">Employee Attrition<br>Prediction System</div>
    <div class="hero-subtitle">
        An end-to-end machine learning analytics platform that predicts employee
        attrition using the IBM HR Analytics dataset. Demonstrates the complete
        ML lifecycle from data exploration to explainable predictions.
    </div>
</div>
""", unsafe_allow_html=True)

render_divider()


# ---------------------------------------------------------------------------
# Feature Cards
# ---------------------------------------------------------------------------

st.markdown("### 🧭 Platform Features")
st.markdown("")

col1, col2, col3, col4 = st.columns(4)

with col1:
    render_feature_card(
        "📊", "Interactive Dashboard",
        "Real-time KPI metrics, attrition rates, and department-level analytics with dynamic Plotly charts."
    )

with col2:
    render_feature_card(
        "🔍", "Data Exploration",
        "Interactive EDA with filters for department, gender, role, and attrition status. Business insights for every chart."
    )

with col3:
    render_feature_card(
        "⚙️", "Data Preprocessing",
        "Step-by-step pipeline: cleaning, encoding, scaling, feature engineering, and SMOTE class balancing."
    )

with col4:
    render_feature_card(
        "🤖", "Model Training",
        "Train 4 ML models (Logistic Regression, Decision Tree, Random Forest, XGBoost) with hyperparameter tuning."
    )

st.markdown("")

col5, col6, col7, col8 = st.columns(4)

with col5:
    render_feature_card(
        "📈", "Model Comparison",
        "Side-by-side metrics, ROC/PR curves, and automatic best model recommendation with written rationale."
    )

with col6:
    render_feature_card(
        "🎯", "Employee Prediction",
        "HR prediction form with risk levels, probability scores, and context-aware retention recommendations."
    )

with col7:
    render_feature_card(
        "🧠", "Explainability",
        "SHAP-based explanations: summary plots, waterfall charts, and human-readable factor analysis."
    )

with col8:
    render_feature_card(
        "📋", "Full Documentation",
        "Complete code documentation, design decisions, algorithm explanations, and deployment guides."
    )


render_divider()


# ---------------------------------------------------------------------------
# System Architecture
# ---------------------------------------------------------------------------

st.markdown("### 🏗️ System Architecture")
st.markdown("")

col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.markdown("""
    <div class="info-card">
        <h4 style="color: #6C63FF; margin-bottom: 1rem;">Machine Learning Pipeline</h4>
        <div style="font-family: monospace; font-size: 0.85rem; line-height: 2; color: rgba(250,250,250,0.8);">
            📁 Data Loading & Validation<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            🔍 Exploratory Data Analysis<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            🧹 Data Cleaning & Preprocessing<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            🔧 Feature Engineering (8 derived features)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            🏷️ Encoding (Label + One-Hot)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            📐 Feature Scaling (StandardScaler)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            ⚖️ Class Balancing (SMOTE)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            🤖 Model Training & Cross-Validation<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            📊 Evaluation & Comparison<br>
            &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            🧠 SHAP Explainability
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div class="info-card">
        <h4 style="color: #00D4AA; margin-bottom: 1rem;">Technology Stack</h4>
        <table style="width: 100%; font-size: 0.85rem; color: rgba(250,250,250,0.8);">
            <tr><td style="padding: 4px 8px;">🐍 Python</td><td style="padding: 4px 8px;">3.12+</td></tr>
            <tr><td style="padding: 4px 8px;">🎈 Streamlit</td><td style="padding: 4px 8px;">Multi-page app</td></tr>
            <tr><td style="padding: 4px 8px;">🐼 Pandas</td><td style="padding: 4px 8px;">Data manipulation</td></tr>
            <tr><td style="padding: 4px 8px;">🔢 NumPy</td><td style="padding: 4px 8px;">Numerical computing</td></tr>
            <tr><td style="padding: 4px 8px;">🧠 Scikit-Learn</td><td style="padding: 4px 8px;">ML models</td></tr>
            <tr><td style="padding: 4px 8px;">⚡ XGBoost</td><td style="padding: 4px 8px;">Gradient boosting</td></tr>
            <tr><td style="padding: 4px 8px;">💡 SHAP</td><td style="padding: 4px 8px;">Explainability</td></tr>
            <tr><td style="padding: 4px 8px;">📊 Plotly</td><td style="padding: 4px 8px;">Interactive charts</td></tr>
            <tr><td style="padding: 4px 8px;">⚖️ Imbalanced-Learn</td><td style="padding: 4px 8px;">SMOTE</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card" style="margin-top: 1rem;">
        <h4 style="color: #FFD93D; margin-bottom: 0.75rem;">Dataset</h4>
        <div style="font-size: 0.85rem; color: rgba(250,250,250,0.8); line-height: 1.6;">
            <strong>IBM HR Analytics Employee Attrition</strong><br>
            • 1,470 employees × 35 features<br>
            • Target: Attrition (Yes / No)<br>
            • Class imbalance: ~16% attrition rate
        </div>
    </div>
    """, unsafe_allow_html=True)


render_divider()


# ---------------------------------------------------------------------------
# Quick Start
# ---------------------------------------------------------------------------

st.markdown("### 🚀 Quick Start")
st.markdown("")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("""
    <div class="info-card">
        <h4 style="color: #6C63FF;">Step 1: Explore</h4>
        <p style="font-size: 0.85rem; color: rgba(250,250,250,0.6); margin-top: 0.5rem;">
            Visit the <strong>Dashboard</strong> and <strong>Data Exploration</strong> pages
            to understand the dataset and key attrition patterns.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <div class="info-card">
        <h4 style="color: #00D4AA;">Step 2: Train</h4>
        <p style="font-size: 0.85rem; color: rgba(250,250,250,0.6); margin-top: 0.5rem;">
            Go to <strong>Model Training</strong> to train ML models with
            hyperparameter tuning. Compare results in <strong>Model Comparison</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_c:
    st.markdown("""
    <div class="info-card">
        <h4 style="color: #FF6B6B;">Step 3: Predict</h4>
        <p style="font-size: 0.85rem; color: rgba(250,250,250,0.6); margin-top: 0.5rem;">
            Use <strong>Employee Prediction</strong> to assess individual employee
            attrition risk with SHAP-powered <strong>Explainability</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("")
st.markdown("")
st.markdown("""
<div style="text-align: center; padding: 2rem 0; color: rgba(250,250,250,0.3); font-size: 0.8rem;">
    Employee Attrition Prediction System • Built with Streamlit & Scikit-Learn<br>
    COM763 Assessment • Machine Learning Analytics Platform
</div>
""", unsafe_allow_html=True)
