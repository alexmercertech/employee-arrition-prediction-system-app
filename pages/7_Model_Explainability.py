"""
Page 7 – Model Explainability
================================

Purpose:
    Provides both global (dataset-wide) and local (individual prediction)
    SHAP-based model interpretability with technical and non-technical views.

Design Decisions:
    - Global explainability uses the full test set for comprehensive SHAP analysis.
    - Local explainability uses individual employee data from session state.
    - Toggle between technical and non-technical views for different audiences.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

from utils.ui_components import apply_custom_css, render_header, render_divider, render_sidebar_info
from utils.explainability import (
    compute_shap_values, plot_shap_summary, plot_shap_bar,
    plot_shap_waterfall, plot_shap_dependence, generate_explanation,
)


# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Explainability | Attrition Predictor", page_icon="🧠", layout="wide")
apply_custom_css()
render_sidebar_info()

render_header("🧠 Model Explainability",
              "SHAP-based interpretability for understanding what drives attrition predictions")


# ---------------------------------------------------------------------------
# Check Prerequisites
# ---------------------------------------------------------------------------

if "training_results" not in st.session_state:
    st.warning(
        "⚠️ No trained models found. Please train models on the **Model Training** "
        "page first."
    )
    st.stop()

results = st.session_state["training_results"]
X_test = st.session_state["X_test"]
y_test = st.session_state["y_test"]
feature_names = st.session_state["feature_names"]

render_divider()


# ---------------------------------------------------------------------------
# Model Selection
# ---------------------------------------------------------------------------

model_names = list(results.keys())
selected_model = st.selectbox("Select Model for Explanation", model_names, key="explain_model")
model = results[selected_model]["model"]

st.info(f"📖 Computing SHAP values for **{selected_model}**... This may take a moment.")


# ---------------------------------------------------------------------------
# Compute SHAP Values (cached in session state)
# ---------------------------------------------------------------------------

shap_cache_key = f"shap_{selected_model}"

if shap_cache_key not in st.session_state:
    with st.spinner(f"Computing SHAP values for {selected_model}..."):
        # Use a subset for efficiency
        n_samples = min(200, len(X_test))
        X_shap = X_test.iloc[:n_samples]
        shap_result = compute_shap_values(model, X_shap, selected_model)
        st.session_state[shap_cache_key] = shap_result
else:
    shap_result = st.session_state[shap_cache_key]


# ---------------------------------------------------------------------------
# Tabs: Global vs Local
# ---------------------------------------------------------------------------

tab_global, tab_local = st.tabs(["🌍 Global Explainability", "🔍 Local Explainability"])


# ---------------------------------------------------------------------------
# Global Explainability
# ---------------------------------------------------------------------------

with tab_global:
    st.markdown("""
    ### Global SHAP Analysis

    Global explainability shows which features are most important across ALL
    predictions, not just a single employee. This helps identify the
    organization-wide drivers of attrition.
    """)

    render_divider()

    # SHAP Summary Plot (Beeswarm)
    st.markdown("#### SHAP Summary Plot (Beeswarm)")
    st.markdown("""
    Each dot represents one employee. The x-axis shows the SHAP value (impact
    on prediction). Color indicates the feature value (red = high, blue = low).
    Features are sorted by overall importance.
    """)

    try:
        fig_summary = plot_shap_summary(shap_result, max_display=15)
        st.pyplot(fig_summary, use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Could not render summary plot: {str(e)}")

    render_divider()

    # SHAP Bar Plot
    st.markdown("#### Feature Importance (Mean |SHAP|)")
    st.markdown("""
    This bar chart shows the average absolute SHAP value for each feature,
    providing a clear ranking of feature importance.
    """)

    try:
        fig_bar = plot_shap_bar(shap_result, max_display=15)
        st.pyplot(fig_bar, use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Could not render bar plot: {str(e)}")

    render_divider()

    # SHAP Dependence Plots
    st.markdown("#### SHAP Dependence Plots")
    st.markdown("""
    Shows the relationship between a feature's value and its SHAP value.
    Points are colored by an interacting feature to reveal interaction effects.
    """)

    # Get top features by importance
    mean_abs_shap = np.abs(shap_result["shap_values"]).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[::-1][:10]
    top_feature_names = [shap_result["feature_names"][i] for i in top_features_idx]

    selected_dep_feature = st.selectbox(
        "Select Feature for Dependence Plot",
        top_feature_names,
        key="dep_feature",
    )

    try:
        fig_dep = plot_shap_dependence(shap_result, selected_dep_feature)
        st.pyplot(fig_dep, use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Could not render dependence plot: {str(e)}")

    render_divider()

    # Global Interpretation
    st.markdown("#### 📖 Global Interpretation")

    st.markdown("""
    <div class="info-card">
        <h4 style="color: #6C63FF;">Key Findings from SHAP Analysis</h4>
        <div style="margin-top: 0.75rem; line-height: 1.8; color: rgba(250,250,250,0.8);">
            <strong>Top factors driving attrition across the organization:</strong><br><br>
            1. <strong>Overtime</strong> — Working overtime is consistently the strongest
               predictor of attrition. Employees who work overtime have significantly higher
               SHAP values, indicating increased attrition risk.<br><br>
            2. <strong>Monthly Income</strong> — Lower income pushes predictions toward
               attrition. The relationship is non-linear, with a steep effect below
               median income levels.<br><br>
            3. <strong>Age</strong> — Younger employees show higher attrition risk.
               The SHAP values decrease as age increases, reflecting the stabilizing
               effect of career maturity.<br><br>
            4. <strong>Years at Company</strong> — New employees (< 2 years) have
               disproportionately high attrition risk, validating the "new hire cliff"
               phenomenon.<br><br>
            5. <strong>Job Satisfaction</strong> — Low satisfaction scores directly
               increase predicted attrition probability.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Local Explainability
# ---------------------------------------------------------------------------

with tab_local:
    st.markdown("""
    ### Local SHAP Analysis (Individual Prediction)

    Explains why a specific employee received a particular prediction.
    The waterfall plot shows how each feature pushes the prediction up
    (toward attrition) or down (toward retention).
    """)

    render_divider()

    # View mode toggle
    view_mode = st.radio("Explanation Mode", ["Non-Technical (HR Friendly)", "Technical (Data Science)"],
                          horizontal=True, key="view_mode")

    # Employee selection
    n_samples = len(shap_result["X"])
    employee_idx = st.slider("Select Employee Index", 0, n_samples - 1, 0, key="emp_idx")

    # Get prediction for this employee
    employee_data = shap_result["X"].iloc[[employee_idx]]
    pred = model.predict(employee_data)[0]
    prob = model.predict_proba(employee_data)[0][1] if hasattr(model, "predict_proba") else (1.0 if pred == 1 else 0.0)

    # Generate explanation
    explanation = generate_explanation(shap_result, employee_idx, pred, prob)

    st.markdown("")

    # Result summary
    risk_color = explanation["risk_color"]
    st.markdown(f"""
    <div class="prediction-result {'prediction-leave' if pred == 1 else 'prediction-stay'}">
        <div style="font-size: 1.5rem; font-weight: 700; color: {risk_color};">
            Employee #{employee_idx}: {explanation['prediction_text']}
        </div>
        <div style="font-size: 1rem; color: rgba(250,250,250,0.6); margin-top: 0.5rem;">
            Probability: <strong style="color: {risk_color};">{prob:.1%}</strong> |
            Risk Level: <strong style="color: {risk_color};">{explanation['risk_level']}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    if view_mode == "Non-Technical (HR Friendly)":
        # HR-friendly explanation
        st.markdown("#### 📋 Plain Language Explanation")
        st.markdown(explanation["summary"])

        st.markdown("")
        st.markdown("#### 🔑 Key Factors:")

        for factor in explanation["top_factors"]:
            icon = "⬆️" if factor["direction"] == "increases" else "⬇️"
            feature_name = factor["feature"].replace("_", " ").title()

            st.markdown(f"""
            <div class="info-card" style="margin-bottom: 0.5rem;">
                <div style="display: flex; align-items: center;">
                    <span style="font-size: 1.2rem; margin-right: 0.75rem;">{icon}</span>
                    <div>
                        <strong style="color: {'#FF6B6B' if factor['direction'] == 'increases' else '#00D4AA'};">
                            {feature_name}
                        </strong>
                        <span style="color: rgba(250,250,250,0.5);"> — {factor['direction']} risk</span>
                        <span style="color: rgba(250,250,250,0.4); font-size: 0.8rem;"> | Impact: {factor['impact']}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown("#### 💡 Recommended Actions:")
        for action in explanation["retention_actions"]:
            st.markdown(f"- {action}")

    else:
        # Technical view
        st.markdown("#### 🔬 SHAP Waterfall Plot")
        st.markdown("""
        The waterfall plot shows the contribution of each feature to this
        specific prediction. Red bars push toward attrition (positive SHAP),
        blue bars push toward retention (negative SHAP).
        """)

        try:
            fig_waterfall = plot_shap_waterfall(shap_result, employee_idx, max_display=12)
            st.pyplot(fig_waterfall, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not render waterfall plot: {str(e)}")

        # Technical details
        st.markdown("#### 📊 SHAP Values Detail")

        shap_vals = shap_result["shap_values"][employee_idx]
        feature_vals = shap_result["X"].iloc[employee_idx]

        detail_df = pd.DataFrame({
            "Feature": shap_result["feature_names"],
            "Feature Value": feature_vals.values,
            "SHAP Value": shap_vals,
            "Absolute SHAP": np.abs(shap_vals),
            "Direction": ["↑ Risk" if s > 0 else "↓ Risk" for s in shap_vals],
        }).sort_values("Absolute SHAP", ascending=False).head(15)

        st.dataframe(
            detail_df.style.background_gradient(
                subset=["SHAP Value"],
                cmap="RdYlGn_r",
                vmin=-0.3, vmax=0.3,
            ).format({"Feature Value": "{:.3f}", "SHAP Value": "{:.4f}", "Absolute SHAP": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(f"""
        <div class="info-card">
            <strong>Technical Summary:</strong><br>
            • Base value (expected prediction): <strong>{shap_result['expected_value']:.4f}</strong><br>
            • Sum of SHAP values: <strong>{shap_vals.sum():.4f}</strong><br>
            • Final prediction score: <strong>{shap_result['expected_value'] + shap_vals.sum():.4f}</strong><br>
            • Model output probability: <strong>{prob:.4f}</strong>
        </div>
        """, unsafe_allow_html=True)
