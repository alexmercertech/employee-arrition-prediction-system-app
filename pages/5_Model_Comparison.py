"""
Page 5 – Model Comparison
===========================

Purpose:
    Provides a comprehensive side-by-side comparison of all trained models
    with metrics tables, ROC/PR curves, feature importance, and automatic
    best model recommendation.

Design Decisions:
    - Requires models to be trained first (checks session state).
    - Uses the evaluation module for all visualizations and metrics.
    - Best model is recommended based on F1 Score with a written rationale.
"""

import streamlit as st
import pandas as pd

from utils.ui_components import apply_custom_css, render_header, render_divider, render_sidebar_info
from utils.evaluation import (
    create_comparison_table, recommend_best_model,
    plot_metrics_comparison, plot_multi_roc_curves, plot_multi_pr_curves,
    plot_confusion_matrix, plot_feature_importance,
)
from utils.model_trainer import detect_overfitting


# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Model Comparison | Attrition Predictor", page_icon="📈", layout="wide")
apply_custom_css()
render_sidebar_info()


# ---------------------------------------------------------------------------
# Check for Training Results
# ---------------------------------------------------------------------------

render_header("📈 Model Comparison", "Side-by-side analysis of trained models with automatic recommendation")

if "training_results" not in st.session_state:
    st.warning(
        "⚠️ No trained models found. Please go to the **Model Training** page "
        "and train your models first."
    )
    st.stop()

results = st.session_state["training_results"]
X_test = st.session_state["X_test"]
y_test = st.session_state["y_test"]
feature_names = st.session_state["feature_names"]

render_divider()


# ---------------------------------------------------------------------------
# Metrics Comparison Table
# ---------------------------------------------------------------------------

st.markdown("### 📋 Performance Metrics Comparison")

comparison_df = create_comparison_table(results, X_test, y_test)
st.dataframe(
    comparison_df.style.highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
                                       color="rgba(108, 99, 255, 0.3)")
    .format({col: "{:.4f}" for col in comparison_df.columns if col != "Training Time (s)"}),
    width="stretch",
)

st.markdown("""
<div class="info-card">
    <strong>📖 Metrics Guide:</strong><br>
    • <strong>Accuracy</strong> — Overall correct predictions (can be misleading for imbalanced data)<br>
    • <strong>Precision</strong> — Of predicted "Leave", how many actually left (avoid false alarms)<br>
    • <strong>Recall</strong> — Of actual "Leave", how many did we catch (avoid missing at-risk employees)<br>
    • <strong>F1 Score</strong> — Harmonic mean of Precision and Recall (our primary metric)<br>
    • <strong>ROC-AUC</strong> — Model's ability to distinguish between classes across all thresholds
</div>
""", unsafe_allow_html=True)

render_divider()


# ---------------------------------------------------------------------------
# Metrics Bar Chart
# ---------------------------------------------------------------------------

st.markdown("### 📊 Visual Comparison")

fig = plot_metrics_comparison(results, X_test, y_test)
st.plotly_chart(fig, width="stretch")

render_divider()


# ---------------------------------------------------------------------------
# ROC Curves
# ---------------------------------------------------------------------------

col_roc, col_pr = st.columns(2)

with col_roc:
    st.markdown("### ROC Curves")
    fig_roc = plot_multi_roc_curves(results, X_test, y_test)
    st.plotly_chart(fig_roc, width="stretch")

with col_pr:
    st.markdown("### Precision-Recall Curves")
    fig_pr = plot_multi_pr_curves(results, X_test, y_test)
    st.plotly_chart(fig_pr, width="stretch")

st.markdown("""
<div class="info-card">
    <strong>💡 Interpretation:</strong><br>
    • <strong>ROC Curve</strong>: A curve closer to the top-left corner indicates better performance.
    AUC = 1.0 is perfect; AUC = 0.5 is random guessing.<br>
    • <strong>PR Curve</strong>: More informative than ROC for imbalanced datasets. A curve closer to
    the top-right indicates the model maintains high precision even at high recall levels.
</div>
""", unsafe_allow_html=True)

render_divider()


# ---------------------------------------------------------------------------
# Confusion Matrices
# ---------------------------------------------------------------------------

st.markdown("### 🔲 Confusion Matrices")

cols = st.columns(min(len(results), 4))
for i, (name, res) in enumerate(results.items()):
    with cols[i % len(cols)]:
        model = res["model"]
        y_pred = model.predict(X_test)
        fig_cm = plot_confusion_matrix(y_test, y_pred, title=name)
        st.plotly_chart(fig_cm, width="stretch")

render_divider()


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------

st.markdown("### 🏆 Feature Importance")

selected_model = st.selectbox("Select Model for Feature Importance",
                               list(results.keys()), key="fi_model")

model = results[selected_model]["model"]
fig_fi = plot_feature_importance(model, feature_names, selected_model)

if fig_fi:
    st.plotly_chart(fig_fi, width="stretch")
else:
    st.info("ℹ️ Feature importance is not available for this model type.")

render_divider()


# ---------------------------------------------------------------------------
# Overfitting Analysis
# ---------------------------------------------------------------------------

st.markdown("### 🔬 Overfitting / Underfitting Analysis")

overfit_df = detect_overfitting(results)
st.dataframe(overfit_df, width="stretch", hide_index=True)

st.markdown("""
<div class="info-card">
    <strong>📖 Diagnosis Guide:</strong><br>
    • <strong>Overfitting</strong> (Train >> Test): Model memorizes training data. Fix: stronger regularization, simpler model, more data.<br>
    • <strong>Underfitting</strong> (Both low): Model is too simple. Fix: more features, complex model, less regularization.<br>
    • <strong>Good Fit</strong> (Small gap, both high): Model generalizes well to unseen data.
</div>
""", unsafe_allow_html=True)

render_divider()


# ---------------------------------------------------------------------------
# Best Model Recommendation
# ---------------------------------------------------------------------------

st.markdown("### 🏅 Best Model Recommendation")

recommendation = recommend_best_model(results, X_test, y_test)

st.markdown(f"""
<div class="prediction-result prediction-stay" style="text-align: left; padding: 2rem;">
    <h2 style="color: #00D4AA; margin-bottom: 1rem;">
        🏆 Recommended: {recommendation['best_model_name']}
    </h2>
</div>
""", unsafe_allow_html=True)

st.markdown("")
st.markdown(recommendation["rationale"])

# Store best model in session state for use by prediction page
st.session_state["best_model_name"] = recommendation["best_model_name"]
st.session_state["best_model"] = recommendation["best_model"]
