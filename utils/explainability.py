"""
Model Explainability Module
=============================

Purpose:
    Provides SHAP-based model interpretation for both global (dataset-wide)
    and local (individual prediction) explanations.

Design Decisions:
    - TreeExplainer is used for tree-based models (Decision Tree, Random
      Forest, XGBoost) for exact SHAP values with O(TLD) complexity.
    - LinearExplainer is used for Logistic Regression.
    - Matplotlib mode is used for rendering SHAP plots because Streamlit
      has limited support for SHAP's JavaScript-based force plots.
    - Human-readable explanations are generated from top SHAP values to
      make the output accessible to non-technical HR stakeholders.

Assumptions:
    - Models have been trained and saved before explainability is invoked.
    - Feature names are preserved from the preprocessing pipeline.

Limitations:
    - KernelExplainer (used as fallback) is slow for large datasets; we
      subsample to 100 background samples for efficiency.
    - SHAP values may not sum exactly to the model output for some models
      due to approximation.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import shap


# ---------------------------------------------------------------------------
# SHAP Value Computation
# ---------------------------------------------------------------------------

def compute_shap_values(model, X: pd.DataFrame, model_name: str = "Model") -> dict:
    """
    Compute SHAP values for a given model and dataset.

    Automatically selects the appropriate explainer based on model type:
        - TreeExplainer for tree-based models
        - LinearExplainer for linear models
        - KernelExplainer as fallback

    Args:
        model: Trained model instance.
        X: Feature DataFrame (typically X_test or a subsample).
        model_name: Name for logging.

    Returns:
        Dictionary containing:
            - 'shap_values': The computed SHAP values (numpy array)
            - 'explainer': The SHAP explainer instance
            - 'expected_value': The base value (expected prediction)
            - 'feature_names': List of feature names
            - 'X': The input DataFrame
    """
    model_type = type(model).__name__

    if model_type in ["RandomForestClassifier", "DecisionTreeClassifier", "XGBClassifier"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # For binary classification, TreeExplainer may return a list of arrays or a 3D array
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # SHAP values for the positive class
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
    elif model_type == "LogisticRegression":
        explainer = shap.LinearExplainer(model, X)
        shap_values = explainer.shap_values(X)
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[0]
    else:
        # Fallback to KernelExplainer (slower)
        background = shap.sample(X, min(100, len(X)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

    return {
        "shap_values": shap_values,
        "explainer": explainer,
        "expected_value": expected_value,
        "feature_names": list(X.columns),
        "X": X,
    }


# ---------------------------------------------------------------------------
# SHAP Visualization Functions
# ---------------------------------------------------------------------------

def plot_shap_summary(shap_result: dict, max_display: int = 15) -> plt.Figure:
    """
    Create a SHAP beeswarm summary plot.

    Shows the distribution of SHAP values for each feature, indicating
    both the magnitude and direction of each feature's impact.

    Args:
        shap_result: Output from compute_shap_values().
        max_display: Maximum number of features to show.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    shap.summary_plot(
        shap_result["shap_values"],
        shap_result["X"],
        max_display=max_display,
        show=False,
        plot_size=None,
    )

    # Style the plot for dark theme
    for text in fig.texts:
        text.set_color("white")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#333333")

    plt.tight_layout()
    return fig


def plot_shap_bar(shap_result: dict, max_display: int = 15) -> plt.Figure:
    """
    Create a SHAP bar plot showing mean absolute SHAP values.

    Provides a clear ranking of feature importance based on their
    average contribution to predictions.

    Args:
        shap_result: Output from compute_shap_values().
        max_display: Maximum number of features to show.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    shap.summary_plot(
        shap_result["shap_values"],
        shap_result["X"],
        plot_type="bar",
        max_display=max_display,
        show=False,
        plot_size=None,
    )

    # Style for dark theme
    for text in fig.texts:
        text.set_color("white")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#333333")

    plt.tight_layout()
    return fig


def plot_shap_waterfall(
    shap_result: dict,
    sample_index: int = 0,
    max_display: int = 12,
) -> plt.Figure:
    """
    Create a SHAP waterfall plot for a single prediction.

    Shows how each feature contributes to moving the prediction from
    the base value (expected value) to the final prediction for a
    specific employee.

    Args:
        shap_result: Output from compute_shap_values().
        sample_index: Index of the sample to explain.
        max_display: Maximum number of features to show.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    # Create SHAP Explanation object
    explanation = shap.Explanation(
        values=shap_result["shap_values"][sample_index],
        base_values=shap_result["expected_value"],
        data=shap_result["X"].iloc[sample_index].values,
        feature_names=shap_result["feature_names"],
    )

    shap.waterfall_plot(explanation, max_display=max_display, show=False)

    # Style for dark theme
    current_fig = plt.gcf()
    current_fig.patch.set_facecolor("#0E1117")
    for ax in current_fig.axes:
        ax.set_facecolor("#0E1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#333333")

    plt.tight_layout()
    return current_fig


def plot_shap_dependence(
    shap_result: dict,
    feature: str,
    interaction_feature: str | None = None,
) -> plt.Figure:
    """
    Create a SHAP dependence plot for a specific feature.

    Shows the relationship between feature values and SHAP values,
    optionally colored by an interaction feature.

    Args:
        shap_result: Output from compute_shap_values().
        feature: Feature name for the x-axis.
        interaction_feature: Feature to use for coloring (auto-selected if None).

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    interaction = interaction_feature if interaction_feature else "auto"

    shap.dependence_plot(
        feature,
        shap_result["shap_values"],
        shap_result["X"],
        interaction_index=interaction,
        ax=ax,
        show=False,
    )

    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#333333")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Human-Readable Explanation
# ---------------------------------------------------------------------------

def generate_explanation(
    shap_result: dict,
    sample_index: int,
    prediction: int,
    probability: float,
    top_n: int = 5,
) -> dict:
    """
    Generate a human-readable explanation for a single prediction.

    Translates SHAP values into plain-language statements that HR
    professionals can understand without machine learning expertise.

    Args:
        shap_result: Output from compute_shap_values().
        sample_index: Index of the sample to explain.
        prediction: The model's prediction (0 or 1).
        probability: Probability of the positive class.
        top_n: Number of top factors to include.

    Returns:
        Dictionary containing:
            - 'prediction_text': "Likely to Leave" or "Likely to Stay"
            - 'probability': Float probability
            - 'risk_level': "High Risk", "Medium Risk", or "Low Risk"
            - 'risk_color': Color code for the risk level
            - 'top_factors': List of dicts with factor details
            - 'summary': Human-readable summary paragraph
            - 'retention_actions': List of recommended actions
    """
    shap_vals = shap_result["shap_values"][sample_index]
    feature_names = shap_result["feature_names"]
    feature_values = shap_result["X"].iloc[sample_index]

    # Sort features by absolute SHAP value
    importance_idx = np.argsort(np.abs(shap_vals))[::-1][:top_n]

    top_factors = []
    for idx in importance_idx:
        fname = feature_names[idx]
        fval = feature_values.iloc[idx]
        shap_val = shap_vals[idx]
        direction = "increases" if shap_val > 0 else "decreases"

        top_factors.append({
            "feature": fname,
            "value": fval,
            "shap_value": round(float(shap_val), 4),
            "direction": direction,
            "impact": "High" if abs(shap_val) > 0.1 else "Medium" if abs(shap_val) > 0.05 else "Low",
        })

    # Determine risk level
    if probability >= 0.6:
        risk_level = "High Risk"
        risk_color = "#FF6B6B"
    elif probability >= 0.3:
        risk_level = "Medium Risk"
        risk_color = "#FFD93D"
    else:
        risk_level = "Low Risk"
        risk_color = "#00D4AA"

    prediction_text = "Likely to Leave" if prediction == 1 else "Likely to Stay"

    # Generate summary
    factor_texts = []
    for f in top_factors[:3]:
        feature_display = f["feature"].replace("_", " ")
        factor_texts.append(f"{feature_display} ({f['direction']} risk)")

    summary = (
        f"This employee is **{prediction_text.lower()}** with a "
        f"**{probability:.1%}** probability of attrition. "
        f"The key factors are: {', '.join(factor_texts)}."
    )

    # Generate retention actions based on top factors
    retention_actions = _generate_retention_actions(top_factors, feature_values)

    return {
        "prediction_text": prediction_text,
        "probability": probability,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "top_factors": top_factors,
        "summary": summary,
        "retention_actions": retention_actions,
    }


def _generate_retention_actions(top_factors: list, feature_values: pd.Series) -> list[str]:
    """
    Generate context-aware retention recommendations based on risk factors.

    Maps specific feature patterns to actionable HR interventions.
    """
    actions = []
    seen_categories = set()

    for factor in top_factors:
        fname = factor["feature"]
        direction = factor["direction"]

        if direction != "increases":
            continue  # Only recommend actions for risk-increasing factors

        if "OverTime" in fname and "overtime" not in seen_categories:
            actions.append("📋 Review overtime workload and consider flexible scheduling or additional resource allocation.")
            seen_categories.add("overtime")
        elif "JobSatisfaction" in fname and "satisfaction" not in seen_categories:
            actions.append("💬 Conduct a one-on-one meeting to discuss job satisfaction and career expectations.")
            seen_categories.add("satisfaction")
        elif "MonthlyIncome" in fname and "income" not in seen_categories:
            actions.append("💰 Review compensation against market benchmarks and consider a salary adjustment.")
            seen_categories.add("income")
        elif "EnvironmentSatisfaction" in fname and "environment" not in seen_categories:
            actions.append("🏢 Assess workplace environment concerns and implement targeted improvements.")
            seen_categories.add("environment")
        elif "WorkLifeBalance" in fname and "balance" not in seen_categories:
            actions.append("⚖️ Offer flexible work arrangements or wellness programs to improve work-life balance.")
            seen_categories.add("balance")
        elif "YearsAtCompany" in fname and "tenure" not in seen_categories:
            actions.append("🎯 Develop a personalized career development plan with clear growth milestones.")
            seen_categories.add("tenure")
        elif "DistanceFromHome" in fname and "distance" not in seen_categories:
            actions.append("🏠 Consider remote work options or relocation assistance to reduce commute burden.")
            seen_categories.add("distance")
        elif "StockOptionLevel" in fname and "stocks" not in seen_categories:
            actions.append("📈 Consider offering stock options or equity participation as a retention incentive.")
            seen_categories.add("stocks")
        elif "TrainingTimesLastYear" in fname and "training" not in seen_categories:
            actions.append("📚 Provide additional training and development opportunities for skill growth.")
            seen_categories.add("training")
        elif "Promotion" in fname and "promotion" not in seen_categories:
            actions.append("🚀 Discuss promotion pathways and set clear performance targets for advancement.")
            seen_categories.add("promotion")

    if not actions:
        actions.append("📊 Schedule a regular check-in to monitor engagement and satisfaction levels.")
        actions.append("🤝 Involve in cross-functional projects to enhance engagement and skill development.")

    return actions[:5]  # Limit to 5 actions
