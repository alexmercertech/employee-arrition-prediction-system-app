"""
Model Evaluation Module
========================

Purpose:
    Computes classification metrics and generates publication-quality
    evaluation visualizations using Plotly.

Design Decisions:
    - F1 Score is the primary model selection metric because the dataset
      is imbalanced (~16% attrition). Accuracy alone would be misleading.
    - Plotly is used for all charts to ensure consistent dark-themed,
      interactive visualizations that match the application's design.
    - ROC curves include AUC annotation for quick comparison.

Assumptions:
    - Models have been trained and can produce both predictions and
      probability estimates (predict_proba).
    - Target variable is binary (0=No, 1=Yes).

Limitations:
    - Feature importance extraction assumes tree-based models or models
      with a coef_ attribute. Other model types may not be supported.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report,
)


# ---------------------------------------------------------------------------
# Plotly Theme
# ---------------------------------------------------------------------------

PLOTLY_TEMPLATE = "plotly_dark"
COLOR_PALETTE = [
    "#6C63FF",  # Indigo (primary)
    "#00D4AA",  # Teal (accent)
    "#FF6B6B",  # Coral (warning)
    "#FFD93D",  # Gold
    "#4ECDC4",  # Cyan
    "#FF8A5C",  # Orange
    "#A8E6CF",  # Mint
    "#DDA0DD",  # Plum
]


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities for the positive class.

    Returns:
        Dictionary of metric name -> value pairs.
    """
    metrics = {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1 Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }

    if y_prob is not None:
        metrics["ROC-AUC"] = round(roc_auc_score(y_true, y_prob), 4)

    return metrics


# ---------------------------------------------------------------------------
# Visualization Functions
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
) -> go.Figure:
    """
    Create an interactive Plotly heatmap of the confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Stay (0)", "Leave (1)"]

    # Calculate percentages
    cm_pct = cm / cm.sum() * 100

    # Create annotation text with both count and percentage
    text = [[f"{cm[i][j]}<br>({cm_pct[i][j]:.1f}%)" for j in range(2)] for i in range(2)]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale=[[0, "#1A1F2E"], [1, "#6C63FF"]],
        showscale=False,
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template=PLOTLY_TEMPLATE,
        height=400,
        margin=dict(l=60, r=40, t=60, b=60),
    )

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
) -> go.Figure:
    """
    Plot the ROC curve with AUC annotation.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities for the positive class.
        model_name: Name for the legend.

    Returns:
        Plotly Figure object.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig = go.Figure()

    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"{model_name} (AUC = {auc:.4f})",
        line=dict(color="#6C63FF", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(108, 99, 255, 0.1)",
    ))

    # Diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random (AUC = 0.5)",
        line=dict(color="#FF6B6B", width=1.5, dash="dash"),
    ))

    fig.update_layout(
        title=dict(text=f"ROC Curve — {model_name}", font=dict(size=16)),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template=PLOTLY_TEMPLATE,
        height=450,
        legend=dict(x=0.55, y=0.05),
        margin=dict(l=60, r=40, t=60, b=60),
    )

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
) -> go.Figure:
    """
    Plot the Precision-Recall curve.

    Particularly useful for imbalanced datasets where ROC can be
    overly optimistic.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities for the positive class.
        model_name: Name for the legend.

    Returns:
        Plotly Figure object.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode="lines",
        name=model_name,
        line=dict(color="#00D4AA", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(0, 212, 170, 0.1)",
    ))

    fig.update_layout(
        title=dict(text=f"Precision-Recall Curve — {model_name}", font=dict(size=16)),
        xaxis_title="Recall",
        yaxis_title="Precision",
        template=PLOTLY_TEMPLATE,
        height=450,
        margin=dict(l=60, r=40, t=60, b=60),
    )

    return fig


def plot_multi_roc_curves(
    results: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> go.Figure:
    """
    Overlay ROC curves for multiple models on a single plot.

    Args:
        results: Dictionary of {model_name: training_result_dict}.
        X_test: Test features.
        y_test: Test target.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    for i, (name, res) in enumerate(results.items()):
        model = res["model"]
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)

            color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                name=f"{name} (AUC = {auc:.4f})",
                line=dict(color=color, width=2),
            ))

    # Diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random",
        line=dict(color="gray", width=1, dash="dash"),
    ))

    fig.update_layout(
        title=dict(text="ROC Curve Comparison", font=dict(size=18)),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template=PLOTLY_TEMPLATE,
        height=500,
        legend=dict(x=0.55, y=0.05),
        margin=dict(l=60, r=40, t=60, b=60),
    )

    return fig


def plot_multi_pr_curves(
    results: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> go.Figure:
    """
    Overlay Precision-Recall curves for multiple models.

    Args:
        results: Dictionary of {model_name: training_result_dict}.
        X_test: Test features.
        y_test: Test target.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    for i, (name, res) in enumerate(results.items()):
        model = res["model"]
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_prob)

            color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                mode="lines",
                name=name,
                line=dict(color=color, width=2),
            ))

    fig.update_layout(
        title=dict(text="Precision-Recall Curve Comparison", font=dict(size=18)),
        xaxis_title="Recall",
        yaxis_title="Precision",
        template=PLOTLY_TEMPLATE,
        height=500,
        margin=dict(l=60, r=40, t=60, b=60),
    )

    return fig


def plot_feature_importance(
    model,
    feature_names: list[str],
    model_name: str = "Model",
    top_n: int = 15,
) -> go.Figure:
    """
    Plot feature importance as a horizontal bar chart.

    Supports tree-based models (feature_importances_) and linear
    models (coef_).

    Args:
        model: Trained model instance.
        feature_names: List of feature names.
        model_name: Name for the chart title.
        top_n: Number of top features to display.

    Returns:
        Plotly Figure object, or None if model doesn't support importance.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return None

    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=True).tail(top_n)

    fig = go.Figure(go.Bar(
        x=importance_df["Importance"],
        y=importance_df["Feature"],
        orientation="h",
        marker=dict(
            color=importance_df["Importance"],
            colorscale=[[0, "#1A1F2E"], [0.5, "#6C63FF"], [1, "#00D4AA"]],
        ),
    ))

    fig.update_layout(
        title=dict(text=f"Feature Importance — {model_name}", font=dict(size=16)),
        xaxis_title="Importance",
        yaxis_title="",
        template=PLOTLY_TEMPLATE,
        height=max(400, top_n * 30),
        margin=dict(l=200, r=40, t=60, b=60),
    )

    return fig


def plot_metrics_comparison(results: dict, X_test: pd.DataFrame, y_test: pd.Series) -> go.Figure:
    """
    Create a grouped bar chart comparing metrics across models.

    Args:
        results: Dictionary of {model_name: training_result_dict}.
        X_test: Test features.
        y_test: Test target.

    Returns:
        Plotly Figure object.
    """
    metrics_data = []

    for name, res in results.items():
        model = res["model"]
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics["Model"] = name
        metrics_data.append(metrics)

    df = pd.DataFrame(metrics_data)
    metric_cols = [c for c in df.columns if c != "Model"]

    fig = go.Figure()

    for i, metric in enumerate(metric_cols):
        fig.add_trace(go.Bar(
            name=metric,
            x=df["Model"],
            y=df[metric],
            marker_color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
            text=df[metric].apply(lambda x: f"{x:.3f}"),
            textposition="auto",
        ))

    fig.update_layout(
        title=dict(text="Model Performance Comparison", font=dict(size=18)),
        barmode="group",
        template=PLOTLY_TEMPLATE,
        height=500,
        yaxis_title="Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=80, b=60),
    )

    return fig


# ---------------------------------------------------------------------------
# Comparison & Recommendation
# ---------------------------------------------------------------------------

def create_comparison_table(results: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Create a comprehensive comparison table of all model metrics.

    Args:
        results: Dictionary of {model_name: training_result_dict}.
        X_test: Test features.
        y_test: Test target.

    Returns:
        DataFrame with models as rows and metrics as columns.
    """
    rows = []

    for name, res in results.items():
        model = res["model"]
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics["Model"] = name
        metrics["CV Mean"] = res.get("cv_mean", 0)
        metrics["CV Std"] = res.get("cv_std", 0)
        metrics["Training Time (s)"] = res.get("training_time", 0)
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index("Model")

    # Reorder columns
    col_order = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "CV Mean", "CV Std", "Training Time (s)"]
    existing_cols = [c for c in col_order if c in df.columns]
    return df[existing_cols]


def recommend_best_model(results: dict, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Automatically select the best model based on F1 Score.

    F1 Score is chosen as the primary metric because:
    1. The dataset is imbalanced (~16% attrition rate).
    2. Both false positives (unnecessary retention costs) and false negatives
       (missed attrition) have business costs.
    3. F1 balances precision and recall, making it more appropriate than
       accuracy for this use case.

    Args:
        results: Dictionary of {model_name: training_result_dict}.
        X_test: Test features.
        y_test: Test target.

    Returns:
        Dictionary with:
            - 'best_model_name': Name of the recommended model
            - 'best_model': The model instance
            - 'metrics': Its metrics
            - 'rationale': Explanation of why it was chosen
    """
    comparison = create_comparison_table(results, X_test, y_test)

    best_name = comparison["F1 Score"].idxmax()
    best_metrics = comparison.loc[best_name].to_dict()
    best_model = results[best_name]["model"]

    # Generate detailed rationale
    f1 = best_metrics["F1 Score"]
    auc = best_metrics.get("ROC-AUC", "N/A")
    recall = best_metrics["Recall"]
    precision = best_metrics["Precision"]

    rationale = (
        f"**{best_name}** is recommended as the best model with an F1 Score "
        f"of **{f1:.4f}** and ROC-AUC of **{auc}**.\n\n"
        f"**Why F1 Score was used for selection:**\n"
        f"- The dataset has a class imbalance (~16% attrition rate)\n"
        f"- Accuracy alone would be misleading (a naive 'no attrition' "
        f"classifier would achieve ~84% accuracy)\n"
        f"- F1 Score balances Precision ({precision:.4f}) and Recall "
        f"({recall:.4f}), capturing both the model's ability to correctly "
        f"identify at-risk employees AND avoid false alarms\n\n"
        f"**Key strengths of {best_name}:**\n"
        f"- Recall of {recall:.4f} means it catches {recall*100:.1f}% of "
        f"employees who will actually leave\n"
        f"- Precision of {precision:.4f} means {precision*100:.1f}% of "
        f"flagged employees are truly at risk\n"
        f"- Cross-validation mean: {best_metrics.get('CV Mean', 'N/A')}"
    )

    return {
        "best_model_name": best_name,
        "best_model": best_model,
        "metrics": best_metrics,
        "rationale": rationale,
    }
