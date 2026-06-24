"""
Page 4 – Model Training
=========================

Purpose:
    Allow users to train ML models directly from the UI with configurable
    options for model selection, hyperparameter tuning, and SMOTE.

Design Decisions:
    - Training runs in the main thread with progress feedback via st.progress.
    - Trained models and results are stored in st.session_state for persistence
      across pages (used by Model Comparison and Prediction pages).
    - Models are also saved to disk via joblib for long-term persistence.
"""

import streamlit as st
import pandas as pd

from utils.ui_components import apply_custom_css, render_header, render_divider, render_sidebar_info
from utils.data_loader import load_data
from utils.preprocessing import preprocess_data
from utils.model_trainer import (
    get_model_configs, train_all_models, save_model, detect_overfitting,
    list_saved_models,
)
from utils.evaluation import compute_metrics


# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Model Training | Attrition Predictor", page_icon="🤖", layout="wide")
apply_custom_css()
render_sidebar_info()


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

df = load_data()
if df is None:
    st.error("❌ Failed to load dataset.")
    st.stop()

render_header("🤖 Model Training", "Train and compare machine learning models with hyperparameter tuning")

render_divider()


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------

st.markdown("### ⚙️ Training Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    available_models = list(get_model_configs().keys())
    selected_models = st.multiselect(
        "Select Models to Train",
        available_models,
        default=available_models,
        key="train_models",
    )

with col2:
    tune_hyperparams = st.toggle("Hyperparameter Tuning", value=True, key="tune_hp")
    use_smote = st.toggle("Apply SMOTE", value=True, key="use_smote")

with col3:
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05, key="test_size")
    st.markdown(f"**Train:** {1-test_size:.0%} | **Test:** {test_size:.0%}")

# Model descriptions
configs = get_model_configs()
with st.expander("📖 Model Descriptions"):
    for name, config in configs.items():
        if name in selected_models:
            st.markdown(f"**{name}:** {config['description']}")
            st.markdown(f"- *Hyperparameters:* `{list(config['params'].keys())}`")
            st.markdown("---")


render_divider()


# ---------------------------------------------------------------------------
# Training Execution
# ---------------------------------------------------------------------------

st.markdown("### 🚀 Train Models")

if not selected_models:
    st.warning("⚠️ Please select at least one model to train.")
    st.stop()

train_button = st.button("🏋️ Start Training", type="primary", width="stretch")

if train_button:
    with st.spinner("Preprocessing data..."):
        prep_result = preprocess_data(df, test_size=test_size, apply_smote_flag=use_smote)

    X_train = prep_result["X_train"]
    X_test = prep_result["X_test"]
    y_train = prep_result["y_train"]
    y_test = prep_result["y_test"]

    # Display preprocessing summary
    st.markdown("#### 📋 Preprocessing Log")
    for msg in prep_result["preprocessing_log"]:
        st.markdown(f"- {msg}")

    if prep_result["balance_info"]:
        bi = prep_result["balance_info"]
        st.info(
            f"⚖️ SMOTE: Before — {bi['before']}, After — {bi['after']} "
            f"({bi['samples_created']} synthetic samples created)"
        )

    st.markdown("---")

    # Train models with progress
    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_callback(name, step, total):
        progress_bar.progress(step / total if total > 0 else 1.0)
        if step < total:
            status_text.markdown(f"🔄 Training **{name}**... ({step+1}/{total})")
        else:
            status_text.markdown("✅ Training complete!")

    results = train_all_models(
        X_train, y_train, X_test, y_test,
        model_names=selected_models,
        tune_hyperparams=tune_hyperparams,
        progress_callback=progress_callback,
    )

    # Store results in session state
    st.session_state["training_results"] = results
    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test
    st.session_state["feature_names"] = prep_result["feature_names"]
    st.session_state["scaler"] = prep_result["scaler"]
    st.session_state["preprocessing_result"] = prep_result

    # Save models to disk
    for name, res in results.items():
        save_model(res["model"], name, metadata={
            "test_score": res["test_score"],
            "test_f1": res["test_f1"],
            "best_params": res["best_params"],
        })

    st.success(f"✅ Successfully trained {len(results)} models!")

    render_divider()

    # ---------------------------------------------------------------------------
    # Display Training Results
    # ---------------------------------------------------------------------------

    st.markdown("### 📊 Training Results")

    for name, res in results.items():
        with st.expander(f"🔹 {name}", expanded=True):
            col_a, col_b, col_c, col_d = st.columns(4)

            col_a.metric("Test Accuracy", f"{res['test_score']:.2%}")
            col_b.metric("Test F1 Score", f"{res['test_f1']:.2%}")
            col_c.metric("CV Mean ± Std", f"{res['cv_mean']:.2%} ± {res['cv_std']:.2%}")
            col_d.metric("Training Time", f"{res['training_time']:.1f}s")

            if res["best_params"]:
                st.markdown("**Best Hyperparameters:**")
                params_df = pd.DataFrame([res["best_params"]])
                st.dataframe(params_df, width="stretch", hide_index=True)

            # Cross-validation scores
            st.markdown("**Cross-Validation Scores (5-Fold):**")
            cv_df = pd.DataFrame({
                "Fold": [f"Fold {i+1}" for i in range(len(res["cv_scores"]))],
                "Accuracy": [f"{s:.4f}" for s in res["cv_scores"]],
            })
            st.dataframe(cv_df.T, width="stretch")

            # Overfitting check
            gap = res["overfitting_gap"]
            if gap > 0.05:
                st.warning(
                    f"⚠️ Potential overfitting detected: Train-Test gap = {gap:.2%}. "
                    f"Train accuracy ({res['train_score']:.2%}) is significantly higher "
                    f"than test accuracy ({res['test_score']:.2%})."
                )
            elif res["test_score"] < 0.70:
                st.warning(
                    f"⚠️ Potential underfitting: Test accuracy = {res['test_score']:.2%}. "
                    f"Consider more features or a more complex model."
                )
            else:
                st.success(
                    f"✅ Good fit: Train-Test gap = {gap:.2%}. "
                    f"The model generalizes well."
                )

    # Overfitting summary table
    render_divider()
    st.markdown("### 🔬 Overfitting / Underfitting Analysis")
    overfit_df = detect_overfitting(results)
    st.dataframe(overfit_df, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Previously Saved Models
# ---------------------------------------------------------------------------

render_divider()
st.markdown("### 💾 Saved Models")

saved_models = list_saved_models()
if saved_models:
    saved_df = pd.DataFrame(saved_models)
    st.dataframe(saved_df, width="stretch", hide_index=True)
else:
    st.info("ℹ️ No models saved yet. Train models above to save them.")


# Display current session state if results exist
if "training_results" in st.session_state and not train_button:
    render_divider()
    st.markdown("### 📊 Previous Training Results (Current Session)")
    st.info("ℹ️ Results from the last training run are available. Navigate to **Model Comparison** for detailed analysis.")

    for name, res in st.session_state["training_results"].items():
        col_a, col_b, col_c = st.columns(3)
        col_a.metric(f"{name} — Accuracy", f"{res['test_score']:.2%}")
        col_b.metric("F1 Score", f"{res['test_f1']:.2%}")
        col_c.metric("CV Mean", f"{res['cv_mean']:.2%}")
