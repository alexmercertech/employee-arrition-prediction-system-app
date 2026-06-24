"""
Model Training CLI Script
===========================

Purpose:
    Standalone script for training all models outside of the Streamlit UI.
    Useful for CI/CD pipelines, pre-deployment model training, or batch
    experimentation.

Usage:
    python train_model.py

    This will:
    1. Load the dataset
    2. Run the full preprocessing pipeline
    3. Train all 4 models with hyperparameter tuning
    4. Evaluate and compare models
    5. Save the best model to disk
    6. Print a comprehensive evaluation report

Design Decisions:
    - Uses the same preprocessing and training modules as the Streamlit app
      to ensure consistency between interactive and batch workflows.
    - Outputs are printed to stdout for easy logging and piping.
"""

import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from utils.data_loader import load_data, validate_data, get_attrition_stats, DATA_PATH
from utils.preprocessing import preprocess_data, analyze_missing_values
from utils.model_trainer import (
    train_all_models, save_model, detect_overfitting, get_model_configs,
)
from utils.evaluation import (
    compute_metrics, create_comparison_table, recommend_best_model,
)


def print_header(text: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_subheader(text: str):
    """Print a formatted subsection header."""
    print(f"\n--- {text} ---\n")


def main():
    """Execute the full training pipeline."""
    start_time = time.time()

    print_header("Employee Attrition Prediction System — Training Pipeline")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # -----------------------------------------------------------------------
    # Step 1: Load and validate data
    # -----------------------------------------------------------------------

    print_subheader("Step 1: Data Loading")

    print(f"Loading data from: {DATA_PATH}")

    # Load without Streamlit caching
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    print(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Validate
    validation = validate_data(df)
    for msg in validation["messages"]:
        print(f"  {msg}")

    if not validation["is_valid"]:
        print("❌ Data validation failed. Aborting.")
        sys.exit(1)

    # Attrition stats
    stats = get_attrition_stats(df)
    print(f"\n  Total Employees: {stats['total_employees']:,}")
    print(f"  Attrition Rate:  {stats['attrition_rate']}%")
    print(f"  Avg Salary:      ${stats['avg_salary']:,.0f}")
    print(f"  Avg Tenure:      {stats['avg_tenure']} years")

    # -----------------------------------------------------------------------
    # Step 2: Preprocessing
    # -----------------------------------------------------------------------

    print_subheader("Step 2: Data Preprocessing")

    # Missing values
    missing_df = analyze_missing_values(df)
    total_missing = missing_df["Missing_Count"].sum()
    print(f"  Missing values: {total_missing}")

    # Run full pipeline
    prep_result = preprocess_data(df, test_size=0.2, apply_smote_flag=True, random_state=42)

    for msg in prep_result["preprocessing_log"]:
        print(f"  • {msg}")

    print(f"\n  Training set: {prep_result['X_train'].shape}")
    print(f"  Test set:     {prep_result['X_test'].shape}")
    print(f"  Features:     {len(prep_result['feature_names'])}")

    if prep_result["balance_info"]:
        bi = prep_result["balance_info"]
        print(f"  SMOTE: {bi['samples_created']} synthetic samples created")
        print(f"    Before: {bi['before']}")
        print(f"    After:  {bi['after']}")

    # -----------------------------------------------------------------------
    # Step 3: Model Training
    # -----------------------------------------------------------------------

    print_subheader("Step 3: Model Training (with Hyperparameter Tuning)")

    X_train = prep_result["X_train"]
    X_test = prep_result["X_test"]
    y_train = prep_result["y_train"]
    y_test = prep_result["y_test"]

    def progress_callback(name, step, total):
        if step < total:
            print(f"  🔄 Training {name}... ({step + 1}/{total})")
        else:
            print(f"  ✅ All models trained!")

    results = train_all_models(
        X_train, y_train, X_test, y_test,
        tune_hyperparams=True,
        progress_callback=progress_callback,
    )

    # -----------------------------------------------------------------------
    # Step 4: Evaluation
    # -----------------------------------------------------------------------

    print_subheader("Step 4: Model Evaluation")

    comparison = create_comparison_table(results, X_test, y_test)
    print(comparison.to_string())

    # -----------------------------------------------------------------------
    # Step 5: Overfitting Analysis
    # -----------------------------------------------------------------------

    print_subheader("Step 5: Overfitting Analysis")

    overfit_df = detect_overfitting(results)
    print(overfit_df.to_string(index=False))

    # -----------------------------------------------------------------------
    # Step 6: Best Model Recommendation
    # -----------------------------------------------------------------------

    print_subheader("Step 6: Best Model Recommendation")

    recommendation = recommend_best_model(results, X_test, y_test)
    print(f"  🏆 Best Model: {recommendation['best_model_name']}")
    print(f"  F1 Score:      {recommendation['metrics']['F1 Score']}")
    print(f"  ROC-AUC:       {recommendation['metrics'].get('ROC-AUC', 'N/A')}")
    print(f"  Accuracy:      {recommendation['metrics']['Accuracy']}")

    # -----------------------------------------------------------------------
    # Step 7: Save Models
    # -----------------------------------------------------------------------

    print_subheader("Step 7: Saving Models")

    for name, res in results.items():
        filepath = save_model(res["model"], name, metadata={
            "test_score": res["test_score"],
            "test_f1": res["test_f1"],
            "best_params": res["best_params"],
            "cv_mean": res["cv_mean"],
        })
        print(f"  💾 Saved {name} → {filepath}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    total_time = time.time() - start_time

    print_header("Training Complete")
    print(f"  Total time:     {total_time:.1f} seconds")
    print(f"  Models trained: {len(results)}")
    print(f"  Best model:     {recommendation['best_model_name']}")
    print(f"  Best F1 Score:  {recommendation['metrics']['F1 Score']}")
    print(f"\n  Models saved to: {os.path.abspath('models/')}")
    print(f"\n  Run the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
