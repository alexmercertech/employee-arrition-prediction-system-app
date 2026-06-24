"""
Model Trainer Module
=====================

Purpose:
    Provides reusable training pipelines for multiple classification models
    with hyperparameter tuning, cross-validation, and model persistence.

Design Decisions:
    - Four models chosen to demonstrate range: Logistic Regression (linear
      baseline), Decision Tree (interpretable), Random Forest (ensemble),
      XGBoost (gradient boosting state-of-art).
    - RandomizedSearchCV preferred over GridSearchCV for efficiency — it
      samples a fixed number of parameter combinations rather than
      exhaustively searching the full grid.
    - Stratified K-Fold (k=5) ensures class proportions are maintained
      in each fold, critical for imbalanced datasets.
    - Models are serialized with joblib for efficient numpy array storage.

Complexity Considerations:
    - XGBoost training with hyperparameter tuning is the most expensive
      operation (~30-60 seconds depending on the search space).
    - Cross-validation multiplies training time by k (5x).

Assumptions:
    - Input data has been preprocessed (encoded, scaled, SMOTE-applied).
    - Target variable is binary (0/1).
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import (
    RandomizedSearchCV,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


# ---------------------------------------------------------------------------
# Model Configurations
# ---------------------------------------------------------------------------

def get_model_configs() -> dict:
    """
    Return model definitions and their hyperparameter search spaces.

    Each entry contains:
        - 'model': An uninitialized sklearn/xgboost estimator instance.
        - 'params': A dictionary of hyperparameter distributions for
          RandomizedSearchCV.
        - 'description': A brief explanation of the model's characteristics.

    Returns:
        Dictionary keyed by model name.
    """
    configs = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "C": [0.01, 0.1, 1, 10, 100],
                "penalty": ["l2"],
                "solver": ["lbfgs", "liblinear"],
            },
            "description": (
                "A linear model that predicts the probability of attrition "
                "using a logistic function. Serves as a strong interpretable "
                "baseline. Regularization parameter C controls the trade-off "
                "between fitting the training data and keeping the model simple."
            ),
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                "max_depth": [3, 5, 7, 10, 15, None],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 5, 10],
                "criterion": ["gini", "entropy"],
            },
            "description": (
                "A tree-based model that learns decision rules from the data. "
                "Highly interpretable but prone to overfitting. max_depth and "
                "min_samples_split are key regularization parameters."
            ),
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42, n_jobs=-1),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, 15, 20, None],
                "min_samples_leaf": [1, 2, 5],
                "min_samples_split": [2, 5, 10],
                "max_features": ["sqrt", "log2"],
            },
            "description": (
                "An ensemble of decision trees trained on random subsets of "
                "the data. Reduces overfitting through bagging and feature "
                "randomization. Generally robust with minimal tuning."
            ),
        },
        "XGBoost": {
            "model": XGBClassifier(
                random_state=42, eval_metric="logloss",
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7, 10],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "min_child_weight": [1, 3, 5],
            },
            "description": (
                "A gradient boosting algorithm that builds trees sequentially, "
                "each correcting errors of the previous ones. State-of-the-art "
                "for tabular data. learning_rate and n_estimators jointly "
                "control model complexity."
            ),
        },
    }
    return configs


# ---------------------------------------------------------------------------
# Training Functions
# ---------------------------------------------------------------------------

def train_single_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tune_hyperparams: bool = True,
    n_iter: int = 20,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Train a single model with optional hyperparameter tuning.

    Args:
        model_name: Name of the model (must match get_model_configs() keys).
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test target.
        tune_hyperparams: Whether to perform RandomizedSearchCV.
        n_iter: Number of parameter settings to sample.
        cv_folds: Number of cross-validation folds.
        random_state: Random seed.

    Returns:
        Dictionary containing:
            - 'model': Trained model instance
            - 'model_name': Name string
            - 'best_params': Best hyperparameters (if tuned)
            - 'train_score': Training accuracy
            - 'test_score': Test accuracy
            - 'train_f1': Training F1 score
            - 'test_f1': Test F1 score
            - 'cv_scores': Cross-validation scores (accuracy)
            - 'cv_mean': Mean CV score
            - 'cv_std': Std of CV scores
            - 'training_time': Time in seconds
            - 'overfitting_gap': train_score - test_score
    """
    configs = get_model_configs()
    if model_name not in configs:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(configs.keys())}")

    config = configs[model_name]
    model = config["model"]
    params = config["params"]

    start_time = time.time()

    if tune_hyperparams and params:
        # Determine search method based on parameter space size
        total_combinations = 1
        for v in params.values():
            total_combinations *= len(v)

        if total_combinations <= n_iter:
            # Use GridSearchCV for small spaces
            search = GridSearchCV(
                model, params,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
                scoring="f1",
                n_jobs=-1,
                verbose=0,
            )
        else:
            search = RandomizedSearchCV(
                model, params,
                n_iter=min(n_iter, total_combinations),
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
                scoring="f1",
                random_state=random_state,
                n_jobs=-1,
                verbose=0,
            )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
    else:
        best_model = model
        best_model.fit(X_train, y_train)
        best_params = {}

    training_time = time.time() - start_time

    # Predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Scores
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="accuracy")

    # Overfitting detection
    overfitting_gap = train_accuracy - test_accuracy

    return {
        "model": best_model,
        "model_name": model_name,
        "best_params": best_params,
        "train_score": round(train_accuracy, 4),
        "test_score": round(test_accuracy, 4),
        "train_f1": round(train_f1, 4),
        "test_f1": round(test_f1, 4),
        "cv_scores": cv_scores.tolist(),
        "cv_mean": round(cv_scores.mean(), 4),
        "cv_std": round(cv_scores.std(), 4),
        "training_time": round(training_time, 2),
        "overfitting_gap": round(overfitting_gap, 4),
    }


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_names: list[str] | None = None,
    tune_hyperparams: bool = True,
    progress_callback=None,
) -> dict:
    """
    Train all (or selected) models and return their results.

    Args:
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        model_names: List of model names to train. None = all models.
        tune_hyperparams: Whether to tune hyperparameters.
        progress_callback: Optional callable(model_name, step, total) for UI updates.

    Returns:
        Dictionary of {model_name: training_result_dict}.
    """
    if model_names is None:
        model_names = list(get_model_configs().keys())

    results = {}
    total = len(model_names)

    for i, name in enumerate(model_names):
        if progress_callback:
            progress_callback(name, i, total)

        result = train_single_model(
            name, X_train, y_train, X_test, y_test,
            tune_hyperparams=tune_hyperparams,
        )
        results[name] = result

    if progress_callback:
        progress_callback("Complete", total, total)

    return results


# ---------------------------------------------------------------------------
# Model Persistence
# ---------------------------------------------------------------------------

def save_model(model, model_name: str, metadata: dict | None = None) -> str:
    """
    Save a trained model to disk using joblib.

    The model is saved with metadata including the timestamp and any
    additional info passed in the metadata dict.

    Args:
        model: Trained model instance.
        model_name: Human-readable model name.
        metadata: Optional dictionary of additional metadata.

    Returns:
        The filepath where the model was saved.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Create a safe filename
    safe_name = model_name.lower().replace(" ", "_")
    filepath = os.path.join(MODELS_DIR, f"{safe_name}.joblib")

    # Bundle model with metadata
    bundle = {
        "model": model,
        "model_name": model_name,
        "saved_at": datetime.now().isoformat(),
        "metadata": metadata or {},
    }

    joblib.dump(bundle, filepath)
    return filepath


def load_model(model_name: str) -> dict | None:
    """
    Load a saved model from disk.

    Args:
        model_name: Human-readable model name.

    Returns:
        The model bundle dict, or None if not found.
    """
    safe_name = model_name.lower().replace(" ", "_")
    filepath = os.path.join(MODELS_DIR, f"{safe_name}.joblib")

    if os.path.exists(filepath):
        return joblib.load(filepath)
    return None


def list_saved_models() -> list[dict]:
    """
    List all saved models in the models directory.

    Returns:
        List of dicts with keys: name, filepath, saved_at, size_kb.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    models = []

    for filename in sorted(os.listdir(MODELS_DIR)):
        if filename.endswith(".joblib"):
            filepath = os.path.join(MODELS_DIR, filename)
            try:
                bundle = joblib.load(filepath)
                models.append({
                    "name": bundle.get("model_name", filename),
                    "filepath": filepath,
                    "saved_at": bundle.get("saved_at", "Unknown"),
                    "size_kb": round(os.path.getsize(filepath) / 1024, 1),
                })
            except Exception:
                models.append({
                    "name": filename,
                    "filepath": filepath,
                    "saved_at": "Error loading",
                    "size_kb": round(os.path.getsize(filepath) / 1024, 1),
                })

    return models


def detect_overfitting(results: dict) -> pd.DataFrame:
    """
    Analyze train-test performance gaps to detect overfitting/underfitting.

    Rules of thumb:
        - Overfitting: train >> test (gap > 5%)
        - Underfitting: both train and test are low (< 70%)
        - Good fit: small gap, both scores are high

    Args:
        results: Dictionary of {model_name: training_result_dict}.

    Returns:
        DataFrame with overfitting analysis.
    """
    rows = []
    for name, res in results.items():
        gap = res["overfitting_gap"]
        train = res["train_score"]
        test = res["test_score"]

        if gap > 0.05:
            status = "⚠️ Overfitting"
            recommendation = "Consider stronger regularization or simpler model."
        elif train < 0.70 and test < 0.70:
            status = "⚠️ Underfitting"
            recommendation = "Consider more features, complex model, or less regularization."
        else:
            status = "✅ Good Fit"
            recommendation = "Model generalizes well."

        rows.append({
            "Model": name,
            "Train Accuracy": f"{train:.2%}",
            "Test Accuracy": f"{test:.2%}",
            "Gap": f"{gap:.2%}",
            "Status": status,
            "Recommendation": recommendation,
        })

    return pd.DataFrame(rows)
