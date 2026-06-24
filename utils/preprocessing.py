"""
Data Preprocessing Module
==========================

Purpose:
    Provides the complete data preprocessing pipeline for the attrition
    prediction system, including cleaning, encoding, scaling, and class
    balancing via SMOTE.

Design Decisions:
    - Label encoding for binary features (Attrition, Gender, OverTime) to
      preserve ordinality in a compact representation.
    - One-hot encoding for multi-category features to avoid imposing false
      ordinal relationships (e.g., Department A > Department B).
    - StandardScaler chosen over MinMaxScaler because tree-based models
      are scale-invariant, while Logistic Regression benefits from
      zero-mean unit-variance features.
    - SMOTE is applied only to training data to prevent data leakage.

Complexity Considerations:
    - One-hot encoding increases dimensionality; we monitor the column count
      and provide feature selection capabilities.
    - SMOTE creates synthetic minority samples; we track the class
      distribution before and after to demonstrate the effect.

Assumptions:
    - The input DataFrame is the raw IBM HR dataset (pre-engineering).
    - Feature engineering is applied before encoding.

Limitations:
    - IQR-based outlier detection may flag legitimate extreme values.
    - SMOTE assumes the feature space supports linear interpolation between
      minority class samples.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

from utils.data_loader import COLUMNS_TO_DROP, BINARY_COLUMNS, MULTI_CATEGORY_COLUMNS
from utils.feature_engineering import create_engineered_features


# ---------------------------------------------------------------------------
# Missing Value Analysis
# ---------------------------------------------------------------------------

def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missing values in the dataset.

    Args:
        df: Raw DataFrame.

    Returns:
        DataFrame with columns: Column, Missing_Count, Missing_Percentage, Data_Type
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    analysis = pd.DataFrame({
        "Column": df.columns,
        "Missing_Count": missing.values,
        "Missing_Percentage": missing_pct.values,
        "Data_Type": df.dtypes.astype(str).values,
    })
    return analysis.sort_values("Missing_Count", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Duplicate Handling
# ---------------------------------------------------------------------------

def handle_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Detect and remove duplicate rows.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (cleaned DataFrame, number of duplicates removed).
    """
    n_duplicates = df.duplicated().sum()
    df_clean = df.drop_duplicates().reset_index(drop=True)
    return df_clean, n_duplicates


# ---------------------------------------------------------------------------
# Outlier Detection
# ---------------------------------------------------------------------------

def detect_outliers(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Detect outliers using the IQR (Interquartile Range) method.

    An observation is considered an outlier if it falls below Q1 - 1.5*IQR
    or above Q3 + 1.5*IQR.

    Args:
        df: Input DataFrame.
        columns: List of numeric columns to check. If None, uses all numeric columns.

    Returns:
        DataFrame with columns: Column, Lower_Bound, Upper_Bound, Outlier_Count, Outlier_Percentage
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_info = []
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_info.append({
            "Column": col,
            "Q1": round(Q1, 2),
            "Q3": round(Q3, 2),
            "IQR": round(IQR, 2),
            "Lower_Bound": round(lower, 2),
            "Upper_Bound": round(upper, 2),
            "Outlier_Count": n_outliers,
            "Outlier_Percentage": round(n_outliers / len(df) * 100, 2),
        })

    return pd.DataFrame(outlier_info).sort_values("Outlier_Count", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Column Cleaning
# ---------------------------------------------------------------------------

def drop_constant_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop columns that are constant or irrelevant.

    The IBM HR dataset contains several columns that provide no predictive
    value: EmployeeCount (always 1), Over18 (always 'Y'),
    StandardHours (always 80), and EmployeeNumber (unique ID).

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (cleaned DataFrame, list of dropped column names).
    """
    existing_drops = [col for col in COLUMNS_TO_DROP if col in df.columns]
    df_clean = df.drop(columns=existing_drops, errors="ignore")
    return df_clean, existing_drops


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_binary_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label encode binary categorical features.

    Maps: Attrition (Yes=1, No=0), Gender (Male=1, Female=0),
    OverTime (Yes=1, No=0).

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (encoded DataFrame, dict of {column: {original: encoded}} mappings).
    """
    df = df.copy()
    mappings = {}

    binary_maps = {
        "Attrition": {"Yes": 1, "No": 0},
        "Gender": {"Male": 1, "Female": 0},
        "OverTime": {"Yes": 1, "No": 0},
    }

    for col, mapping in binary_maps.items():
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].map(mapping)
            mappings[col] = mapping

    return df, mappings


def encode_multi_category_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    One-hot encode multi-category features.

    Uses pd.get_dummies with drop_first=True to avoid multicollinearity.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (encoded DataFrame, list of new column names created).
    """
    original_cols = set(df.columns)
    existing_multi = [col for col in MULTI_CATEGORY_COLUMNS if col in df.columns]

    df = pd.get_dummies(df, columns=existing_multi, drop_first=True, dtype=int)

    new_cols = list(set(df.columns) - original_cols)
    return df, sorted(new_cols)


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    exclude_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Apply StandardScaler to numeric features.

    The scaler is fit only on training data and then applied to both train
    and test sets to prevent data leakage.

    Args:
        X_train: Training feature DataFrame.
        X_test: Test feature DataFrame.
        exclude_cols: Columns to exclude from scaling (e.g., already-encoded binary cols).

    Returns:
        Tuple of (scaled X_train, scaled X_test, fitted StandardScaler).
    """
    scaler = StandardScaler()

    if exclude_cols is None:
        exclude_cols = []

    # Identify numeric columns to scale
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_scale = [c for c in numeric_cols if c not in exclude_cols]

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    if cols_to_scale:
        X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
        X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
# SMOTE (Class Balancing)
# ---------------------------------------------------------------------------

def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance classes.

    SMOTE generates synthetic samples for the minority class (Attrition=Yes)
    by interpolating between existing minority samples. This is applied only
    to the training set to prevent data leakage.

    Args:
        X_train: Training features.
        y_train: Training target.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (resampled X_train, resampled y_train, dict with before/after class counts).
    """
    before_counts = y_train.value_counts().to_dict()

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Convert back to DataFrame/Series to preserve column names
    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_resampled = pd.Series(y_resampled, name=y_train.name)

    after_counts = y_resampled.value_counts().to_dict()

    balance_info = {
        "before": before_counts,
        "after": after_counts,
        "samples_created": len(y_resampled) - len(y_train),
    }

    return X_resampled, y_resampled, balance_info


# ---------------------------------------------------------------------------
# Full Preprocessing Pipeline
# ---------------------------------------------------------------------------

def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    apply_smote_flag: bool = True,
    random_state: int = 42,
) -> dict:
    """
    Execute the complete preprocessing pipeline.

    Steps:
        1. Drop constant/irrelevant columns
        2. Handle duplicates
        3. Feature engineering (derived features)
        4. Binary encoding (Label Encoding)
        5. One-hot encoding
        6. Train/test split (stratified)
        7. Feature scaling (StandardScaler)
        8. SMOTE (optional, on training data only)

    Args:
        df: Raw DataFrame (original IBM HR dataset).
        test_size: Proportion of data for test set.
        apply_smote_flag: Whether to apply SMOTE for class balancing.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary containing:
            - 'X_train': Preprocessed training features
            - 'X_test': Preprocessed test features
            - 'y_train': Training target
            - 'y_test': Test target
            - 'scaler': Fitted StandardScaler
            - 'feature_names': List of final feature names
            - 'preprocessing_log': List of step descriptions
            - 'balance_info': SMOTE class balance info (if applied)
            - 'binary_mappings': Label encoding mappings
            - 'new_onehot_cols': One-hot encoded column names
            - 'dropped_cols': Constant columns that were dropped
            - 'n_duplicates': Number of duplicates removed
            - 'df_engineered': DataFrame after feature engineering (before encoding)
    """
    log = []

    # Step 1: Drop constant columns
    df_clean, dropped_cols = drop_constant_columns(df)
    log.append(f"Dropped {len(dropped_cols)} constant columns: {dropped_cols}")

    # Step 2: Handle duplicates
    df_clean, n_duplicates = handle_duplicates(df_clean)
    log.append(f"Removed {n_duplicates} duplicate rows")

    # Step 3: Feature engineering
    df_engineered = create_engineered_features(df_clean)
    log.append("Created 8 engineered features")

    # Step 4: Binary encoding
    df_encoded, binary_mappings = encode_binary_features(df_engineered)
    log.append(f"Label encoded {len(binary_mappings)} binary columns: {list(binary_mappings.keys())}")

    # Step 5: One-hot encoding
    df_encoded, new_onehot_cols = encode_multi_category_features(df_encoded)
    log.append(f"One-hot encoded multi-category columns, creating {len(new_onehot_cols)} new features")

    # Step 6: Split features and target
    target_col = "Attrition"
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    log.append(
        f"Train/test split: {len(X_train)} train, {len(X_test)} test "
        f"(test_size={test_size})"
    )

    # Step 7: Scale features
    # Exclude binary/one-hot encoded columns from scaling
    binary_encoded = [c for c in ["Gender", "OverTime", "IsNewEmployee"] if c in X_train.columns]
    exclude_from_scaling = binary_encoded + new_onehot_cols
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, exclude_cols=exclude_from_scaling,
    )
    log.append("Applied StandardScaler to numeric features")

    # Step 8: SMOTE (optional)
    balance_info = None
    if apply_smote_flag:
        X_train_scaled, y_train, balance_info = apply_smote(
            X_train_scaled, y_train, random_state=random_state,
        )
        log.append(
            f"Applied SMOTE: {balance_info['samples_created']} synthetic "
            f"samples created"
        )

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": list(X_train_scaled.columns),
        "preprocessing_log": log,
        "balance_info": balance_info,
        "binary_mappings": binary_mappings,
        "new_onehot_cols": new_onehot_cols,
        "dropped_cols": dropped_cols,
        "n_duplicates": n_duplicates,
        "df_engineered": df_engineered,
    }
