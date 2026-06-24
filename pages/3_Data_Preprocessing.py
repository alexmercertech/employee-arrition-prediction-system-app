"""
Page 3 – Data Preprocessing
==============================

Purpose:
    Displays the step-by-step data preprocessing pipeline with before/after
    transformations and explanations for each decision.

Design Decisions:
    - Each preprocessing step is shown in an expander for clean organization.
    - Before/after views let assessors verify transformation correctness.
    - Explanations justify each design decision (e.g., why one-hot vs label).
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from utils.ui_components import apply_custom_css, render_header, render_divider, render_sidebar_info
from utils.data_loader import load_data, COLUMNS_TO_DROP
from utils.preprocessing import (
    analyze_missing_values, handle_duplicates, detect_outliers,
    drop_constant_columns, encode_binary_features, encode_multi_category_features,
    apply_smote,
)
from utils.feature_engineering import create_engineered_features, get_feature_documentation


# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Data Preprocessing | Attrition Predictor", page_icon="⚙️", layout="wide")
apply_custom_css()
render_sidebar_info()

PLOTLY_TEMPLATE = "plotly_dark"


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

df = load_data()
if df is None:
    st.error("❌ Failed to load dataset.")
    st.stop()

render_header("⚙️ Data Preprocessing Pipeline",
              "Step-by-step walkthrough of data cleaning, transformation, and preparation")

render_divider()


# ---------------------------------------------------------------------------
# Step 1: Raw Data Overview
# ---------------------------------------------------------------------------

with st.expander("📁 Step 1: Raw Data Overview", expanded=True):
    st.markdown("""
    **Purpose:** Examine the raw dataset before any transformations.
    Understanding the data shape, types, and initial values is the foundation
    of any ML pipeline.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]}")
    col3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    st.markdown("**First 5 Rows:**")
    st.dataframe(df.head(), use_container_width=True, height=220)

    st.markdown("**Data Types:**")
    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Non-Null Count": df.notnull().sum().values,
        "Unique Values": df.nunique().values,
        "Sample Value": [str(df[col].iloc[0]) for col in df.columns],
    })
    st.dataframe(dtype_df, use_container_width=True, hide_index=True, height=400)


# ---------------------------------------------------------------------------
# Step 2: Missing Value Analysis
# ---------------------------------------------------------------------------

with st.expander("🔍 Step 2: Missing Value Analysis"):
    st.markdown("""
    **Purpose:** Identify missing values that could cause model errors or bias.

    **Decision:** The IBM HR dataset has no missing values, which is expected
    for a synthetic dataset. In production systems, we would impute missing
    values using median (numeric) or mode (categorical) strategies.
    """)

    missing_df = analyze_missing_values(df)
    total_missing = missing_df["Missing_Count"].sum()

    if total_missing == 0:
        st.success("✅ No missing values found in the dataset!")
    else:
        st.warning(f"⚠️ Found {total_missing} missing values across {(missing_df['Missing_Count'] > 0).sum()} columns.")

    st.dataframe(missing_df, use_container_width=True, hide_index=True)

    # Heatmap visualization of nulls (will be all zeros for this dataset)
    null_matrix = df.isnull().astype(int)
    fig = go.Figure(data=go.Heatmap(
        z=null_matrix.values[:100].T,  # Show first 100 rows
        y=null_matrix.columns,
        colorscale=[[0, "#1A1F2E"], [1, "#FF6B6B"]],
        showscale=False,
    ))
    fig.update_layout(
        title="Missing Value Heatmap (First 100 Rows)",
        template=PLOTLY_TEMPLATE, height=400,
        margin=dict(l=200, r=40, t=60, b=40),
        xaxis_title="Row Index",
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Step 3: Duplicate Detection
# ---------------------------------------------------------------------------

with st.expander("📋 Step 3: Duplicate Detection"):
    st.markdown("""
    **Purpose:** Remove duplicate records that could bias model training.

    **Decision:** Duplicates are removed to ensure each employee is
    represented exactly once. Duplicates can artificially inflate
    the importance of certain patterns.
    """)

    df_deduped, n_dupes = handle_duplicates(df)

    if n_dupes == 0:
        st.success("✅ No duplicate rows found!")
    else:
        st.warning(f"⚠️ Removed {n_dupes} duplicate rows.")

    col1, col2 = st.columns(2)
    col1.metric("Before", f"{len(df):,} rows")
    col2.metric("After", f"{len(df_deduped):,} rows")


# ---------------------------------------------------------------------------
# Step 4: Constant Column Removal
# ---------------------------------------------------------------------------

with st.expander("🗑️ Step 4: Constant Column Removal"):
    st.markdown("""
    **Purpose:** Remove columns that provide no predictive value.

    **Decision:** The following columns are dropped because they contain
    a single constant value across all rows or are non-predictive identifiers:
    """)

    df_cleaned, dropped = drop_constant_columns(df_deduped)

    for col in COLUMNS_TO_DROP:
        if col in df.columns:
            unique_vals = df[col].unique()
            st.markdown(f"- **{col}**: {len(unique_vals)} unique value(s) → `{unique_vals[:3]}`")

    st.markdown("""
    **Rationale:**
    - `EmployeeCount` — Always 1 (provides zero information)
    - `Over18` — Always 'Y' (all employees are adults)
    - `StandardHours` — Always 80 (standard work hours are identical)
    - `EmployeeNumber` — Unique ID (not a predictive feature; would cause overfitting)
    """)

    col1, col2 = st.columns(2)
    col1.metric("Columns Before", f"{len(df_deduped.columns)}")
    col2.metric("Columns After", f"{len(df_cleaned.columns)}")


# ---------------------------------------------------------------------------
# Step 5: Outlier Detection
# ---------------------------------------------------------------------------

with st.expander("📊 Step 5: Outlier Detection"):
    st.markdown("""
    **Purpose:** Identify extreme values that may be errors or legitimate
    exceptional cases.

    **Method:** IQR (Interquartile Range) — values below Q1 - 1.5×IQR or
    above Q3 + 1.5×IQR are flagged as outliers.

    **Decision:** Outliers are identified but NOT removed, because:
    1. Tree-based models (Random Forest, XGBoost) are robust to outliers
    2. Extreme values may be legitimate (e.g., high-income executives)
    3. Removing outliers could reduce the already-small dataset further
    """)

    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    outlier_df = detect_outliers(df_cleaned, numeric_cols)
    st.dataframe(outlier_df, use_container_width=True, hide_index=True)

    # Box plot for selected column
    selected_col = st.selectbox("View Outlier Distribution", numeric_cols,
                                 index=numeric_cols.index("MonthlyIncome") if "MonthlyIncome" in numeric_cols else 0)

    fig = px.box(
        df_cleaned, y=selected_col, color="Attrition",
        color_discrete_map={"Yes": "#FF6B6B", "No": "#00D4AA"},
        template=PLOTLY_TEMPLATE, points="outliers",
    )
    fig.update_layout(height=400, margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Step 6: Feature Engineering
# ---------------------------------------------------------------------------

with st.expander("🔧 Step 6: Feature Engineering"):
    st.markdown("""
    **Purpose:** Create derived features that capture meaningful business
    patterns not directly available in the raw data. Each feature is
    designed to provide additional predictive signal.
    """)

    df_engineered = create_engineered_features(df_cleaned)
    feature_docs = get_feature_documentation()

    st.markdown("#### Engineered Features")
    st.dataframe(feature_docs, use_container_width=True, hide_index=True, height=350)

    st.markdown("#### Preview of New Features")
    new_feature_cols = feature_docs["Feature"].tolist()
    existing_new = [c for c in new_feature_cols if c in df_engineered.columns]
    st.dataframe(
        df_engineered[["Age", "MonthlyIncome", "YearsAtCompany"] + existing_new].head(10),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    col1.metric("Features Before", f"{len(df_cleaned.columns)}")
    col2.metric("Features After", f"{len(df_engineered.columns)}")


# ---------------------------------------------------------------------------
# Step 7: Encoding
# ---------------------------------------------------------------------------

with st.expander("🏷️ Step 7: Feature Encoding"):
    st.markdown("""
    **Purpose:** Convert categorical features to numerical format for ML models.

    **Two encoding strategies are used:**

    1. **Label Encoding** for binary features (2 categories):
       - `Attrition`: Yes→1, No→0
       - `Gender`: Male→1, Female→0
       - `OverTime`: Yes→1, No→0

       *Rationale:* Binary features have a natural ordering (presence/absence),
       so label encoding is appropriate and compact.

    2. **One-Hot Encoding** for multi-category features:
       - `BusinessTravel`, `Department`, `EducationField`, `JobRole`, `MaritalStatus`

       *Rationale:* Multi-category features have no ordinal relationship
       (e.g., "Sales" is not greater than "HR"). One-hot encoding prevents
       the model from assuming false ordinal relationships. `drop_first=True`
       avoids multicollinearity.
    """)

    df_encoded, binary_maps = encode_binary_features(df_engineered)
    df_encoded, onehot_cols = encode_multi_category_features(df_encoded)

    st.markdown("#### Label Encoding Mappings")
    for col, mapping in binary_maps.items():
        st.markdown(f"- **{col}**: {mapping}")

    st.markdown(f"#### One-Hot Encoded Columns ({len(onehot_cols)} new)")
    st.code(", ".join(onehot_cols[:10]) + ("..." if len(onehot_cols) > 10 else ""))

    col1, col2 = st.columns(2)
    col1.metric("Features Before Encoding", f"{len(df_engineered.columns)}")
    col2.metric("Features After Encoding", f"{len(df_encoded.columns)}")

    st.markdown("#### Encoded Data Preview")
    st.dataframe(df_encoded.head(), use_container_width=True, height=220)


# ---------------------------------------------------------------------------
# Step 8: Class Imbalance Analysis
# ---------------------------------------------------------------------------

with st.expander("⚖️ Step 8: Class Imbalance & SMOTE"):
    st.markdown("""
    **Purpose:** Address class imbalance in the target variable.

    **Problem:** The dataset has approximately 16% attrition rate (minority class),
    which can cause models to be biased toward predicting "No Attrition" since
    that maximizes accuracy without actually learning attrition patterns.

    **Solution:** SMOTE (Synthetic Minority Over-sampling Technique)
    - Creates synthetic samples for the minority class
    - Works by interpolating between existing minority samples in feature space
    - Applied ONLY to the training set to prevent data leakage
    """)

    # Show class distribution
    attrition_counts = df["Attrition"].value_counts()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Stayed (No)", "Left (Yes)"],
        y=attrition_counts.values,
        marker_color=["#00D4AA", "#FF6B6B"],
        text=[f"{v} ({v/len(df)*100:.1f}%)" for v in attrition_counts.values],
        textposition="auto",
    ))
    fig.update_layout(
        title="Class Distribution (Before SMOTE)",
        template=PLOTLY_TEMPLATE, height=350,
        yaxis_title="Count",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **After SMOTE (applied during training):**
    - The minority class (Attrition=Yes) is upsampled to match the majority class
    - This creates a balanced training set, improving the model's ability to
      detect at-risk employees
    - Test set remains unchanged to provide realistic evaluation metrics

    **Why SMOTE over other methods:**
    - **vs Random Oversampling:** SMOTE creates NEW synthetic samples rather than
      duplicating existing ones, reducing overfitting risk
    - **vs Random Undersampling:** We preserve ALL majority class samples, keeping
      valuable information from our already-small dataset
    - **vs Class Weights:** SMOTE works at the data level rather than the algorithm
      level, making it compatible with any classifier
    """)


# ---------------------------------------------------------------------------
# Step 9: Summary
# ---------------------------------------------------------------------------

st.markdown("### 📝 Preprocessing Pipeline Summary")

summary_data = {
    "Step": [
        "1. Raw Data Loading",
        "2. Missing Value Analysis",
        "3. Duplicate Removal",
        "4. Constant Column Removal",
        "5. Outlier Detection",
        "6. Feature Engineering",
        "7. Feature Encoding",
        "8. Feature Scaling",
        "9. SMOTE (Training Only)",
    ],
    "Action": [
        f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns",
        "No missing values found — no action needed",
        f"Removed {n_dupes} duplicates",
        f"Dropped {len(dropped)} constant columns: {', '.join(dropped)}",
        "Identified outliers via IQR — retained for tree models",
        f"Created {len(existing_new)} derived features",
        f"Label encoded 3 binary + One-hot encoded creating {len(onehot_cols)} features",
        "StandardScaler applied to numeric features (train-fit, test-transform)",
        "Synthetic minority oversampling to balance Attrition classes",
    ],
    "Rationale": [
        "Validate data integrity and schema compliance",
        "Missing values can bias models; clean data ensures reliability",
        "Prevent duplicate records from inflating model confidence",
        "Zero-variance features add noise without predictive value",
        "Tree models handle outliers; removing would lose valid data",
        "Domain-driven features capture business patterns models can't infer",
        "ML models require numerical input; encoding strategy preserves semantics",
        "Normalizes feature magnitudes for Logistic Regression convergence",
        "Balances training data to prevent majority-class bias",
    ],
}

summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df, use_container_width=True, hide_index=True, height=380)
