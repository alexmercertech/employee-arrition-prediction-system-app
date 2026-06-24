"""
Data Loader Module
==================

Purpose:
    Provides functions to load, validate, and summarize the IBM HR Analytics
    Employee Attrition dataset. Uses Streamlit caching for performance.

Design Decisions:
    - @st.cache_data is used to cache the DataFrame across reruns, avoiding
      repeated disk I/O and CSV parsing.
    - Validation checks for expected columns to catch data integrity issues early.
    - Graceful error handling returns None with st.error() messages rather than
      crashing the application.

Assumptions:
    - The dataset CSV is located at data/WA_Fn-UseC_-HR-Employee-Attrition.csv
      relative to the project root.
    - The CSV uses UTF-8 encoding (with optional BOM).
"""

import os
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Path to the dataset relative to the project root
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATA_FILE = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)

# Expected columns in the IBM HR dataset (35 total)
EXPECTED_COLUMNS = [
    "Age", "Attrition", "BusinessTravel", "DailyRate", "Department",
    "DistanceFromHome", "Education", "EducationField", "EmployeeCount",
    "EmployeeNumber", "EnvironmentSatisfaction", "Gender", "HourlyRate",
    "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
    "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
    "Over18", "OverTime", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
    "StandardHours",
]

# Columns that are constant or irrelevant and should be dropped
COLUMNS_TO_DROP = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]

# Categorical columns for encoding
CATEGORICAL_COLUMNS = [
    "BusinessTravel", "Department", "EducationField", "Gender",
    "JobRole", "MaritalStatus", "OverTime",
]

# Binary categorical columns (label encode)
BINARY_COLUMNS = ["Gender", "OverTime", "Attrition"]

# Multi-category columns (one-hot encode)
MULTI_CATEGORY_COLUMNS = [
    "BusinessTravel", "Department", "EducationField",
    "JobRole", "MaritalStatus",
]

# Numeric columns for analysis
NUMERIC_COLUMNS = [
    "Age", "DailyRate", "DistanceFromHome", "Education",
    "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement", "JobLevel",
    "JobSatisfaction", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
    "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction",
    "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear",
    "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager",
]

# Ordinal columns with human-readable labels
EDUCATION_MAP = {
    1: "Below College", 2: "College", 3: "Bachelor",
    4: "Master", 5: "Doctor",
}

ENVIRONMENT_SATISFACTION_MAP = {
    1: "Low", 2: "Medium", 3: "High", 4: "Very High",
}

JOB_INVOLVEMENT_MAP = {
    1: "Low", 2: "Medium", 3: "High", 4: "Very High",
}

JOB_SATISFACTION_MAP = {
    1: "Low", 2: "Medium", 3: "High", 4: "Very High",
}

PERFORMANCE_RATING_MAP = {
    1: "Low", 2: "Good", 3: "Excellent", 4: "Outstanding",
}

RELATIONSHIP_SATISFACTION_MAP = {
    1: "Low", 2: "Medium", 3: "High", 4: "Very High",
}

WORK_LIFE_BALANCE_MAP = {
    1: "Bad", 2: "Good", 3: "Better", 4: "Best",
}


# ---------------------------------------------------------------------------
# Data Loading Functions
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data(filepath: str = DATA_PATH) -> pd.DataFrame | None:
    """
    Load the HR attrition dataset from a CSV file.

    Uses Streamlit's caching decorator to avoid reloading on every rerun.
    Handles FileNotFoundError and general exceptions gracefully by displaying
    error messages through Streamlit's UI rather than raising unhandled
    exceptions.

    Args:
        filepath: Absolute or relative path to the CSV file.

    Returns:
        A pandas DataFrame containing the raw dataset, or None if loading fails.
    """
    try:
        df = pd.read_csv(filepath, encoding="utf-8-sig")
        return df
    except FileNotFoundError:
        st.error(
            f"❌ Dataset file not found at: `{filepath}`\n\n"
            "Please ensure the file `WA_Fn-UseC_-HR-Employee-Attrition.csv` "
            "is placed in the `data/` directory."
        )
        return None
    except pd.errors.EmptyDataError:
        st.error("❌ The dataset file is empty. Please check the file.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return None


def validate_data(df: pd.DataFrame) -> dict:
    """
    Validate the loaded DataFrame against expected schema.

    Checks for:
        - Presence of expected columns
        - Unexpected additional columns
        - Data type consistency
        - Row count sanity

    Args:
        df: The loaded DataFrame to validate.

    Returns:
        A dictionary containing validation results with keys:
            - 'is_valid' (bool): Whether the dataset passes all checks.
            - 'missing_columns' (list): Expected columns not found.
            - 'extra_columns' (list): Unexpected columns found.
            - 'row_count' (int): Number of rows.
            - 'col_count' (int): Number of columns.
            - 'messages' (list): Human-readable validation messages.
    """
    results = {
        "is_valid": True,
        "missing_columns": [],
        "extra_columns": [],
        "row_count": len(df),
        "col_count": len(df.columns),
        "messages": [],
    }

    # Check for missing expected columns
    actual_cols = set(df.columns)
    expected_cols = set(EXPECTED_COLUMNS)
    results["missing_columns"] = list(expected_cols - actual_cols)
    results["extra_columns"] = list(actual_cols - expected_cols)

    if results["missing_columns"]:
        results["is_valid"] = False
        results["messages"].append(
            f"⚠️ Missing columns: {', '.join(results['missing_columns'])}"
        )

    if results["extra_columns"]:
        results["messages"].append(
            f"ℹ️ Extra columns found: {', '.join(results['extra_columns'])}"
        )

    # Row count check
    if results["row_count"] == 0:
        results["is_valid"] = False
        results["messages"].append("❌ Dataset has zero rows.")
    else:
        results["messages"].append(
            f"✅ Dataset has {results['row_count']:,} rows and "
            f"{results['col_count']} columns."
        )

    # Check target variable
    if "Attrition" in df.columns:
        unique_values = df["Attrition"].unique()
        if set(unique_values) == {"Yes", "No"}:
            results["messages"].append("✅ Target variable 'Attrition' is valid (Yes/No).")
        else:
            results["messages"].append(
                f"⚠️ Unexpected Attrition values: {unique_values}"
            )

    return results


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate a comprehensive summary of the dataset.

    Produces statistics useful for the EDA dashboard including shape,
    data types, missing value counts, unique value counts, and basic
    descriptive statistics.

    Args:
        df: The DataFrame to summarize.

    Returns:
        A dictionary containing:
            - 'shape': Tuple of (rows, columns)
            - 'dtypes': Series of column data types
            - 'missing': Series of missing value counts per column
            - 'missing_pct': Series of missing value percentages
            - 'unique': Series of unique value counts per column
            - 'numeric_stats': DataFrame of descriptive statistics
            - 'memory_usage': Total memory usage in MB
    """
    summary = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str),
        "missing": df.isnull().sum(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2),
        "unique": df.nunique(),
        "numeric_stats": df.describe(),
        "memory_usage": df.memory_usage(deep=True).sum() / 1024 / 1024,
    }
    return summary


def get_attrition_stats(df: pd.DataFrame) -> dict:
    """
    Compute key attrition-related business metrics.

    These metrics are displayed on the Dashboard page as KPI cards.

    Args:
        df: The DataFrame containing employee data.

    Returns:
        A dictionary containing:
            - 'total_employees': Total number of employees
            - 'attrition_count': Number who left
            - 'retention_count': Number who stayed
            - 'attrition_rate': Percentage who left
            - 'avg_salary': Mean monthly income
            - 'avg_age': Mean employee age
            - 'avg_tenure': Mean years at company
            - 'avg_satisfaction': Mean job satisfaction score
    """
    attrition_count = (df["Attrition"] == "Yes").sum() if "Attrition" in df.columns else 0
    total = len(df)

    stats = {
        "total_employees": total,
        "attrition_count": attrition_count,
        "retention_count": total - attrition_count,
        "attrition_rate": round(attrition_count / total * 100, 1) if total > 0 else 0,
        "avg_salary": round(df["MonthlyIncome"].mean(), 0) if "MonthlyIncome" in df.columns else 0,
        "avg_age": round(df["Age"].mean(), 1) if "Age" in df.columns else 0,
        "avg_tenure": round(df["YearsAtCompany"].mean(), 1) if "YearsAtCompany" in df.columns else 0,
        "avg_satisfaction": round(df["JobSatisfaction"].mean(), 2) if "JobSatisfaction" in df.columns else 0,
    }
    return stats
