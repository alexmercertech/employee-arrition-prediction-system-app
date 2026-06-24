"""
Feature Engineering Module
===========================

Purpose:
    Creates derived features from the raw IBM HR dataset that provide
    additional predictive signal for attrition modeling. Each feature is
    documented with its business rationale.

Design Decisions:
    - Features are computed as simple, interpretable transformations to
      maintain explainability (important for HR use cases).
    - Division operations use (denominator + 1) to prevent division by zero.
    - Composite scores use equal weighting of satisfaction dimensions.

Assumptions:
    - Input DataFrame contains the original IBM HR columns prior to encoding.
    - Numeric columns have already been cleaned (no nulls in key fields).

Limitations:
    - Feature interactions are linear combinations; non-linear relationships
      are left to the model to discover.
    - Domain-specific thresholds (e.g., "new employee" <= 1 year) are
      heuristic and may not generalize to all organizations.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Feature Definitions with Business Rationale
# ---------------------------------------------------------------------------

FEATURE_DOCS = {
    "YearsPerCompany": {
        "formula": "TotalWorkingYears / (NumCompaniesWorked + 1)",
        "rationale": (
            "Measures average tenure at each employer. A low value indicates "
            "a pattern of frequent job changes (potential flight risk), while "
            "a high value suggests loyalty and stability."
        ),
        "type": "Loyalty Indicator",
    },
    "IncomePerYearWorked": {
        "formula": "MonthlyIncome / (TotalWorkingYears + 1)",
        "rationale": (
            "Captures salary progression relative to experience. Employees "
            "with lower income-per-year may feel underpaid relative to their "
            "experience, increasing attrition risk."
        ),
        "type": "Compensation Indicator",
    },
    "SatisfactionIndex": {
        "formula": "mean(EnvironmentSatisfaction, JobSatisfaction, RelationshipSatisfaction, WorkLifeBalance)",
        "rationale": (
            "A composite satisfaction score averaging four satisfaction "
            "dimensions. Provides a single holistic measure of employee "
            "well-being. Low scores correlate with higher attrition."
        ),
        "type": "Well-being Indicator",
    },
    "IsNewEmployee": {
        "formula": "1 if YearsAtCompany <= 1, else 0",
        "rationale": (
            "Flags employees in their first year at the company. New hires "
            "are statistically more likely to leave (the 'new hire cliff'). "
            "This binary feature helps the model capture this pattern."
        ),
        "type": "Tenure Indicator",
    },
    "PromotionStagnation": {
        "formula": "YearsSinceLastPromotion - YearsInCurrentRole",
        "rationale": (
            "Measures career stagnation. A positive value means the employee "
            "has been waiting for a promotion longer than they have been in "
            "their current role — a potential frustration indicator."
        ),
        "type": "Growth Indicator",
    },
    "OvertimeDistance": {
        "formula": "DistanceFromHome × OverTime_encoded",
        "rationale": (
            "Interaction between commute distance and overtime work. Employees "
            "who work overtime AND have long commutes face compounded burnout "
            "risk, making this combination a strong attrition predictor."
        ),
        "type": "Burnout Indicator",
    },
    "TenureRatio": {
        "formula": "YearsAtCompany / (TotalWorkingYears + 1)",
        "rationale": (
            "Proportion of total career spent at the current company. A very "
            "low ratio despite many working years may indicate the employee "
            "is a recent hire who has been in the workforce for a long time "
            "— and may be more likely to leave again."
        ),
        "type": "Career Indicator",
    },
    "ManagerTenureRatio": {
        "formula": "YearsWithCurrManager / (YearsAtCompany + 1)",
        "rationale": (
            "How long an employee has been with their current manager relative "
            "to their tenure. A low ratio may indicate frequent manager changes, "
            "which can reduce trust and engagement."
        ),
        "type": "Management Indicator",
    },
}


# ---------------------------------------------------------------------------
# Feature Engineering Functions
# ---------------------------------------------------------------------------

def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all derived features and append them to the DataFrame.

    This function is idempotent — if features already exist, they are
    overwritten (not duplicated).

    Args:
        df: DataFrame with original IBM HR columns.

    Returns:
        DataFrame with additional engineered feature columns.
    """
    df = df.copy()

    # Years per company — loyalty indicator
    df["YearsPerCompany"] = (
        df["TotalWorkingYears"] / (df["NumCompaniesWorked"] + 1)
    ).round(2)

    # Income per year of experience — salary progression
    df["IncomePerYearWorked"] = (
        df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)
    ).round(2)

    # Composite satisfaction index (mean of 4 satisfaction scores)
    satisfaction_cols = [
        "EnvironmentSatisfaction", "JobSatisfaction",
        "RelationshipSatisfaction", "WorkLifeBalance",
    ]
    df["SatisfactionIndex"] = df[satisfaction_cols].mean(axis=1).round(2)

    # New employee flag (first year)
    df["IsNewEmployee"] = (df["YearsAtCompany"] <= 1).astype(int)

    # Promotion stagnation score
    df["PromotionStagnation"] = (
        df["YearsSinceLastPromotion"] - df["YearsInCurrentRole"]
    )

    # Overtime × Distance interaction
    overtime_encoded = (df["OverTime"] == "Yes").astype(int) if not pd.api.types.is_numeric_dtype(df["OverTime"]) else df["OverTime"]
    df["OvertimeDistance"] = df["DistanceFromHome"] * overtime_encoded

    # Tenure ratio — proportion of career at this company
    df["TenureRatio"] = (
        df["YearsAtCompany"] / (df["TotalWorkingYears"] + 1)
    ).round(2)

    # Manager tenure ratio
    df["ManagerTenureRatio"] = (
        df["YearsWithCurrManager"] / (df["YearsAtCompany"] + 1)
    ).round(2)

    return df


def get_engineered_feature_names() -> list[str]:
    """Return the names of all engineered features."""
    return list(FEATURE_DOCS.keys())


def get_feature_documentation() -> pd.DataFrame:
    """
    Return a DataFrame documenting all engineered features.

    Useful for display in the Data Preprocessing page to explain
    each feature's purpose and formula to assessors.

    Returns:
        DataFrame with columns: Feature, Formula, Rationale, Type
    """
    rows = []
    for name, info in FEATURE_DOCS.items():
        rows.append({
            "Feature": name,
            "Formula": info["formula"],
            "Rationale": info["rationale"],
            "Type": info["type"],
        })
    return pd.DataFrame(rows)
