"""
Page 2 – Data Exploration
===========================

Purpose:
    Interactive EDA dashboard with dynamic filters for exploring employee
    attrition patterns across multiple dimensions.

Design Decisions:
    - Tab-based layout to organize different types of analysis.
    - Sidebar filters update all charts simultaneously for cohort analysis.
    - Business insights accompany each visualization to demonstrate
      analytical thinking.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from utils.ui_components import apply_custom_css, render_header, render_divider, render_sidebar_info
from utils.data_loader import load_data, NUMERIC_COLUMNS


# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Data Exploration | Attrition Predictor", page_icon="🔍", layout="wide")
apply_custom_css()
render_sidebar_info()

PLOTLY_TEMPLATE = "plotly_dark"
COLORS = ["#6C63FF", "#00D4AA", "#FF6B6B", "#FFD93D", "#4ECDC4", "#FF8A5C", "#A8E6CF", "#DDA0DD"]


# ---------------------------------------------------------------------------
# Data Loading & Filters
# ---------------------------------------------------------------------------

df = load_data()
if df is None:
    st.error("❌ Failed to load dataset.")
    st.stop()

render_header("🔍 Data Exploration", "Interactive exploratory data analysis with dynamic filtering")

# Sidebar filters
st.sidebar.markdown("### 🔎 Filters")

departments = st.sidebar.multiselect(
    "Department", df["Department"].unique().tolist(),
    default=df["Department"].unique().tolist(), key="eda_dept",
)
genders = st.sidebar.multiselect(
    "Gender", df["Gender"].unique().tolist(),
    default=df["Gender"].unique().tolist(), key="eda_gender",
)
job_roles = st.sidebar.multiselect(
    "Job Role", sorted(df["JobRole"].unique().tolist()),
    default=sorted(df["JobRole"].unique().tolist()), key="eda_role",
)
attrition_status = st.sidebar.multiselect(
    "Attrition Status", ["Yes", "No"],
    default=["Yes", "No"], key="eda_attrition",
)

# Apply filters
filtered = df[
    (df["Department"].isin(departments)) &
    (df["Gender"].isin(genders)) &
    (df["JobRole"].isin(job_roles)) &
    (df["Attrition"].isin(attrition_status))
]

st.sidebar.markdown(f"**Showing:** {len(filtered):,} / {len(df):,} employees")

render_divider()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Distributions", "🔗 Correlations", "📈 Relationships", "⏰ Overtime Impact"
])


# ---------------------------------------------------------------------------
# Tab 1: Distributions
# ---------------------------------------------------------------------------

with tab1:
    st.markdown("### Numeric Feature Distributions")
    st.markdown("Explore how numeric features are distributed, colored by attrition status.")

    col1, col2 = st.columns(2)

    feature_options = [c for c in NUMERIC_COLUMNS if c in filtered.columns]
    selected_feature = col1.selectbox("Select Feature", feature_options, index=feature_options.index("Age") if "Age" in feature_options else 0)

    chart_type = col2.selectbox("Chart Type", ["Histogram", "Box Plot", "Violin Plot"])

    if chart_type == "Histogram":
        fig = px.histogram(
            filtered, x=selected_feature, color="Attrition",
            barmode="overlay", opacity=0.7, nbins=30,
            color_discrete_map={"Yes": "#FF6B6B", "No": "#00D4AA"},
            template=PLOTLY_TEMPLATE,
        )
    elif chart_type == "Box Plot":
        fig = px.box(
            filtered, x="Attrition", y=selected_feature, color="Attrition",
            color_discrete_map={"Yes": "#FF6B6B", "No": "#00D4AA"},
            template=PLOTLY_TEMPLATE,
        )
    else:
        fig = px.violin(
            filtered, x="Attrition", y=selected_feature, color="Attrition",
            box=True, points="outliers",
            color_discrete_map={"Yes": "#FF6B6B", "No": "#00D4AA"},
            template=PLOTLY_TEMPLATE,
        )

    fig.update_layout(height=450, margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    st.markdown("#### 📋 Descriptive Statistics")
    stats_by_attrition = filtered.groupby("Attrition")[selected_feature].describe().round(2)
    st.dataframe(stats_by_attrition, use_container_width=True)

    render_divider()

    # Categorical distributions
    st.markdown("### Categorical Feature Distributions")

    cat_cols = ["Department", "JobRole", "EducationField", "MaritalStatus", "BusinessTravel"]
    selected_cat = st.selectbox("Select Categorical Feature", cat_cols)

    cat_data = filtered.groupby([selected_cat, "Attrition"]).size().reset_index(name="Count")
    fig = px.bar(
        cat_data, x=selected_cat, y="Count", color="Attrition",
        barmode="group",
        color_discrete_map={"Yes": "#FF6B6B", "No": "#00D4AA"},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(
        height=450, margin=dict(l=40, r=40, t=40, b=40),
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Attrition rate per category
    rate_data = filtered.groupby(selected_cat)["Attrition"].apply(
        lambda x: (x == "Yes").mean() * 100
    ).round(1).sort_values(ascending=False).reset_index(name="Attrition Rate (%)")

    st.dataframe(rate_data, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 2: Correlations
# ---------------------------------------------------------------------------

with tab2:
    st.markdown("### Correlation Heatmap")
    st.markdown("Shows Pearson correlations between numeric features. Stronger correlations indicate features that move together.")

    numeric_df = filtered.select_dtypes(include=[np.number])
    corr = numeric_df.corr().round(2)

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale=[[0, "#FF6B6B"], [0.5, "#1A1F2E"], [1, "#00D4AA"]],
        zmid=0,
        text=corr.values,
        texttemplate="%{text:.1f}",
        textfont={"size": 8},
        hoverongaps=False,
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=700,
        margin=dict(l=100, r=40, t=40, b=100),
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="info-card">
        <strong>💡 Key Correlations:</strong><br>
        • <strong>MonthlyIncome & JobLevel</strong>: Strong positive — seniority drives pay<br>
        • <strong>TotalWorkingYears & MonthlyIncome</strong>: Positive — experience correlates with earnings<br>
        • <strong>YearsAtCompany & YearsInCurrentRole</strong>: Positive — tenure metrics co-occur<br>
        • <strong>Age & TotalWorkingYears</strong>: Strong positive — expected demographic pattern
    </div>
    """, unsafe_allow_html=True)

    # Top correlations table
    st.markdown("#### Top Correlations (Absolute)")
    corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            corr_pairs.append({
                "Feature 1": corr.columns[i],
                "Feature 2": corr.columns[j],
                "Correlation": corr.iloc[i, j],
                "Absolute": abs(corr.iloc[i, j]),
            })

    corr_df = pd.DataFrame(corr_pairs).sort_values("Absolute", ascending=False).head(15)
    corr_df = corr_df.drop(columns=["Absolute"])
    st.dataframe(corr_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 3: Relationships
# ---------------------------------------------------------------------------

with tab3:
    st.markdown("### Feature Relationships")
    st.markdown("Explore bivariate relationships between features.")

    col_x, col_y = st.columns(2)

    x_feature = col_x.selectbox("X-Axis Feature", feature_options, index=0, key="rel_x")
    y_feature = col_y.selectbox("Y-Axis Feature", feature_options,
                                 index=min(3, len(feature_options) - 1), key="rel_y")

    fig = px.scatter(
        filtered, x=x_feature, y=y_feature, color="Attrition",
        color_discrete_map={"Yes": "#FF6B6B", "No": "#00D4AA"},
        template=PLOTLY_TEMPLATE,
        opacity=0.6,
        hover_data=["Department", "JobRole"],
    )
    fig.update_layout(height=500, margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)

    # Additional relationship views
    st.markdown("#### Job Role vs Attrition Rate")

    role_rates = filtered.groupby("JobRole").agg(
        Total=("Attrition", "count"),
        Attrition_Count=("Attrition", lambda x: (x == "Yes").sum()),
    ).reset_index()
    role_rates["Attrition_Rate"] = (role_rates["Attrition_Count"] / role_rates["Total"] * 100).round(1)
    role_rates = role_rates.sort_values("Attrition_Rate", ascending=True)

    fig = go.Figure(go.Bar(
        x=role_rates["Attrition_Rate"],
        y=role_rates["JobRole"],
        orientation="h",
        marker=dict(
            color=role_rates["Attrition_Rate"],
            colorscale=[[0, "#00D4AA"], [0.5, "#FFD93D"], [1, "#FF6B6B"]],
        ),
        text=role_rates["Attrition_Rate"].apply(lambda x: f"{x}%"),
        textposition="outside",
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=400,
        xaxis_title="Attrition Rate (%)", yaxis_title="",
        margin=dict(l=200, r=80, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="info-card">
        <strong>💡 Insight:</strong> Sales Representatives and Laboratory Technicians
        typically show the highest attrition rates. These roles often have lower pay
        relative to workload, limited career progression, and higher market demand —
        all contributing to turnover.
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 4: Overtime Impact
# ---------------------------------------------------------------------------

with tab4:
    st.markdown("### Overtime Impact Analysis")
    st.markdown("Detailed analysis of how overtime work affects employee attrition.")

    col_ot1, col_ot2 = st.columns(2)

    with col_ot1:
        # Overtime attrition rate
        ot_data = filtered.groupby("OverTime").agg(
            Total=("Attrition", "count"),
            Left=("Attrition", lambda x: (x == "Yes").sum()),
        ).reset_index()
        ot_data["Attrition_Rate"] = (ot_data["Left"] / ot_data["Total"] * 100).round(1)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=ot_data["OverTime"].map({"Yes": "Works Overtime", "No": "No Overtime"}),
            y=ot_data["Attrition_Rate"],
            marker_color=["#FF6B6B", "#00D4AA"],
            text=ot_data["Attrition_Rate"].apply(lambda x: f"{x}%"),
            textposition="auto",
            width=0.5,
        ))

        fig.update_layout(
            title="Attrition Rate by Overtime Status",
            template=PLOTLY_TEMPLATE, height=400,
            yaxis_title="Attrition Rate (%)",
            margin=dict(l=40, r=40, t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_ot2:
        # Overtime by department
        ot_dept = filtered.groupby(["Department", "OverTime"]).size().reset_index(name="Count")

        fig = px.bar(
            ot_dept, x="Department", y="Count", color="OverTime",
            barmode="group",
            color_discrete_map={"Yes": "#FF6B6B", "No": "#00D4AA"},
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(
            title="Overtime Distribution by Department",
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Overtime statistics
    st.markdown("#### 📋 Overtime Statistics")

    ot_stats = filtered.groupby("OverTime").agg({
        "MonthlyIncome": "mean",
        "WorkLifeBalance": "mean",
        "JobSatisfaction": "mean",
        "DistanceFromHome": "mean",
        "YearsAtCompany": "mean",
    }).round(2)

    ot_stats.columns = ["Avg Monthly Income", "Avg Work-Life Balance", "Avg Job Satisfaction",
                         "Avg Distance From Home", "Avg Years at Company"]

    st.dataframe(ot_stats, use_container_width=True)

    st.markdown("""
    <div class="info-card">
        <strong>💡 Insight:</strong> Overtime employees often report lower work-life balance
        scores and job satisfaction. The combination of extended hours and potential commute
        stress creates a compounded burnout effect, making overtime management one of the
        most impactful levers for reducing attrition.
    </div>
    """, unsafe_allow_html=True)
