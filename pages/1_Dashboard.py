"""
Page 1 – Dashboard
====================

Purpose:
    Displays key HR metrics and attrition analytics through interactive
    KPI cards and Plotly charts. Provides a high-level overview of
    employee attrition patterns across the organization.

Design Decisions:
    - KPI cards use glassmorphism styling for a premium enterprise look.
    - All charts use the Plotly dark template for consistency.
    - Sidebar filters allow dynamic exploration of specific segments.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from utils.ui_components import apply_custom_css, render_kpi_card, render_header, render_divider, render_sidebar_info
from utils.data_loader import load_data, get_attrition_stats


# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Dashboard | Attrition Predictor", page_icon="📊", layout="wide")
apply_custom_css()
render_sidebar_info()

PLOTLY_TEMPLATE = "plotly_dark"
COLORS = {"primary": "#6C63FF", "accent": "#00D4AA", "warning": "#FF6B6B", "gold": "#FFD93D"}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

df = load_data()

if df is None:
    st.error("❌ Failed to load dataset. Please ensure the data file exists.")
    st.stop()

# Sidebar filters
st.sidebar.markdown("### 🔎 Filters")
departments = ["All"] + sorted(df["Department"].unique().tolist())
selected_dept = st.sidebar.selectbox("Department", departments, key="dash_dept")

genders = ["All"] + sorted(df["Gender"].unique().tolist())
selected_gender = st.sidebar.selectbox("Gender", genders, key="dash_gender")

attrition_filter = st.sidebar.selectbox("Attrition Status", ["All", "Yes", "No"], key="dash_attrition")

# Apply filters
filtered_df = df.copy()
if selected_dept != "All":
    filtered_df = filtered_df[filtered_df["Department"] == selected_dept]
if selected_gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]
if attrition_filter != "All":
    filtered_df = filtered_df[filtered_df["Attrition"] == attrition_filter]


# ---------------------------------------------------------------------------
# Header & KPIs
# ---------------------------------------------------------------------------

render_header("📊 HR Analytics Dashboard", "Real-time overview of employee attrition metrics and organizational health")

stats = get_attrition_stats(filtered_df)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    render_kpi_card("👥", f"{stats['total_employees']:,}", "Total Employees")
with col2:
    render_kpi_card("🚪", f"{stats['attrition_count']:,}", "Left Company",
                    delta=f"{stats['attrition_rate']}%", delta_positive=False)
with col3:
    render_kpi_card("📈", f"{stats['attrition_rate']}%", "Attrition Rate")
with col4:
    render_kpi_card("💰", f"${stats['avg_salary']:,.0f}", "Avg Monthly Salary")
with col5:
    render_kpi_card("⏱️", f"{stats['avg_tenure']} yrs", "Avg Tenure")

st.markdown("")

render_divider()


# ---------------------------------------------------------------------------
# Charts Row 1: Attrition Distribution & Department Analysis
# ---------------------------------------------------------------------------

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### Attrition Distribution")

    attrition_counts = filtered_df["Attrition"].value_counts()
    fig = go.Figure(data=[go.Pie(
        labels=attrition_counts.index.map({"Yes": "Left", "No": "Stayed"}),
        values=attrition_counts.values,
        hole=0.55,
        marker=dict(colors=[COLORS["warning"], COLORS["accent"]]),
        textinfo="label+percent",
        textfont=dict(size=14),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
    )])

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        annotations=[dict(text=f"<b>{stats['attrition_rate']}%</b><br>Attrition",
                          x=0.5, y=0.5, font_size=16, showarrow=False,
                          font_color="#FAFAFA")],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="info-card">
        <strong>💡 Insight:</strong> The attrition rate reflects the proportion of employees
        who have left the organization. Industry benchmarks suggest rates above 15% warrant
        immediate attention to retention strategies.
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown("#### Department-wise Attrition")

    dept_attrition = filtered_df.groupby(["Department", "Attrition"]).size().reset_index(name="Count")
    fig = px.bar(
        dept_attrition, x="Department", y="Count", color="Attrition",
        barmode="group",
        color_discrete_map={"Yes": COLORS["warning"], "No": COLORS["accent"]},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        xaxis_title="", yaxis_title="Employee Count",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Business insight
    dept_rates = filtered_df.groupby("Department")["Attrition"].apply(
        lambda x: (x == "Yes").mean() * 100
    ).round(1)
    highest_dept = dept_rates.idxmax()
    st.markdown(f"""
    <div class="info-card">
        <strong>💡 Insight:</strong> <strong>{highest_dept}</strong> has the highest
        attrition rate at <strong>{dept_rates.max()}%</strong>. This department may need
        targeted retention initiatives such as improved career development opportunities.
    </div>
    """, unsafe_allow_html=True)


render_divider()


# ---------------------------------------------------------------------------
# Charts Row 2: Age Distribution & Income Analysis
# ---------------------------------------------------------------------------

col_left2, col_right2 = st.columns(2)

with col_left2:
    st.markdown("#### Age Distribution by Attrition")

    fig = go.Figure()
    for status, color in [("No", COLORS["accent"]), ("Yes", COLORS["warning"])]:
        subset = filtered_df[filtered_df["Attrition"] == status]
        fig.add_trace(go.Histogram(
            x=subset["Age"], name=f"{'Stayed' if status == 'No' else 'Left'}",
            marker_color=color, opacity=0.7,
            nbinsx=20,
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=380,
        barmode="overlay",
        xaxis_title="Age", yaxis_title="Count",
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)

    young_attrition = filtered_df[
        (filtered_df["Age"] < 35) & (filtered_df["Attrition"] == "Yes")
    ].shape[0]
    total_attrition = (filtered_df["Attrition"] == "Yes").sum()
    young_pct = round(young_attrition / max(total_attrition, 1) * 100, 1)

    st.markdown(f"""
    <div class="info-card">
        <strong>💡 Insight:</strong> <strong>{young_pct}%</strong> of attrition occurs
        among employees under 35, suggesting younger employees are more likely to leave.
        Targeted engagement programs for early-career professionals could reduce this.
    </div>
    """, unsafe_allow_html=True)

with col_right2:
    st.markdown("#### Monthly Income by Job Role")

    fig = px.box(
        filtered_df, x="JobRole", y="MonthlyIncome",
        color="Attrition",
        color_discrete_map={"Yes": COLORS["warning"], "No": COLORS["accent"]},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="", yaxis_title="Monthly Income ($)",
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)

    avg_income_left = filtered_df[filtered_df["Attrition"] == "Yes"]["MonthlyIncome"].mean()
    avg_income_stayed = filtered_df[filtered_df["Attrition"] == "No"]["MonthlyIncome"].mean()
    income_diff = round(avg_income_stayed - avg_income_left, 0)

    st.markdown(f"""
    <div class="info-card">
        <strong>💡 Insight:</strong> Employees who left earned on average
        <strong>${income_diff:,.0f} less</strong> per month than those who stayed.
        Compensation competitiveness is a key factor in retention.
    </div>
    """, unsafe_allow_html=True)


render_divider()


# ---------------------------------------------------------------------------
# Charts Row 3: Overtime Impact & Job Satisfaction
# ---------------------------------------------------------------------------

col_left3, col_right3 = st.columns(2)

with col_left3:
    st.markdown("#### Overtime Impact on Attrition")

    ot_attrition = filtered_df.groupby(["OverTime", "Attrition"]).size().reset_index(name="Count")
    ot_total = filtered_df.groupby("OverTime").size().reset_index(name="Total")
    ot_attrition = ot_attrition.merge(ot_total, on="OverTime")
    ot_attrition["Percentage"] = (ot_attrition["Count"] / ot_attrition["Total"] * 100).round(1)

    fig = px.bar(
        ot_attrition, x="OverTime", y="Count", color="Attrition",
        barmode="group",
        color_discrete_map={"Yes": COLORS["warning"], "No": COLORS["accent"]},
        template=PLOTLY_TEMPLATE,
        text="Count",
    )
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Works Overtime?", yaxis_title="Employee Count",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)

    ot_yes_attrition = filtered_df[
        (filtered_df["OverTime"] == "Yes") & (filtered_df["Attrition"] == "Yes")
    ].shape[0]
    ot_yes_total = (filtered_df["OverTime"] == "Yes").sum()
    ot_rate = round(ot_yes_attrition / max(ot_yes_total, 1) * 100, 1)

    st.markdown(f"""
    <div class="info-card">
        <strong>💡 Insight:</strong> Employees who work overtime have a <strong>{ot_rate}%</strong>
        attrition rate — significantly higher than average. Overtime management and
        workload distribution should be prioritized.
    </div>
    """, unsafe_allow_html=True)

with col_right3:
    st.markdown("#### Job Satisfaction vs Attrition")

    sat_map = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
    temp_df = filtered_df.copy()
    temp_df["SatisfactionLabel"] = temp_df["JobSatisfaction"].map(sat_map)

    sat_attrition = temp_df.groupby(["SatisfactionLabel", "Attrition"]).size().reset_index(name="Count")

    fig = px.bar(
        sat_attrition, x="SatisfactionLabel", y="Count", color="Attrition",
        barmode="group",
        color_discrete_map={"Yes": COLORS["warning"], "No": COLORS["accent"]},
        template=PLOTLY_TEMPLATE,
        category_orders={"SatisfactionLabel": ["Low", "Medium", "High", "Very High"]},
    )
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Job Satisfaction Level", yaxis_title="Employee Count",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)

    low_sat_attrition = filtered_df[
        (filtered_df["JobSatisfaction"] == 1) & (filtered_df["Attrition"] == "Yes")
    ].shape[0]
    low_sat_total = (filtered_df["JobSatisfaction"] == 1).sum()
    low_sat_rate = round(low_sat_attrition / max(low_sat_total, 1) * 100, 1)

    st.markdown(f"""
    <div class="info-card">
        <strong>💡 Insight:</strong> Low job satisfaction has an attrition rate of
        <strong>{low_sat_rate}%</strong>. Regular employee satisfaction surveys and
        prompt action on feedback are essential retention tools.
    </div>
    """, unsafe_allow_html=True)
