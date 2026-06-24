"""
Page 6 – Employee Prediction
================================

Purpose:
    Professional HR prediction form that accepts employee attributes and
    returns attrition prediction with probability, risk level, top factors,
    and context-aware retention recommendations.

Design Decisions:
    - Form inputs are organized into logical sections matching HR workflows.
    - Prediction uses the best model from the comparison page (or loads
      the most recent saved model as fallback).
    - Results include both technical metrics and actionable HR insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

from utils.ui_components import (
    apply_custom_css, render_header, render_divider, render_sidebar_info,
    render_risk_badge,
)
from utils.data_loader import load_data
from utils.preprocessing import preprocess_data
from utils.model_trainer import load_model, list_saved_models
from utils.explainability import compute_shap_values, generate_explanation


# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Employee Prediction | Attrition Predictor", page_icon="🎯", layout="wide")
apply_custom_css()
render_sidebar_info()

render_header("🎯 Employee Attrition Prediction",
              "Enter employee details to predict attrition risk with AI-powered explainability")


# ---------------------------------------------------------------------------
# Load Model
# ---------------------------------------------------------------------------

def get_prediction_model():
    """Get the best available model for predictions."""
    # Priority 1: Best model from current session
    if "best_model" in st.session_state:
        return st.session_state["best_model"], st.session_state.get("best_model_name", "Best Model")

    # Priority 2: Load from disk
    saved = list_saved_models()
    if saved:
        # Try to load XGBoost first, then Random Forest, etc.
        priority = ["XGBoost", "Random Forest", "Logistic Regression", "Decision Tree"]
        for name in priority:
            bundle = load_model(name)
            if bundle:
                return bundle["model"], name

        # Load the first available
        bundle = load_model(saved[0]["name"])
        if bundle:
            return bundle["model"], saved[0]["name"]

    return None, None


model, model_name = get_prediction_model()

if model is None:
    st.warning(
        "⚠️ No trained model available. Please go to the **Model Training** page "
        "and train at least one model before making predictions."
    )
    st.stop()

st.info(f"🤖 Using model: **{model_name}**")

render_divider()


# ---------------------------------------------------------------------------
# Preprocessing Setup
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_preprocessing_artifacts():
    """Load data and compute preprocessing artifacts once."""
    df = load_data()
    if df is None:
        return None
    prep = preprocess_data(df, apply_smote_flag=False)
    return prep


prep = get_preprocessing_artifacts()
if prep is None:
    st.error("❌ Failed to prepare preprocessing artifacts.")
    st.stop()


# ---------------------------------------------------------------------------
# Employee Input Form
# ---------------------------------------------------------------------------

st.markdown("### 📝 Employee Information")

with st.form("prediction_form"):

    # -- Section 1: Personal Information --
    st.markdown("#### 👤 Personal Information")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)

    with p_col1:
        age = st.number_input("Age", min_value=18, max_value=65, value=35, step=1)
    with p_col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with p_col3:
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    with p_col4:
        distance = st.number_input("Distance From Home (miles)", min_value=1, max_value=30, value=10)

    st.markdown("---")

    # -- Section 2: Job Information --
    st.markdown("#### 💼 Job Information")
    j_col1, j_col2, j_col3, j_col4 = st.columns(4)

    with j_col1:
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    with j_col2:
        job_role = st.selectbox("Job Role", [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative",
            "Manager", "Sales Representative", "Research Director",
            "Human Resources",
        ])
    with j_col3:
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5], index=1)
    with j_col4:
        education = st.selectbox("Education", [1, 2, 3, 4, 5], index=2,
                                  format_func=lambda x: {
                                      1: "Below College", 2: "College",
                                      3: "Bachelor", 4: "Master", 5: "Doctor"
                                  }[x])

    j_col5, j_col6, j_col7, j_col8 = st.columns(4)

    with j_col5:
        education_field = st.selectbox("Education Field", [
            "Life Sciences", "Medical", "Marketing",
            "Technical Degree", "Human Resources", "Other",
        ])
    with j_col6:
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
    with j_col7:
        years_in_role = st.number_input("Years in Current Role", min_value=0, max_value=20, value=3)
    with j_col8:
        years_with_manager = st.number_input("Years with Current Manager", min_value=0, max_value=20, value=3)

    st.markdown("---")

    # -- Section 3: Compensation --
    st.markdown("#### 💰 Compensation & Benefits")
    c_col1, c_col2, c_col3, c_col4 = st.columns(4)

    with c_col1:
        monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000, step=100)
    with c_col2:
        daily_rate = st.number_input("Daily Rate", min_value=100, max_value=1500, value=800)
    with c_col3:
        hourly_rate = st.number_input("Hourly Rate", min_value=30, max_value=100, value=65)
    with c_col4:
        monthly_rate = st.number_input("Monthly Rate", min_value=2000, max_value=27000, value=14000)

    c_col5, c_col6, c_col7, c_col8 = st.columns(4)

    with c_col5:
        percent_salary_hike = st.number_input("Salary Hike (%)", min_value=11, max_value=25, value=15)
    with c_col6:
        stock_option = st.selectbox("Stock Option Level", [0, 1, 2, 3])
    with c_col7:
        performance_rating = st.selectbox("Performance Rating", [1, 2, 3, 4], index=2,
                                           format_func=lambda x: {
                                               1: "Low", 2: "Good", 3: "Excellent", 4: "Outstanding"
                                           }[x])
    with c_col8:
        years_since_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=1)

    st.markdown("---")

    # -- Section 4: Satisfaction Scores --
    st.markdown("#### 😊 Satisfaction & Work-Life Balance")
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)

    with s_col1:
        env_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], index=2,
                                         format_func=lambda x: {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}[x])
    with s_col2:
        job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], index=2,
                                         format_func=lambda x: {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}[x])
    with s_col3:
        relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4], index=2,
                                                   format_func=lambda x: {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}[x])
    with s_col4:
        work_life_balance = st.selectbox("Work-Life Balance", [1, 2, 3, 4], index=2,
                                          format_func=lambda x: {1: "Bad", 2: "Good", 3: "Better", 4: "Best"}[x])

    st.markdown("---")

    # -- Section 5: Work History --
    st.markdown("#### 📋 Work History & Status")
    w_col1, w_col2, w_col3, w_col4 = st.columns(4)

    with w_col1:
        total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=10)
    with w_col2:
        num_companies = st.number_input("Companies Worked At", min_value=0, max_value=10, value=2)
    with w_col3:
        training_times = st.number_input("Training Times Last Year", min_value=0, max_value=6, value=3)
    with w_col4:
        job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4], index=2,
                                        format_func=lambda x: {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}[x])

    w_col5, w_col6 = st.columns(2)

    with w_col5:
        overtime = st.selectbox("Works Overtime?", ["No", "Yes"])
    with w_col6:
        business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])

    st.markdown("")
    submit = st.form_submit_button("🔮 Predict Attrition Risk", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Prediction Logic
# ---------------------------------------------------------------------------

if submit:
    render_divider()

    with st.spinner("Analyzing employee data..."):
        # Build raw feature dictionary matching the original dataset columns
        raw_input = {
            "Age": age, "BusinessTravel": business_travel, "DailyRate": daily_rate,
            "Department": department, "DistanceFromHome": distance, "Education": education,
            "EducationField": education_field, "EnvironmentSatisfaction": env_satisfaction,
            "Gender": gender, "HourlyRate": hourly_rate, "JobInvolvement": job_involvement,
            "JobLevel": job_level, "JobRole": job_role, "JobSatisfaction": job_satisfaction,
            "MaritalStatus": marital_status, "MonthlyIncome": monthly_income,
            "MonthlyRate": monthly_rate, "NumCompaniesWorked": num_companies,
            "OverTime": overtime, "PercentSalaryHike": percent_salary_hike,
            "PerformanceRating": performance_rating,
            "RelationshipSatisfaction": relationship_satisfaction,
            "StockOptionLevel": stock_option, "TotalWorkingYears": total_working_years,
            "TrainingTimesLastYear": training_times, "WorkLifeBalance": work_life_balance,
            "YearsAtCompany": years_at_company, "YearsInCurrentRole": years_in_role,
            "YearsSinceLastPromotion": years_since_promotion,
            "YearsWithCurrManager": years_with_manager,
        }

        input_df = pd.DataFrame([raw_input])

        # Apply feature engineering
        from utils.feature_engineering import create_engineered_features
        input_engineered = create_engineered_features(input_df)

        # Binary encoding
        binary_maps = {"Gender": {"Male": 1, "Female": 0}, "OverTime": {"Yes": 1, "No": 0}}
        for col, mapping in binary_maps.items():
            if col in input_engineered.columns:
                input_engineered[col] = input_engineered[col].map(mapping)

        # One-hot encoding
        from utils.data_loader import MULTI_CATEGORY_COLUMNS
        existing_multi = [col for col in MULTI_CATEGORY_COLUMNS if col in input_engineered.columns]
        input_encoded = pd.get_dummies(input_engineered, columns=existing_multi, drop_first=True, dtype=int)

        # Align columns with training data
        training_features = prep["feature_names"]
        for col in training_features:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[training_features]

        # Scale features
        scaler = prep["scaler"]
        from utils.data_loader import BINARY_COLUMNS
        numeric_cols = input_encoded.select_dtypes(include=[np.number]).columns.tolist()
        # Scale same columns as training
        input_scaled = input_encoded.copy()
        try:
            cols_to_scale = [c for c in numeric_cols if c not in
                             [col for col in input_encoded.columns if
                              input_encoded[col].nunique() <= 2 and input_encoded[col].max() <= 1]]
            if cols_to_scale:
                input_scaled[cols_to_scale] = scaler.transform(input_encoded[cols_to_scale])
        except Exception:
            # If scaling fails (column mismatch), use unscaled data
            input_scaled = input_encoded

        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else (1.0 if prediction == 1 else 0.0)

        # Risk level
        if probability >= 0.6:
            risk_level = "High Risk"
            risk_color = "#FF6B6B"
            result_class = "prediction-leave"
        elif probability >= 0.3:
            risk_level = "Medium Risk"
            risk_color = "#FFD93D"
            result_class = "prediction-leave"
        else:
            risk_level = "Low Risk"
            risk_color = "#00D4AA"
            result_class = "prediction-stay"

        prediction_text = "Likely to Leave" if prediction == 1 else "Likely to Stay"

    # ---------------------------------------------------------------------------
    # Display Results
    # ---------------------------------------------------------------------------

    st.markdown("### 📊 Prediction Results")

    # Result card
    st.markdown(f"""
    <div class="prediction-result {result_class}">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">
            {'🚪' if prediction == 1 else '✅'}
        </div>
        <div style="font-size: 1.8rem; font-weight: 700; color: {risk_color}; margin-bottom: 0.5rem;">
            {prediction_text}
        </div>
        <div style="font-size: 1rem; color: rgba(250,250,250,0.6); margin-bottom: 1rem;">
            Attrition Probability: <strong style="color: {risk_color};">{probability:.1%}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    col_risk, col_prob, col_model = st.columns(3)

    with col_risk:
        render_risk_badge(risk_level)
    with col_prob:
        st.metric("Probability Score", f"{probability:.1%}")
    with col_model:
        st.metric("Model Used", model_name)

    st.markdown("")

    # Probability gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 40, "color": risk_color}},
        title={"text": "Attrition Probability", "font": {"size": 16, "color": "#FAFAFA"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#FAFAFA"},
            "bar": {"color": risk_color},
            "bgcolor": "#1A1F2E",
            "steps": [
                {"range": [0, 30], "color": "rgba(0, 212, 170, 0.15)"},
                {"range": [30, 60], "color": "rgba(255, 217, 61, 0.15)"},
                {"range": [60, 100], "color": "rgba(255, 107, 107, 0.15)"},
            ],
            "threshold": {
                "line": {"color": "#FAFAFA", "width": 2},
                "thickness": 0.75,
                "value": probability * 100,
            },
        },
    ))

    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=40, r=40, t=60, b=20),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
    )
    st.plotly_chart(fig, use_container_width=True)

    render_divider()

    # SHAP Explanation
    st.markdown("### 🧠 Key Factors Influencing This Prediction")

    try:
        shap_result = compute_shap_values(model, input_scaled, model_name)
        explanation = generate_explanation(shap_result, 0, prediction, probability)

        st.markdown(explanation["summary"])

        st.markdown("")
        st.markdown("#### Top Contributing Factors:")

        for factor in explanation["top_factors"]:
            icon = "🔴" if factor["direction"] == "increases" else "🟢"
            impact_badge = f"**{factor['impact']} Impact**"
            st.markdown(
                f"{icon} **{factor['feature'].replace('_', ' ')}** = {factor['value']:.2f} "
                f"→ {factor['direction']} attrition risk ({impact_badge}, SHAP: {factor['shap_value']:.4f})"
            )

        render_divider()

        # Retention Recommendations
        st.markdown("### 💡 Recommended Retention Actions")

        for action in explanation["retention_actions"]:
            st.markdown(f"- {action}")

    except Exception as e:
        st.warning(f"⚠️ Could not generate SHAP explanation: {str(e)}")
        st.markdown("Basic prediction results are still available above.")

    # Store prediction for explainability page
    st.session_state["last_prediction"] = {
        "input_scaled": input_scaled,
        "prediction": prediction,
        "probability": probability,
        "risk_level": risk_level,
        "raw_input": raw_input,
    }
