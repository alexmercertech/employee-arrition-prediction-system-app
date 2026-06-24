# Employee Attrition Prediction System

An end-to-end Machine Learning Analytics Platform built with Python and Streamlit that predicts employee attrition using the IBM HR Analytics dataset.

## 🎯 Overview

This application demonstrates the complete machine learning lifecycle:
1. **Problem Definition** — Predicting employee attrition for proactive HR intervention
2. **Data Pipeline** — Cleaning, encoding, scaling, and SMOTE balancing
3. **Model Implementation** — 4 classifiers with hyperparameter tuning
4. **Experimental Evaluation** — Comprehensive metrics, curves, and model comparison
5. **Explainable AI** — SHAP-based global and local interpretability

## 🚀 Features

| Page | Description |
|------|-------------|
| **Dashboard** | KPI cards, attrition analytics, and department-level insights |
| **Data Exploration** | Interactive EDA with dynamic filters and correlation analysis |
| **Data Preprocessing** | Step-by-step pipeline walkthrough with before/after views |
| **Model Training** | UI-based training with hyperparameter tuning and SMOTE |
| **Model Comparison** | Metrics tables, ROC/PR curves, and best model recommendation |
| **Employee Prediction** | HR prediction form with risk levels and retention actions |
| **Explainability** | SHAP summary, waterfall, and dependence plots |

## 📁 Project Structure

```
employee_attrition_app/
├── app.py                          # Main entry point (Home page)
├── requirements.txt                # Dependencies
├── train_model.py                  # CLI training script
├── README.md                       # This file
├── .streamlit/
│   └── config.toml                 # Theme configuration
├── utils/
│   ├── __init__.py                 # Package init
│   ├── data_loader.py              # Data loading & validation
│   ├── preprocessing.py            # Cleaning, encoding, scaling, SMOTE
│   ├── feature_engineering.py      # 8 derived features
│   ├── model_trainer.py            # Training pipelines & persistence
│   ├── evaluation.py               # Metrics & Plotly visualizations
│   ├── explainability.py           # SHAP wrapper functions
│   └── ui_components.py            # Reusable UI elements & CSS
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Data_Exploration.py
│   ├── 3_Data_Preprocessing.py
│   ├── 4_Model_Training.py
│   ├── 5_Model_Comparison.py
│   ├── 6_Employee_Prediction.py
│   └── 7_Model_Explainability.py
├── models/                         # Saved .joblib models
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
└── assets/
```

## 🛠️ Installation

### Local Setup

```bash
# Clone the repository
git clone <repository-url>
cd employee-arrition-prediction-system-app

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
# Option 1: Start the Streamlit app directly
streamlit run app.py

# Option 2: Train models first via CLI, then start the app
python train_model.py
streamlit run app.py
```

The application will open at `http://localhost:8501`.

## ☁️ Deployment to Streamlit Community Cloud

1. **Push to GitHub**: Ensure all files (including `data/WA_Fn-UseC_-HR-Employee-Attrition.csv`) are committed.

2. **Go to** [share.streamlit.io](https://share.streamlit.io)

3. **Deploy**:
   - Repository: `your-username/your-repo-name`
   - Branch: `main`
   - Main file path: `app.py`

4. **Click Deploy** — Streamlit Community Cloud will install dependencies from `requirements.txt` automatically.

> **Note**: Pre-train models locally by running `python train_model.py` before deploying, or train via the UI after deployment.

## 🔬 Machine Learning Methodology

### Dataset
- **Source**: IBM HR Analytics Employee Attrition & Performance (Kaggle)
- **Size**: 1,470 employees × 35 features
- **Target**: Attrition (Yes/No) — Binary classification
- **Class Imbalance**: ~16% attrition rate (237 Yes / 1,233 No)

### Preprocessing Pipeline
1. Drop constant columns (EmployeeCount, Over18, StandardHours, EmployeeNumber)
2. Handle duplicates
3. Feature engineering (8 derived features)
4. Label encoding (binary features)
5. One-hot encoding (multi-category features)
6. StandardScaler normalization
7. SMOTE class balancing (training data only)

### Models
| Model | Type | Key Hyperparameters |
|-------|------|-------------------|
| Logistic Regression | Linear | C, penalty, solver |
| Decision Tree | Tree | max_depth, min_samples_split, criterion |
| Random Forest | Ensemble | n_estimators, max_depth, max_features |
| XGBoost | Boosting | learning_rate, n_estimators, max_depth |

### Evaluation Metrics
- **Primary**: F1 Score (appropriate for imbalanced data)
- **Secondary**: Accuracy, Precision, Recall, ROC-AUC
- **Validation**: 5-Fold Stratified Cross-Validation

### Feature Engineering
| Feature | Formula | Type |
|---------|---------|------|
| YearsPerCompany | TotalWorkingYears / (NumCompaniesWorked + 1) | Loyalty |
| IncomePerYearWorked | MonthlyIncome / (TotalWorkingYears + 1) | Compensation |
| SatisfactionIndex | Mean of 4 satisfaction scores | Well-being |
| IsNewEmployee | 1 if YearsAtCompany ≤ 1 | Tenure |
| PromotionStagnation | YearsSinceLastPromotion - YearsInCurrentRole | Growth |
| OvertimeDistance | DistanceFromHome × OverTime | Burnout |
| TenureRatio | YearsAtCompany / (TotalWorkingYears + 1) | Career |
| ManagerTenureRatio | YearsWithCurrManager / (YearsAtCompany + 1) | Management |

## 📊 Technology Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.12+ | Core language |
| Streamlit | ≥1.28.0 | Web application framework |
| Pandas | ≥2.0.0 | Data manipulation |
| NumPy | ≥1.24.0 | Numerical computing |
| Scikit-Learn | ≥1.3.0 | ML models & preprocessing |
| XGBoost | ≥2.0.0 | Gradient boosting |
| SHAP | ≥0.42.0 | Model explainability |
| Plotly | ≥5.18.0 | Interactive visualizations |
| Imbalanced-Learn | ≥0.11.0 | SMOTE oversampling |
| Seaborn | ≥0.13.0 | Statistical visualizations |
| Matplotlib | ≥3.8.0 | SHAP plot rendering |
| Joblib | ≥1.3.0 | Model serialization |

## 📋 Assessment Criteria Mapping

| Criterion | Where Demonstrated |
|-----------|-------------------|
| Problem Definition | Home page (app.py), README |
| Data Pipeline | Pages 2-3, utils/preprocessing.py, utils/feature_engineering.py |
| Model Implementation | Page 4, utils/model_trainer.py |
| Debugging & Iteration | Page 4 (overfitting analysis), Page 5 (comparison) |
| Evaluation | Page 5, utils/evaluation.py |
| Communication | All pages (business insights), Page 7 (SHAP explanations) |

## 📜 License

This project uses the IBM HR Analytics Employee Attrition dataset, a fictional dataset created by IBM data scientists for educational purposes.

## 🙏 Credits

- **Dataset**: IBM HR Analytics Employee Attrition & Performance — [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Framework**: [Streamlit](https://streamlit.io/)
- **Explainability**: [SHAP](https://shap.readthedocs.io/)
