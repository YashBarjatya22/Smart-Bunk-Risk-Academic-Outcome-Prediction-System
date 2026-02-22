from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("dataset.csv")
MODEL_DIR = Path("models")
RISK_MODEL_PATH = MODEL_DIR / "risk_model.joblib"
MARKS_MODEL_PATH = MODEL_DIR / "marks_model.joblib"
ARTIFACTS_PATH = MODEL_DIR / "artifacts.joblib"

FEATURE_COLUMNS = [
    "Attendance_Percentage",
    "Internal_Marks",
    "Assignment_Submission_Rate",
    "Study_Hours_Per_Week",
    "Previous_Sem_GPA",
    "Monthly_Bunks",
]

RISK_MAPPING = {
    "Graduate": "Low Risk",
    "Enrolled": "Medium Risk",
    "Dropout": "High Risk",
}

RISK_COLOR = {
    "Low Risk": "#16a34a",
    "Medium Risk": "#f59e0b",
    "High Risk": "#dc2626",
}


st.set_page_config(
    page_title="Smart Bunk Risk & Academic Outcome System",
    page_icon="🎓",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f8fbff 0%, #eef3fb 100%);
    }
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #14b8a6 100%);
        border-radius: 18px;
        padding: 2rem 2.2rem;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.26);
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
    }
    .risk-badge {
        font-weight: 700;
        font-size: 1rem;
        padding: 0.5rem 0.9rem;
        border-radius: 999px;
        color: white;
        display: inline-block;
    }
    .section-title {
        margin-top: 0.2rem;
        margin-bottom: 0.5rem;
        color: #0f172a;
    }
    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


@st.cache_data(show_spinner=False)
def prepare_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    eval_1 = df["Curricular units 1st sem (evaluations)"].replace(0, np.nan)
    eval_2 = df["Curricular units 2nd sem (evaluations)"].replace(0, np.nan)
    approved_total = (
        df["Curricular units 1st sem (approved)"]
        + df["Curricular units 2nd sem (approved)"]
    )
    evaluations_total = eval_1.fillna(0) + eval_2.fillna(0)
    enrolled_total = (
        df["Curricular units 1st sem (enrolled)"]
        + df["Curricular units 2nd sem (enrolled)"]
    ).replace(0, np.nan)

    attendance_ratio = approved_total / evaluations_total.replace(0, np.nan)
    assignment_ratio = approved_total / enrolled_total

    df["Attendance_Percentage"] = (attendance_ratio * 100).replace([np.inf, -np.inf], np.nan)
    df["Attendance_Percentage"] = df["Attendance_Percentage"].fillna(
        df["Attendance_Percentage"].median()
    ).clip(25, 100)

    df["Internal_Marks"] = (
        0.45 * df["Curricular units 1st sem (grade)"]
        + 0.55 * df["Curricular units 2nd sem (grade)"]
    ).fillna(df["Curricular units 1st sem (grade)"].median()).clip(0, 20)

    df["Assignment_Submission_Rate"] = (assignment_ratio * 100).replace(
        [np.inf, -np.inf], np.nan
    )
    df["Assignment_Submission_Rate"] = df["Assignment_Submission_Rate"].fillna(
        df["Assignment_Submission_Rate"].median()
    ).clip(20, 100)

    age = df["Age at enrollment"]
    scholarship = df["Scholarship holder"]
    tuition = df["Tuition fees up to date"]

    study_hours = (
        8
        + (approved_total * 1.2)
        + (scholarship * 3)
        + (tuition * 1.5)
        - np.maximum(age - 18, 0) * 0.15
    )
    df["Study_Hours_Per_Week"] = study_hours.clip(2, 45)

    df["Previous_Sem_GPA"] = ((df["Curricular units 1st sem (grade)"] / 20) * 4).clip(0, 4)

    without_eval = (
        df["Curricular units 1st sem (without evaluations)"]
        + df["Curricular units 2nd sem (without evaluations)"]
    )

    bunk_score = ((100 - df["Attendance_Percentage"]) / 4) + (without_eval * 0.5) + (
        df["Debtor"] * 2
    )
    df["Monthly_Bunks"] = bunk_score.fillna(bunk_score.median()).clip(0, 30)

    df["Risk_Level"] = df["Target"].map(RISK_MAPPING).fillna("Medium Risk")

    df["Final_Marks"] = (
        0.5 * df["Curricular units 1st sem (grade)"]
        + 0.5 * df["Curricular units 2nd sem (grade)"]
    ).clip(0, 20)

    for col in FEATURE_COLUMNS + ["Final_Marks"]:
        df[col] = df[col].fillna(df[col].median())

    return df


@st.cache_resource(show_spinner=False)
def train_or_load_models(df: pd.DataFrame):
    MODEL_DIR.mkdir(exist_ok=True)

    if RISK_MODEL_PATH.exists() and MARKS_MODEL_PATH.exists() and ARTIFACTS_PATH.exists():
        risk_model = joblib.load(RISK_MODEL_PATH)
        marks_model = joblib.load(MARKS_MODEL_PATH)
        artifacts = joblib.load(ARTIFACTS_PATH)
        return risk_model, marks_model, artifacts

    X = df[FEATURE_COLUMNS]
    y_risk = df["Risk_Level"]
    y_marks = df["Final_Marks"]

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X,
        y_risk,
        test_size=0.2,
        random_state=42,
        stratify=y_risk,
    )

    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X,
        y_marks,
        test_size=0.2,
        random_state=42,
    )

    risk_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=2000, random_state=42),
            ),
        ]
    )

    marks_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", LinearRegression()),
        ]
    )

    risk_model.fit(X_train_r, y_train_r)
    marks_model.fit(X_train_m, y_train_m)

    y_pred_r = risk_model.predict(X_test_r)
    y_pred_m = marks_model.predict(X_test_m)

    artifacts = {
        "risk_accuracy": accuracy_score(y_test_r, y_pred_r),
        "marks_r2": r2_score(y_test_m, y_pred_m),
        "marks_mae": mean_absolute_error(y_test_m, y_pred_m),
        "feature_means": X.mean().to_dict(),
        "feature_stds": X.std().replace(0, 1).to_dict(),
    }

    joblib.dump(risk_model, RISK_MODEL_PATH)
    joblib.dump(marks_model, MARKS_MODEL_PATH)
    joblib.dump(artifacts, ARTIFACTS_PATH)

    return risk_model, marks_model, artifacts


def build_sidebar_inputs(df: pd.DataFrame) -> Dict[str, float]:
    st.sidebar.header("🎛️ Student Input Panel")
    st.sidebar.caption("Set student profile values for real-time risk and marks prediction.")

    defaults = df[FEATURE_COLUMNS].median()

    attendance = st.sidebar.slider(
        "Attendance Percentage",
        min_value=0.0,
        max_value=100.0,
        value=float(defaults["Attendance_Percentage"]),
        step=0.5,
    )
    internal_marks = st.sidebar.slider(
        "Internal Marks (out of 20)",
        min_value=0.0,
        max_value=20.0,
        value=float(defaults["Internal_Marks"]),
        step=0.1,
    )
    assignment_rate = st.sidebar.slider(
        "Assignment Submission Rate",
        min_value=0.0,
        max_value=100.0,
        value=float(defaults["Assignment_Submission_Rate"]),
        step=0.5,
    )
    study_hours = st.sidebar.slider(
        "Study Hours Per Week",
        min_value=0.0,
        max_value=60.0,
        value=float(defaults["Study_Hours_Per_Week"]),
        step=0.5,
    )
    prev_gpa = st.sidebar.slider(
        "Previous Sem GPA (out of 4)",
        min_value=0.0,
        max_value=4.0,
        value=float(defaults["Previous_Sem_GPA"]),
        step=0.05,
    )
    monthly_bunks = st.sidebar.slider(
        "Monthly Bunks",
        min_value=0,
        max_value=40,
        value=int(round(defaults["Monthly_Bunks"])),
        step=1,
    )

    input_payload = {
        "Attendance_Percentage": attendance,
        "Internal_Marks": internal_marks,
        "Assignment_Submission_Rate": assignment_rate,
        "Study_Hours_Per_Week": study_hours,
        "Previous_Sem_GPA": prev_gpa,
        "Monthly_Bunks": monthly_bunks,
    }

    st.sidebar.markdown("### Live Input Preview")
    st.sidebar.dataframe(
        pd.DataFrame([input_payload]).T.rename(columns={0: "Value"}),
        use_container_width=True,
        height=250,
    )

    return input_payload


def generate_recommendations(row: pd.Series, risk_level: str, predicted_marks: float) -> List[str]:
    suggestions: List[str] = []

    if row["Attendance_Percentage"] < 75:
        suggestions.append("Increase daily class attendance by 10-15% to reduce risk quickly.")
    if row["Assignment_Submission_Rate"] < 80:
        suggestions.append("Create a weekly assignment tracker and submit before deadlines.")
    if row["Study_Hours_Per_Week"] < 12:
        suggestions.append("Plan at least 2 focused study blocks per day to reach 14+ hrs/week.")
    if row["Previous_Sem_GPA"] < 2.5:
        suggestions.append("Meet a faculty mentor for subject-wise remediation and exam strategy.")
    if row["Monthly_Bunks"] > 8:
        suggestions.append("Limit bunks to under 6 per month to prevent attendance penalties.")
    if row["Internal_Marks"] < 12:
        suggestions.append("Prioritize internals and tutorials to improve marks before final evaluation.")

    if predicted_marks >= 15 and risk_level == "Low Risk":
        suggestions.append("Keep current momentum and target advanced topics for distinction.")
    elif predicted_marks < 10:
        suggestions.append("Start a 4-week recovery plan with daily revision and peer study sessions.")

    if not suggestions:
        suggestions.append("Performance is stable; maintain consistency and monitor weak subjects weekly.")

    return suggestions


def risk_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    weights = {"Low Risk": 20, "Medium Risk": 60, "High Risk": 100}
    score = 0.0
    for p, label in zip(probabilities, labels):
        score += p * weights.get(label, 60)
    return float(np.clip(score, 0, 100))


def render_footer() -> None:
    st.markdown("---")
    st.markdown(
        "<center><b>Smart Bunk Risk & Academic Outcome Prediction System</b> · CHRIST University · Built with Streamlit, Plotly & scikit-learn</center>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1 style="margin:0;">🎓 Smart Bunk Risk & Academic Outcome Prediction System</h1>
            <p style="font-size:1.1rem;margin:0.5rem 0 0.2rem 0;"><b>CHRIST University</b></p>
            <p style="margin:0.2rem 0 0 0;opacity:0.92;">
            Real-time academic risk intelligence that combines attendance behavior, study patterns,
            and semester history to forecast risk level and expected final marks.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not DATA_PATH.exists():
        st.error("dataset.csv was not found in the app root directory.")
        return

    raw_df = load_data(DATA_PATH)
    df = prepare_dataset(raw_df)
    risk_model, marks_model, artifacts = train_or_load_models(df)

    user_input = build_sidebar_inputs(df)

    input_df = pd.DataFrame([user_input], columns=FEATURE_COLUMNS)

    st.subheader("Prediction Control")
    predict_clicked = st.button("🔮 Predict Academic Risk & Outcome", type="primary", use_container_width=True)

    if predict_clicked:
        proba = risk_model.predict_proba(input_df)[0]
        pred_risk = risk_model.predict(input_df)[0]
        pred_marks = float(marks_model.predict(input_df)[0])
        pred_marks = float(np.clip(pred_marks, 0, 20))

        st.session_state["prediction"] = {
            "risk": pred_risk,
            "marks": pred_marks,
            "proba": proba,
            "labels": risk_model.classes_,
            "risk_score": risk_score(proba, risk_model.classes_),
        }

    if "prediction" in st.session_state:
        prediction = st.session_state["prediction"]
        pred_risk = prediction["risk"]
        pred_marks = prediction["marks"]
        proba = prediction["proba"]
        labels = prediction["labels"]

        st.markdown("### Results Dashboard")
        col_a, col_b, col_c = st.columns([1.2, 1, 1])

        with col_a:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("#### Risk Level")
            st.markdown(
                f'<span class="risk-badge" style="background:{RISK_COLOR[pred_risk]};">{pred_risk}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_b:
            st.metric("Predicted Final Marks", f"{pred_marks:.2f} / 20")

        with col_c:
            st.metric("Risk Score", f"{prediction['risk_score']:.1f} / 100")

        gauge_value = float(proba[list(labels).index(pred_risk)] * 100)
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=gauge_value,
                title={"text": f"Probability: {pred_risk}"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": RISK_COLOR[pred_risk]},
                    "steps": [
                        {"range": [0, 35], "color": "#dcfce7"},
                        {"range": [35, 70], "color": "#fef3c7"},
                        {"range": [70, 100], "color": "#fee2e2"},
                    ],
                },
            )
        )
        gauge.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))

        radar_features = FEATURE_COLUMNS
        radar_values = input_df.iloc[0].values.tolist()
        radar_values.append(radar_values[0])
        radar_theta = radar_features + [radar_features[0]]

        radar = go.Figure()
        radar.add_trace(
            go.Scatterpolar(
                r=radar_values,
                theta=radar_theta,
                fill="toself",
                name="Student Profile",
                line_color="#2563eb",
            )
        )
        radar.update_layout(height=360, polar=dict(radialaxis=dict(visible=True)))

        scaler = risk_model.named_steps["scaler"]
        clf = risk_model.named_steps["classifier"]
        transformed = scaler.transform(input_df)[0]
        class_index = list(clf.classes_).index(pred_risk)
        coeff = clf.coef_[class_index]
        contrib = pd.DataFrame(
            {
                "Feature": FEATURE_COLUMNS,
                "Contribution": transformed * coeff,
            }
        ).sort_values("Contribution", ascending=False)

        bars = px.bar(
            contrib,
            x="Feature",
            y="Contribution",
            color="Contribution",
            color_continuous_scale="Tealrose",
            title="Factor Contribution for Predicted Risk",
        )
        bars.update_layout(height=360)

        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.plotly_chart(gauge, use_container_width=True)
        with row1_col2:
            st.plotly_chart(radar, use_container_width=True)

        st.plotly_chart(bars, use_container_width=True)

        st.markdown("### Smart Recommendations")
        recs = generate_recommendations(input_df.iloc[0], pred_risk, pred_marks)
        for rec in recs:
            st.markdown(f"- {rec}")

        report_df = pd.DataFrame(
            [
                {
                    **user_input,
                    "Predicted_Risk_Level": pred_risk,
                    "Predicted_Final_Marks": round(pred_marks, 2),
                    "Risk_Score": round(prediction["risk_score"], 2),
                }
            ]
        )

        st.download_button(
            "⬇️ Download Prediction Report (CSV)",
            data=report_df.to_csv(index=False).encode("utf-8"),
            file_name="student_prediction_report.csv",
            mime="text/csv",
        )

    st.markdown("### Data Analytics")

    analytics_col1, analytics_col2 = st.columns(2)

    with analytics_col1:
        attendance_hist = px.histogram(
            df,
            x="Attendance_Percentage",
            nbins=30,
            title="Attendance Distribution",
            color_discrete_sequence=["#1d4ed8"],
        )
        st.plotly_chart(attendance_hist, use_container_width=True)

    with analytics_col2:
        scatter = px.scatter(
            df,
            x="Previous_Sem_GPA",
            y="Final_Marks",
            color="Risk_Level",
            title="GPA vs Final Marks",
            color_discrete_map={
                "Low Risk": "#16a34a",
                "Medium Risk": "#f59e0b",
                "High Risk": "#dc2626",
            },
            opacity=0.75,
        )
        st.plotly_chart(scatter, use_container_width=True)

    corr_cols = FEATURE_COLUMNS + ["Final_Marks"]
    corr = df[corr_cols].corr(numeric_only=True)
    heat = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Correlation Heatmap",
    )
    st.plotly_chart(heat, use_container_width=True)

    clf = risk_model.named_steps["classifier"]
    risk_importance = np.mean(np.abs(clf.coef_), axis=0)
    reg_coef = np.abs(marks_model.named_steps["regressor"].coef_)

    combined_importance = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Risk_Model_Importance": risk_importance,
            "Marks_Model_Importance": reg_coef,
        }
    )

    importance_fig = px.bar(
        combined_importance.melt(id_vars="Feature", var_name="Model", value_name="Importance"),
        x="Feature",
        y="Importance",
        color="Model",
        barmode="group",
        title="Feature Importance Across Models",
        color_discrete_sequence=["#2563eb", "#14b8a6"],
    )
    st.plotly_chart(importance_fig, use_container_width=True)

    st.markdown("### Dataset Statistics & Model Performance")
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.metric("Rows", f"{len(df):,}")
        st.metric("Columns", f"{len(df.columns):,}")
    with stat_col2:
        st.metric("Risk Classification Accuracy", f"{artifacts['risk_accuracy'] * 100:.2f}%")
        st.metric("Final Marks R²", f"{artifacts['marks_r2']:.3f}")
    with stat_col3:
        st.metric("Final Marks MAE", f"{artifacts['marks_mae']:.3f}")
        st.metric("Avg Attendance", f"{df['Attendance_Percentage'].mean():.1f}%")

    with st.expander("View Engineered Dataset Snapshot"):
        st.dataframe(df[FEATURE_COLUMNS + ["Risk_Level", "Final_Marks"]].head(20), use_container_width=True)

    render_footer()


if __name__ == "__main__":
    main()
