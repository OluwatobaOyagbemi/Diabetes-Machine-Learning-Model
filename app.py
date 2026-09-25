from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from utils import (
    BINARY_STANDARDS,
    CLINICAL_THRESHOLD,
    DATASET_INFO,
    DIABETES_OVERVIEW,
    FEATURE_COLUMNS,
    FEATURE_EXPLANATIONS,
    GENDER_SOURCE_URL,
    MODEL_CARD,
    SMOKING_RISK,
    SMOKING_SOURCE_URL,
    TARGET_COLUMN,
    WHO_STANDARDS,
    AI_ETHICS_SOURCE_URL,
    classify_value,
    cross_validated_metrics,
    display_name,
    get_background_sample,
    global_feature_importance,
    load_or_train_model,
    local_feature_contribution,
    make_test_predictions,
    threshold_metrics,
)


BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "diabetes_prediction_dataset.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
st.set_page_config(page_title="Diabetes Screening Dashboard", page_icon="+", layout="wide")

# Clinically aesthetic palette: white/off-white surfaces, one primary clinical blue/teal,
# muted gray text, and amber/red reserved strictly for caution / high-risk flags.
PRIMARY_BLUE = "#0B5FA5"
TEAL = "#0C8599"
TEXT_GRAY = "#5f6b7a"
SAFE_GREEN = "#2e8b57"
CAUTION_AMBER = "#d98c10"
ALERT_RED = "#c53b32"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@500;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: #2c3e50; }}
    h1, h2, h3 {{ font-family: 'Poppins', sans-serif; color: {PRIMARY_BLUE}; font-weight: 600; letter-spacing: 0.01em; }}
    p, li, span, label, div[data-testid="stMarkdownContainer"] {{ color: #33414f; }}
    .stCaption, [data-testid="stCaptionContainer"] p {{ color: {TEXT_GRAY} !important; }}

    /* Clinical background: soft moving gradient plus a subtle glucose-droplet motif */
    [data-testid="stAppViewContainer"] {{
        background-image:
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120' viewBox='0 0 120 120'%3E%3Cg fill='%230C8599' fill-opacity='0.05'%3E%3Cpath d='M30 8c0 0-13 16-13 26a13 13 0 0026 0C43 24 30 8 30 8z'/%3E%3Cpath d='M92 54c0 0-11 13-11 22a11 11 0 0022 0c0-9-11-22-11-22z'/%3E%3Cpath d='M58 90c0 0-9 11-9 18a9 9 0 0018 0c0-7-9-18-9-18z'/%3E%3C/g%3E%3C/svg%3E"),
            linear-gradient(160deg, #eaf3fb 0%, #ffffff 32%, #ffffff 68%, #eef8f6 100%);
        background-attachment: fixed, fixed;
        background-size: 120px 120px, 200% 200%;
        animation: driftGradient 24s ease-in-out infinite;
    }}
    @keyframes driftGradient {{
        0% {{ background-position: 0 0, 0% 0%; }}
        50% {{ background-position: 0 0, 100% 60%; }}
        100% {{ background-position: 0 0, 0% 0%; }}
    }}

    /* Advanced motion graphics: a scrolling ECG/glucose-monitor pulse line and drifting droplets */
    .pulse-strip {{
        position: fixed; top: 0; left: 0; width: 100%; height: 34px; z-index: 999;
        pointer-events: none; opacity: 0.55;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='34' viewBox='0 0 300 34'%3E%3Cpolyline points='0,17 60,17 72,4 84,30 96,17 150,17 162,4 174,30 186,17 300,17' fill='none' stroke='%230C8599' stroke-width='2'/%3E%3C/svg%3E");
        background-repeat: repeat-x;
        background-size: 300px 34px;
        animation: scrollPulse 6s linear infinite;
    }}
    @keyframes scrollPulse {{
        from {{ background-position-x: 0; }}
        to {{ background-position-x: -300px; }}
    }}
    .floating-drops {{
        position: fixed; inset: 0; z-index: -1; pointer-events: none; overflow: hidden;
    }}
    .floating-drops span {{
        position: absolute; bottom: -40px; display: block; border-radius: 50% 50% 50% 0;
        transform: rotate(45deg); background: rgba(11, 95, 165, 0.10);
        animation: floatUp linear infinite;
    }}
    .floating-drops span:nth-child(1) {{ left: 6%;  width: 14px; height: 14px; animation-duration: 14s; animation-delay: 0s; }}
    .floating-drops span:nth-child(2) {{ left: 22%; width: 10px; height: 10px; animation-duration: 18s; animation-delay: 2s; }}
    .floating-drops span:nth-child(3) {{ left: 40%; width: 18px; height: 18px; animation-duration: 16s; animation-delay: 4s; }}
    .floating-drops span:nth-child(4) {{ left: 58%; width: 12px; height: 12px; animation-duration: 20s; animation-delay: 1s; }}
    .floating-drops span:nth-child(5) {{ left: 74%; width: 16px; height: 16px; animation-duration: 15s; animation-delay: 3s; }}
    .floating-drops span:nth-child(6) {{ left: 90%; width: 11px; height: 11px; animation-duration: 19s; animation-delay: 5s; }}
    @keyframes floatUp {{
        0% {{ transform: translateY(0) rotate(45deg); opacity: 0; }}
        10% {{ opacity: 1; }}
        90% {{ opacity: 1; }}
        100% {{ transform: translateY(-115vh) rotate(45deg); opacity: 0; }}
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0B5FA5 0%, #0C8599 100%);
    }}
    [data-testid="stSidebar"] * {{ color: #eaf4fb !important; }}
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{ color: #cfe6f5 !important; }}

    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(14px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes pulseGlow {{
        0% {{ box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.55); }}
        70% {{ box-shadow: 0 0 0 10px rgba(255, 255, 255, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(255, 255, 255, 0); }}
    }}
    [data-testid="stAppViewContainer"] h1 {{
        animation: fadeInUp 0.5s ease-out;
    }}
    [data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]) {{
        animation: fadeInUp 0.45s ease-out;
    }}
    [data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]):nth-of-type(2) {{ animation-delay: 0.05s; }}
    [data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]):nth-of-type(3) {{ animation-delay: 0.1s; }}
    [data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]):nth-of-type(4) {{ animation-delay: 0.15s; }}
    [data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]):nth-of-type(5) {{ animation-delay: 0.2s; }}

    [data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"] {{
        gap: 0.6rem;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]) {{
        border-radius: 12px !important;
        background: rgba(255, 255, 255, 0.88);
        box-shadow: 0 1px 3px rgba(15, 45, 75, 0.08);
        margin-bottom: 1.1rem;
        transition: box-shadow 0.25s ease, transform 0.25s ease;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]):hover {{
        box-shadow: 0 8px 22px rgba(15, 45, 75, 0.16);
        transform: translateY(-3px);
    }}
    .stPlotlyChart, [data-testid="stImage"] {{ animation: fadeInUp 0.6s ease-out; }}
    div[data-testid="stMetric"] {{ transition: transform 0.2s ease; }}
    div[data-testid="stMetric"]:hover {{ transform: scale(1.03); }}
    .risk-badge {{
        display: inline-block; padding: 5px 16px; border-radius: 999px;
        color: white; font-weight: 600; font-size: 1rem; letter-spacing: 0.02em;
        animation: pulseGlow 2.2s infinite;
    }}
    .clinical-note {{
        border-left: 4px solid {TEAL}; padding: 10px 16px; background: #f4faf9;
        border-radius: 6px; margin-bottom: 10px; color: {TEXT_GRAY};
        animation: fadeInUp 0.4s ease-out;
    }}
    .result-headline {{
        font-family: 'Poppins', sans-serif;
        font-size: 2.6rem; font-weight: 700; color: {PRIMARY_BLUE}; line-height: 1.1;
        animation: fadeInUp 0.5s ease-out;
    }}
    .result-caption {{
        color: {TEXT_GRAY}; font-size: 0.95rem; margin-top: 4px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    return data.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()


@st.cache_resource
def get_model(model_path: str, data: pd.DataFrame):
    return load_or_train_model(model_path, data)


@st.cache_data
def get_evaluation_data(data: pd.DataFrame):
    return make_test_predictions(data)


@st.cache_data
def get_cv_metrics(data: pd.DataFrame):
    return cross_validated_metrics(data)


@st.cache_data
def get_background_sample_cached(data: pd.DataFrame):
    return get_background_sample(data)


@st.cache_data
def get_global_importance(_model, background: pd.DataFrame):
    return global_feature_importance(_model, background)


def risk_tier(score: float) -> tuple[str, str]:
    if score < 0.15:
        return "Low", SAFE_GREEN
    if score < 0.35:
        return "Moderate", CAUTION_AMBER
    return "High", ALERT_RED


def who_gauge(feature: str, value: float) -> go.Figure:
    spec = WHO_STANDARDS[feature]
    low, high = spec["range"]
    label, color = classify_value(feature, value)
    steps = [
        {"range": [max(band_low, low), min(band_high, high)], "color": band_color}
        for band_low, band_high, _, band_color in spec["bands"]
    ]
    figure = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": f" {spec['unit']}"},
        title={"text": f"{spec['label']}<br><span style='font-size:0.8em;color:{color}'>{label}</span>"},
        gauge={
            "axis": {"range": [low, high]},
            "bar": {"color": "#1f2933"},
            "steps": steps,
            "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.85, "value": value},
        },
    ))
    figure.update_layout(margin=dict(l=20, r=20, t=70, b=10), height=260)
    return figure


def render_dashboard(data: pd.DataFrame) -> None:
    st.title("WHO Standards Feature Dashboard")
    st.caption("Every input is compared against WHO / ADA / IDF reference ranges so each value carries clinical context, not just a raw number.")
    patient = patient_form(data)
    if patient.empty:
        st.info("Submit the form above to see your values plotted against WHO reference standards.")
        return

    row = patient.iloc[0]

    with st.container(border=True):
        st.subheader("Continuous clinical measures vs. WHO reference bands")
        gauge_cols = st.columns(4)
        for column, feature in zip(gauge_cols, ["bmi", "HbA1c_level", "blood_glucose_level", "age"]):
            with column:
                st.plotly_chart(who_gauge(feature, float(row[feature])), use_container_width=True)
                spec = WHO_STANDARDS[feature]
                with st.expander(f"About {spec['label']}"):
                    st.write(spec["explanation"])
                    st.caption(f"Reference: [{spec['source']}]({spec['source_url']})")

    with st.container(border=True):
        st.subheader("Risk-factor flags")
        flag_cols = st.columns(3)
        with flag_cols[0]:
            is_hyper = row["hypertension"] == 1
            st.metric("Hypertension", "Present" if is_hyper else "Not present")
            st.markdown(f"<div style='height:10px;border-radius:4px;background:{ALERT_RED if is_hyper else SAFE_GREEN}'></div>", unsafe_allow_html=True)
            with st.expander("About hypertension"):
                st.write(BINARY_STANDARDS["hypertension"]["explanation"])
                st.caption(f"Reference: [{BINARY_STANDARDS['hypertension']['source']}]({BINARY_STANDARDS['hypertension']['source_url']})")
        with flag_cols[1]:
            is_heart = row["heart_disease"] == 1
            st.metric("Heart disease", "Present" if is_heart else "Not present")
            st.markdown(f"<div style='height:10px;border-radius:4px;background:{ALERT_RED if is_heart else SAFE_GREEN}'></div>", unsafe_allow_html=True)
            with st.expander("About heart disease"):
                st.write(BINARY_STANDARDS["heart_disease"]["explanation"])
                st.caption(f"Reference: [{BINARY_STANDARDS['heart_disease']['source']}]({BINARY_STANDARDS['heart_disease']['source_url']})")
        with flag_cols[2]:
            smoking = row["smoking_history"]
            smoke_label, smoke_color = SMOKING_RISK.get(smoking, ("Unknown", "#9aa0a6"))
            st.metric("Smoking history", smoking)
            st.markdown(f"<div style='height:10px;border-radius:4px;background:{smoke_color}'></div>", unsafe_allow_html=True)
            st.caption(smoke_label)
            with st.expander("About smoking history"):
                st.write("WHO identifies tobacco use as a major modifiable risk factor for cardiovascular disease and a contributor to insulin resistance.")
                st.caption(f"Reference: [WHO Tobacco Fact Sheet]({SMOKING_SOURCE_URL})")

    with st.container(border=True):
        st.subheader("Demographic context")
        demo_cols = st.columns(2)
        with demo_cols[0]:
            st.metric("Gender", row["gender"])
            st.caption(f"Dataset field: gender. Diabetes prevalence and presentation can differ by sex, per [WHO gender and health reporting]({GENDER_SOURCE_URL}).")
        with demo_cols[1]:
            age_label, age_color = classify_value("age", float(row["age"]))
            st.metric("Age group", age_label)
            st.markdown(f"<div style='height:10px;border-radius:4px;background:{age_color}'></div>", unsafe_allow_html=True)


def patient_form(data: pd.DataFrame) -> pd.DataFrame:
    with st.form("patient_risk_form"):
        first, second, third = st.columns(3)
        with first:
            gender = st.selectbox("Gender", sorted(data["gender"].unique()), help="Dataset field: gender.")
            age = st.number_input("Age (years)", 18, 100, 45, 1, help="Dataset field: age. The source data includes children, but this screener is configured for adults.")
            bmi = st.number_input("BMI (kg/m2)", 12.0, 70.0, 27.0, 0.1, help="Dataset field: bmi. Weight in kilograms divided by height in meters squared.")
        with second:
            hypertension = st.selectbox("Hypertension diagnosis", ["No", "Yes"], help="Dataset field: hypertension. A known diagnosis, not a blood-pressure measurement.")
            heart_disease = st.selectbox("Heart disease diagnosis", ["No", "Yes"], help="Dataset field: heart_disease. A known diagnosis of heart disease.")
            smoking = st.selectbox("Smoking history", sorted(data["smoking_history"].unique()), help="Dataset field: smoking_history. 'No Info' means the source dataset has no recorded history.")
        with third:
            hba1c = st.number_input("HbA1c (%)", 3.0, 14.0, 5.6, 0.1, help="Dataset field: HbA1c_level. A laboratory result, not a home measurement.")
            glucose = st.number_input("Blood glucose (mg/dL)", 50, 400, 120, 1, help="Dataset field: blood_glucose_level. The dataset does not specify fasting status.")
            submitted = st.form_submit_button("Estimate screening risk", type="primary")
    if submitted:
        st.session_state["patient_input"] = {
            "gender": gender,
            "age": age,
            "hypertension": int(hypertension == "Yes"),
            "heart_disease": int(heart_disease == "Yes"),
            "smoking_history": smoking,
            "bmi": bmi,
            "HbA1c_level": hba1c,
            "blood_glucose_level": glucose,
        }
    if "patient_input" not in st.session_state:
        return pd.DataFrame()
    return pd.DataFrame([st.session_state["patient_input"]])


def render_screener(data: pd.DataFrame, model) -> None:
    st.title("Patient Risk Screener")
    st.markdown(
        "<div class='clinical-note'><strong>Screening tool only - not a diagnosis.</strong> "
        "Abnormal results require confirmatory lab testing (HbA1c, FPG, or OGTT).</div>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.subheader("Patient inputs")
        patient = patient_form(data)

    if patient.empty:
        return

    score = float(model.predict_proba(patient)[:, 1][0])
    tier, color = risk_tier(score)

    with st.container(border=True):
        st.subheader("Screening result")
        left, right = st.columns([1, 1.4])
        with left:
            st.markdown(f"<div class='result-headline'>{score:.1%}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-caption'>estimated diabetes risk</div>", unsafe_allow_html=True)
            st.markdown(f"<span class='risk-badge' style='background:{color}'>{tier} risk</span>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='background:{color};height:20px;border-radius:4px;width:{score * 100:.1f}%;min-width:2px;margin-top:12px'></div>",
                unsafe_allow_html=True,
            )
            if score >= CLINICAL_THRESHOLD:
                st.error(f"**Screen-positive** (at or above the {CLINICAL_THRESHOLD:.0%} clinical threshold). Recommend confirmatory HbA1c, fasting plasma glucose, or OGTT testing.")
            else:
                st.success(f"**Screen-negative** (below the {CLINICAL_THRESHOLD:.0%} clinical threshold). Reassess screening in about 3 years, per clinician guidance.")
            st.caption(f"The {CLINICAL_THRESHOLD:.0%} threshold is set low on purpose, to catch about 90% of true diabetes cases (sensitivity) even though it means more people get sent for an unnecessary confirmatory test.")
            st.warning("This estimate is not a diagnostic substitute. It supports the decision to refer for confirmatory testing - it does not replace clinical judgment.")
        with right:
            background = get_background_sample_cached(data)
            local = local_feature_contribution(model, patient, background)
            view = st.radio("Explanation view", ["Plain-language", "Technical (feature importance)"], horizontal=True, key="screener_view")
            if view == "Technical (feature importance)":
                ordered = local.sort_values("contribution")
                figure = go.Figure(go.Bar(
                    x=ordered["contribution"], y=[display_name(feature) for feature in ordered["feature"]], orientation="h",
                    marker_color=[ALERT_RED if value > 0 else SAFE_GREEN for value in ordered["contribution"]],
                ))
                figure.update_layout(
                    title="Feature contribution to this patient's risk score",
                    xaxis_title="Contribution (impact on model output)",
                    yaxis_title=None,
                    margin=dict(l=10, r=10, t=45, b=10),
                )
                st.plotly_chart(figure, use_container_width=True)
                st.caption("Positive values push risk higher; negative values push risk lower, relative to a background sample of the dataset.")
            else:
                st.caption("Top factors behind this patient's estimate, ranked by impact:")
                for _, item in local.head(6).iterrows():
                    direction = "increases" if item["contribution"] > 0 else "decreases"
                    explanation = FEATURE_EXPLANATIONS.get(item["feature"], "")
                    st.markdown(f"- **{display_name(item['feature'])}** {direction} this patient's estimated risk. {explanation}")


def render_transparency(data: pd.DataFrame) -> None:
    st.title("Model & Data Transparency")
    st.caption(f"Model Card, dataset provenance, and performance evidence, presented in the spirit of [WHO's 2021 guidance on ethics and governance of AI for health]({AI_ETHICS_SOURCE_URL}).")

    with st.container(border=True):
        st.subheader("Model Card")
        st.markdown(f"**Intended use.** {MODEL_CARD['intended_use']}")
        st.markdown(f"**Population trained on.** {MODEL_CARD['population']}")
        col_limits, col_scope = st.columns(2)
        with col_limits:
            st.markdown("**Known limitations**")
            for item in MODEL_CARD["limitations"]:
                st.markdown(f"- {item}")
        with col_scope:
            st.markdown("**Out-of-scope use**")
            for item in MODEL_CARD["out_of_scope"]:
                st.markdown(f"- {item}")

    with st.container(border=True):
        st.subheader("Dataset provenance")
        total = len(data)
        positive = int(data[TARGET_COLUMN].sum())
        negative = total - positive
        gender_counts = data["gender"].value_counts()
        stat_cols = st.columns(4)
        stat_cols[0].metric("Source", DATASET_INFO["name"])
        stat_cols[1].metric("Sample size", f"{total:,}")
        stat_cols[2].metric("Diabetes prevalence", f"{positive / total:.1%}")
        stat_cols[3].metric("Class balance", f"{negative:,} / {positive:,}")
        st.caption(DATASET_INFO["provenance"])
        st.markdown("**Gender breakdown:** " + ", ".join(f"{gender}: {count:,}" for gender, count in gender_counts.items()))
        st.markdown("**Known dataset limitations:**")
        for item in MODEL_CARD["limitations"]:
            st.markdown(f"- {item}")

    with st.container(border=True):
        st.subheader("Model performance")
        x_test, y_test, probabilities, evaluation_model = get_evaluation_data(data)
        threshold = st.slider("Classification threshold", 0.05, 0.95, CLINICAL_THRESHOLD, 0.01, help="The notebook recommends 0.10 for clinical screening, targeting about 90% sensitivity while accepting more confirmatory tests.")
        metrics = threshold_metrics(y_test, probabilities, threshold)
        cv_metrics = get_cv_metrics(data)
        cv_auc_mean, cv_auc_std = cv_metrics["roc_auc"]

        metrics_table = pd.DataFrame({
            "Metric": ["Sensitivity", "Specificity", "PPV", "NPV", "AUC"],
            "This evaluation": [metrics["sensitivity"], metrics["specificity"], metrics["ppv"], metrics["npv"], metrics["auc"]],
            "Published ADA risk-calculator reference": ["Not specified", "Not specified", "~10%", "96-99%", "0.78-0.83"],
        })
        st.dataframe(metrics_table.style.format({"This evaluation": "{:.1%}"}), use_container_width=True, hide_index=True)
        st.caption(f"5-fold cross-validated ROC-AUC: {cv_auc_mean:.3f} +/- {cv_auc_std:.3f} (mean +/- std across folds, default 0.5 threshold).")

        charts_left, charts_right = st.columns(2)
        matrix = metrics["confusion_matrix"]
        with charts_left:
            matrix_figure = px.imshow(matrix, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual", color="Count"), x=["No diabetes", "Diabetes"], y=["No diabetes", "Diabetes"])
            matrix_figure.update_layout(title=f"Confusion matrix at threshold {threshold:.2f}")
            st.plotly_chart(matrix_figure, use_container_width=True)
        with charts_right:
            fpr, tpr, _ = roc_curve(y_test, probabilities)
            roc_figure = go.Figure()
            roc_figure.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Model (AUC {metrics['auc']:.3f})", line=dict(color=PRIMARY_BLUE)))
            roc_figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="gray")))
            roc_figure.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
            st.plotly_chart(roc_figure, use_container_width=True)

        pr_precision, pr_recall, _ = precision_recall_curve(y_test, probabilities)
        calibration_observed, calibration_predicted = calibration_curve(y_test, probabilities, n_bins=10)
        lower, upper = st.columns(2)
        with lower:
            pr_figure = go.Figure(go.Scatter(x=pr_recall, y=pr_precision, mode="lines", name=f"AP {metrics['average_precision']:.3f}", line=dict(color=TEAL)))
            pr_figure.update_layout(title="Precision-recall curve", xaxis_title="Recall", yaxis_title="Precision")
            st.plotly_chart(pr_figure, use_container_width=True)
        with upper:
            calibration_figure = go.Figure()
            calibration_figure.add_trace(go.Scatter(x=calibration_predicted, y=calibration_observed, mode="lines+markers", name="Model", line=dict(color=PRIMARY_BLUE)))
            calibration_figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line=dict(dash="dash", color="gray")))
            calibration_figure.update_layout(title="Calibration curve", xaxis_title="Predicted probability", yaxis_title="Observed frequency")
            st.plotly_chart(calibration_figure, use_container_width=True)
        st.caption("Evaluated on a deterministic 80/20 stratified split of the raw-data Gradient Boosting model, matching the notebook's selected deployment approach.")

    with st.container(border=True):
        st.subheader("Top features driving predictions")
        background = get_background_sample_cached(data)
        global_importance = get_global_importance(evaluation_model, background)
        view = st.radio("View", ["Plain-language", "Technical (feature importance)"], horizontal=True, key="global_importance_view")
        ordered = global_importance.sort_values("importance")
        importance_figure = px.bar(
            ordered, x="importance", y=[display_name(feature) for feature in ordered["feature"]], orientation="h",
            title="Model feature importance", color_discrete_sequence=[TEAL],
        )
        importance_figure.update_layout(yaxis_title=None, xaxis_title="Relative importance")
        st.plotly_chart(importance_figure, use_container_width=True)
        if view == "Plain-language":
            st.markdown("**Why these features matter clinically:**")
            for _, item in global_importance.iterrows():
                explanation = FEATURE_EXPLANATIONS.get(item["feature"], "")
                st.markdown(f"- **{display_name(item['feature'])}**: {explanation}")

    with st.container(border=True):
        st.subheader("Clinical importance")
        st.markdown(
            "This tool supports earlier referral decisions by flagging patients whose profile resembles "
            "confirmed diabetes cases in the training data, so clinicians can prioritize confirmatory lab testing "
            "(HbA1c, fasting plasma glucose, or OGTT) sooner rather than waiting for symptoms to appear. "
            "It augments routine screening workflows by surfacing risk earlier and explaining which factors drove "
            "each estimate. It does not replace a clinical diagnosis, does not interpret lab results on its own, "
            "and should not be used to defer testing for a patient who is otherwise due for screening."
        )


def render_overview() -> None:
    st.title("Diabetes Overview")
    st.caption("A short, sourced primer before you use the screening tools.")

    with st.container(border=True):
        st.subheader("What is diabetes? (World Health Organization)")
        for sentence in DIABETES_OVERVIEW["who_summary"]:
            st.markdown(f"- {sentence}")
        st.caption(f"Source: [{DIABETES_OVERVIEW['who_source_label']}]({DIABETES_OVERVIEW['who_source_url']})")


def main() -> None:
    st.markdown('<div class="pulse-strip"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="floating-drops"><span></span><span></span><span></span><span></span><span></span><span></span></div>',
        unsafe_allow_html=True,
    )
    data = load_data(str(DATA_PATH))
    model = get_model(str(MODEL_PATH), data)
    st.sidebar.title("Diabetes Screening")
    page = st.sidebar.radio("Navigate", ["Diabetes Overview", "Patient Risk Screener", "WHO Standards Dashboard", "Model & Data Transparency"])
    if page == "Diabetes Overview":
        render_overview()
    elif page == "Patient Risk Screener":
        render_screener(data, model)
    elif page == "WHO Standards Dashboard":
        render_dashboard(data)
    else:
        render_transparency(data)


if __name__ == "__main__":
    main()