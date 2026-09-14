from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "diabetes"
CLINICAL_THRESHOLD = 0.10
FEATURE_COLUMNS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]
CATEGORICAL_COLUMNS = ["gender", "smoking_history"]
NUMERIC_COLUMNS = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]


# WHO / IDF / ADA reference standards used to contextualize each input feature.
# Sources: WHO BMI classification, WHO/IDF HbA1c and glucose diagnostic criteria,
# WHO STEPS cardiovascular risk factors (hypertension, heart disease, tobacco use).
WHO_STANDARDS = {
    "bmi": {
        "label": "Body Mass Index (BMI)",
        "unit": "kg/m2",
        "explanation": "Weight (kg) divided by height squared (m2). WHO uses BMI as the standard screen for weight-related metabolic risk, including type 2 diabetes.",
        "source": "WHO BMI Classification",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight",
        "bands": [
            (0, 18.5, "Underweight", "#4C9BE8"),
            (18.5, 25.0, "Normal weight", "#2e8b57"),
            (25.0, 30.0, "Overweight", "#d98c10"),
            (30.0, 60.0, "Obese", "#c53b32"),
        ],
        "range": (12.0, 60.0),
    },
    "HbA1c_level": {
        "label": "HbA1c (Glycated Hemoglobin)",
        "unit": "%",
        "explanation": "Reflects average blood glucose over the past 2-3 months. WHO/ADA use HbA1c as a primary diagnostic marker for diabetes and prediabetes.",
        "source": "WHO / ADA HbA1c Diagnostic Criteria",
        "source_url": "https://www.who.int/publications/i/item/use-of-glycated-haemoglobin-(-hba1c)-in-the-diagnosis-of-diabetes-mellitus",
        "bands": [
            (3.0, 5.7, "Normal", "#2e8b57"),
            (5.7, 6.5, "Prediabetes", "#d98c10"),
            (6.5, 14.0, "Diabetes range", "#c53b32"),
        ],
        "range": (3.0, 14.0),
    },
    "blood_glucose_level": {
        "label": "Blood Glucose",
        "unit": "mg/dL",
        "explanation": "A direct measurement of glucose in the blood. WHO fasting plasma glucose thresholds distinguish normal, impaired (prediabetes), and diabetic ranges. The dataset does not record fasting status.",
        "source": "WHO Fasting Plasma Glucose Criteria",
        "source_url": "https://www.who.int/publications/i/item/definition-and-diagnosis-of-diabetes-mellitus-and-intermediate-hyperglycaemia",
        "bands": [
            (50, 100, "Normal (fasting)", "#2e8b57"),
            (100, 126, "Impaired / Prediabetes", "#d98c10"),
            (126, 400, "Diabetes range", "#c53b32"),
        ],
        "range": (50, 400),
    },
    "age": {
        "label": "Age",
        "unit": "years",
        "explanation": "WHO and ADA screening guidance flags age over 45 as a non-modifiable risk factor for type 2 diabetes, with risk rising further after 65.",
        "source": "WHO / ADA Age-Related Risk Guidance",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/noncommunicable-diseases",
        "bands": [
            (18, 45, "Lower age-related risk", "#2e8b57"),
            (45, 65, "Elevated age-related risk", "#d98c10"),
            (65, 100, "High age-related risk", "#c53b32"),
        ],
        "range": (18, 100),
    },
}

BINARY_STANDARDS = {
    "hypertension": {
        "label": "Hypertension",
        "explanation": "WHO defines hypertension as blood pressure >=140/90 mmHg. It roughly doubles cardiometabolic and diabetes-related risk when present alongside other factors.",
        "source": "WHO Hypertension Guidelines",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/hypertension",
    },
    "heart_disease": {
        "label": "Heart Disease",
        "explanation": "A diagnosed cardiovascular condition. WHO lists cardiovascular disease and diabetes as closely linked, compounding comorbidity risk.",
        "source": "WHO Noncommunicable Disease Guidance",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)",
    },
}

SMOKING_RISK = {
    "never": ("Lower risk", "#2e8b57"),
    "No Info": ("Unknown", "#9aa0a6"),
    "former": ("Reduced risk (quit)", "#d98c10"),
    "not current": ("Reduced risk (quit)", "#d98c10"),
    "ever": ("Elevated risk (history)", "#d98c10"),
    "current": ("Elevated risk", "#c53b32"),
}

SMOKING_SOURCE_URL = "https://www.who.int/news-room/fact-sheets/detail/tobacco"
GENDER_SOURCE_URL = "https://www.who.int/news-room/fact-sheets/detail/gender"
AI_ETHICS_SOURCE_URL = "https://www.who.int/publications/i/item/9789240029200"


def classify_value(feature: str, value: float) -> tuple[str, str]:
    """Return (category label, color) for a numeric feature based on WHO bands."""
    for low, high, label, color in WHO_STANDARDS[feature]["bands"]:
        if low <= value < high:
            return label, color
    last_low, _, label, color = WHO_STANDARDS[feature]["bands"][-1]
    return label, color


# Model Card metadata, following the spirit of WHO/FDA-aligned AI transparency guidance
# (WHO "Ethics and governance of artificial intelligence for health", 2021).
MODEL_CARD = {
    "intended_use": "Pre-screening aid to flag adults who may benefit from confirmatory diabetes testing (HbA1c, fasting plasma glucose, or OGTT). Intended to support earlier referral decisions, not to diagnose.",
    "population": "Trained on 100,000 de-identified records from a public Kaggle-style diabetes prediction dataset spanning ages, both diagnosed hypertension/heart-disease status, and self-reported smoking history. Gender categories: Female, Male, Other.",
    "limitations": [
        "Single public dataset of unknown clinical/institutional origin - not validated against a real health-system cohort.",
        "Severe class imbalance (8.5% positive) - precision falls sharply as the decision threshold is lowered for higher recall.",
        "'Other' gender category has only 18 records in the full dataset - too few to assess model behavior reliably for that group.",
        "Blood glucose values are not labeled as fasting or random, so readings are not directly comparable to WHO fasting-plasma-glucose criteria.",
        "Smoking history is self-reported and 'No Info' entries may hide real smoking status.",
    ],
    "out_of_scope": [
        "Not a diagnostic device - must not be used to confirm or rule out diabetes without laboratory testing.",
        "Not validated for pediatric populations, pregnancy (gestational diabetes), or Type 1 diabetes screening.",
        "Not intended for autonomous clinical decision-making without a clinician in the loop.",
    ],
}

DATASET_INFO = {
    "name": "Diabetes Prediction Dataset",
    "source": "diabetes_prediction_dataset.csv (public Kaggle-style dataset bundled with this project)",
    "provenance": "Single dataset, deterministically split 80/20 (stratified) into training and held-out test data for all reported metrics. No separate external validation cohort is used.",
}

# Plain-language, feature-specific clinical explanations for the SHAP "why it matters" view.
FEATURE_EXPLANATIONS = {
    "age": "Diabetes risk rises steadily with age as insulin sensitivity declines over time.",
    "bmi": "Higher BMI reflects excess adiposity, a major driver of insulin resistance.",
    "HbA1c_level": "Elevated HbA1c reflects average blood glucose over months, a core diabetes marker.",
    "blood_glucose_level": "Elevated glucose reflects impaired insulin regulation at the time of measurement.",
    "hypertension": "Hypertension frequently co-occurs with insulin resistance as part of metabolic syndrome.",
    "heart_disease": "Existing cardiovascular disease shares risk pathways with type 2 diabetes.",
    "smoking_history": "Tobacco use worsens insulin resistance and compounds cardiometabolic risk.",
    "gender": "Diabetes prevalence and risk-factor patterns differ somewhat by sex in population data.",
}

# Human-readable, properly capitalized labels for display purposes only.
FEATURE_DISPLAY_NAMES = {
    "age": "Age",
    "bmi": "BMI",
    "HbA1c_level": "HbA1c",
    "blood_glucose_level": "Blood Glucose",
    "hypertension": "Hypertension",
    "heart_disease": "Heart Disease",
    "smoking_history": "Smoking Status",
    "gender": "Gender",
}


def display_name(feature: str) -> str:
    """Return a properly capitalized, clinician-friendly label for a raw feature name."""
    return FEATURE_DISPLAY_NAMES.get(feature, feature.replace("_", " ").title())


# Diabetes Overview content, sourced from the WHO diabetes fact sheet, for the
# app's educational overview section.
DIABETES_OVERVIEW = {
    "who_summary": [
        "Diabetes is a chronic disease that occurs when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces, leading to raised blood glucose (hyperglycaemia).",
        "The number of people living with diabetes rose from 200 million in 1990 to 830 million in 2022, with prevalence rising fastest in low- and middle-income countries.",
        "More than 95% of people with diabetes have type 2 diabetes, which is largely preventable through a healthy diet, regular physical activity, a normal body weight, and avoiding tobacco use.",
        "Uncontrolled diabetes causes serious damage to the body's nerves and blood vessels over time, contributing to blindness, kidney failure, heart attacks, stroke, and lower-limb amputation.",
        "In 2021, diabetes and diabetes-related kidney disease caused over 2 million deaths, and diabetes can be treated and its complications avoided or delayed through diet, physical activity, medication, and regular screening.",
    ],
    "who_source_label": "World Health Organization, \"Diabetes\" Fact Sheet (14 November 2024)",
    "who_source_url": "https://www.who.int/news-room/fact-sheets/detail/diabetes",
}


def make_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=100,
                    random_state=42,
                ),
            ),
        ]
    )


def load_or_train_model(model_path: str, data: pd.DataFrame) -> tuple[Pipeline, str]:
    path = Path(model_path)
    if path.exists():
        with path.open("rb") as model_file:
            return pickle.load(model_file), f"Loaded trained artifact: {path.name}"

    model = make_pipeline()
    model.fit(data[FEATURE_COLUMNS], data[TARGET_COLUMN])
    return model, "Gradient Boosting clinical model trained from raw CSV data for this session (no model.pkl found)."


def make_test_predictions(data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, Pipeline]:
    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=42
    )
    model = make_pipeline()
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    return x_test.reset_index(drop=True), y_test.to_numpy(), probabilities, model


def threshold_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    return {
        "confusion_matrix": np.array([[tn, fp], [fn, tp]]),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
    }


def feature_importance(model: Pipeline) -> pd.DataFrame:
    classifier = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]
    names = preprocessor.get_feature_names_out()
    return (
        pd.DataFrame({"feature": names, "importance": classifier.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _raw_feature_name(transformed_name: str) -> str:
    """Map a ColumnTransformer output name (e.g. 'categorical__gender_Female') back to its raw feature."""
    suffix = transformed_name.split("__", 1)[-1]
    for column in CATEGORICAL_COLUMNS:
        if suffix == column or suffix.startswith(column + "_"):
            return column
    return suffix


def get_background_sample(data: pd.DataFrame, sample_size: int = 500) -> pd.DataFrame:
    """A small, fixed reference sample used to contextualize a patient's feature values."""
    return data[FEATURE_COLUMNS].sample(n=min(sample_size, len(data)), random_state=42)


def global_feature_importance(model: Pipeline, background: pd.DataFrame, top_n: int = 7) -> pd.DataFrame:
    """Model feature-importance, aggregated from one-hot columns back to each raw feature."""
    classifier = model.named_steps["model"]
    names = model.named_steps["preprocessor"].get_feature_names_out()
    frame = pd.DataFrame({"transformed_feature": names, "importance": classifier.feature_importances_})
    frame["feature"] = frame["transformed_feature"].map(_raw_feature_name)
    return (
        frame.groupby("feature", as_index=False)["importance"].sum()
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def local_feature_contribution(model: Pipeline, patient: pd.DataFrame, background: pd.DataFrame) -> pd.DataFrame:
    """Per-raw-feature contribution proxy for a single patient: importance weighted by how far the
    patient's value sits from a background sample's average, aggregated back to raw feature names."""
    transformed_patient = model.named_steps["preprocessor"].transform(patient)
    if hasattr(transformed_patient, "toarray"):
        transformed_patient = transformed_patient.toarray()
    transformed_background = model.named_steps["preprocessor"].transform(background)
    if hasattr(transformed_background, "toarray"):
        transformed_background = transformed_background.toarray()
    background_mean = np.asarray(transformed_background).mean(axis=0).ravel()
    classifier = model.named_steps["model"]
    names = model.named_steps["preprocessor"].get_feature_names_out()
    contribution = classifier.feature_importances_ * (np.asarray(transformed_patient).ravel() - background_mean)
    frame = pd.DataFrame({"transformed_feature": names, "contribution": contribution})
    frame["feature"] = frame["transformed_feature"].map(_raw_feature_name)
    grouped = frame.groupby("feature", as_index=False)["contribution"].sum()
    return grouped.reindex(grouped["contribution"].abs().sort_values(ascending=False).index).head(10)


def cross_validated_metrics(data: pd.DataFrame, n_splits: int = 5) -> dict:
    """5-fold stratified CV mean +/- std for headline metrics, at the default 0.5 threshold."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_validate(
        make_pipeline(),
        data[FEATURE_COLUMNS],
        data[TARGET_COLUMN],
        cv=cv,
        scoring=["roc_auc", "recall", "precision", "accuracy"],
        n_jobs=-1,
    )
    return {
        metric: (scores[f"test_{metric}"].mean(), scores[f"test_{metric}"].std())
        for metric in ["roc_auc", "recall", "precision", "accuracy"]
    }