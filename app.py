import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "diabetes_rf_model.joblib"
FEATURES_PATH = "model_features.joblib"

st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, feature_columns


def prepare_input(user_data: dict, feature_columns: list[str]) -> pd.DataFrame:
    input_df = pd.DataFrame([user_data])
    input_encoded = pd.get_dummies(input_df, drop_first=True)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
    return input_encoded


st.title("🩺 Diabetes Prediction App")
st.write("Random Forest model trained from your diabetes prediction notebook.")

try:
    model, feature_columns = load_model()
except FileNotFoundError:
    st.error("Model files not found. Run `python train_model.py` first.")
    st.stop()

with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=35.0, step=1.0)
    hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    heart_disease = st.selectbox("Heart disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    smoking_history = st.selectbox(
        "Smoking history",
        ["never", "No Info", "current", "former", "ever", "not current"],
    )
    bmi = st.number_input("BMI", min_value=5.0, max_value=80.0, value=25.0, step=0.1)
    hba1c_level = st.number_input("HbA1c level", min_value=3.0, max_value=15.0, value=5.7, step=0.1)
    blood_glucose_level = st.number_input("Blood glucose level", min_value=50, max_value=400, value=140, step=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    user_data = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": hba1c_level,
        "blood_glucose_level": blood_glucose_level,
    }

    X_input = prepare_input(user_data, feature_columns)
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("High diabetes risk predicted")
    else:
        st.success("Low diabetes risk predicted")

    st.metric("Predicted diabetes probability", f"{probability * 100:.2f}%")
    st.caption("This app is for learning/demo purposes, not medical diagnosis.")
