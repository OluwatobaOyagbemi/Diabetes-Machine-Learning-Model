import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = "diabetes_prediction_dataset.csv"
MODEL_PATH = "diabetes_rf_model.joblib"
FEATURES_PATH = "model_features.joblib"


def main():
    data = pd.read_csv(DATA_PATH)

    X = data.drop("diabetes", axis=1)
    y = data["diabetes"]

    X_encoded = pd.get_dummies(X, drop_first=True)

    smote = SMOTE(sampling_strategy="minority", random_state=42)
    X_sm, y_sm = smote.fit_resample(X_encoded, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sm,
        y_sm,
        test_size=0.2,
        random_state=42,
        stratify=y_sm,
    )

    rf_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    rf_pipeline.fit(X_train, y_train)

    y_pred = rf_pipeline.predict(X_test)
    y_proba = rf_pipeline.predict_proba(X_test)[:, 1]

    print("Random Forest results")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("F1 Score:", round(f1_score(y_test, y_pred), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))
    print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

    joblib.dump(rf_pipeline, MODEL_PATH)
    joblib.dump(list(X_encoded.columns), FEATURES_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved feature columns to {FEATURES_PATH}")


if __name__ == "__main__":
    main()
