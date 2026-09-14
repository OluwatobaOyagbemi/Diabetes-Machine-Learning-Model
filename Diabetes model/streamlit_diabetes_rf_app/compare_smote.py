from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_FILENAME = "diabetes_prediction_dataset.csv"


def find_data_path() -> Path:
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / DATA_FILENAME,
        base_dir.parent / DATA_FILENAME,
        base_dir.parent / ".venv-1" / DATA_FILENAME,
        Path.cwd() / DATA_FILENAME,
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find {DATA_FILENAME}. Put it in {base_dir} or the workspace root."
    )


def load_data(data_path: Path) -> pd.DataFrame:
    return pd.read_csv(data_path)


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = data.drop("diabetes", axis=1)
    y = data["diabetes"]
    X_encoded = pd.get_dummies(X, drop_first=True)
    return X_encoded, y


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_curve": (fpr, tpr),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["No Diabetes", "Diabetes"]
        ),
    }


def plot_class_distribution(y_train, y_train_smote, y_test, output_dir: Path) -> None:
    counts = {
        "Train Before SMOTE": y_train.value_counts().sort_index(),
        "Train After SMOTE": y_train_smote.value_counts().sort_index(),
        "Test Set": y_test.value_counts().sort_index(),
    }
    labels = [0, 1]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (label, series) in enumerate(counts.items()):
        ax.bar(x + idx * width, [series.get(0, 0), series.get(1, 0)], width, label=label)

    ax.set_xticks(x + width)
    ax.set_xticklabels(["No Diabetes", "Diabetes"])
    ax.set_ylabel("Sample Count")
    ax.set_title("Class Distribution Before and After SMOTE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution_comparison.png")
    plt.close(fig)


def plot_confusion_matrices(results: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    titles = ["Baseline Model", "SMOTE Model"]

    for ax, result, title in zip(axes, results.values(), titles):
        cm = result["confusion_matrix"]
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

        ax.set_title(title)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Diabetes", "Diabetes"])
        ax.set_yticklabels(["No Diabetes", "Diabetes"])

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_comparison.png")
    plt.close(fig)


def plot_roc_curves(results: dict, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, result in results.items():
        fpr, tpr = result["roc_curve"]
        ax.plot(fpr, tpr, label=f"{label} (AUC={result['roc_auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_title("ROC Curve Comparison")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve_comparison.png")
    plt.close(fig)


def print_summary(results: dict) -> None:
    print("\n=== Model Comparison ===\n")
    for label, result in results.items():
        print(f"{label} results:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  F1 Score: {result['f1']:.4f}")
        print(f"  ROC AUC: {result['roc_auc']:.4f}")
        print("  Classification Report:")
        print(result["classification_report"])
        print("---------------------------")


def main() -> None:
    data_path = find_data_path()
    data = load_data(data_path)
    X_encoded, y = prepare_features(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    baseline_model = build_pipeline()
    baseline_model.fit(X_train, y_train)
    baseline_results = evaluate_model(baseline_model, X_test, y_test)

    smote = SMOTE(sampling_strategy="minority", random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    smote_model = build_pipeline()
    smote_model.fit(X_train_smote, y_train_smote)
    smote_results = evaluate_model(smote_model, X_test, y_test)

    results = {
        "Baseline": baseline_results,
        "SMOTE": smote_results,
    }

    output_dir = Path(".").resolve()
    plot_class_distribution(y_train, y_train_smote, y_test, output_dir)
    plot_confusion_matrices(results, output_dir)
    plot_roc_curves(results, output_dir)

    print_summary(results)
    print(f"\nSaved plots to {output_dir}")


if __name__ == "__main__":
    main()
