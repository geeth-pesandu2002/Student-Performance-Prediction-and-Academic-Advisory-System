"""Evaluate the local model on a held-out set before it can replace the prototype."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.sri_lankan_schema import LOCAL_FEATURES, categorize_gpa

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "sri_lankan_university.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "sri_lankan_university_pipeline.joblib"
REPORT_PATH = PROJECT_ROOT / "docs" / "sri_lankan_model_evaluation.txt"


def evaluate_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Add anonymized records to {DATA_PATH} before evaluation."
        )
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Train the local model first: {MODEL_PATH} was not found."
        )

    data = pd.read_csv(DATA_PATH)
    data["performance_category"] = data["semester_gpa"].apply(categorize_gpa)
    features = data[LOCAL_FEATURES]
    target = data["performance_category"]
    _, test_features, _, test_target = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    model = joblib.load(MODEL_PATH)
    predictions = model.predict(test_features)
    report = classification_report(test_target, predictions, zero_division=0)
    output = (
        "Sri Lankan university model evaluation\n"
        "========================================\n"
        f"Records: {len(data)}\n"
        f"Held-out records: {len(test_target)}\n"
        f"Accuracy: {accuracy_score(test_target, predictions):.3f}\n\n"
        f"{report}"
    )
    REPORT_PATH.write_text(output, encoding="utf-8")
    return output


if __name__ == "__main__":
    print(evaluate_model())