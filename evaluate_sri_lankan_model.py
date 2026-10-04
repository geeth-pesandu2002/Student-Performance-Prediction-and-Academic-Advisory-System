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
    methodology_notes = """


========================================================================
DATASET LIMITATIONS AND METHODOLOGY NOTES
========================================================================

1. SYNTHETIC DATA LIMITATIONS:
    - This dataset contains 360 simulated records, not real student data.
    - It was generated for assignment development purposes only.
    - Its patterns may not represent real-world university student performance.

2. SAMPLE SIZE AND CLASS IMBALANCE:
    - 360 records is relatively small for robust machine learning.
    - Class imbalance exists: Average (221) > At Risk (84) > High Performance (55).
    - We used stratified splitting so the training and held-out sets preserve approximately the same class proportions.
    - Stratification does not remove class imbalance; it remains a limitation of this evaluation.

3. MODEL AND PREPROCESSING METHODOLOGY:
    - A scikit-learn Pipeline applies preprocessing consistently before prediction.
    - Categorical features use OneHotEncoder.
    - Numeric features use StandardScaler.
    - The selected Random Forest model can provide feature-importance values for transformed features.
    - The confusion matrix and classification metrics are calculated on the deterministic held-out set of 72 records.
    - Feature importance indicates model reliance, not causation.

4. FEATURE LIMITATIONS:
    - Current features do not fully capture socioeconomic background, mental-health context, module difficulty, or teaching quality.

5. ETHICAL AND DEPLOYMENT CONSIDERATIONS:
    - This model is an advisory prototype, not a definitive assessment.
    - Academic-advisor oversight is required before intervention.
    - Approved anonymized real data and fairness audits are required before deployment.

6. RECOMMENDATIONS FOR FUTURE WORK:
    - Replace synthetic data with approved anonymized university records.
    - Re-evaluate class-balance methods such as class weighting or SMOTE only after inspecting real data.
    - Compare additional models and tune parameters using a reproducible validation strategy.
"""
    output = (
        "Sri Lankan university model evaluation\n"
        "========================================\n"
        f"Records: {len(data)}\n"
        f"Held-out records: {len(test_target)}\n"
        f"Accuracy: {accuracy_score(test_target, predictions):.3f}\n\n"
                f"{report}"
                f"{methodology_notes}"
    )
    REPORT_PATH.write_text(output, encoding="utf-8")
    return output


if __name__ == "__main__":
    print(evaluate_model())