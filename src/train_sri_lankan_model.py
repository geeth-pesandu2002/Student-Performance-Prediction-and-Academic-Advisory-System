"""Train the future local model from anonymized university records."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.sri_lankan_schema import (
    LOCAL_CATEGORICAL_FEATURES,
    LOCAL_FEATURES,
    LOCAL_NUMERIC_FEATURES,
    categorize_gpa,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "sri_lankan_university.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "sri_lankan_university_pipeline.joblib"
COMPARISON_PATH = PROJECT_ROOT / "docs" / "model_comparison.txt"


def train_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Add anonymized records to {DATA_PATH} before training the local model."
        )

    data = pd.read_csv(DATA_PATH)
    required_columns = set(LOCAL_FEATURES + ["semester_gpa"])
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError("Missing dataset columns: " + ", ".join(missing_columns))
    if len(data) < 30:
        raise ValueError("Collect at least 30 anonymized records before training.")

    data["performance_category"] = data["semester_gpa"].apply(categorize_gpa)
    if data["performance_category"].nunique() < 3:
        raise ValueError("The dataset must contain examples in all three GPA categories.")

    train_features, test_features, train_target, test_target = train_test_split(
        data[LOCAL_FEATURES],
        data["performance_category"],
        test_size=0.2,
        random_state=42,
        stratify=data["performance_category"],
    )
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_split=5, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight="balanced",
        ),
    }
    pipelines = {}
    scores = {}
    for name, estimator in estimators.items():
        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), LOCAL_NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), LOCAL_CATEGORICAL_FEATURES),
        ])
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        pipeline.fit(train_features, train_target)
        pipelines[name] = pipeline
        scores[name] = accuracy_score(test_target, pipeline.predict(test_features))

    selected_name = max(scores, key=scores.get)
    selected_pipeline = pipelines[selected_name]
    joblib.dump(selected_pipeline, MODEL_PATH)
    comparison = ["Model comparison on the deterministic held-out split", ""]
    comparison.extend(f"{name}: {score:.3f}" for name, score in scores.items())
    comparison.append(f"Selected model: {selected_name}")
    COMPARISON_PATH.write_text("\n".join(comparison) + "\n", encoding="utf-8")
    return selected_pipeline


if __name__ == "__main__":
    model = train_model()
    print(f"Saved local model with {len(model.feature_names_in_)} inputs to {MODEL_PATH}")