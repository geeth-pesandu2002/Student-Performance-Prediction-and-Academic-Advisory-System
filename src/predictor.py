from pathlib import Path
import joblib
import pandas as pd


# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to the saved model
MODEL_PATH = PROJECT_ROOT / "models" / "sri_lankan_university_pipeline.joblib"


def load_model():
    """
    Load and return the trained Random Forest pipeline.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found at: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    return model


def predict_student_performance(student_data):
    """
    Predict the performance category of a student.

    Parameters:
    student_data : dict or pandas DataFrame
        Student information containing the required input features.

    Returns:
    prediction : str
        Predicted performance category.
    """

    # Load the trained model
    model = load_model()

    # Convert dictionary input into a DataFrame
    if isinstance(student_data, dict):
        student_data = pd.DataFrame([student_data])
    elif not isinstance(student_data, pd.DataFrame):
        raise TypeError(
            "student_data must be a dict or pandas DataFrame."
        )

    if student_data.empty:
        raise ValueError("student_data cannot be empty.")

    required_features = list(getattr(model, "feature_names_in_", []))
    if required_features:
        missing_features = [
            feature
            for feature in required_features
            if feature not in student_data.columns
        ]

        if missing_features:
            raise ValueError(
                "Missing required input features: "
                + ", ".join(missing_features)
            )

        # Keep only expected features in training order.
        student_data = student_data[required_features]

    if student_data.isnull().any().any():
        raise ValueError(
            "Input contains missing values. Please fill all required fields."
        )

    # Make prediction
    prediction = model.predict(student_data)

    return prediction[0]


def predict_student_with_probabilities(student_data):
    """Return the prediction and model probabilities for one student."""
    model = load_model()

    if isinstance(student_data, dict):
        student_data = pd.DataFrame([student_data])
    elif not isinstance(student_data, pd.DataFrame):
        raise TypeError("student_data must be a dict or pandas DataFrame.")

    if student_data.empty:
        raise ValueError("student_data cannot be empty.")

    required_features = list(getattr(model, "feature_names_in_", []))
    if not required_features:
        required_features = list(
            model.named_steps["preprocessor"].feature_names_in_
        )

    missing_features = [
        feature for feature in required_features
        if feature not in student_data.columns
    ]
    if missing_features:
        raise ValueError(
            "Missing required input features: "
            + ", ".join(missing_features)
        )

    student_data = student_data[required_features].copy()
    if student_data.isnull().any().any():
        raise ValueError(
            "Input contains missing values. Please fill all required fields."
        )

    prediction = model.predict(student_data)[0]
    probabilities = model.predict_proba(student_data)[0]
    probability_by_class = dict(zip(model.classes_, probabilities))

    return prediction, probability_by_class