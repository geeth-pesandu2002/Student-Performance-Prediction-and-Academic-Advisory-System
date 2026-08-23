from pathlib import Path
import joblib
import pandas as pd


# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to the saved model
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_pipeline.joblib"


def load_model():
    """
    Load and return the trained Random Forest pipeline.
    """
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

    # Make prediction
    prediction = model.predict(student_data)

    return prediction[0]