# Student Performance AI and Academic Advisory System

Flask application that predicts a university performance category and provides academic recommendations.

## Requirements

- Python 3.10 or newer
- Packages listed in `requirement.txt`

## Setup

From the project directory, create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirement.txt
```

## Run

```powershell
python app.py
```

Open `http://localhost:5000` in a browser.

The trained model is included in `models/`, so no model training is required to run the application.

## How the system works

The model was trained with Python scripts, not a Jupyter notebook:

1. `src/generate_synthetic_sri_lankan_data.py` creates the development dataset in `data/raw/`.
2. `src/sri_lankan_schema.py` defines the model input fields and converts semester GPA into `At Risk`, `Average`, or `High Performance`.
3. `src/train_sri_lankan_model.py` loads the CSV, splits the data into training and test sets, preprocesses numeric and categorical fields, compares Logistic Regression, Decision Tree, and Random Forest pipelines, and saves the selected pipeline to `models/`.
4. `src/evaluate_sri_lankan_model.py` evaluates the saved model on the same held-out split.
5. `src/predictor.py` loads the saved pipeline and generates predictions and probabilities for the Flask API.

To regenerate the development data, retrain, and evaluate:

```powershell
python -m src.generate_synthetic_sri_lankan_data
python -m src.train_sri_lankan_model
python -m src.evaluate_sri_lankan_model
```

The academic recommendations are rule-based, not produced by the machine-learning model. They are implemented in `src/university_advisory.py`: for example, low attendance produces an attendance recommendation, failed modules produce a recovery-plan recommendation, and low assignment completion produces an assignment recommendation. `app.py` combines the model prediction with these recommendations.

For visual training and evaluation evidence, open `notebooks/model_training.ipynb` in Jupyter or VS Code after installing the requirements. The notebook demonstrates the same workflow; the Python scripts under `src/` remain the authoritative implementation.

## API

Health check:

```text
GET http://localhost:5000/api/health
```

Prediction:

```text
POST http://localhost:5000/api/predict
Content-Type: application/json
```

Example request:

```json
{
  "academic_year": 2,
  "semester": 1,
  "faculty": "Computing",
  "programme": "Computer Science",
  "previous_gpa": 2.6,
  "attendance_percentage": 82,
  "study_hours_per_week": 10,
  "failed_modules": 0,
  "assignment_completion_percentage": 78,
  "assessment_average_percentage": 58,
  "lecture_participation_percentage": 78,
  "tutorial_participation": "yes",
  "lms_active": "yes",
  "internet_access": "yes",
  "financial_work_pressure": "no",
  "wellbeing_rating": 3
}
```

## Tests

```powershell
python -m unittest discover -s tests
```
