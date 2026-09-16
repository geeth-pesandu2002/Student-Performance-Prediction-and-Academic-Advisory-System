# Student Performance AI and Academic Advisory System

This Flask application predicts a university performance category and generates practical academic actions. The active interface is designed around university indicators rather than the original secondary-school dataset.

## Important assignment note

The active model uses 360 simulated Sri Lankan university records generated for assignment development. These are not real student records and must not be presented as research evidence. For real deployment, replace the simulated data with approved, anonymized university data.

The current evaluation baseline is documented in `docs/sri_lankan_model_evaluation.txt`.
The candidate model comparison is documented in `docs/model_comparison.txt`.

See [the contribution and workflow record](docs/contribution_and_workflow.md) for the role mapping and examiner explanation. The detailed training and model-saving workflow is in [model_training_workflow.md](docs/model_training_workflow.md).

## Features

- University-focused performance prediction
- Categories: At Risk, Average, and High Performance
- Attendance, GPA, assessment, assignment, LMS, and study indicators
- Personalized recovery actions and weekly timetable
- Flask JSON API and browser interface

## Requirements

- Python 3.10 or newer
- A virtual environment is recommended
- Packages listed in `requirement.txt`

## Installation on Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirement.txt
```

If PowerShell blocks activation, run the commands from Command Prompt instead:

```cmd
.venv\Scripts\activate
```

## Run the application

```bash
pip install -r requirement.txt
python app.py
```

Open `http://localhost:5000` in a browser.

Use the `Average`, `At risk`, and `High performer` buttons to load example profiles. You can also enter a custom university profile and select **Generate insight**. The result includes the predicted category, model probabilities, profile checks, recommended actions, and a weekly timetable.

## Input fields

The form uses these university-focused fields:

`academic_year`, `semester`, `faculty`, `programme`, `previous_gpa`, `attendance_percentage`, `study_hours_per_week`, `failed_modules`, `assignment_completion_percentage`, `assessment_average_percentage`, `lecture_participation_percentage`, `tutorial_participation`, `lms_active`, `internet_access`, `financial_work_pressure`, and `wellbeing_rating`.

Do not enter names, student numbers, addresses, phone numbers, or other identifying information.

## API usage

Health check:

```text
GET http://localhost:5000/api/health
```

Prediction endpoint:

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

The detailed advisory endpoint is also available at `POST /api/advisory`.

## Retrain and evaluate

The generated dataset is available at `data/raw/sri_lankan_university.csv`. To regenerate the assignment dataset, retrain, and evaluate:

```bash
python -m src.generate_synthetic_sri_lankan_data
python -m src.train_sri_lankan_model
python -m src.evaluate_sri_lankan_model
```

Use `data/raw/sri_lankan_university_template.csv` and `docs/sri_lankan_data_collection.md` when replacing simulated records with approved local data.

## Explore the data in the notebook

Open `notebooks/data_analysis_and_model_training.ipynb` in VS Code or Jupyter. Select the project `.venv` kernel and run all cells. The notebook demonstrates data quality checks, category distribution, academic relationship charts, correlations, and model training. If the notebook is opened after cloning, install dependencies first with `pip install -r requirement.txt`.

## Tests

```bash
python -m unittest tests/test_predictor.py
```

The standalone API smoke test expects the Flask server to be running in another terminal:

```bash
python test_flask_backend.py
```

To verify Python syntax and imports:

```bash
python -m compileall -q app.py src test_flask_backend.py
```

## Project structure

```text
app.py                         Flask application and API routes
src/sri_lankan_schema.py       University feature contract and labels
src/predictor.py               Saved-model loading and predictions
src/university_advisory.py     Academic recommendations and timetable
src/train_sri_lankan_model.py  Model comparison, training, and saving
src/evaluate_sri_lankan_model.py Held-out evaluation report
data/raw/                      Synthetic dataset and collection template
models/                        Active trained pipeline
templates/ and static/         Browser interface
notebooks/                     Exploratory analysis notebook
tests/                         Automated tests
docs/                          Reports, limitations, and contribution notes
```

The project deliberately does not include the previous Portuguese secondary-school datasets, notebooks, processed files, or legacy model. The active training and evaluation path is implemented in `src/`.
