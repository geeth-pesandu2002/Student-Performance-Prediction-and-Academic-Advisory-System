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

## Run the application

```bash
pip install -r requirement.txt
python app.py
```

Open `http://localhost:5000` in a browser.

## Retrain and evaluate

The generated dataset is available at `data/raw/sri_lankan_university.csv`. To regenerate the assignment dataset, retrain, and evaluate:

```bash
python -m src.generate_synthetic_sri_lankan_data
python -m src.train_sri_lankan_model
python -m src.evaluate_sri_lankan_model
```

Use `data/raw/sri_lankan_university_template.csv` and `docs/sri_lankan_data_collection.md` when replacing simulated records with approved local data.

## Tests

```bash
python -m unittest tests/test_predictor.py
```

The project deliberately does not include the previous Portuguese secondary-school datasets, notebooks, processed files, or legacy model. The active training and evaluation path is implemented in `src/`.
