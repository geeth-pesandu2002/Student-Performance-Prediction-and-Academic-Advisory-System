# Contribution and Workflow Record

## Project status

The repository contains the integrated implementation of the proposed group system. The active model uses 360 simulated Sri Lankan university records created for assignment development. The records are synthetic and are not evidence from real students.



## Work completed in the repository

### G.P. Thrikawala - Data and system integration

- Defined the university-focused input contract in `src/sri_lankan_schema.py`.
- Prepared the synthetic university dataset and reusable generator.
- Connected the data contract, trained model, Flask API, advisory module, and browser interface.
- Implemented API input validation and prediction flow in `app.py` and `src/predictor.py`.
- Updated the application documentation and supported system testing.

### WAMD Wijethunga - User interface and application integration

The proposed role is represented in the integrated implementation by:

- University student input form in `templates/index.html`.
- Demo profiles, form conversion, loading states, and result rendering in `static/script.js`.
- Dashboard layout and responsive presentation in `static/style.css`.
- Display of prediction probabilities, profile checks, recommendations, and weekly timetable.

### KBKG Senasana - Data science and model evaluation

The proposed role is represented in:

- Feature and label design in `src/sri_lankan_schema.py`.
- Held-out accuracy and precision/recall/F1 evaluation in `src/evaluate_sri_lankan_model.py`.
- Evaluation output in `docs/sri_lankan_model_evaluation.txt`.
- Data collection assumptions and limitations in `docs/sri_lankan_data_collection.md`.

### MFM Amhar - Machine-learning model development

The proposed role is represented in:

- Preprocessing pipeline and Random Forest training in `src/train_sri_lankan_model.py`.
- Model persistence with `joblib`.
- Model loading and prediction in `src/predictor.py`.
- Reproducible synthetic-data generation in `src/generate_synthetic_sri_lankan_data.py`.

### HAT Himadhya - Architecture, testing, and software quality

The proposed role is represented in:

- Flask route structure in `app.py`.
- Separation of schema, training, evaluation, prediction, and advisory modules.
- Predictor tests in `tests/test_predictor.py`.
- API smoke-test profiles in `test_flask_backend.py`.
- Compilation and end-to-end validation performed before submission.

## Examiner explanation

The system follows this workflow:

1. Synthetic or approved university records are loaded.
2. Semester GPA is converted into three performance categories.
3. Numeric features are scaled and categorical features are one-hot encoded.
4. Candidate model development is performed in the training pipeline; the Random Forest pipeline is selected for the active prototype.
5. The model is saved as `models/sri_lankan_university_pipeline.joblib`.
6. Flask loads the saved model and validates incoming form data.
7. The prediction is combined with academic rules to generate recovery actions and a weekly timetable.
8. Tests verify prediction behavior and the API verifies the integrated path.

## Important limitation

The current dataset is simulated. The evaluation score is an assignment-development baseline, not proof that the model predicts real Sri Lankan university performance. A real deployment requires approved anonymized records, retraining, and a fresh evaluation.