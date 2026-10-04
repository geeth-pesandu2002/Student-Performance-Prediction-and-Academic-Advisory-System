# Proposal Alignment Review

## Overall result

The current system fulfills the proposal's core expected outcomes. It predicts academic performance, identifies at-risk students, displays results through a web UI, provides academic guidance, and demonstrates a complete machine-learning application workflow.

## Requirement mapping

| Proposal requirement | Current evidence | Status |
|---|---|---|
| Predict student academic performance | Saved classification pipeline and `/api/predict` | Complete |
| Identify students at risk | `At Risk` class and probability output | Complete |
| Display prediction results using a UI | `templates/index.html` and `static/script.js` | Complete |
| Provide meaningful academic guidance | `src/university_advisory.py` and weekly timetable | Complete |
| Demonstrate practical ML in education | Data, training, evaluation, API, and UI workflow | Complete |
| Data processing | CSV loading, validation, label creation, preprocessing pipeline | Complete |
| Exploratory data analysis | `notebooks/data_analysis_and_model_training.ipynb` | Complete for assignment evidence |
| Classification | Logistic Regression, Decision Tree, and Random Forest comparison | Complete |
| Model evaluation | Accuracy, precision, recall, F1 report | Complete |
| Software development and integration | Flask application and separated source modules | Complete |
| AI-based decision support | Prediction plus rule-based academic actions | Complete |

## Dataset change from the proposal

The proposal originally referenced the UCI Student Performance dataset. The current system uses a synthetic Sri Lankan university-style dataset because the UCI records represented Portuguese secondary-school students and included fields that were not appropriate for the intended university context.

The active dataset uses academic indicators such as previous GPA, attendance, study time, failed modules, assignment completion, assessments, lecture participation, LMS activity, internet access, work pressure, and wellbeing.

The synthetic records are suitable for demonstrating the assignment workflow. They are not real student evidence and should be described honestly during the presentation. A real deployment would require approved and anonymized university records.

## Current model evidence

- Logistic Regression accuracy: 0.472
- Decision Tree accuracy: 0.417
- Random Forest accuracy: 0.625
- Selected model: Random Forest
- Held-out evaluation accuracy: 0.625

These scores are synthetic-data baselines. They should not be interpreted as real-world Sri Lankan university performance.

## Proposal limitations that still apply

- Prediction quality depends on dataset quality.
- Synthetic data cannot represent the full university environment.
- The selected features do not cover every personal, financial, psychological, or social factor.
- Recommendations are supportive guidance, not decisions about a student's future.
- Inputs such as previous GPA and assessment marks are available relatively late in the academic journey, so this is not a very-early-warning model.
- Teachers and academic advisors remain responsible for human decisions.

## Presentation wording

A suitable explanation is:

> "The original proposal used the UCI Student Performance dataset as an initial reference. During implementation, we replaced it with a synthetic Sri Lankan university-style dataset because the UCI data represented a different educational context. The replacement allowed us to demonstrate the same machine-learning workflow using features more relevant to university students. The resulting model is an assignment prototype and is not claimed to be trained on real Sri Lankan student records."
