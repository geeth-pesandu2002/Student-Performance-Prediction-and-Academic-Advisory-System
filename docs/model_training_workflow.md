# Model Training and Saving Workflow

The project uses Python scripts instead of relying on notebooks for the executable training workflow. This keeps training repeatable from a clean environment.

## Training

```bash
python -m src.generate_synthetic_sri_lankan_data
python -m src.train_sri_lankan_model
```

`src/train_sri_lankan_model.py` loads the university CSV, validates its columns, creates the performance label from `semester_gpa`, splits the data with stratification, scales numeric features, one-hot encodes categorical features, and compares balanced Logistic Regression, Decision Tree, and Random Forest pipelines. The highest-scoring pipeline is saved as the active model and the comparison is written to `docs/model_comparison.txt`.

## Evaluation

```bash
python -m src.evaluate_sri_lankan_model
```

The evaluator uses the same deterministic held-out split and writes accuracy, precision, recall, F1 score, and class support to `docs/sri_lankan_model_evaluation.txt`.

## Saving and loading

The training script saves the complete preprocessing-plus-model pipeline to:

```text
models/sri_lankan_university_pipeline.joblib
```

The application loads that file through `src/predictor.py`. Saving the complete pipeline is important because production prediction must apply exactly the same scaling and categorical encoding used during training.

## Why notebooks are optional

`model_training.ipynb` and `model_saving.ipynb` are not required for the model to work. They would be presentation artifacts duplicating the scripts. The scripts are the authoritative implementation because they can be executed reproducibly and are covered by the project validation commands. The workflow can be demonstrated to examiners using this document and the command outputs.