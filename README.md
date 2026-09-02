# 🎓 AI Student Performance Prediction and Academic Advisory System

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Status](https://img.shields.io/badge/status-In%20Development-yellow)
![Python](https://img.shields.io/badge/python-3.8%2B-green)
![License](https://img.shields.io/badge/license-MIT-blue)

## 📋 Project Overview

This system uses machine learning to predict student academic performance based on 30 features including study habits, family background, social activities, and previous academic history. The goal is to provide early warnings and personalized recommendations to help students succeed academically.

**Key Features:**
- 🤖 **AI-Powered Predictions** - Random Forest classifier trained on 1,000+ real student records
- 📊 **Data-Driven Insights** - Analyzes 30 student features to make predictions
- 🎯 **Three Performance Categories** - At Risk, Average, High Performance
- ✅ **Input Validation** - Robust error handling with user-friendly messages
- 📈 **Reproducible Results** - Full data pipeline from raw data to predictions

---

## 🎯 Quick Start (5 minutes)

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook
- Required packages: pandas, scikit-learn, joblib, numpy

### Installation

```bash
# 1. Clone or download the project
git clone https://github.com/geeth-pesandu2002/Student-Performance-Prediction-and-Academic-Advisory-System.git
cd student-performance-prediction

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Notebook
jupyter notebook
```

### Run the Complete Pipeline (First Time Only)

Follow these steps in order to understand the complete data science workflow:

```bash
# Step 1: Explore the data
jupyter notebook notebooks/01_data_understanding.ipynb

# Step 2: Clean and prepare data
jupyter notebook notebooks/02_data_preprocessing.ipynb

# Step 3: Train the ML model
jupyter notebook notebooks/03_model_training.ipynb

# Step 4: Save the trained model
jupyter notebook notebooks/04_model_saving.ipynb

# Step 5: Test predictions
jupyter notebook notebooks/05_prediction_testing.ipynb
```

### Make a Prediction (Using Saved Model)

```python
from src.predictor import predict_student_performance

# Student data with all 30 required features
student = {
    "school": "GP",           # Gabriel Pereira or Mousinho da Silveira
    "sex": "F",               # Male or Female
    "age": 17,                # 15-22
    "address": "U",           # Urban or Rural
    "famsize": "GT3",         # LE3 (<=3) or GT3 (>3)
    "Pstatus": "T",           # Living Together or Apart
    "Medu": 4,                # Mother's education (0-4)
    "Fedu": 4,                # Father's education (0-4)
    "Mjob": "at_home",        # Mother's job
    "Fjob": "teacher",        # Father's job
    "reason": "course",       # Reason to choose school
    "guardian": "mother",     # Student's guardian
    "traveltime": 2,          # Travel time to school (1-4)
    "studytime": 2,           # Weekly study time (1-4)
    "failures": 0,            # Number of past failures (0-4)
    "schoolsup": "yes",       # Extra school support
    "famsup": "no",           # Family support
    "paid": "no",             # Paid extra classes
    "activities": "no",       # Extra-curricular activities
    "nursery": "yes",         # Attended nursery
    "higher": "yes",          # Wants higher education
    "internet": "no",         # Internet access at home
    "romantic": "no",         # In romantic relationship
    "famrel": 4,              # Family relationship quality (1-5)
    "freetime": 3,            # Free time after school (1-5)
    "goout": 4,               # Going out with friends (1-5)
    "Dalc": 1,                # Workday alcohol consumption (1-5)
    "Walc": 1,                # Weekend alcohol consumption (1-5)
    "health": 3,              # Current health status (1-5)
    "absences": 6             # Number of absences (0-93)
}

# Make prediction
prediction = predict_student_performance(student)
print(f"Predicted Performance: {prediction}")
# Output: "At Risk", "Average", or "High Performance"
```

---

## 📁 Project Structure

```
student-performance-prediction/
├── README.md                          # This file - project documentation
├── requirements.txt                   # Python package dependencies
│
├── data/
│   ├── raw/
│   │   ├── student-mat.csv           # Math course student data (395 students)
│   │   ├── student-por.csv           # Portuguese course student data (649 students)
│   │   └── student.txt               # Data dictionary - feature descriptions
│   │
│   └── processed/
│       ├── X_train.csv               # Training features (80% of data)
│       ├── X_test.csv                # Test features (20% of data)
│       ├── y_train.csv               # Training targets (grades)
│       ├── y_test.csv                # Test targets (grades)
│       └── preprocessor.joblib       # Saved data preprocessing pipeline
│
├── models/
│   └── random_forest_pipeline.joblib # Trained AI model (production ready)
│
├── notebooks/
│   ├── 01_data_understanding.ipynb   # EDA - explore, visualize, validate data
│   ├── 02_data_preprocessing.ipynb   # Clean, transform, prepare for ML
│   ├── 03_model_training.ipynb       # Train 3 models, select best (Random Forest)
│   ├── 04_model_saving.ipynb         # Serialize trained model for production
│   └── 05_prediction_testing.ipynb   # Test predictor with sample students
│
├── src/
│   ├── predictor.py                  # Core prediction function (MAIN INTERFACE)
│   └── __init__.py                   # Python package marker
│
├── tests/
│   ├── test_predictor.py             # Unit tests for predictor.py
│   └── __pycache__/                  # Python cache (auto-generated)
│
├── docs/
│   ├── data_dictionary.md            # Detailed feature descriptions
│   └── model_evaluation.md           # Model performance metrics (future)
│
└── .gitignore                        # Files to exclude from version control
```

---

## 🔍 Detailed Workflow

### 1. Data Understanding (`01_data_understanding.ipynb`)

**Purpose:** Explore and validate raw data

**What It Does:**
- Loads student datasets (Math and Portuguese courses)
- Checks for missing values (Result: 0 missing ✓)
- Identifies duplicate records (Result: 0 duplicates ✓)
- Analyzes feature distributions
- Creates summary statistics

**Output:**
- Data quality assessment
- Feature distributions
- Basic visualizations

---

### 2. Data Preprocessing (`02_data_preprocessing.ipynb`)

**Purpose:** Clean and prepare data for machine learning

**What It Does:**
- Encodes categorical variables (text → numbers)
- Scales numerical features (standardization)
- Splits data: 80% training, 20% testing
- Creates preprocessing pipeline for reuse

**Output:**
- Processed data files in `data/processed/`
- Preprocessing pipeline saved

**Key Statistics:**
- Training samples: ~838 students
- Test samples: ~206 students
- Total features: 30

---

### 3. Model Training (`03_model_training.ipynb`)

**Purpose:** Train and compare machine learning models

**What It Does:**
- Implements 3 candidate algorithms:
  - Logistic Regression
  - Decision Tree
  - **Random Forest** (selected)
- Uses GridSearchCV for hyperparameter tuning
- Compares models using cross-validation
- Selects best model: **Random Forest**

**Model Configuration:**
```python
RandomForestClassifier(
    n_estimators=100,      # 100 decision trees voting
    max_depth=10,          # Tree depth limit
    min_samples_split=5,   # Min samples to split node
    random_state=42        # Reproducibility
)
```

**Performance:**
- Training Accuracy: ~65%
- Test Accuracy: ~48%
- 3 Output Classes: At Risk, Average, High Performance

---

### 4. Model Saving (`04_model_saving.ipynb`)

**Purpose:** Save trained model for production use

**What It Does:**
- Serializes trained model to file
- Defines performance categories:
  - **At Risk:** Grade < 8
  - **Average:** Grade 8-12
  - **High Performance:** Grade > 12
- Saves pipeline for reproducibility

**Output:**
- `models/random_forest_pipeline.joblib` (1.77 MB)

---

### 5. Prediction Testing (`05_prediction_testing.ipynb`)

**Purpose:** Validate predictions with sample students

**What It Does:**
- Tests predictor with 3 diverse student profiles
- Verifies output format and validity
- Demonstrates real-world usage

**Sample Results:**
```
Student A: At Risk
Student B: Average
Student C: High Performance
```

---

## 🔧 Using the Predictor Function

### Basic Usage

```python
from src.predictor import predict_student_performance

# Create student data dictionary
student_data = {
    "school": "GP",
    "sex": "F",
    "age": 17,
    # ... (all 30 features required)
}

# Get prediction
result = predict_student_performance(student_data)
print(result)  # Output: "At Risk", "Average", or "High Performance"
```

### Error Handling

The predictor includes robust error handling:

```python
# Example 1: Missing required fields
try:
    bad_data = {"school": "GP", "sex": "F"}  # Missing 28 fields!
    result = predict_student_performance(bad_data)
except ValueError as e:
    print(f"Error: {e}")
    # Output: "Missing required input features: age, address, famsize, ..."

# Example 2: Wrong data type
try:
    wrong_data = {
        "school": "GP",
        "sex": "F",
        "age": "seventeen"  # Should be integer!
        # ... (other fields)
    }
    result = predict_student_performance(wrong_data)
except ValueError as e:
    print(f"Error: {e}")

# Example 3: Null/missing values
try:
    null_data = {
        "school": "GP",
        # Some fields are None
    }
    result = predict_student_performance(null_data)
except ValueError as e:
    print(f"Error: {e}")
    # Output: "Input contains missing values. Please fill all required fields."
```

### Using with Pandas DataFrame

```python
import pandas as pd
from src.predictor import predict_student_performance

# Load student data from CSV
df = pd.read_csv("students.csv")

# Predict for single student
student = df.iloc[0].to_dict()
prediction = predict_student_performance(student)

# Predict for multiple students
predictions = []
for idx, row in df.iterrows():
    pred = predict_student_performance(row.to_dict())
    predictions.append(pred)

df['Prediction'] = predictions
```

---

## 📊 Data Dictionary

### Required Input Features (30 total)

All 30 features must be provided to make a prediction. See `docs/data_dictionary.md` for detailed descriptions.

**Example of Top Important Features:**
1. **failures** - Number of past class failures (most important)
2. **absences** - Number of school absences
3. **goout** - Frequency going out with friends
4. **Medu** - Mother's education level
5. **Fedu** - Father's education level

---

## 🧪 Testing

### Run Unit Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_predictor.py -v

# Run with coverage report
python -m pytest tests/ --cov=src
```

### Test Coverage

The `test_predictor.py` includes:
- ✅ Valid input prediction (returns known label)
- ✅ Missing fields error handling
- ✅ Data type validation
- ✅ Null value detection

**Current Test Status:** All tests passing ✓

---

## 📈 Model Performance

### Accuracy Metrics

- **Overall Accuracy:** 48%
- **Baseline (Random Guessing):** 33%
- **Improvement over Baseline:** +15%

### What This Means

- Out of 100 predictions, ~48 are completely accurate
- The model is significantly better than random guessing
- **Use as a suggestion tool, not a definitive decision**

### Performance by Category

See `docs/model_evaluation.md` for detailed breakdown by class.

### How to Improve Accuracy

1. **Collect more data** - 5,000+ students instead of 1,000
2. **Add new features** - Sleep hours, mental health, nutrition status
3. **Try different models** - Neural Networks, XGBoost, SVM
4. **Feature engineering** - Create derived features (study intensity, absence rate)
5. **Hyperparameter tuning** - Further optimize Random Forest parameters

---

## 👥 Team Roles and Responsibilities

| Member | Role | Responsibilities | Contact |
|--------|------|------------------|---------|
| **G.P Thrikawala** (D/BIT/24/0036) | Data & System Integration | Dataset collection, data cleaning, predictor.py | - |
| **WAMD Wijethunga** (D/BIT/24/0017) | UI & Application | Web interface, form design, integration | - |
| **KBKG Senasana** (D/DBA/25/0028) | Data Analysis & Evaluation | EDA, model evaluation, metrics | - |
| **MFM Amhar** (D/BCS/25/0029) | ML Model Development | Model selection, training, hyperparameter tuning | - |
| **HAT Himadhya** (D/BSE/25/0027) | Architecture & Testing | System design, test cases, integration testing | - |

---

## 📚 Technology Stack

### Core Libraries
- **pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning algorithms
- **joblib** - Model serialization
- **numpy** - Numerical computing

### Development Tools
- **Jupyter Notebook** - Interactive analysis and documentation
- **pytest** - Unit testing framework
- **Python 3.8+** - Programming language

### Data
- **student-mat.csv** - 395 Math course students
- **student-por.csv** - 649 Portuguese course students
- **Total:** 1,044 student records

---

## 🔐 Data Privacy

This project uses **anonymized student data**. No personally identifiable information (names, IDs, addresses) is included in the dataset.

**Data Source:** UCI Machine Learning Repository - Student Performance Dataset

---

## 📝 File Descriptions

### Key Files

| File | Purpose | Created By |
|------|---------|-----------|
| `src/predictor.py` | Main prediction function | GP Thrikawala |
| `models/random_forest_pipeline.joblib` | Trained AI model | MFM Amhar |
| `notebooks/01_data_understanding.ipynb` | Data exploration | KBKG Senasana |
| `notebooks/02_data_preprocessing.ipynb` | Data cleaning | GP Thrikawala |
| `notebooks/03_model_training.ipynb` | Model training | MFM Amhar |
| `notebooks/04_model_saving.ipynb` | Model serialization | MFM Amhar |
| `notebooks/05_prediction_testing.ipynb` | Prediction validation | GP Thrikawala |

---

## 🚀 Next Steps

### Phase 1: User Interface (Current)
- [ ] Design web interface
- [ ] Build student input form (30 fields)
- [ ] Create results display
- [ ] Integrate with predictor.py

### Phase 2: Testing & Validation
- [ ] Complete integration testing
- [ ] System testing with full workflow
- [ ] Performance optimization
- [ ] Bug fixes and refinements

### Phase 3: Deployment
- [ ] Documentation finalization
- [ ] Presentation preparation
- [ ] Model versioning
- [ ] Production deployment plan

---

## 🐛 Troubleshooting

### Common Issues

**Issue 1: ModuleNotFoundError when importing predictor**
```bash
# Solution: Ensure you're in the project root directory
cd /path/to/student-performance-prediction
python -c "from src.predictor import predict_student_performance"
```

**Issue 2: Model file not found**
```bash
# Solution: Ensure random_forest_pipeline.joblib exists in models/
ls models/random_forest_pipeline.joblib  # Should return the file path
```

**Issue 3: Missing dependencies**
```bash
# Solution: Reinstall requirements
pip install -r requirements.txt
```

**Issue 4: Jupyter notebook kernel issues**
```bash
# Solution: Restart kernel and run cells from top
# In Jupyter: Kernel → Restart & Run All
```

---

## 📞 Support & Questions

For questions about specific components:

- **Data & Integration:** GP Thrikawala
- **ML Model:** MFM Amhar
- **Data Analysis:** KBKG Senasana
- **UI/Application:** WAMD Wijethunga
- **Architecture & Testing:** HAT Himadhya

---

## 📄 License

This project is provided as-is for educational purposes.

---

## ✅ Checklist for Using This Project

Before deploying or sharing:

- [ ] All notebooks run without errors
- [ ] Model file exists and loads correctly
- [ ] Predictor function works with test data
- [ ] All 30 features are properly validated
- [ ] Error messages are user-friendly
- [ ] README documentation is complete
- [ ] Unit tests pass
- [ ] Team has reviewed the code
- [ ] Performance expectations are documented
- [ ] Privacy concerns have been addressed

---

## 📊 Quick Reference

### Prediction Performance

```
Input: 30 student features
Process: Random Forest model (100 trees voting)
Output: Performance category + recommendations

Accuracy: 48% (better than random 33%)
Categories: At Risk | Average | High Performance
```

### Data Characteristics

```
Students: 1,044 total (395 Math + 649 Portuguese)
Features: 30 (demographics, academic, lifestyle)
Missing Values: 0 (clean data)
Duplicates: 0 (validated)
Training/Test Split: 80/20
```

### System Requirements

```
Python: 3.8 or higher
RAM: 2GB minimum (4GB recommended)
Storage: 500MB (with all data and models)
Time to Run Full Pipeline: ~5-10 minutes
```

---

## 🎓 Learning Resources

This project demonstrates:
- **Data Science Workflow** - From raw data to predictions
- **Feature Engineering** - Preparing data for ML
- **Model Selection** - Comparing algorithms
- **Model Evaluation** - Measuring performance
- **Production Code** - Error handling, validation
- **Testing** - Unit tests and validation
- **Documentation** - Writing clear technical docs

---

## 📋 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Sept 2024 | Initial release - core functionality complete |

---

## 🙏 Acknowledgments

- Dataset source: UCI Machine Learning Repository
- Team collaboration: All 5 team members
- Guidance: Machine Learning instructors

---

**Last Updated:** September 2, 2026  
**Status:** Active Development  
**Next Review:** End of September 2026

---

*For the latest version and updates, refer to the main project repository.*
