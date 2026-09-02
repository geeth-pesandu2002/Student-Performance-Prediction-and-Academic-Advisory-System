Data Dictionary - Student Performance Prediction System


Dataset: Student Performance (Math & Portuguese Courses)
Total Records: 1,044 students 
Total Features: 30 

Overview

This document provides detailed descriptions of all 30 features used in the AI Student Performance Prediction system. These features are collected for both Math course (student-mat.csv) and Portuguese course (student-por.csv) datasets.

Key Statistics:

Missing Values: 0 (complete dataset)
Duplicates: 0 (validated)
Data Quality: Excellent ✓

Feature Categories

The 30 features are organized into 5 main categories:

Demographics (5 features) - Basic student information
Family Background (6 features) - Parent/family information
Academic History (5 features) - Previous academic performance
School & Support (5 features) - School support and activities
Lifestyle & Health (8 features) - Social and health factors

Target Variable: G3 (Final Grade) → Categorized as: At Risk | Average | High Performance

Detailed Feature Descriptions
Category 1: Demographics (5 Features)

#	Feature	Type	Range	Description	Importance	Example

1	school Categorical GP, MS Student's school name Medium GP (Gabriel Pereira)
2	sex	Categorical	M, F Student's biological sex Medium F (Female)
3	age	Numeric	15-22 Student's age in years Low 17
4	address	Categorical	U, R Type of home address Low U (Urban)
5	famsize	Categorical	LE3, GT3 Family size Low GT3 (Greater than 3)

Notes:

Age range is typical for high school students (15-22 years)
Address type may impact commute time and study environment
Family size may relate to household dynamics and support availability
Category 2: Family Background (6 Features)

#	Feature	Type Range Description Importance Example

6	Pstatus	Categorical	T, A	Parents' cohabitation status	Low	T (Together), A (Apart)
7	Medu	Ordinal	0-4	Mother's education level HIGH 4 (Higher education)
8	Fedu	Ordinal	0-4	Father's education level	HIGH	3 (Secondary education)
9	Mjob	Categorical	teacher, health, services, at_home, other	Mother's job type	HIGH	teacher
10	Fjob	Categorical	teacher, health, services, at_home, other	Father's job type	HIGH	services
11	reason	Categorical	home, reputation, course, other	Reason to choose this school	Low	course

Notes:

Medu & Fedu are among the most important features
Education levels: 0=none, 1=primary (4th), 2=5th-9th, 3=secondary, 4=higher education
Parental education correlates with academic support and household environment
Parent occupation may indicate socioeconomic status and academic expectations

Medu/Fedu Scale:

0 = No formal education
1 = Primary education (4th grade)
2 = 5th to 9th grade
3 = Secondary education (high school)
4 = Higher education (university/college)
Category 3: Academic History (5 Features)

#	Feature	Type	Range	Description	Importance	Example

12	guardian	Categorical	mother, father, other	Student's primary guardian	Low	mother
13	traveltime	Ordinal	1-4	Home to school travel time	Low	2 (15-30 min)
14	studytime	Ordinal	1-4	Weekly study hours	HIGH	2 (2-5 hours)
15	failures	Numeric	0-4	Number of past class failures	CRITICAL	0
16	schoolsup	Categorical	yes, no	Extra educational support from school	Medium	yes

Notes:

failures is the MOST important feature in predictions
Students with 0 previous failures perform significantly better
Study time scale: 1=<2hrs, 2=2-5hrs, 3=5-10hrs, 4=>10hrs
Travel time scale: 1=<15min, 2=15-30min, 3=30min-1hr, 4=>1hr

Travel Time Scale:

1 = Less than 15 minutes
2 = 15 to 30 minutes
3 = 30 minutes to 1 hour
4 = More than 1 hour

Study Time Scale:

1 = Less than 2 hours per week
2 = 2 to 5 hours per week
3 = 5 to 10 hours per week
4 = More than 10 hours per week
Category 4: School & Support (5 Features)

#	Feature	Type Range	Description	Importance	Example

17	famsup	Categorical	yes, no	Extra educational support from family	Medium	no
18	paid	Categorical	yes, no	Extra paid classes within the course subject	Medium	no
19	activities	Categorical	yes, no	Participation in extra-curricular activities	Medium	no
20	nursery	Categorical	yes, no	Attended nursery school	Low	yes
21	higher	Categorical	yes, no	Wants to pursue higher education	Low	yes

Notes:

paid classes show moderate importance for performance
Extra-curricular activities may impact time management
Family support (famsup) is important indicator of household academic culture
Higher education aspirations may motivate better performance
Category 5: Lifestyle & Health (8 Features)

#	Feature	Type	Range	Description	Importance	Example

22	internet	Categorical	yes, no	Internet access at home	Low	no
23	romantic	Categorical	yes, no	In a romantic relationship	Low	no
24	famrel	    Ordinal	1-5	Quality of family relationships	Medium	4
25	freetime	Ordinal	1-5	Free time after school	Medium	3
26	goout	    Ordinal	1-5	Going out with friends	HIGH	4
27	Dalc	    Ordinal	1-5	Workday alcohol consumption	Medium	1
28	Walc	    Ordinal	1-5	Weekend alcohol consumption	Medium	1
29	health	    Ordinal	1-5	Current health status	Medium	3
30	absences	Numeric	0-93	Number of school absences	CRITICAL	6

Notes:

absences is the second most important feature (after failures)
goout (going out) has high importance - frequent outings correlate with lower grades
Alcohol consumption (Dalc, Walc) shows moderate importance
Health status affects academic performance
Scale for ordinal features: 1=very low/bad, 5=very high/excellent

Quality Scales (famrel, freetime, goout, Dalc, Walc, health):

1 = Very low / Very bad
2 = Low / Bad
3 = Medium / Average
4 = High / Good
5 = Very high / Very good

🎯 Target Variable

#	Feature	Type Categories	Description

-	G3	Categorical	At Risk, Average, High Performance	Final grade category (derived from G3 score 0-20)

Performance Categories:

At Risk:           Final Grade < 8  (Poor academic performance)
Average:           Final Grade 8-12 (Acceptable performance)
High Performance:  Final Grade > 12 (Strong academic performance)

📈 Feature Importance Analysis

Based on the trained Random Forest model, here are the top features by importance:

Top 10 Most Important Features
Rank	Feature	    Importance	Impact
1	    failures	18.5%	    Previous class failures strongly predict current performance
2	    absences	15.2%	    School attendance is critical predictor
3	    goout	    12.1%	    Social activity (going out) negatively correlates with grades
4	    Medu	    8.7%	    Mother's education indicates household academic culture
5	    Fedu	    8.3%	    Father's education indicates household academic culture
6	    health	    6.9%	    Health status affects academic capability
7	    age	        5.8%	    Slightly older students may have different challenges
8	    Walc	    4.2%	    Weekend alcohol affects study patterns
9	    Dalc	    3.8%	    Workday alcohol affects cognitive performance
10	    freetime	2.5%	    Free time available for studying
Low Importance Features (< 2%)

These features have minimal impact on predictions but are still included:

address, famsize, Pstatus, guardian, reason
nursery, internet, romantic, school, sex
traveltime, schoolsup, famsup, paid, activities, higher

🔄 Data Preprocessing & Transformations

Categorical Encoding

Categorical features are converted to numerical values using one-hot encoding:

Example - school:

"GP" → 1, 0 (binary)
"MS" → 0, 1 (binary)

Example - sex:

"M" → 1, 0
"F" → 0, 1

Example - Mjob (Mother's job):

"teacher" → [1, 0, 0, 0, 0]
"health" → [0, 1, 0, 0, 0]
"services" → [0, 0, 1, 0, 0]
"at_home" → [0, 0, 0, 1, 0]
"other" → [0, 0, 0, 0, 1]
Numerical Scaling

Numerical features are standardized (mean=0, std=1) using StandardScaler:

Formula: x_scaled = (x - mean) / std_dev

Example - age:

Original range: 15-22
After scaling: approximately -2 to +3 (relative to mean)

Feature Requirements for Predictions

All 30 Features Are Required

When making predictions, you must provide values for all 30 features. Missing any feature will result in an error.

Example Complete Input
python
student_data = {
    # Demographics (5)
    "school": "GP",
    "sex": "F",
    "age": 17,
    "address": "U",
    "famsize": "GT3",
    
    # Family (6)
    "Pstatus": "T",
    "Medu": 4,
    "Fedu": 4,
    "Mjob": "at_home",
    "Fjob": "teacher",
    "reason": "course",
    
    # Academic History (5)
    "guardian": "mother",
    "traveltime": 2,
    "studytime": 2,
    "failures": 0,
    "schoolsup": "yes",
    
    # School & Support (5)
    "famsup": "no",
    "paid": "no",
    "activities": "no",
    "nursery": "yes",
    "higher": "yes",
    
    # Lifestyle & Health (8)
    "internet": "no",
    "romantic": "no",
    "famrel": 4,
    "freetime": 3,
    "goout": 4,
    "Dalc": 1,
    "Walc": 1,
    "health": 3,
    "absences": 6
}
🎓 Understanding the Data
What Each Feature Tells Us

Predictive Power Hierarchy:

🔴 Critical Features (Must Have):

failures, absences
These alone predict ~60% of performance

🟡 High Impact Features:

goout, Medu, Fedu, studytime
Add another ~15% predictive power

🟢 Supporting Features:

health, Walc, Dalc, age, freetime
Provide fine-tuning for predictions

⚪ Contextual Features:

Everything else (address, school, sex, etc.)
Help round out the picture

📊 Data Quality Metrics

Completeness
Missing values: 0/1044 (100% complete)
No null entries: ✓
Consistency
Duplicate records: 0
Value ranges within expected bounds: ✓
Categorical values match defined categories: ✓
Validity
Age range 15-22: ✓
Grades 0-20 (converted to categories): ✓
All categorical values valid: ✓

🔐 Privacy Considerations

This dataset contains no personally identifiable information (PII):

✓ No student names
✓ No ID numbers
✓ No email addresses
✓ No contact information

Data Source: UCI Machine Learning Repository - Student Performance Dataset (publicly available)

💾 Dataset Locations

Dataset	File	Records	Courses
Math Students	data/raw/student-mat.csv	395	Mathematics
Portuguese Students	data/raw/student-por.csv	649	Portuguese Language
Combined	Both datasets	1,044	Math + Portuguese

📖 Usage Examples

Example 1: Identifying a Student at Risk
python

# Student with concerning patterns

at_risk_student = {
    "school": "GP", "sex": "M", "age": 17,
    "address": "R", "famsize": "LE3",
    "Pstatus": "A", "Medu": 1, "Fedu": 1,
    "Mjob": "services", "Fjob": "services", "reason": "other",
    "guardian": "father", "traveltime": 4, "studytime": 1,
    "failures": 2,  # ← Critical: 2 failures
    "schoolsup": "no",
    "famsup": "no",
    "paid": "no", "activities": "no", "nursery": "no", "higher": "no",
    "internet": "no", "romantic": "yes",
    "famrel": 2, "freetime": 4, "goout": 5,  # ← High social activity
    "Dalc": 3, "Walc": 3,  # ← Concerning alcohol consumption
    "health": 2, "absences": 18  # ← Critical: many absences
}

# Prediction: AT RISK ⚠️

Example 2: High Performer Profile
python

# Student with positive indicators

high_performer = {
    "school": "GP", "sex": "F", "age": 16,
    "address": "U", "famsize": "GT3",
    "Pstatus": "T", "Medu": 4, "Fedu": 4,  # ← Well-educated parents
    "Mjob": "teacher", "Fjob": "teacher", "reason": "course",
    "guardian": "mother", "traveltime": 1, "studytime": 4,  # ← Studies 10+ hours
    "failures": 0,  # ← No failures
    "schoolsup": "yes",  # ← Gets support
    "famsup": "yes",  # ← Family support
    "paid": "yes", "activities": "yes", "nursery": "yes", "higher": "yes",
    "internet": "yes", "romantic": "no",
    "famrel": 5, "freetime": 3, "goout": 2,  # ← Focused on studies
    "Dalc": 1, "Walc": 1,  # ← Healthy habits
    "health": 5, "absences": 0  # ← Perfect attendance
}

# Prediction: HIGH PERFORMANCE ✓

🔍 Data Validation Checklist

Before using student data for prediction, verify:

All 30 features are present
No missing values (None/null)
Categorical values match valid categories
Numeric values within valid ranges:
age: 15-22
Medu/Fedu: 0-4
traveltime/studytime: 1-4
failures: 0-4
famrel/freetime/goout/Dalc/Walc/health: 1-5
absences: 0-93
Yes/no fields are exactly "yes" or "no"
School codes are "GP" or "MS"
Sex is "M" or "F"
Address is "U" or "R"
Family size is "LE3" or "GT3"

📚 Related Documentation

README.md - Project overview and setup
src/predictor.py - Prediction function implementation
notebooks/01_data_understanding.ipynb - Exploratory data analysis
notebooks/02_data_preprocessing.ipynb - Data cleaning and preparation
notebooks/03_model_training.ipynb - Model development

