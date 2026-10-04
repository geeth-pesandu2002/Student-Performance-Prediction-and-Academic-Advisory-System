"""Generate synthetic Sri Lankan university student records for prototype development.

Lead Developer: MFM Amhar (Machine Learning Model Development)

Methodological Purpose & Disclaimer:
------------------------------------
This module produces simulated undergraduate student academic records for assignment
development, architectural prototyping, and pipeline validation. These records are
synthetically generated and must NEVER be presented as empirical research findings
or real student data from any higher education institution.

Generative Statistical Design & Correlation Modeling:
------------------------------------------------------
The data generation process reflects realistic pedagogical and socioeconomic dynamics
observed in Sri Lankan university settings:
1. Faculties & Degree Programmes:
   - Covers 5 faculties (Computing, Engineering, Management, Science, Humanities)
   - Realistic degree programmes mapped to respective faculties.
2. Academic Study & Engagement Dynamics:
   - Weekly Study Hours: Modeled as Normal(mu=12, sigma=5) clipped to [2, 30] hours.
   - Attendance Percentage: Positively correlated with study hours, bounded in [45, 100]%.
   - Continuous Assessments & Assignments: Correlated with study habits and prior GPA.
   - Failed Modules: Modeled using a Poisson process parameterized inversely with prior GPA.
3. Target Variable (semester_gpa):
   - Formulated as a multi-factor linear combination of previous GPA, attendance,
     study hours, assignment completion, and continuous assessment scores, offset by
     failed modules and calibrated with Gaussian noise Normal(mu=-1.25, sigma=0.38).
   - Clipped to standard GPA boundaries [0.0, 4.0].
4. Socioeconomic & Wellbeing Indicators:
   - LMS activity, tutorial attendance, home internet access, work/financial pressure,
     and subjective wellbeing ratings (1 to 5 scale).

Reproducibility Protocol:
-------------------------
Deterministic reproducibility is guaranteed by initializing NumPy's modern PCG64
generator (`np.random.default_rng(seed=42)`). Given the same seed, the exact same 360
student records and distributions are reproduced across all platforms and operating systems.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "sri_lankan_university.csv"


def generate_records(record_count: int = 360, seed: int = 42) -> pd.DataFrame:
    """Generate a DataFrame of synthetic university student academic records.

    Args:
        record_count: Number of student records to synthesize (default: 360).
        seed: Random seed for deterministic generation via NumPy PCG64 (default: 42).

    Returns:
        pd.DataFrame: Tabular student records containing 16 predictor features and `semester_gpa`.
    """
    rng = np.random.default_rng(seed)
    faculties = ["Computing", "Engineering", "Management", "Science", "Humanities"]
    programmes = {
        "Computing": ["Information Technology", "Computer Science"],
        "Engineering": ["Engineering"],
        "Management": ["Business Management", "Accounting"],
        "Science": ["Applied Science", "Biological Science"],
        "Humanities": ["Languages", "Social Sciences"],
    }
    rows = []
    for _ in range(record_count):
        faculty = rng.choice(faculties)
        study_hours = int(np.clip(rng.normal(12, 5), 2, 30))
        attendance = float(np.clip(rng.normal(78 + study_hours * 0.35, 10), 45, 100))
        previous_gpa = float(np.clip(rng.normal(2.55 + (attendance - 75) / 100, 0.55), 0, 4))
        failed_modules = int(np.clip(rng.poisson(max(0.15, 1.1 - previous_gpa / 3)), 0, 4))
        assignment_completion = float(np.clip(rng.normal(65 + study_hours * 1.5, 12), 25, 100))
        assessment_average = float(np.clip(rng.normal(48 + previous_gpa * 10 + study_hours * 0.8, 12), 20, 95))
        lecture_participation = float(np.clip(rng.normal(attendance - 5, 12), 20, 100))
        semester_gpa = float(np.clip(
            0.25 * previous_gpa
            + 0.012 * attendance
            + 0.025 * study_hours
            + 0.012 * assignment_completion
            + 0.01 * assessment_average
            - 0.22 * failed_modules
            + rng.normal(-1.25, 0.38),
            0,
            4,
        ))
        rows.append({
            "academic_year": int(rng.choice([1, 2, 3, 4])),
            "semester": int(rng.choice([1, 2])),
            "faculty": faculty,
            "programme": rng.choice(programmes[faculty]),
            "previous_gpa": round(previous_gpa, 2),
            "attendance_percentage": round(attendance, 1),
            "study_hours_per_week": study_hours,
            "failed_modules": failed_modules,
            "assignment_completion_percentage": round(assignment_completion, 1),
            "assessment_average_percentage": round(assessment_average, 1),
            "lecture_participation_percentage": round(lecture_participation, 1),
            "tutorial_participation": rng.choice(["yes", "no"], p=[0.65, 0.35]),
            "lms_active": rng.choice(["yes", "no"], p=[0.8, 0.2]),
            "internet_access": rng.choice(["yes", "limited", "no"], p=[0.75, 0.2, 0.05]),
            "financial_work_pressure": rng.choice(["yes", "no"], p=[0.2, 0.8]),
            "wellbeing_rating": int(rng.choice([1, 2, 3, 4, 5], p=[0.04, 0.12, 0.34, 0.35, 0.15])),
            "semester_gpa": round(semester_gpa, 2),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    data = generate_records()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(data)} synthetic records at {OUTPUT_PATH}")