"""Feature contract for the future Sri Lankan university model."""

LOCAL_NUMERIC_FEATURES = [
    "academic_year",
    "semester",
    "previous_gpa",
    "attendance_percentage",
    "study_hours_per_week",
    "failed_modules",
    "assignment_completion_percentage",
    "assessment_average_percentage",
    "lecture_participation_percentage",
    "wellbeing_rating",
]

LOCAL_CATEGORICAL_FEATURES = [
    "faculty",
    "programme",
    "tutorial_participation",
    "lms_active",
    "internet_access",
    "financial_work_pressure",
]

LOCAL_FEATURES = LOCAL_NUMERIC_FEATURES + LOCAL_CATEGORICAL_FEATURES


def categorize_gpa(gpa):
    """Map a 4.0-scale semester GPA to the product's three categories."""
    if gpa < 2.0:
        return "At Risk"
    if gpa < 3.0:
        return "Average"
    return "High Performance"