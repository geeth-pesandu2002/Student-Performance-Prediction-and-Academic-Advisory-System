"""
Test Flask Backend with Multiple Student Profiles
Run this to verify the backend handles different student types correctly
"""
 
import requests
 
BASE_URL = "http://localhost:5000"
 
# Test Profile 1: Average Student (Already tested above)
student_average = {
    "academic_year": 2, "semester": 1, "faculty": "Computing", "programme": "Computer Science",
    "previous_gpa": 2.6, "attendance_percentage": 82, "study_hours_per_week": 10,
    "failed_modules": 0, "assignment_completion_percentage": 78,
    "assessment_average_percentage": 58, "lecture_participation_percentage": 78,
    "tutorial_participation": "yes", "lms_active": "yes", "internet_access": "yes",
    "financial_work_pressure": "no", "wellbeing_rating": 3
}
 
# Test Profile 2: At Risk Student
student_at_risk = {
    "academic_year": 2, "semester": 1, "faculty": "Computing", "programme": "Computer Science",
    "previous_gpa": 1.7, "attendance_percentage": 62, "study_hours_per_week": 5,
    "failed_modules": 2, "assignment_completion_percentage": 48,
    "assessment_average_percentage": 35, "lecture_participation_percentage": 55,
    "tutorial_participation": "no", "lms_active": "no", "internet_access": "limited",
    "financial_work_pressure": "yes", "wellbeing_rating": 2
}
 
# Test Profile 3: High Performer
student_high_performer = {
    "academic_year": 3, "semester": 1, "faculty": "Engineering", "programme": "Engineering",
    "previous_gpa": 3.5, "attendance_percentage": 94, "study_hours_per_week": 18,
    "failed_modules": 0, "assignment_completion_percentage": 96,
    "assessment_average_percentage": 82, "lecture_participation_percentage": 92,
    "tutorial_participation": "yes", "lms_active": "yes", "internet_access": "yes",
    "financial_work_pressure": "no", "wellbeing_rating": 4
}
 
students = [
    ("Average Student", student_average),
    ("At Risk Student", student_at_risk),
    ("High Performer", student_high_performer)
]
 
def run_smoke_tests():
    """Exercise the running Flask server with representative profiles."""
    print("=" * 70)
    print("FLASK BACKEND TEST - Multiple Student Profiles")
    print("=" * 70)

    for profile_name, student_data in students:
        print(f"\n{'=' * 70}")
        print(f"Testing: {profile_name}")
        print(f"{'=' * 70}")

        try:
            response = requests.post(
                f"{BASE_URL}/api/predict",
                json=student_data,
                timeout=10,
            )
            print(f"Status Code: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"Prediction: {data['prediction']}")
                print(f"Success Probability: {data['success_probability']}%")
                print(f"Summary: {data['summary'][:100]}...")
            else:
                print(f"Error: {response.json()}")
        except requests.RequestException as error:
            print(f"Request failed: {error}")
            print("Make sure Flask server is running: python app.py")


if __name__ == "__main__":
    run_smoke_tests()