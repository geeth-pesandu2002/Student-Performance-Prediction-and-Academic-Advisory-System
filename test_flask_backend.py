"""Unit and smoke tests for the Flask prediction backend."""

import unittest

import requests

from app import app

BASE_URL = "http://localhost:5000"
LABELS = {"At Risk", "Average", "High Performance"}

student_average = {
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
    "wellbeing_rating": 3,
}

student_at_risk = {
    **student_average,
    "previous_gpa": 1.7,
    "attendance_percentage": 62,
    "study_hours_per_week": 5,
    "failed_modules": 2,
    "assignment_completion_percentage": 48,
    "assessment_average_percentage": 35,
    "lecture_participation_percentage": 55,
    "tutorial_participation": "no",
    "lms_active": "no",
    "internet_access": "limited",
    "financial_work_pressure": "yes",
    "wellbeing_rating": 2,
}

student_high_performer = {
    **student_average,
    "academic_year": 3,
    "faculty": "Engineering",
    "programme": "Engineering",
    "previous_gpa": 3.5,
    "attendance_percentage": 94,
    "study_hours_per_week": 18,
    "assignment_completion_percentage": 96,
    "assessment_average_percentage": 82,
    "lecture_participation_percentage": 92,
    "wellbeing_rating": 4,
}


class TestFlaskBackend(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.valid_student = student_average

    def test_health_check(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "healthy")

    def test_predict_missing_fields_returns_400(self):
        response = self.client.post("/api/predict", json={"faculty": "Computing"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Missing required fields", response.get_json()["message"])

    def test_predict_empty_input_returns_400(self):
        response = self.client.post("/api/predict", json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("No data received", response.get_json()["message"])

    def test_predict_invalid_input_returns_error(self):
        invalid_student = self.valid_student.copy()
        invalid_student["previous_gpa"] = "not-a-number"

        response = self.client.post("/api/predict", json=invalid_student)
        self.assertEqual(response.status_code, 500)
        self.assertIn("Prediction failed", response.get_json()["message"])

    def test_successful_prediction_response(self):
        response = self.client.post("/api/predict", json=self.valid_student)
        self.assertEqual(response.status_code, 200)

        payload = response.get_json()
        self.assertEqual(payload["status"], "success")
        self.assertIn(payload["prediction"], LABELS)
        self.assertGreater(payload["model_probability"], 0)
        self.assertIn("summary", payload)

    def test_prediction_profiles_return_valid_labels(self):
        for student in (
            student_at_risk,
            student_average,
            student_high_performer,
        ):
            with self.subTest(student=student):
                response = self.client.post("/api/predict", json=student)
                self.assertEqual(response.status_code, 200)
                self.assertIn(response.get_json()["prediction"], LABELS)


def run_smoke_tests():
    """Exercise a separately running Flask server with sample profiles."""
    students = [
        ("Average Student", student_average),
        ("At Risk Student", student_at_risk),
        ("High Performer", student_high_performer),
    ]
    for profile_name, student_data in students:
        try:
            response = requests.post(
                f"{BASE_URL}/api/predict",
                json=student_data,
                timeout=10,
            )
            print(f"{profile_name}: {response.status_code}")
        except requests.RequestException as error:
            print(f"{profile_name}: request failed: {error}")


if __name__ == "__main__":
    run_smoke_tests()
