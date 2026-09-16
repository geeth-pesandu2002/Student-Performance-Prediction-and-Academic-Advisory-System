import unittest

from src.predictor import predict_student_performance


class TestPredictor(unittest.TestCase):
    def setUp(self):
        self.valid_student = {
            "academic_year": 2, "semester": 1, "faculty": "Computing",
            "programme": "Computer Science", "previous_gpa": 2.6,
            "attendance_percentage": 82, "study_hours_per_week": 10,
            "failed_modules": 0, "assignment_completion_percentage": 78,
            "assessment_average_percentage": 58,
            "lecture_participation_percentage": 78,
            "tutorial_participation": "yes", "lms_active": "yes",
            "internet_access": "yes", "financial_work_pressure": "no",
            "wellbeing_rating": 3,
        }

    def test_valid_prediction_returns_known_label(self):
        prediction = predict_student_performance(self.valid_student)
        self.assertIn(
            prediction,
            {"At Risk", "Average", "High Performance"},
        )

    def test_missing_features_raise_clear_error(self):
        with self.assertRaises(ValueError) as context:
            predict_student_performance({"faculty": "Computing"})

        self.assertIn("Missing required input features", str(context.exception))


if __name__ == "__main__":
    unittest.main()
