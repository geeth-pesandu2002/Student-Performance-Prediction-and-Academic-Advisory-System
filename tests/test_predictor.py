import unittest

from src.predictor import (
    predict_student_performance,
    predict_student_with_probabilities,
)


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
        self.at_risk_student = {
            "academic_year": 2,
            "semester": 1,
            "faculty": "Computing",
            "programme": "Computer Science",
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
        self.high_performance_student = {
            "academic_year": 3,
            "semester": 1,
            "faculty": "Engineering",
            "programme": "Engineering",
            "previous_gpa": 3.5,
            "attendance_percentage": 94,
            "study_hours_per_week": 18,
            "failed_modules": 0,
            "assignment_completion_percentage": 96,
            "assessment_average_percentage": 82,
            "lecture_participation_percentage": 92,
            "tutorial_participation": "yes",
            "lms_active": "yes",
            "internet_access": "yes",
            "financial_work_pressure": "no",
            "wellbeing_rating": 4,
        }

    def test_valid_prediction_returns_known_label(self):
        prediction = predict_student_performance(self.valid_student)
        self.assertIn(
            prediction,
            {"At Risk", "Average", "High Performance"},
        )

        predicted_label, probabilities = predict_student_with_probabilities(
            self.valid_student
        )
        self.assertEqual(predicted_label, prediction)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=6)

    def test_missing_features_raise_clear_error(self):
        with self.assertRaises(ValueError) as context:
            predict_student_performance({"faculty": "Computing"})

        self.assertIn("Missing required input features", str(context.exception))

    def test_empty_input_raises_value_error(self):
        for predictor in (
            predict_student_performance,
            predict_student_with_probabilities,
        ):
            with self.subTest(predictor=predictor.__name__):
                with self.assertRaises(ValueError):
                    predictor({})

    def test_invalid_input_raises_type_error(self):
        with self.assertRaises(TypeError):
            predict_student_performance(["not", "a", "student"])

        with self.assertRaises(TypeError):
            predict_student_with_probabilities("not a dictionary")

    def test_profiles_return_valid_predictions(self):
        labels = {"At Risk", "Average", "High Performance"}
        for student in (
            self.at_risk_student,
            self.valid_student,
            self.high_performance_student,
        ):
            with self.subTest(student=student):
                self.assertIn(predict_student_performance(student), labels)


if __name__ == "__main__":
    unittest.main()
