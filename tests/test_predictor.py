import unittest

from src.predictor import predict_student_performance


class TestPredictor(unittest.TestCase):
    def setUp(self):
        self.valid_student = {
            "school": "GP",
            "sex": "F",
            "age": 18,
            "address": "U",
            "famsize": "GT3",
            "Pstatus": "A",
            "Medu": 4,
            "Fedu": 4,
            "Mjob": "at_home",
            "Fjob": "teacher",
            "reason": "course",
            "guardian": "mother",
            "traveltime": 2,
            "studytime": 2,
            "failures": 0,
            "schoolsup": "yes",
            "famsup": "no",
            "paid": "no",
            "activities": "no",
            "nursery": "yes",
            "higher": "yes",
            "internet": "no",
            "romantic": "no",
            "famrel": 4,
            "freetime": 3,
            "goout": 4,
            "Dalc": 1,
            "Walc": 1,
            "health": 3,
            "absences": 6,
        }

    def test_valid_prediction_returns_known_label(self):
        prediction = predict_student_performance(self.valid_student)
        self.assertIn(
            prediction,
            {"At Risk", "Average", "High Performance"},
        )

    def test_missing_features_raise_clear_error(self):
        with self.assertRaises(ValueError) as context:
            predict_student_performance({"school": "GP", "sex": "F"})

        self.assertIn("Missing required input features", str(context.exception))


if __name__ == "__main__":
    unittest.main()
