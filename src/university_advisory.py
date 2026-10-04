"""Recommendations based on Sri Lankan university academic indicators."""


def _item(number, action, detail, impact, timeline):
    return {
        "priority": number,
        "action": action,
        "detail": detail,
        "impact_points": impact,
        "effort_level": "Manageable",
        "timeline": timeline,
        "success_probability": 0.75,
    }


def generate_advisory(student_data, prediction):
    priorities = []
    if student_data["failed_modules"] > 0:
        priorities.append(_item(1, "Create a failed-module recovery plan", "Meet the lecturer or advisor and divide each weak topic into weekly targets.", 5, "This week"))
    if student_data["attendance_percentage"] < 80:
        priorities.append(_item(2, "Raise attendance before the next assessment", "Set an attendance target and catch up on notes or recordings after every missed class.", 4, "Next 2 weeks"))
    if student_data["assignment_completion_percentage"] < 80:
        priorities.append(_item(3, "Finish assignments earlier", "Break each assignment into research, draft, and review blocks with an earlier personal deadline.", 4, "Before the next deadline"))
    if student_data["study_hours_per_week"] < 10:
        priorities.append(_item(4, "Follow a five-block weekly timetable", "Schedule five focused 60-minute study blocks and start with the module needing the most support.", 4, "Start today"))
    if student_data["assessment_average_percentage"] < 50:
        priorities.append(_item(5, "Use active assessment practice", "Review mistakes and complete practice questions within 24 hours of each lecture.", 3, "Before the next test"))
    if student_data["lms_active"] != "yes":
        priorities.append(_item(6, "Use the learning management system weekly", "Check announcements, download resources, and complete available quizzes every week.", 2, "Within 7 days"))
    if not priorities:
        priorities.append(_item(1, "Maintain and stretch your current routine", "Add one practice session for the subject closest to your target grade.", 2, "Ongoing"))

    critical = []
    warnings = []
    strengths = []
    if student_data["failed_modules"] > 0:
        critical.append({"factor": "failed_modules", "description": "Previous failed modules need a recovery plan."})
    if student_data["attendance_percentage"] < 70:
        critical.append({"factor": "attendance", "description": "Low attendance is putting learning continuity at risk."})
    elif student_data["attendance_percentage"] < 80:
        warnings.append({"factor": "attendance", "description": "Attendance should improve before the next assessment."})
    if student_data["assignment_completion_percentage"] < 80:
        warnings.append({"factor": "assignments", "description": "Completing more assignments will create more practice evidence."})
    if student_data["assessment_average_percentage"] >= 70:
        strengths.append({"factor": "assessment_average", "description": "Assessment performance is a useful foundation."})
    if student_data["attendance_percentage"] >= 85:
        strengths.append({"factor": "attendance", "description": "Consistent attendance supports learning continuity."})

    timetable = [
        "Monday: 60 min review of the hardest module",
        "Tuesday: 60 min assignment or tutorial work",
        "Thursday: 60 min practice questions and error review",
        "Saturday: 90 min revision and next-week planning",
    ]
    summary = (f"The model places this student in the {prediction} category. "
               f"The clearest route toward High Performance is to complete the top "
               f"{min(3, len(priorities))} actions consistently, starting with the first one.")
    return {
        "prediction": prediction,
        "analysis": {"critical_issues": critical, "warnings": warnings, "strengths": strengths,
                     "opportunities": [{"factor": "academic_plan", "description": item["detail"]} for item in priorities[:3]]},
        "impact_analysis": {"priorities": priorities[:3], "total_potential": sum(item["impact_points"] for item in priorities[:3]), "scenarios": []},
        "resources": [],
        "weekly_timetable": timetable,
        "success_probability": min(95, 55 + len(strengths) * 10 - len(critical) * 8),
        "summary": summary,
    }