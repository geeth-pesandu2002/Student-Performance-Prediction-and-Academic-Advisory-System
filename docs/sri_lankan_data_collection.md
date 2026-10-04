# Sri Lankan University Data Collection

The current production model is still a prototype trained on the bundled UCI secondary-school data. This document defines the data needed before replacing it with a Sri Lankan university model.

## Collection rules

- Collect only with approval from the university or project supervisor.
- Explain the purpose and obtain informed consent.
- Do not collect names, registration numbers, phone numbers, addresses, or free-text personal stories.
- Store the data securely and restrict access to the project team.
- Use an anonymous row identifier only if duplicate checking is needed.
- Allow students to skip optional wellbeing or financial questions.
- Report model accuracy separately for faculties and programmes where the sample size allows it.

## Template columns

The empty template is `data/raw/sri_lankan_university_template.csv`. Use one row per anonymized student-semester.

- `academic_year`, `semester`: academic period.
- `faculty`, `programme`: broad academic grouping, not a student identity.
- `previous_gpa`: GPA before the current semester, on a 0.0-4.0 scale.
- `attendance_percentage`: percentage of scheduled classes attended.
- `study_hours_per_week`: approximate independent study hours.
- `failed_modules`: modules failed before the current semester.
- `assignment_completion_percentage`: completed assignments as a percentage.
- `assessment_average_percentage`: average mark from assessments available before prediction.
- `lecture_participation_percentage`: estimated participation in learning activities.
- `tutorial_participation`: `yes` or `no`.
- `lms_active`: `yes` or `no` based on meaningful LMS activity.
- `internet_access`: `yes`, `limited`, or `no`.
- `financial_work_pressure`: `yes` or `no`; keep this optional and avoid asking for income.
- `wellbeing_rating`: optional self-rating from 1 to 5.
- `semester_gpa`: final outcome used only to create the training label, never as a prediction input.

## Labels

The initial labels are deliberately simple and should be reviewed with an academic advisor:

- `At Risk`: semester GPA below 2.0
- `Average`: semester GPA from 2.0 up to 3.0
- `High Performance`: semester GPA of 3.0 or above

These thresholds are not universal rules. Confirm them against the grading policy of the target university before deployment.