const form = document.getElementById('studentForm');
const results = document.getElementById('results');
const emptyState = document.getElementById('emptyState');
const loadingState = document.getElementById('loadingState');
const errorBox = document.getElementById('errorBox');
const numericFields = new Set(['academic_year', 'semester', 'previous_gpa', 'attendance_percentage', 'study_hours_per_week', 'failed_modules', 'assignment_completion_percentage', 'assessment_average_percentage', 'lecture_participation_percentage', 'wellbeing_rating']);

const profiles = {
    average: { academic_year: 2, semester: 1, faculty: 'Computing', programme: 'Computer Science', previous_gpa: 2.6, attendance_percentage: 82, study_hours_per_week: 10, failed_modules: 0, assignment_completion_percentage: 78, assessment_average_percentage: 58, lecture_participation_percentage: 78, tutorial_participation: 'yes', lms_active: 'yes', internet_access: 'yes', financial_work_pressure: 'no', wellbeing_rating: 3 },
    risk: { academic_year: 2, semester: 1, faculty: 'Computing', programme: 'Computer Science', previous_gpa: 1.7, attendance_percentage: 62, study_hours_per_week: 5, failed_modules: 2, assignment_completion_percentage: 48, assessment_average_percentage: 35, lecture_participation_percentage: 55, tutorial_participation: 'no', lms_active: 'no', internet_access: 'limited', financial_work_pressure: 'yes', wellbeing_rating: 2 },
    high: { academic_year: 3, semester: 1, faculty: 'Engineering', programme: 'Engineering', previous_gpa: 3.5, attendance_percentage: 94, study_hours_per_week: 18, failed_modules: 0, assignment_completion_percentage: 96, assessment_average_percentage: 82, lecture_participation_percentage: 92, tutorial_participation: 'yes', lms_active: 'yes', internet_access: 'yes', financial_work_pressure: 'no', wellbeing_rating: 4 }
};

for (const button of document.querySelectorAll('[data-demo]')) {
    button.addEventListener('click', () => fillForm(profiles[button.dataset.demo]));
}
document.getElementById('resetButton').addEventListener('click', resetDashboard);
form.addEventListener('submit', submitPrediction);

function fillForm(profile) {
    Object.entries(profile).forEach(([name, value]) => {
        const field = form.elements[name];
        if (field) field.value = value;
    });
    errorBox.classList.remove('show');
    form.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function resetDashboard() {
    form.reset();
    results.hidden = true;
    loadingState.hidden = true;
    emptyState.hidden = false;
    errorBox.classList.remove('show');
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

async function submitPrediction(event) {
    event.preventDefault();
    if (!form.reportValidity()) return;
    const data = Object.fromEntries(new FormData(form).entries());
    Object.keys(data).forEach((key) => { if (numericFields.has(key)) data[key] = Number(data[key]); });
    setLoading(true);
    try {
        const response = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
        const payload = await response.json();
        if (!response.ok || payload.status !== 'success') throw new Error(payload.message || 'The prediction could not be completed.');
        renderResults(payload);
    } catch (error) {
        errorBox.textContent = error.message;
        errorBox.classList.add('show');
    } finally { setLoading(false); }
}

function setLoading(isLoading) {
    emptyState.hidden = isLoading;
    results.hidden = isLoading;
    loadingState.hidden = !isLoading;
    if (isLoading) errorBox.classList.remove('show');
}

function renderResults(data) {
    emptyState.hidden = true;
    results.hidden = false;
    const label = data.prediction || 'Unknown';
    const badge = document.getElementById('predictionBadge');
    const hero = document.getElementById('resultHero');
    document.getElementById('predictionLabel').textContent = label;
    badge.textContent = label === 'High Performance' ? '↑' : label === 'Average' ? '→' : '↓';
    badge.className = `result-badge ${label === 'At Risk' ? 'risk' : label === 'Average' ? 'average' : ''}`;
    hero.dataset.state = label;
    const modelProbability = Number(data.model_probability || 0);
    document.getElementById('modelProbability').textContent = `${modelProbability.toFixed(1)}%`;
    document.getElementById('confidenceBar').style.width = `${modelProbability}%`;
    const probabilityLabels = [['At Risk', 'risk'], ['Average', 'average'], ['High Performance', 'high']];
    document.getElementById('classProbabilities').innerHTML = probabilityLabels.map(([name, key]) => `<div>${name}<strong>${Number((data.model_probabilities || {})[name] || 0).toFixed(1)}%</strong></div>`).join('');
    document.getElementById('summaryText').textContent = data.summary || 'No summary returned.';
    const analysis = data.analysis || {};
    const items = [['critical_issues', 'Critical'], ['warnings', 'Warnings'], ['strengths', 'Strengths'], ['opportunities', 'Opportunities']];
    document.getElementById('analysisGrid').innerHTML = items.map(([key, name]) => `<div class="analysis-item"><strong>${analysis[key] || 0}</strong><span>${name}</span></div>`).join('');
    document.getElementById('analysisTotal').textContent = `${Object.values(analysis).reduce((sum, value) => sum + Number(value || 0), 0)} signals`;
    const priorities = data.top_priorities || [];
    document.getElementById('priorityCount').textContent = `${priorities.length} ${priorities.length === 1 ? 'priority' : 'priorities'}`;
    document.getElementById('priorities').innerHTML = priorities.length ? priorities.map((priority, index) => `<div class="priority"><span class="priority-number">0${index + 1}</span><div><p>${priority.action}</p><small>${priority.detail || 'Build this habit consistently.'}</small><small class="priority-meta">${priority.timeline || 'Ongoing'} · +${priority.impact_points} potential points</small></div></div>`).join('') : '<div class="priority"><span class="priority-number">✓</span><div><p>No urgent priorities.</p><small>Keep the current rhythm going.</small></div></div>';
    const timetable = data.weekly_timetable || [];
    document.getElementById('timetable').innerHTML = timetable.map((session, index) => `<div class="timetable-row"><span>0${index + 1}</span><p>${session}</p></div>`).join('');
}
