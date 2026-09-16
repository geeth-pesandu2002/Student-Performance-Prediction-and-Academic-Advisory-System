from flask import Flask, request, jsonify, render_template
from src.predictor import (
    predict_student_performance,
    predict_student_with_probabilities,
)
from src.university_advisory import generate_advisory
from src.sri_lankan_schema import LOCAL_FEATURES
import json

# ============================================================================
# INITIALIZE FLASK APP
# ============================================================================

app = Flask(__name__)


@app.route('/')
def index():
    """Serve the student performance dashboard."""
    return render_template('index.html')

# Allow CORS (Cross-Origin Resource Sharing) - for UI communication
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# ============================================================================
# ROUTE 1: HEALTH CHECK (Test if server is running)
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Simple endpoint to test if server is running
    
    Request: GET /api/health
    Response: {"status": "healthy"}
    """
    return jsonify({
        'status': 'healthy',
        'message': 'Student Performance Prediction API is running'
    }), 200

# ============================================================================
# ROUTE 2: PREDICT ENDPOINT (Main prediction + advisory)
# ============================================================================

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    
    Request: POST /api/predict
    Body: JSON with the 16 university-focused student fields
    
    Response: JSON with prediction + advisory
    """
    
    try:
        # Get data from request
        student_data = request.get_json()
        
        # Validate that data was received
        if not student_data:
            return jsonify({
                'status': 'error',
                'message': 'No data received. Please send student data in JSON format.'
            }), 400
        
        required_fields = LOCAL_FEATURES
        
        missing_fields = [field for field in required_fields if field not in student_data]
        
        if missing_fields:
            return jsonify({
                'status': 'error',
                'message': f'Missing required fields: {", ".join(missing_fields)}',
                'missing_count': len(missing_fields)
            }), 400
        
        # Get prediction from predictor.py
        try:
            prediction, model_probabilities = predict_student_with_probabilities(
                student_data
            )
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Prediction failed: {str(e)}'
            }), 500
        
        # Get advisory from advisory.py
        try:
            advisory = generate_advisory(student_data, prediction)
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Advisory generation failed: {str(e)}'
            }), 500
        
        # Return success response
        return jsonify({
            'status': 'success',
            'prediction': advisory['prediction'],
            'model_probability': round(
                model_probabilities.get(prediction, 0) * 100, 2
            ),
            'model_probabilities': {
                label: round(probability * 100, 2)
                for label, probability in model_probabilities.items()
            },
            'success_probability': advisory['success_probability'],
            'summary': advisory['summary'],
            'analysis': {
                'critical_issues': len(advisory['analysis']['critical_issues']),
                'warnings': len(advisory['analysis']['warnings']),
                'strengths': len(advisory['analysis']['strengths']),
                'opportunities': len(advisory['analysis']['opportunities'])
            },
            'top_priorities': [
                {
                    'priority': p['priority'],
                    'action': p['action'],
                    'detail': p['detail'],
                    'timeline': p['timeline'],
                    'impact_points': p['impact_points'],
                    'success_probability': int(p['success_probability'] * 100)
                }
                for p in advisory['impact_analysis']['priorities'][:3]
            ],
            'total_potential_improvement': advisory['impact_analysis']['total_potential'],
            'weekly_timetable': advisory['weekly_timetable'],
            'resources_count': len(advisory['resources']),
            'full_advisory': advisory  # Send complete data if needed
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Unexpected error: {str(e)}'
        }), 500

# ============================================================================
# ROUTE 3: DETAILED ADVISORY (Get complete advisory data)
# ============================================================================

@app.route('/api/advisory', methods=['POST'])
def get_detailed_advisory():
    """
    Get complete advisory data (for detailed view)
    
    Request: POST /api/advisory
    Body: JSON with 30 student fields
    
    Response: JSON with complete advisory
    """
    
    try:
        student_data = request.get_json()
        
        if not student_data:
            return jsonify({
                'status': 'error',
                'message': 'No data received'
            }), 400
        
        # Get prediction
        prediction = predict_student_performance(student_data)
        
        # Get advisory
        advisory = generate_advisory(student_data, prediction)
        
        return jsonify({
            'status': 'success',
            'advisory': advisory
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found',
        'available_endpoints': [
            'GET /api/health',
            'POST /api/predict',
            'POST /api/advisory'
        ]
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors (wrong HTTP method)"""
    return jsonify({
        'status': 'error',
        'message': 'Method not allowed'
    }), 405

# ============================================================================
# MAIN - Run the server
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("🚀 STUDENT PERFORMANCE PREDICTION API")
    print("="*70)
    print("\n📍 Server running at: http://localhost:5000")
    print("\n📋 Available endpoints:")
    print("   GET  /api/health")
    print("   POST /api/predict")
    print("   POST /api/advisory")
    print("\n💡 To test, use curl or Postman:")
    print("   curl http://localhost:5000/api/health")
    print("\n⏹️  Press CTRL+C to stop the server")
    print("="*70 + "\n")
    
    # Run the Flask app
    app.run(debug=True, host='localhost', port=5000)