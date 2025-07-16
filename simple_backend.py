#!/usr/bin/env python3
"""
Simple backend for testing
"""
from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

app = Flask(__name__)
CORS(app)

def create_response(status='success', message=None, data=None, error=None):
    """Create standardized API response"""
    response = {
        'status': status,
        'timestamp': datetime.now().isoformat()
    }
    
    if message:
        response['message'] = message
    if data:
        response['data'] = data
    if error:
        response['error'] = error
        
    return response

@app.route('/')
def index():
    """Root endpoint"""
    return jsonify(create_response(
        status='success',
        message='CekAjaYuk Simple API is running',
        data={'version': '1.0.0'}
    ))

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify(create_response(
        status='success',
        message='API is healthy',
        data={
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        }
    ))

@app.route('/api/init')
def force_init():
    """Force initialize the application"""
    return jsonify(create_response(
        status='success',
        message='Application initialized successfully'
    ))

@app.route('/api/models/info')
def models_info():
    """Get information about loaded models"""
    return jsonify(create_response(
        status='success',
        data={
            'models_loaded': False,
            'available_models': {
                'random_forest': {'loaded': False, 'status': '❌ Not Loaded'},
                'deep_learning': {'loaded': False, 'status': '❌ Not Loaded'},
                'feature_scaler': {'loaded': False, 'status': '❌ Not Loaded'},
                'text_vectorizer': {'loaded': False, 'status': '❌ Not Loaded'}
            },
            'summary': {
                'loaded_count': 0,
                'total_count': 4,
                'load_percentage': 0.0,
                'status': 'Demo Mode'
            }
        }
    ))

@app.route('/api/test-ocr')
def test_ocr():
    """Test OCR functionality"""
    return jsonify(create_response(
        status='success',
        data={
            'tesseract_available': False,
            'version': None,
            'languages': 'ind+eng',
            'test_result': 'OCR in demo mode',
            'status': 'demo'
        }
    ))

@app.route('/api/dataset/info')
def dataset_info():
    """Get information about dataset"""
    return jsonify(create_response(
        status='success',
        data={
            'dataset_type': 'synthetic',
            'total_samples': 1000,
            'genuine_samples': 500,
            'fake_samples': 500,
            'balance_ratio': 1.0,
            'ready_for_training': True,
            'quality': 'demo',
            'last_updated': datetime.now().isoformat()
        }
    ))

@app.errorhandler(404)
def not_found(e):
    """Handle not found error"""
    return jsonify(create_response(
        status='error',
        error='Endpoint not found'
    )), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    return jsonify(create_response(
        status='error',
        error='Internal server error'
    )), 500

if __name__ == '__main__':
    print("🚀 Starting Simple CekAjaYuk Backend...")
    print("📍 Running on http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)
