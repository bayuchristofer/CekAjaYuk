"""
CekAjaYuk Backend API
Flask application untuk menangani analisis iklan lowongan kerja palsu
"""

import os
import json
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logging

# Import custom modules
from models import ModelManager
from utils import (
    allowed_file, validate_image, preprocess_image,
    extract_traditional_features, extract_text_ocr,
    create_response, safe_filename, cleanup_old_files
)
from text_analyzer import JobPostingTextAnalyzer
from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=['*'], allow_headers=['Content-Type'], methods=['GET', 'POST', 'OPTIONS'])

# Load configuration
config_name = os.environ.get('FLASK_ENV', 'development')
app.config.from_object(config[config_name])

# Initialize components
model_manager = None
text_analyzer = None

def initialize_app():
    """Initialize application components"""
    global model_manager, text_analyzer

    try:
        # Create necessary directories
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)

        # Initialize model manager
        model_manager = ModelManager(app.config['MODELS_FOLDER'])
        model_manager.load_models()

        # Initialize text analyzer
        text_analyzer = JobPostingTextAnalyzer()

        logger.info("Application initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing application: {e}")
        # Continue with limited functionality

# Routes
@app.route('/')
def index():
    """Root endpoint"""
    return jsonify(create_response(
        status='success',
        message='CekAjaYuk API is running',
        data={
            'version': '1.0.0',
            'models_loaded': model_manager.models_loaded if model_manager else False
        }
    ))

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify(create_response(
        status='success',
        message='API is healthy',
        data={
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'models_loaded': model_manager.models_loaded if model_manager else False
        }
    ))

@app.route('/api/init')
def force_init():
    """Force initialize the application"""
    try:
        initialize_app()
        return jsonify(create_response(
            status='success',
            message='Application initialized successfully'
        ))
    except Exception as e:
        return jsonify(create_response(
            status='error',
            error=f'Initialization failed: {str(e)}'
        )), 500

@app.route('/api/models/info')
def models_info():
    """Get information about loaded models"""
    global model_manager

    # Try to initialize if not already done
    if model_manager is None:
        try:
            models_dir = Path(__file__).parent.parent / 'models'
            model_manager = ModelManager(str(models_dir))
            model_manager.load_models()
        except Exception as e:
            logger.error(f"Failed to initialize model manager: {e}")

    if model_manager:
        return jsonify(create_response(
            status='success',
            data=model_manager.get_model_info()
        ))
    else:
        return jsonify(create_response(
            status='error',
            error='Model manager not initialized'
        )), 500

@app.route('/api/test-ocr')
def test_ocr():
    """Test OCR functionality"""
    try:
        # Check if Tesseract is available
        import pytesseract
        from PIL import Image
        import numpy as np

        # Create a simple test image with text
        test_image = Image.new('RGB', (200, 50), color='white')

        # Try to extract text (should return empty for blank image)
        try:
            test_text = pytesseract.image_to_string(test_image, config=r'--oem 3 --psm 6 -l ind+eng')

            # Get Tesseract version
            version = pytesseract.get_tesseract_version()

            return jsonify(create_response(
                status='success',
                data={
                    'tesseract_available': True,
                    'version': str(version),
                    'languages': 'ind+eng',
                    'test_result': 'OCR is working',
                    'supported_languages': ['ind', 'eng']
                }
            ))

        except Exception as e:
            return jsonify(create_response(
                status='success',
                data={
                    'tesseract_available': False,
                    'error': str(e),
                    'test_result': 'OCR test failed'
                }
            ))

    except ImportError:
        return jsonify(create_response(
            status='error',
            error='Tesseract not installed'
        )), 500
    except Exception as e:
        return jsonify(create_response(
            status='error',
            error=f'OCR test failed: {str(e)}'
        )), 500

@app.route('/api/dataset/info')
def dataset_info():
    """Get information about dataset"""
    try:
        # Check if real dataset exists
        dataset_dir = Path(__file__).parent.parent / 'dataset'
        genuine_dir = dataset_dir / 'genuine'
        fake_dir = dataset_dir / 'fake'

        if genuine_dir.exists() and fake_dir.exists():
            # Count real images
            supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

            genuine_count = len([f for f in genuine_dir.iterdir()
                               if f.suffix.lower() in supported_formats])
            fake_count = len([f for f in fake_dir.iterdir()
                            if f.suffix.lower() in supported_formats])

            dataset_data = {
                'dataset_type': 'real',
                'total_samples': genuine_count + fake_count,
                'genuine_samples': genuine_count,
                'fake_samples': fake_count,
                'balance_ratio': genuine_count / max(fake_count, 1),
                'ready_for_training': (genuine_count + fake_count) >= 200,
                'quality': 'excellent' if (genuine_count + fake_count) >= 800 else 'good',
                'last_updated': datetime.now().isoformat()
            }
        else:
            # Fallback to synthetic data info
            dataset_data = {
                'dataset_type': 'synthetic',
                'total_samples': 1000,
                'genuine_samples': 500,
                'fake_samples': 500,
                'balance_ratio': 1.0,
                'ready_for_training': True,
                'quality': 'demo',
                'last_updated': datetime.now().isoformat()
            }

        return jsonify(create_response(
            status='success',
            data=dataset_data
        ))

    except Exception as e:
        logger.error(f"Error getting dataset info: {e}")
        return jsonify(create_response(
            status='error',
            error=str(e)
        )), 500

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    """Analyze uploaded image using ML/DL models"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify(create_response(
                status='error',
                error='No file uploaded'
            )), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify(create_response(
                status='error',
                error='No file selected'
            )), 400

        if not allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
            return jsonify(create_response(
                status='error',
                error='File type not allowed'
            )), 400

        # Save uploaded file
        filename = safe_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Validate image
        if not validate_image(filepath):
            return jsonify(create_response(
                status='error',
                error='Invalid image file'
            )), 400

        # Initialize results
        results = {
            'filename': filename,
            'timestamp': datetime.now().isoformat()
        }

        # Extract features for Random Forest
        features = extract_traditional_features(filepath)
        rf_result = None
        if features is not None and model_manager:
            rf_result = model_manager.predict_with_random_forest(features)

        # Preprocess for Deep Learning
        image_array = preprocess_image(filepath)
        dl_result = None
        if image_array is not None and model_manager:
            dl_result = model_manager.predict_with_deep_learning(image_array)

        # If no models available, create mock predictions
        if not rf_result and not dl_result:
            rf_result = model_manager.create_mock_prediction() if model_manager else {
                'prediction': 'fake',
                'confidence': 0.7,
                'model': 'Mock Model (No models loaded)'
            }

        # Combine predictions
        combined_result = None
        if model_manager:
            combined_result = model_manager.combine_predictions(rf_result, dl_result, None)

        results.update({
            'random_forest': rf_result,
            'deep_learning': dl_result,
            'combined': combined_result
        })

        # Clean up old files
        cleanup_old_files(app.config['UPLOAD_FOLDER'])

        return jsonify(create_response(
            status='success',
            data=results
        ))

    except Exception as e:
        logger.error(f"Error in analyze_image: {e}")
        return jsonify(create_response(
            status='error',
            error=str(e)
        )), 500

@app.route('/api/extract-text', methods=['POST'])
def extract_text():
    """Extract text from uploaded image using OCR"""
    try:
        if 'file' not in request.files:
            return jsonify(create_response(
                status='error',
                error='No file uploaded'
            )), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify(create_response(
                status='error',
                error='No file selected'
            )), 400

        if not allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
            return jsonify(create_response(
                status='error',
                error='File type not allowed'
            )), 400

        # Save uploaded file
        filename = safe_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Validate image
        if not validate_image(filepath):
            return jsonify(create_response(
                status='error',
                error='Invalid image file'
            )), 400

        # Extract text using OCR
        extracted_text = extract_text_ocr(filepath, app.config.get('TESSERACT_CONFIG'))

        return jsonify(create_response(
            status='success',
            data={
                'text': extracted_text,
                'filename': filename,
                'character_count': len(extracted_text),
                'word_count': len(extracted_text.split()) if extracted_text else 0
            }
        ))

    except Exception as e:
        logger.error(f"Error in extract_text: {e}")
        return jsonify(create_response(
            status='error',
            error=str(e)
        )), 500

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    """Analyze text content for suspicious patterns"""
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify(create_response(
                status='error',
                error='No text provided'
            )), 400

        text = data['text']

        # Analyze text content using advanced analyzer
        if text_analyzer:
            analysis_result = text_analyzer.analyze_text(text)
        else:
            # Fallback to basic analysis
            analysis_result = {
                'score': 0.5,
                'prediction': 'uncertain',
                'confidence': 0.3,
                'note': 'Text analyzer not available'
            }

        return jsonify(create_response(
            status='success',
            data=analysis_result
        ))

    except Exception as e:
        logger.error(f"Error in analyze_text: {e}")
        return jsonify(create_response(
            status='error',
            error=str(e)
        )), 500

@app.route('/api/analyze-complete', methods=['POST'])
def analyze_complete():
    """Complete analysis: image + OCR + text analysis"""
    try:
        if 'file' not in request.files:
            return jsonify(create_response(
                status='error',
                error='No file uploaded'
            )), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify(create_response(
                status='error',
                error='No file selected'
            )), 400

        if not allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
            return jsonify(create_response(
                status='error',
                error='File type not allowed'
            )), 400

        # Save uploaded file
        filename = safe_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Validate image
        if not validate_image(filepath):
            return jsonify(create_response(
                status='error',
                error='Invalid image file'
            )), 400

        results = {
            'filename': filename,
            'timestamp': datetime.now().isoformat()
        }

        # Step 1: Image Analysis
        features = extract_traditional_features(filepath)
        image_array = preprocess_image(filepath)

        rf_result = None
        dl_result = None

        if model_manager:
            if features is not None:
                rf_result = model_manager.predict_with_random_forest(features)
            if image_array is not None:
                dl_result = model_manager.predict_with_deep_learning(image_array)

        # Step 2: OCR Text Extraction
        extracted_text = extract_text_ocr(filepath, app.config.get('TESSERACT_CONFIG'))

        # Step 3: Text Analysis
        text_result = None
        if text_analyzer and extracted_text:
            text_result = text_analyzer.analyze_text(extracted_text)

        # Step 4: Combined Analysis
        combined_result = None
        if model_manager:
            combined_result = model_manager.combine_predictions(rf_result, dl_result, text_result)

        results.update({
            'image_analysis': {
                'random_forest': rf_result,
                'deep_learning': dl_result
            },
            'ocr_extraction': {
                'text': extracted_text,
                'character_count': len(extracted_text) if extracted_text else 0,
                'word_count': len(extracted_text.split()) if extracted_text else 0
            },
            'text_analysis': text_result,
            'final_prediction': combined_result
        })

        return jsonify(create_response(
            status='success',
            data=results
        ))

    except Exception as e:
        logger.error(f"Error in analyze_complete: {e}")
        return jsonify(create_response(
            status='error',
            error=str(e)
        )), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify(create_response(
        status='error',
        error='File too large. Maximum size is 16MB.'
    )), 413

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
    # Initialize application
    initialize_app()

    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)
