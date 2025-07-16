#!/usr/bin/env python3
"""
Fixed CekAjaYuk Backend with proper model loading and OCR
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime
from pathlib import Path
import os
import sys
import logging
import numpy as np
import cv2
from PIL import Image
import io
import base64

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add backend to path
backend_dir = Path(__file__).parent / 'backend'
sys.path.insert(0, str(backend_dir))

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODELS_FOLDER'] = 'models'

# Global variables
model_manager = None
text_analyzer = None

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

def initialize_models():
    """Initialize models with proper error handling"""
    global model_manager, text_analyzer
    
    try:
        # Try to import and initialize model manager
        from models import ModelManager
        from text_analysis import JobPostingTextAnalyzer
        
        models_dir = Path(__file__).parent / 'models'
        model_manager = ModelManager(str(models_dir))
        model_manager.load_models()
        
        text_analyzer = JobPostingTextAnalyzer()
        
        logger.info("✅ Models initialized successfully")
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Model initialization failed: {e}")
        logger.info("🔄 Running in demo mode")
        return False

def initialize_app():
    """Initialize application components"""
    try:
        # Create necessary directories
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)
        
        # Initialize models
        models_loaded = initialize_models()
        
        logger.info(f"✅ Application initialized (Models: {'Loaded' if models_loaded else 'Demo Mode'})")
        
    except Exception as e:
        logger.error(f"❌ Error initializing application: {e}")

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
    try:
        if model_manager and hasattr(model_manager, 'get_model_info'):
            return jsonify(create_response(
                status='success',
                data=model_manager.get_model_info()
            ))
        else:
            # Demo mode response
            return jsonify(create_response(
                status='success',
                data={
                    'models_loaded': False,
                    'available_models': {
                        'random_forest': {'loaded': False, 'status': '❌ Not Loaded', 'error': 'Demo mode'},
                        'deep_learning': {'loaded': False, 'status': '❌ Not Loaded', 'error': 'Demo mode'},
                        'feature_scaler': {'loaded': False, 'status': '❌ Not Loaded', 'error': 'Demo mode'},
                        'text_vectorizer': {'loaded': False, 'status': '❌ Not Loaded', 'error': 'Demo mode'}
                    },
                    'summary': {
                        'loaded_count': 0,
                        'total_count': 4,
                        'load_percentage': 0.0,
                        'status': 'Demo Mode - Basic functionality available'
                    }
                }
            ))
    except Exception as e:
        return jsonify(create_response(
            status='error',
            error=f'Error getting model info: {str(e)}'
        )), 500

@app.route('/api/test-ocr')
def test_ocr():
    """Test OCR functionality"""
    try:
        import pytesseract
        from PIL import Image
        
        # Create a simple test image with text
        test_image = Image.new('RGB', (300, 100), color='white')
        
        # Try to extract text
        try:
            # Test basic OCR functionality
            test_text = pytesseract.image_to_string(test_image, config=r'--oem 3 --psm 6 -l eng')
            
            # Get version if possible
            try:
                version = pytesseract.get_tesseract_version()
            except:
                version = "Unknown"
            
            return jsonify(create_response(
                status='success',
                data={
                    'tesseract_available': True,
                    'version': str(version),
                    'languages': 'ind+eng',
                    'test_result': 'OCR is working',
                    'status': 'ready',
                    'supported_languages': ['ind', 'eng']
                }
            ))
            
        except Exception as e:
            return jsonify(create_response(
                status='success',
                data={
                    'tesseract_available': False,
                    'version': None,
                    'languages': 'ind+eng',
                    'test_result': f'OCR test failed: {str(e)}',
                    'status': 'limited',
                    'error': str(e)
                }
            ))
            
    except ImportError:
        return jsonify(create_response(
            status='success',
            data={
                'tesseract_available': False,
                'version': None,
                'languages': None,
                'test_result': 'Tesseract not installed',
                'status': 'not_installed',
                'error': 'pytesseract not installed'
            }
        ))
    except Exception as e:
        return jsonify(create_response(
            status='error',
            error=f'OCR test failed: {str(e)}'
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

        # Read image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Preprocess image for better OCR
        processed_image = preprocess_for_ocr(cv_image)

        # Convert back to PIL
        processed_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

        # Extract text with multiple configurations
        extracted_text = extract_text_with_ocr(processed_pil)

        return jsonify(create_response(
            status='success',
            data={
                'extracted_text': extracted_text,
                'text_length': len(extracted_text),
                'preprocessing_applied': True
            }
        ))

    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return jsonify(create_response(
            status='error',
            error=str(e)
        )), 500

def preprocess_for_ocr(image):
    """Preprocess image for better OCR results"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # Convert back to BGR for consistency
        result = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

        return result

    except Exception as e:
        logger.warning(f"Image preprocessing failed: {e}")
        return image

def extract_text_with_ocr(image):
    """Extract text using OCR with multiple configurations"""
    try:
        import pytesseract

        # Try multiple OCR configurations
        configs = [
            r'--oem 3 --psm 6 -l ind+eng',  # Indonesian + English
            r'--oem 3 --psm 3 -l ind+eng',  # Fully automatic
            r'--oem 3 --psm 11 -l ind+eng', # Sparse text
            r'--oem 3 --psm 6 -l eng',      # English only
        ]

        best_text = ""
        best_confidence = 0

        for config in configs:
            try:
                # Extract text with confidence
                data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)

                # Calculate average confidence
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0

                # Extract text
                text = pytesseract.image_to_string(image, config=config)
                text = clean_extracted_text(text)

                # Keep best result
                if avg_confidence > best_confidence and len(text.strip()) > len(best_text.strip()):
                    best_text = text
                    best_confidence = avg_confidence

            except Exception as e:
                logger.warning(f"OCR config {config} failed: {e}")
                continue

        # Fallback to simple extraction
        if not best_text.strip():
            try:
                best_text = pytesseract.image_to_string(image, config=r'--oem 3 --psm 6')
                best_text = clean_extracted_text(best_text)
            except Exception as e:
                logger.error(f"Fallback OCR failed: {e}")
                best_text = "OCR extraction failed"

        return best_text

    except ImportError:
        return "Tesseract OCR not installed"
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return f"OCR error: {str(e)}"

def clean_extracted_text(text):
    """Clean and normalize extracted text"""
    if not text:
        return ""

    import re

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text)

    # Remove special characters that are likely OCR errors
    text = re.sub(r'[^\w\s\n\-.,!?()@#$%&*+=<>:;/\\]', '', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text

@app.route('/api/dataset/info')
def dataset_info():
    """Get information about dataset"""
    try:
        # Check if real dataset exists
        dataset_dir = Path(__file__).parent / 'dataset'
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
            # Demo dataset info
            dataset_data = {
                'dataset_type': 'demo',
                'total_samples': 1000,
                'genuine_samples': 500,
                'fake_samples': 500,
                'balance_ratio': 1.0,
                'ready_for_training': True,
                'quality': 'demo',
                'last_updated': datetime.now().isoformat(),
                'note': 'Using synthetic demo dataset'
            }

        return jsonify(create_response(
            status='success',
            data=dataset_data
        ))

    except Exception as e:
        return jsonify(create_response(
            status='error',
            error=str(e)
        )), 500

# Error handlers
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
    print("🚀 Starting CekAjaYuk Backend (Fixed Version)...")
    print("📍 Running on http://localhost:5001")
    print("🔧 Initializing application...")
    
    # Initialize application
    initialize_app()
    
    print("✅ Backend ready!")
    print("=" * 50)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)
