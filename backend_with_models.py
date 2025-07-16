#!/usr/bin/env python3
"""
CekAjaYuk Backend with Real Model Loading
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
import joblib
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODELS_FOLDER'] = 'models'

# Global variables for models
rf_model = None
dl_model = None
feature_scaler = None
text_vectorizer = None
models_loaded = False

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

def load_random_forest():
    """Load Random Forest model"""
    global rf_model
    try:
        models_dir = Path('models')
        
        # Try different RF model files
        rf_files = [
            'random_forest_real_20250704_020314.pkl',
            'random_forest_classifier_latest.pkl',
            'random_forest_model.pkl'
        ]
        
        for rf_file in rf_files:
            rf_path = models_dir / rf_file
            if rf_path.exists():
                rf_model = joblib.load(rf_path)
                logger.info(f"✅ Random Forest loaded from {rf_file}")
                return True
                
        logger.warning("⚠️ No Random Forest model found")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error loading Random Forest: {e}")
        return False

def load_deep_learning():
    """Load Deep Learning model"""
    global dl_model
    try:
        # Try to import TensorFlow
        try:
            import tensorflow as tf
            logger.info("✅ TensorFlow imported successfully")
        except ImportError as e:
            logger.warning(f"⚠️ TensorFlow not available: {e}")
            return False
        
        models_dir = Path('models')
        dl_path = models_dir / 'cnn_best_real.h5'
        
        if dl_path.exists():
            dl_model = tf.keras.models.load_model(str(dl_path))
            logger.info("✅ Deep Learning model loaded from cnn_best_real.h5")
            return True
        else:
            logger.warning("⚠️ Deep Learning model file not found")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error loading Deep Learning model: {e}")
        return False

def load_feature_scaler():
    """Load Feature Scaler"""
    global feature_scaler
    try:
        models_dir = Path('models')
        scaler_path = models_dir / 'feature_scaler.pkl'
        
        if scaler_path.exists():
            feature_scaler = joblib.load(scaler_path)
            logger.info("✅ Feature Scaler loaded")
            return True
        else:
            logger.warning("⚠️ Feature Scaler not found")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error loading Feature Scaler: {e}")
        return False

def load_text_vectorizer():
    """Load Text Vectorizer"""
    global text_vectorizer
    try:
        models_dir = Path('models')
        vectorizer_files = [
            'text_vectorizer.pkl',
            'tfidf_vectorizer.pkl',
            'count_vectorizer.pkl'
        ]
        
        for vec_file in vectorizer_files:
            vec_path = models_dir / vec_file
            if vec_path.exists():
                text_vectorizer = joblib.load(vec_path)
                logger.info(f"✅ Text Vectorizer loaded from {vec_file}")
                return True
                
        logger.warning("⚠️ Text Vectorizer not found")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error loading Text Vectorizer: {e}")
        return False

def configure_tesseract():
    """Configure Tesseract OCR"""
    try:
        import pytesseract
        
        # Try common Tesseract paths on Windows
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe'.format(os.environ.get('USERNAME', '')),
            'tesseract'  # If in PATH
        ]
        
        for path in possible_paths:
            try:
                if path != 'tesseract' and not os.path.exists(path):
                    continue
                    
                pytesseract.pytesseract.tesseract_cmd = path
                version = pytesseract.get_tesseract_version()
                logger.info(f"✅ Tesseract found at: {path} (version: {version})")
                return True, str(version), path
                
            except Exception as e:
                logger.debug(f"Tesseract not found at {path}: {e}")
                continue
        
        logger.warning("⚠️ Tesseract not found in common locations")
        return False, None, None
        
    except ImportError:
        logger.warning("⚠️ pytesseract not installed")
        return False, None, None

def initialize_models():
    """Initialize all models"""
    global models_loaded
    
    logger.info("🔄 Loading models...")
    
    loaded_count = 0
    total_count = 4
    
    # Load Random Forest
    if load_random_forest():
        loaded_count += 1
    
    # Load Deep Learning
    if load_deep_learning():
        loaded_count += 1
    
    # Load Feature Scaler
    if load_feature_scaler():
        loaded_count += 1
    
    # Load Text Vectorizer
    if load_text_vectorizer():
        loaded_count += 1
    
    models_loaded = loaded_count > 0
    
    logger.info(f"📊 Models loaded: {loaded_count}/{total_count}")
    
    if loaded_count == 0:
        logger.error("❌ No models could be loaded")
    elif loaded_count < total_count:
        logger.warning(f"⚠️ Partial loading: {loaded_count}/{total_count} models loaded")
    else:
        logger.info("✅ All models loaded successfully")
    
    return loaded_count

def initialize_app():
    """Initialize application components"""
    try:
        # Create necessary directories
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)
        
        # Initialize models
        loaded_count = initialize_models()
        
        # Configure OCR
        ocr_available, ocr_version, ocr_path = configure_tesseract()
        
        logger.info(f"✅ Application initialized")
        logger.info(f"   Models: {loaded_count}/4 loaded")
        logger.info(f"   OCR: {'Available' if ocr_available else 'Not Available'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error initializing application: {e}")
        return False

# Routes
@app.route('/')
def index():
    """Root endpoint"""
    return jsonify(create_response(
        status='success',
        message='CekAjaYuk API is running',
        data={
            'version': '1.0.0',
            'models_loaded': models_loaded
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
            'models_loaded': models_loaded
        }
    ))

@app.route('/api/init')
def force_init():
    """Force initialize the application"""
    try:
        success = initialize_app()
        return jsonify(create_response(
            status='success' if success else 'warning',
            message='Application initialized successfully' if success else 'Partial initialization'
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
        loaded_count = 0
        total_count = 4

        available_models = {}

        # Random Forest Model
        if rf_model is not None:
            available_models['random_forest'] = {
                'type': 'RandomForestClassifier',
                'n_estimators': getattr(rf_model, 'n_estimators', 'unknown'),
                'loaded': True,
                'status': '✅ Ready'
            }
            loaded_count += 1
        else:
            available_models['random_forest'] = {
                'type': 'RandomForestClassifier',
                'loaded': False,
                'status': '❌ Not Loaded',
                'error': 'Model file not found or failed to load'
            }

        # Deep Learning Model
        if dl_model is not None:
            try:
                available_models['deep_learning'] = {
                    'type': 'TensorFlow/Keras CNN',
                    'input_shape': str(dl_model.input_shape) if hasattr(dl_model, 'input_shape') else 'unknown',
                    'output_shape': str(dl_model.output_shape) if hasattr(dl_model, 'output_shape') else 'unknown',
                    'loaded': True,
                    'status': '✅ Ready'
                }
                loaded_count += 1
            except Exception as e:
                available_models['deep_learning'] = {
                    'type': 'TensorFlow/Keras CNN',
                    'loaded': True,
                    'status': '⚠️ Loaded with issues',
                    'error': str(e)
                }
                loaded_count += 1
        else:
            available_models['deep_learning'] = {
                'type': 'TensorFlow/Keras CNN',
                'loaded': False,
                'status': '❌ Not Loaded',
                'error': 'TensorFlow not available or model file not found'
            }

        # Feature Scaler
        if feature_scaler is not None:
            available_models['feature_scaler'] = {
                'type': type(feature_scaler).__name__,
                'loaded': True,
                'status': '✅ Ready'
            }
            loaded_count += 1
        else:
            available_models['feature_scaler'] = {
                'type': 'StandardScaler/MinMaxScaler',
                'loaded': False,
                'status': '❌ Not Loaded',
                'error': 'Scaler file not found'
            }

        # Text Vectorizer
        if text_vectorizer is not None:
            available_models['text_vectorizer'] = {
                'type': type(text_vectorizer).__name__,
                'loaded': True,
                'status': '✅ Ready'
            }
            loaded_count += 1
        else:
            available_models['text_vectorizer'] = {
                'type': 'TfidfVectorizer/CountVectorizer',
                'loaded': False,
                'status': '❌ Not Loaded',
                'error': 'Vectorizer file not found'
            }

        # Summary
        summary = {
            'loaded_count': loaded_count,
            'total_count': total_count,
            'load_percentage': round((loaded_count / total_count) * 100, 1),
            'status': 'Ready' if loaded_count > 0 else 'No models loaded'
        }

        return jsonify(create_response(
            status='success',
            data={
                'models_loaded': loaded_count > 0,
                'available_models': available_models,
                'summary': summary,
                'timestamp': datetime.now().isoformat()
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

            # Try to get available languages
            try:
                langs = pytesseract.get_languages()
                languages = '+'.join(langs[:5]) if langs else 'eng+ind'
            except:
                languages = 'eng+ind'

            return jsonify(create_response(
                status='success',
                data={
                    'tesseract_available': True,
                    'version': str(version),
                    'languages': languages,
                    'test_result': 'OCR is working',
                    'status': 'ready',
                    'supported_languages': langs if 'langs' in locals() else ['eng', 'ind']
                }
            ))

        except Exception as e:
            return jsonify(create_response(
                status='success',
                data={
                    'tesseract_available': False,
                    'version': None,
                    'languages': 'eng+ind',
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

@app.route('/api/dataset/info')
def dataset_info():
    """Get information about dataset"""
    try:
        # Check if real dataset exists
        dataset_dir = Path('dataset')
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
    print("🚀 Starting CekAjaYuk Backend with Model Loading...")
    print("📍 Running on http://localhost:5001")
    print("🔧 Initializing application...")
    
    # Initialize application
    initialize_app()
    
    print("✅ Backend ready!")
    print("=" * 50)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)
