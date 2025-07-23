#!/usr/bin/env python3
"""
VPS Configuration for CekAjaYuk
Optimized settings for Hostinger VPS deployment
"""

import os
import logging
from pathlib import Path

# VPS Environment Configuration
class VPSConfig:
    # Memory Management
    LAZY_LOAD_CNN = True  # Load CNN model only when needed
    MAX_MEMORY_USAGE = 1.5  # GB - Maximum memory usage
    ENABLE_GARBAGE_COLLECTION = True
    
    # Model Loading Strategy
    ESSENTIAL_MODELS_ONLY = True  # Load only RF + Vectorizer initially
    CNN_ON_DEMAND = True  # Load CNN only when image analysis is requested
    
    # Performance Settings
    GUNICORN_WORKERS = 2  # Reduced for VPS memory constraints
    GUNICORN_TIMEOUT = 120
    GUNICORN_MAX_REQUESTS = 500  # Restart workers to prevent memory leaks
    
    # File Paths (VPS-specific)
    MODELS_DIR = Path('/var/www/cekajayuk/models')
    UPLOADS_DIR = Path('/var/www/cekajayuk/uploads')
    LOGS_DIR = Path('/var/log/cekajayuk')
    
    # Tesseract Configuration (VPS Linux)
    TESSERACT_CMD = '/usr/bin/tesseract'  # Standard Linux path
    TESSERACT_CONFIG = '--psm 6 --oem 3'
    
    # Flask Settings
    FLASK_ENV = 'production'
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'cekajayuk-vps-production-key')
    
    # Database (if needed later)
    DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///cekajayuk.db')
    
    # Logging Configuration
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Security Settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    
    # OCR Settings
    OCR_TIMEOUT = 30  # seconds
    OCR_MAX_IMAGE_SIZE = (2048, 2048)  # Resize large images
    
    # Model Loading Priorities
    MODEL_PRIORITY = {
        'random_forest': ['random_forest_retrained.pkl', 'random_forest_production.pkl'],
        'vectorizer': ['tfidf_vectorizer_retrained.pkl', 'text_vectorizer_production.pkl'],
        'scaler': ['feature_scaler_production.pkl'],
        'cnn': ['cnn_production.h5']  # Lazy loaded
    }
    
    @classmethod
    def setup_vps_environment(cls):
        """Setup VPS-specific environment"""
        # Create necessary directories
        for directory in [cls.MODELS_DIR, cls.UPLOADS_DIR, cls.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=cls.LOG_LEVEL,
            format=cls.LOG_FORMAT,
            handlers=[
                logging.FileHandler(cls.LOGS_DIR / 'cekajayuk.log'),
                logging.StreamHandler()
            ]
        )
        
        # Set environment variables
        os.environ['FLASK_ENV'] = cls.FLASK_ENV
        os.environ['PYTHONPATH'] = str(Path.cwd())
        
        return True
    
    @classmethod
    def get_memory_info(cls):
        """Get current memory usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'total': memory.total / (1024**3),
                'available': memory.available / (1024**3),
                'used': memory.used / (1024**3),
                'percent': memory.percent
            }
        except ImportError:
            return {'error': 'psutil not available'}
    
    @classmethod
    def should_load_cnn(cls):
        """Check if CNN model should be loaded based on memory"""
        if not cls.CNN_ON_DEMAND:
            return True
            
        memory_info = cls.get_memory_info()
        if 'available' in memory_info:
            return memory_info['available'] > 1.0  # Need at least 1GB free
        return True  # Default to True if can't check memory

# VPS-specific model loader
class VPSModelLoader:
    def __init__(self):
        self.config = VPSConfig()
        self.loaded_models = {}
        self.logger = logging.getLogger(__name__)
    
    def load_essential_models(self):
        """Load only essential models (RF + Vectorizer)"""
        import joblib
        
        models_loaded = 0
        
        # Load Random Forest
        for rf_file in self.config.MODEL_PRIORITY['random_forest']:
            rf_path = self.config.MODELS_DIR / rf_file
            if rf_path.exists():
                try:
                    self.loaded_models['rf'] = joblib.load(rf_path)
                    self.logger.info(f"✅ Random Forest loaded: {rf_file}")
                    models_loaded += 1
                    break
                except Exception as e:
                    self.logger.error(f"❌ Failed to load {rf_file}: {e}")
        
        # Load Vectorizer
        for vec_file in self.config.MODEL_PRIORITY['vectorizer']:
            vec_path = self.config.MODELS_DIR / vec_file
            if vec_path.exists():
                try:
                    self.loaded_models['vectorizer'] = joblib.load(vec_path)
                    self.logger.info(f"✅ Vectorizer loaded: {vec_file}")
                    models_loaded += 1
                    break
                except Exception as e:
                    self.logger.error(f"❌ Failed to load {vec_file}: {e}")
        
        # Load Scaler
        for scaler_file in self.config.MODEL_PRIORITY['scaler']:
            scaler_path = self.config.MODELS_DIR / scaler_file
            if scaler_path.exists():
                try:
                    self.loaded_models['scaler'] = joblib.load(scaler_path)
                    self.logger.info(f"✅ Scaler loaded: {scaler_file}")
                    models_loaded += 1
                    break
                except Exception as e:
                    self.logger.error(f"❌ Failed to load {scaler_file}: {e}")
        
        # Force garbage collection
        if self.config.ENABLE_GARBAGE_COLLECTION:
            import gc
            gc.collect()
        
        self.logger.info(f"📊 Essential models loaded: {models_loaded}/3")
        return models_loaded
    
    def load_cnn_on_demand(self):
        """Load CNN model only when needed"""
        if 'cnn' in self.loaded_models:
            return self.loaded_models['cnn']
        
        if not self.config.should_load_cnn():
            self.logger.warning("⚠️ Insufficient memory for CNN model")
            return None
        
        for cnn_file in self.config.MODEL_PRIORITY['cnn']:
            cnn_path = self.config.MODELS_DIR / cnn_file
            if cnn_path.exists():
                try:
                    import tensorflow as tf
                    
                    # Memory optimization for TensorFlow
                    try:
                        tf.config.experimental.set_memory_growth(
                            tf.config.experimental.list_physical_devices('GPU')[0], True
                        )
                    except:
                        pass
                    
                    self.loaded_models['cnn'] = tf.keras.models.load_model(
                        str(cnn_path), compile=False
                    )
                    self.logger.info(f"✅ CNN loaded on demand: {cnn_file}")
                    return self.loaded_models['cnn']
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to load CNN {cnn_file}: {e}")
        
        return None
    
    def get_model(self, model_name):
        """Get model with lazy loading"""
        if model_name == 'cnn':
            return self.load_cnn_on_demand()
        return self.loaded_models.get(model_name)
    
    def get_status(self):
        """Get loading status of all models"""
        status = {}
        for model_name in ['rf', 'vectorizer', 'scaler', 'cnn']:
            if model_name in self.loaded_models:
                status[model_name] = {'loaded': True, 'status': '✅ Ready'}
            elif model_name == 'cnn':
                status[model_name] = {'loaded': False, 'status': '⏳ Lazy Loading'}
            else:
                status[model_name] = {'loaded': False, 'status': '❌ Not Loaded'}
        return status

# Global VPS model loader instance
vps_loader = None

def get_vps_loader():
    """Get or create VPS model loader"""
    global vps_loader
    if vps_loader is None:
        vps_loader = VPSModelLoader()
    return vps_loader
