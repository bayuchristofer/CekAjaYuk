"""
Configuration file for CekAjaYuk backend
"""

import os
from datetime import timedelta

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'cekajayuk-secret-key-2024'
    
    # File upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uploads')
    MODELS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
    
    # OCR settings
    TESSERACT_CONFIG = r'--oem 3 --psm 6 -l ind+eng'
    
    # Model settings
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    
    # API settings
    API_RATE_LIMIT = "100 per hour"
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'cekajayuk.log'

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Enhanced security for production
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    WTF_CSRF_ENABLED = False

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
