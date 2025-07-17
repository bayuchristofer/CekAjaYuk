#!/usr/bin/env python3
"""
CekAjaYuk Backend - Railway Deployment Version
Lightweight version for cloud deployment
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime
from pathlib import Path
import os
import sys
import logging
import numpy as np
import io
import base64

# Try to import optional dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("Tesseract not available")

try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODELS_FOLDER'] = 'models'

# Global model variables
rf_model = None
feature_scaler = None
text_vectorizer = None
models_loaded_count = 0

# Indonesian Keywords Dictionary for Job Posting Analysis
INDONESIAN_KEYWORDS = {
    'legitimate_indicators': [
        'perusahaan', 'company', 'pt', 'cv', 'tbk', 'persero', 'terbuka', 'swasta',
        'kantor', 'alamat', 'lokasi', 'cabang', 'pusat', 'regional', 'divisi',
        'departemen', 'bagian', 'unit', 'tim', 'grup', 'holding', 'korporat',
        'posisi', 'jabatan', 'lowongan', 'vacancy', 'karir', 'career', 'pekerjaan',
        'staff', 'karyawan', 'pegawai', 'manager', 'supervisor', 'koordinator',
        'gaji', 'salary', 'upah', 'tunjangan', 'benefit', 'asuransi', 'bpjs',
        'pengalaman', 'experience', 'pendidikan', 'education', 'sarjana', 'diploma',
        'skill', 'keahlian', 'kemampuan', 'kualifikasi', 'requirement', 'syarat',
        'interview', 'wawancara', 'seleksi', 'test', 'tes', 'recruitment',
        'kontrak', 'permanent', 'tetap', 'freelance', 'partime', 'fulltime'
    ],
    'suspicious_indicators': [
        'mudah', 'cepat', 'instant', 'tanpa', 'modal', 'investasi', 'untung',
        'profit', 'bonus', 'komisi', 'mlm', 'bisnis', 'peluang', 'kesempatan',
        'jutaan', 'milyar', 'kaya', 'sukses', 'freedom', 'bebas', 'fleksibel',
        'rumah', 'online', 'internet', 'smartphone', 'hp', 'whatsapp', 'wa',
        'transfer', 'kirim', 'bayar', 'deposit', 'daftar', 'registrasi', 'fee'
    ]
}

def create_directories():
    """Create necessary directories"""
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)
        logger.info("Directories created successfully")
    except Exception as e:
        logger.error(f"Error creating directories: {e}")

def load_models():
    """Load ML models if available"""
    global rf_model, feature_scaler, text_vectorizer, models_loaded_count
    
    if not SKLEARN_AVAILABLE:
        logger.warning("Scikit-learn not available, models cannot be loaded")
        return False
    
    try:
        models_dir = Path(app.config['MODELS_FOLDER'])
        
        # Try to load Random Forest model
        rf_path = models_dir / 'random_forest_production.pkl'
        if rf_path.exists():
            rf_model = joblib.load(rf_path)
            models_loaded_count += 1
            logger.info("Random Forest model loaded")
        
        # Try to load feature scaler
        scaler_path = models_dir / 'feature_scaler_production.pkl'
        if scaler_path.exists():
            feature_scaler = joblib.load(scaler_path)
            models_loaded_count += 1
            logger.info("Feature scaler loaded")
        
        # Try to load text vectorizer
        vectorizer_path = models_dir / 'text_vectorizer_production.pkl'
        if vectorizer_path.exists():
            text_vectorizer = joblib.load(vectorizer_path)
            models_loaded_count += 1
            logger.info("Text vectorizer loaded")
        
        logger.info(f"Loaded {models_loaded_count} models successfully")
        return models_loaded_count > 0
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

def analyze_text_keywords(text):
    """Analyze text for suspicious/legitimate keywords"""
    if not text:
        return 0.5, "No text to analyze"
    
    text_lower = text.lower()
    
    # Count legitimate indicators
    legitimate_count = sum(1 for keyword in INDONESIAN_KEYWORDS['legitimate_indicators'] 
                          if keyword in text_lower)
    
    # Count suspicious indicators
    suspicious_count = sum(1 for keyword in INDONESIAN_KEYWORDS['suspicious_indicators'] 
                          if keyword in text_lower)
    
    # Calculate confidence score
    total_keywords = legitimate_count + suspicious_count
    if total_keywords == 0:
        confidence = 0.5  # Neutral if no keywords found
    else:
        confidence = legitimate_count / total_keywords
    
    # Generate analysis details
    analysis = f"Found {legitimate_count} legitimate and {suspicious_count} suspicious indicators"
    
    return confidence, analysis

def simple_ocr_fallback(image_data):
    """Simple fallback OCR when Tesseract is not available"""
    return "OCR not available in this deployment. Please ensure image contains clear, readable text."

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for Railway"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': models_loaded_count,
        'dependencies': {
            'opencv': CV2_AVAILABLE,
            'pil': PIL_AVAILABLE,
            'tesseract': TESSERACT_AVAILABLE,
            'sklearn': SKLEARN_AVAILABLE
        }
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_job_posting():
    """Analyze job posting for authenticity"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read image
        image_data = file.read()
        
        # Extract text (fallback if OCR not available)
        if TESSERACT_AVAILABLE and PIL_AVAILABLE:
            try:
                image = Image.open(io.BytesIO(image_data))
                extracted_text = pytesseract.image_to_string(image, lang='eng+ind')
            except Exception as e:
                logger.error(f"OCR error: {e}")
                extracted_text = simple_ocr_fallback(image_data)
        else:
            extracted_text = simple_ocr_fallback(image_data)
        
        # Analyze text
        confidence, analysis = analyze_text_keywords(extracted_text)
        
        # Determine status based on confidence
        if confidence < 0.4:
            status = "FAKE"
            recommendation = "This appears to be a fake job posting. Exercise extreme caution."
        elif confidence < 0.8:
            status = "CAUTION"
            recommendation = "This posting requires manual verification. Check company details carefully."
        else:
            status = "GENUINE"
            recommendation = "This appears to be a legitimate job posting."
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, 'wb') as f:
            f.write(image_data)
        
        return jsonify({
            'status': status,
            'confidence': round(confidence * 100, 2),
            'recommendation': recommendation,
            'extracted_text': extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
            'analysis': analysis,
            'filename': filename,
            'disclaimer': "⚠️ DISCLAIMER: This analysis is for educational purposes only. Always verify job postings through official company channels and trusted job platforms.",
            'models_used': models_loaded_count,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({
            'error': 'Analysis failed',
            'details': str(e),
            'disclaimer': "⚠️ DISCLAIMER: This analysis is for educational purposes only. Always verify job postings through official company channels and trusted job platforms."
        }), 500

@app.route('/')
def serve_frontend():
    """Serve the main frontend page"""
    try:
        return send_from_directory('frontend', 'index.html')
    except Exception as e:
        return f"Frontend not available: {e}", 404

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    try:
        return send_from_directory('frontend/static', filename)
    except Exception as e:
        return f"Static file not found: {e}", 404

if __name__ == '__main__':
    create_directories()
    load_models()
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
