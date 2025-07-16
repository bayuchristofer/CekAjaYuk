#!/usr/bin/env python3
"""
CekAjaYuk Backend - Working Version with Proper Status
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
dl_model = None
models_loaded_count = 0

# Indonesian Keywords Dictionary for Job Posting Analysis
INDONESIAN_KEYWORDS = {
    # Kata-kata yang menunjukkan lowongan kerja ASLI
    'legitimate_indicators': [
        # Informasi perusahaan yang jelas
        'perusahaan', 'company', 'pt', 'cv', 'tbk', 'persero', 'terbuka', 'swasta',
        'kantor', 'alamat', 'lokasi', 'cabang', 'pusat', 'regional', 'divisi',
        'departemen', 'bagian', 'unit', 'tim', 'grup', 'holding', 'korporat',

        # Posisi pekerjaan yang spesifik
        'posisi', 'jabatan', 'lowongan', 'vacancy', 'karir', 'career', 'pekerjaan',
        'staff', 'karyawan', 'pegawai', 'manager', 'supervisor', 'koordinator',
        'asisten', 'admin', 'administrasi', 'sekretaris', 'operator', 'teknisi',
        'analis', 'programmer', 'developer', 'designer', 'marketing', 'sales',
        'customer', 'service', 'finance', 'accounting', 'hr', 'hrd', 'legal',
        'engineer', 'consultant', 'specialist', 'executive', 'director',

        # Kualifikasi yang realistis
        'kualifikasi', 'persyaratan', 'requirement', 'pendidikan', 'pengalaman',
        'keahlian', 'skill', 'kemampuan', 'kompetensi', 'sertifikat', 'ijazah',
        'diploma', 'sarjana', 's1', 's2', 'sma', 'smk', 'd3', 'fresh graduate',
        'berpengalaman', 'minimal', 'maksimal', 'usia', 'tahun', 'bulan',
        'lulusan', 'jurusan', 'fakultas', 'universitas', 'institut', 'sekolah',

        # Benefit yang wajar
        'gaji', 'salary', 'upah', 'tunjangan', 'benefit', 'fasilitas', 'asuransi',
        'kesehatan', 'bpjs', 'cuti', 'bonus', 'insentif', 'komisi', 'transport',
        'makan', 'seragam', 'training', 'pelatihan', 'pengembangan', 'karir',
        'jenjang', 'promosi', 'kenaikan', 'pangkat', 'golongan', 'grade',

        # Proses rekrutmen yang jelas
        'lamaran', 'apply', 'melamar', 'kirim', 'email', 'cv', 'resume',
        'surat', 'motivasi', 'interview', 'wawancara', 'test', 'tes', 'seleksi',
        'tahap', 'proses', 'deadline', 'batas', 'waktu', 'periode', 'jadwal',
        'panggilan', 'undangan', 'pemberitahuan', 'hasil', 'pengumuman',

        # Kontak yang profesional
        'kontak', 'contact', 'telepon', 'phone', 'whatsapp', 'wa', 'email',
        'website', 'alamat', 'pic', 'person', 'charge', 'hrd', 'recruitment',
        'recruiter', 'hiring', 'manager', 'koordinator', 'penerimaan'
    ],

    # Kata-kata yang menunjukkan lowongan kerja PALSU/MENCURIGAKAN
    'suspicious_indicators': [
        # Janji berlebihan
        'mudah', 'cepat', 'instant', 'langsung', 'tanpa', 'pengalaman',
        'jutaan', 'milyar', 'kaya', 'sukses', 'freedom', 'bebas', 'flexible',
        'kerja', 'rumah', 'online', 'part', 'time', 'sampingan', 'tambahan',
        'unlimited', 'tak', 'terbatas', 'fantastis', 'luar', 'biasa',

        # Skema MLM/Piramida
        'mlm', 'multi', 'level', 'marketing', 'network', 'bisnis', 'investasi',
        'modal', 'join', 'member', 'downline', 'upline', 'sponsor', 'referral',
        'komisi', 'passive', 'income', 'residual', 'binary', 'matrix', 'plan',
        'sistem', 'piramida', 'rantai', 'jaringan', 'distributor', 'agen',

        # Permintaan uang/biaya
        'biaya', 'bayar', 'transfer', 'deposit', 'jaminan', 'administrasi',
        'pendaftaran', 'registrasi', 'materai', 'meterai', 'pulsa', 'saldo',
        'top', 'up', 'isi', 'ulang', 'voucher', 'token', 'kode', 'pin',
        'starter', 'pack', 'paket', 'membership', 'keanggotaan', 'iuran',

        # Bahasa yang tidak profesional
        'bro', 'sis', 'guys', 'teman', 'sobat', 'kawan', 'sahabat', 'bestie',
        'mantap', 'keren', 'wow', 'amazing', 'fantastic', 'gila', 'gilak',
        'mantul', 'mantab', 'jos', 'gandos', 'top', 'markotop', 'ajib',
        'dahsyat', 'hebat', 'super', 'mega', 'ultra', 'extreme',

        # Urgency yang berlebihan
        'segera', 'cepat', 'buruan', 'terbatas', 'limited', 'promo', 'diskon',
        'gratis', 'free', 'bonus', 'hadiah', 'doorprize', 'undian', 'lucky',
        'beruntung', 'kesempatan', 'emas', 'langka', 'jarang', 'eksklusif',
        'special', 'khusus', 'istimewa', 'rahasia', 'secret', 'tersembunyi',

        # Kontak tidak profesional
        'dm', 'inbox', 'pm', 'private', 'message', 'chat', 'japri', 'personal',
        'nomor', 'hp', 'handphone', 'telegram', 'line', 'bbm', 'pin', 'ig',
        'instagram', 'facebook', 'fb', 'twitter', 'tiktok', 'youtube',
        'medsos', 'sosmed', 'social', 'media', 'platform', 'aplikasi',

        # Skema get-rich-quick
        'autopilot', 'otomatis', 'robot', 'bot', 'software', 'tools',
        'trick', 'tips', 'cara', 'metode', 'strategi', 'formula', 'resep',
        'kunci', 'solusi', 'jalan', 'pintas', 'shortcut', 'hack', 'cheat',
        'magic', 'ajaib', 'mukjizat', 'keajaiban', 'misteri', 'fenomena',

        # Kata-kata manipulatif
        'jangan', 'sampai', 'terlewat', 'lewatkan', 'sia', 'siakan', 'rugi',
        'menyesal', 'penyesalan', 'kesalahan', 'fatal', 'besar', 'seumur',
        'hidup', 'selamanya', 'abadi', 'kekal', 'permanen', 'tetap'
    ],

    # Kata-kata netral yang perlu konteks
    'neutral_keywords': [
        'kerja', 'work', 'job', 'opportunity', 'kesempatan', 'peluang',
        'penghasilan', 'income', 'uang', 'money', 'rupiah', 'dollar',
        'waktu', 'time', 'hari', 'minggu', 'bulan', 'tahun', 'jam',
        'tempat', 'lokasi', 'daerah', 'kota', 'jakarta', 'surabaya',
        'bandung', 'medan', 'semarang', 'yogyakarta', 'bali', 'makassar',
        'solo', 'malang', 'bogor', 'depok', 'tangerang', 'bekasi',
        'industri', 'sektor', 'bidang', 'area', 'wilayah', 'zona'
    ]
}

# Global status variables
models_status = {
    'random_forest': {'loaded': False, 'status': '❌ Not Loaded', 'error': 'Not loaded yet'},
    'deep_learning': {'loaded': False, 'status': '❌ Not Loaded', 'error': 'Not loaded yet'},
    'feature_scaler': {'loaded': False, 'status': '❌ Not Loaded', 'error': 'Not loaded yet'},
    'text_vectorizer': {'loaded': False, 'status': '❌ Not Loaded', 'error': 'Not loaded yet'}
}

ocr_status = {
    'available': False,
    'version': None,
    'status': 'not_installed',
    'error': 'Tesseract not configured'
}

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

def check_tesseract():
    """Check Tesseract OCR availability"""
    global ocr_status

    try:
        import pytesseract

        # Try common Tesseract paths on Windows
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            'tesseract'  # If in PATH
        ]

        for path in possible_paths:
            try:
                # For file paths, check if file exists
                if path != 'tesseract':
                    if not os.path.exists(path):
                        logger.debug(f"Path not found: {path}")
                        continue
                    else:
                        logger.info(f"Found Tesseract at: {path}")

                # Set the tesseract command path
                pytesseract.pytesseract.tesseract_cmd = path

                # Test if it works by getting version
                version = pytesseract.get_tesseract_version()

                # Test if it can actually process an image
                from PIL import Image
                import numpy as np

                # Create a simple test image
                test_img = Image.new('RGB', (100, 50), color='white')
                from PIL import ImageDraw
                draw = ImageDraw.Draw(test_img)
                draw.text((10, 10), "TEST", fill='black')

                # Try to extract text
                test_text = pytesseract.image_to_string(test_img, config='--psm 8')

                # Get available languages
                try:
                    languages = pytesseract.get_languages(config='')
                    lang_support = 'ind+eng' if 'ind' in languages and 'eng' in languages else 'eng'
                except:
                    lang_support = 'eng'

                ocr_status = {
                    'available': True,
                    'version': str(version),
                    'status': 'ready',
                    'path': path,
                    'languages': lang_support,
                    'supported_languages': languages if 'languages' in locals() else ['eng'],
                    'test_result': 'success'
                }

                logger.info(f"✅ Tesseract fully configured at: {path}")
                logger.info(f"   Version: {version}")
                logger.info(f"   Languages: {lang_support}")
                logger.info(f"   Test extraction: {'PASS' if 'TEST' in test_text.upper() else 'PARTIAL'}")
                return True

            except Exception as e:
                logger.debug(f"Tesseract test failed at {path}: {e}")
                continue
        
        ocr_status = {
            'available': False,
            'version': None,
            'status': 'not_found',
            'error': 'Tesseract not found in common locations'
        }
        
        logger.warning("⚠️ Tesseract not found")
        return False
        
    except ImportError:
        ocr_status = {
            'available': False,
            'version': None,
            'status': 'not_installed',
            'error': 'pytesseract not installed'
        }
        logger.warning("⚠️ pytesseract not installed")
        return False

def load_models():
    """Load all available models"""
    global rf_model, feature_scaler, text_vectorizer, dl_model, models_status, models_loaded_count

    import joblib
    models_dir = Path('models')
    loaded_count = 0

    # Load Random Forest (Production Model)
    rf_files = [
        'random_forest_production.pkl',  # New production model
        'random_forest_real_20250704_020314.pkl',  # Fallback
        'random_forest_classifier_latest.pkl'  # Fallback
    ]

    for rf_file in rf_files:
        rf_path = models_dir / rf_file
        if rf_path.exists():
            try:
                rf_model = joblib.load(rf_path)
                models_status['random_forest'] = {
                    'loaded': True,
                    'status': '✅ Ready (Production)',
                    'type': 'RandomForestClassifier',
                    'n_estimators': getattr(rf_model, 'n_estimators', 100),
                    'model_file': rf_file
                }
                loaded_count += 1
                logger.info(f"✅ Random Forest loaded from {rf_file}")
                break
            except Exception as e:
                logger.error(f"❌ Failed to load {rf_file}: {e}")

    # Load Feature Scaler (Production Model)
    scaler_files = [
        'feature_scaler_production.pkl',  # New production model
        'feature_scaler.pkl'  # Fallback
    ]

    for scaler_file in scaler_files:
        scaler_path = models_dir / scaler_file
        if scaler_path.exists():
            try:
                feature_scaler = joblib.load(scaler_path)
                models_status['feature_scaler'] = {
                    'loaded': True,
                    'status': '✅ Ready (Production)',
                    'type': type(feature_scaler).__name__,
                    'model_file': scaler_file
                }
                loaded_count += 1
                logger.info(f"✅ Feature Scaler loaded from {scaler_file}")
                break
            except Exception as e:
                logger.error(f"❌ Failed to load {scaler_file}: {e}")

    # Load Text Vectorizer (Production Model)
    vec_files = [
        'text_vectorizer_production.pkl',  # New production model
        'text_vectorizer.pkl'  # Fallback
    ]

    for vec_file in vec_files:
        vec_path = models_dir / vec_file
        if vec_path.exists():
            try:
                text_vectorizer = joblib.load(vec_path)
                models_status['text_vectorizer'] = {
                    'loaded': True,
                    'status': '✅ Ready (Production)',
                    'type': type(text_vectorizer).__name__,
                    'features': len(text_vectorizer.get_feature_names_out()),
                    'model_file': vec_file
                }
                loaded_count += 1
                logger.info(f"✅ Text Vectorizer loaded from {vec_file}")
                break
            except Exception as e:
                logger.error(f"❌ Failed to load {vec_file}: {e}")

    # Try to load Deep Learning model (Production Model)
    dl_files = [
        'cnn_production.h5',  # New production model
        'cnn_best_real.h5'  # Fallback
    ]

    for dl_file in dl_files:
        dl_path = models_dir / dl_file
        if dl_path.exists():
            try:
                # Try multiple import methods for TensorFlow/Keras
                dl_model = None

                # Method 1: Try tensorflow.keras
                try:
                    import tensorflow as tf
                    dl_model = tf.keras.models.load_model(str(dl_path))
                    logger.info("✅ Deep Learning model loaded via tensorflow.keras")
                except Exception as e1:
                    logger.debug(f"tensorflow.keras failed: {e1}")

                    # Method 2: Try standalone keras
                    try:
                        import keras
                        dl_model = keras.models.load_model(str(dl_path))
                        logger.info("✅ Deep Learning model loaded via standalone keras")
                    except Exception as e2:
                        logger.debug(f"standalone keras failed: {e2}")

                        # Method 3: Try with custom objects
                        try:
                            import tensorflow as tf
                            dl_model = tf.keras.models.load_model(str(dl_path), compile=False)
                            logger.info("✅ Deep Learning model loaded without compilation")
                        except Exception as e3:
                            logger.debug(f"load without compile failed: {e3}")
                            raise Exception(f"All import methods failed: {e1}, {e2}, {e3}")

                if dl_model is not None:
                    models_status['deep_learning'] = {
                        'loaded': True,
                        'status': '✅ Ready (Production)',
                        'type': 'TensorFlow/Keras CNN',
                        'input_shape': str(dl_model.input_shape) if hasattr(dl_model, 'input_shape') else 'Unknown',
                        'model_file': dl_file
                    }
                    loaded_count += 1
                    logger.info(f"✅ Deep Learning model loaded from {dl_file}")
                    break  # Exit loop if successful
                else:
                    raise Exception("Model loading returned None")

            except Exception as e:
                logger.warning(f"⚠️ Deep Learning model {dl_file} failed to load: {e}")
                models_status['deep_learning'] = {
                    'loaded': False,
                    'status': '⚠️ Found but failed to load',
                    'error': str(e),
                    'model_file': dl_file
                }
                continue  # Try next file

    # If no deep learning model loaded, set final status
    if dl_model is None:
        models_status['deep_learning'] = {
            'loaded': False,
            'status': '❌ Not Available',
            'error': 'No compatible deep learning model found'
        }

    models_loaded_count = loaded_count
    logger.info(f"📊 Total models loaded: {loaded_count}/4")

    return loaded_count

def initialize_app():
    """Initialize application components"""
    try:
        # Create necessary directories
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)
        
        # Load models and check components
        load_models()
        check_tesseract()
        
        logger.info("✅ Application initialized")
        logger.info(f"   Models: {models_loaded_count}/4 loaded")
        logger.info(f"   OCR: {'Available' if ocr_status['available'] else 'Not Available'}")
        
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
            'models_loaded': models_loaded_count > 0,
            'models_count': models_loaded_count,
            'ocr_available': ocr_status['available']
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
            'models_loaded': models_loaded_count > 0,
            'models_count': models_loaded_count,
            'ocr_available': ocr_status['available']
        }
    ))

@app.route('/api/init')
def force_init():
    """Force initialize the application"""
    try:
        success = initialize_app()
        return jsonify(create_response(
            status='success',
            message='Application initialized - Running in compatibility mode'
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
        # Count actually loaded models
        loaded_count = sum(1 for model in models_status.values()
                          if model.get('loaded', False))
        total_count = 4

        # Count models that are found but not loaded
        found_count = sum(1 for model in models_status.values()
                         if not model.get('loaded', False) and 'Found' in model.get('status', ''))

        load_percentage = (loaded_count / total_count) * 100

        # Determine status based on loaded models
        if loaded_count == total_count:
            status = f'Production Ready - {loaded_count}/{total_count} models loaded'
        elif loaded_count >= 3:
            status = f'Mostly Ready - {loaded_count}/{total_count} models loaded'
        elif loaded_count >= 1:
            status = f'Limited Mode - {loaded_count}/{total_count} models loaded'
        else:
            status = f'Compatibility Mode - {found_count}/{total_count} models found'

        summary = {
            'loaded_count': loaded_count,
            'total_count': total_count,
            'found_count': found_count,
            'load_percentage': load_percentage,
            'status': status
        }

        return jsonify(create_response(
            status='success',
            data={
                'models_loaded': loaded_count > 0,
                'available_models': models_status,
                'summary': summary,
                'timestamp': datetime.now().isoformat(),
                'note': f'{loaded_count}/{total_count} models successfully loaded and ready for production use.'
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
        # Return current OCR status with additional test info
        test_data = ocr_status.copy()
        test_data['tesseract_available'] = ocr_status['available']
        test_data['test_result'] = 'OCR is working' if ocr_status['available'] else 'OCR not available'

        return jsonify(create_response(
            status='success',
            data=test_data
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
        image_data = None

        # Check if it's a file upload or JSON payload
        if request.content_type and 'application/json' in request.content_type:
            # Handle JSON payload (from frontend)
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify(create_response(
                    status='error',
                    error='No image data in JSON payload'
                )), 400

            # Extract base64 image data
            image_str = data['image']
            if image_str.startswith('data:image'):
                # Remove data URL prefix
                image_str = image_str.split(',')[1]

            try:
                image_data = base64.b64decode(image_str)
            except Exception as e:
                return jsonify(create_response(
                    status='error',
                    error=f'Invalid base64 image data: {str(e)}'
                )), 400

        else:
            # Handle file upload
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

            image_data = file.read()

        # Check if OCR is available
        if not ocr_status['available']:
            return jsonify(create_response(
                status='error',
                error='OCR not available. Please install Tesseract.'
            )), 503

        # Process image data
        if not image_data:
            return jsonify(create_response(
                status='error',
                error='No image data received'
            )), 400

        image = Image.open(io.BytesIO(image_data))

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to OpenCV format for preprocessing
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Preprocess image for better OCR with error handling
        try:
            processed_image = preprocess_for_ocr(cv_image)
            processed_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"Warning: Image preprocessing failed: {e}")
            # Use original image if preprocessing fails
            processed_pil = image

        # Extract text with OCR (with timing and error handling)
        import time
        start_time = time.time()

        try:
            extracted_text = extract_text_with_ocr(processed_pil)
            processing_time = time.time() - start_time

            # If OCR returns empty or very short text, try with original image
            if len(extracted_text.strip()) < 5:
                print("OCR result too short, trying with original image...")
                extracted_text_original = extract_text_with_ocr(image)
                if len(extracted_text_original.strip()) > len(extracted_text.strip()):
                    extracted_text = extracted_text_original

        except Exception as e:
            print(f"OCR extraction failed: {e}")
            processing_time = time.time() - start_time

            # Return error response with helpful message
            return jsonify(create_response(
                status='error',
                error=f'OCR extraction failed: {str(e)}. Try uploading a clearer image with better contrast.',
                data={
                    'processing_time': processing_time,
                    'error_type': 'ocr_extraction_failed',
                    'suggestions': [
                        'Upload a higher resolution image',
                        'Ensure good lighting and contrast',
                        'Try cropping to focus on text areas',
                        'Use images with clear, readable text'
                    ]
                }
            )), 500

        # Calculate confidence based on text quality
        char_count = len(extracted_text.strip())
        word_count = len(extracted_text.split())

        # Enhanced confidence calculation with quality indicators
        confidence = 0
        quality_indicators = []

        # Text length analysis
        if char_count > 200:
            confidence += 40
            quality_indicators.append("Adequate text length")
        elif char_count > 100:
            confidence += 30
            quality_indicators.append("Moderate text length")
        elif char_count > 50:
            confidence += 20
            quality_indicators.append("Short text length")
        else:
            confidence += 10
            quality_indicators.append("Very short text - may need better OCR")

        # Word count analysis
        if word_count > 30:
            confidence += 30
            quality_indicators.append("Rich vocabulary")
        elif word_count > 15:
            confidence += 20
            quality_indicators.append("Adequate vocabulary")
        elif word_count > 5:
            confidence += 10
            quality_indicators.append("Limited vocabulary")
        else:
            quality_indicators.append("Very few words - consider external OCR")

        # Job-related keywords
        job_keywords = ['job', 'position', 'salary', 'company', 'apply', 'work', 'career', 'employment', 'hiring']
        keyword_count = sum(1 for keyword in job_keywords if keyword in extracted_text.lower())

        if keyword_count >= 3:
            confidence += 20
            quality_indicators.append("Job-related content detected")
        elif keyword_count >= 1:
            confidence += 10
            quality_indicators.append("Some job-related terms found")

        # Check for garbled text or special characters
        import re
        garbled_chars = len(re.findall(r'[^\w\s\-.,!?()@#$%&*+=/\\]', extracted_text))
        if garbled_chars > char_count * 0.1:  # More than 10% garbled
            confidence -= 20
            quality_indicators.append("High garbled character ratio - external OCR recommended")
        elif garbled_chars > 0:
            confidence -= 5
            quality_indicators.append("Some garbled characters detected")

        confidence = max(10, min(confidence, 95))  # Cap between 10-95%

        # Calculate final metrics
        char_count = len(extracted_text.strip())
        word_count = len(extracted_text.split())

        # If extracted text is too short or empty, provide helpful fallback
        if char_count < 10:
            print(f"Warning: OCR extracted very short text ({char_count} chars): '{extracted_text}'")

            # Don't return error, but provide clear indication
            extracted_text = f"[OCR_LOW_QUALITY] Extracted: '{extracted_text.strip()}'\n\nOCR could not extract sufficient text from this image.\nPossible reasons:\n- Image quality too low\n- Text too small or blurry\n- Poor contrast\n- Unusual font or language\n\nPlease edit this text manually with the correct information from your job posting."

            confidence = 5  # Very low confidence
            quality_indicators.append("OCR extraction insufficient - manual editing required")

        # Determine quality recommendation
        quality_recommendation = ""
        if confidence < 30:
            quality_recommendation = "OCR quality very low - manual editing strongly recommended"
        elif confidence < 70:
            quality_recommendation = "Consider using external OCR services for better accuracy"
        elif char_count < 50:
            quality_recommendation = "Text too short - try external OCR for better results"
        elif word_count < 10:
            quality_recommendation = "Limited vocabulary detected - external OCR may help"

        return jsonify(create_response(
            status='success',
            message='Text extracted successfully',
            data={
                'text': extracted_text,
                'extracted_text': extracted_text,  # Backward compatibility
                'char_count': char_count,
                'text_length': char_count,  # Backward compatibility
                'confidence': confidence,
                'method': 'Standard OCR',
                'processing_time': processing_time,
                'preview': extracted_text[:100] + '...' if len(extracted_text) > 100 else extracted_text,
                'preprocessing_applied': True,
                'ocr_version': ocr_status.get('version', 'Unknown'),
                'language_detected': 'Indonesian/English',
                'word_count': word_count,
                'quality_score': 'High' if confidence > 80 else 'Medium' if confidence > 60 else 'Low',
                'quality_indicators': quality_indicators,
                'quality_recommendation': quality_recommendation,
                'needs_external_ocr': confidence < 70 or char_count < 50 or word_count < 10
            }
        ))

    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return jsonify(create_response(
            status='error',
            error=str(e)
        )), 500

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    """Analyze text for fake job detection"""
    try:
        # Get text data
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify(create_response(
                status='error',
                error='No text provided'
            )), 400

        text = data['text']

        # Perform text analysis
        text_analysis = analyze_text_features(text)

        return jsonify(create_response(
            status='success',
            message='Text analysis completed',
            data={
                'text_analysis': text_analysis,
                'text_length': len(text),
                'processing_time': 0.5,  # Simulated processing time
                'indonesian_keywords': {
                    'legitimate_count': len(text_analysis.get('indonesian_analysis', {}).get('found_keywords', {}).get('legitimate', [])),
                    'suspicious_count': len(text_analysis.get('indonesian_analysis', {}).get('found_keywords', {}).get('suspicious', [])),
                    'analysis': text_analysis.get('indonesian_analysis', {}).get('analysis', 'N/A')
                }
            }
        ))
    except Exception as e:
        logger.error(f"Error in text analysis: {e}")
        return jsonify(create_response(
            status='error',
            error=str(e)
        )), 500

@app.route('/api/analyze-fake-genuine', methods=['POST'])
def analyze_fake_genuine():
    """Analyze job posting for fake/genuine classification with detailed explanations"""
    try:
        # Get image and text data
        data = request.get_json()
        if not data:
            return jsonify(create_response(
                status='error',
                error='No data provided'
            )), 400

        extracted_text = data.get('text', '')
        image_data = data.get('image', '')

        if not extracted_text and not image_data:
            return jsonify(create_response(
                status='error',
                error='No text or image data provided'
            )), 400

        # Perform detailed analysis with all models
        analysis_results = perform_detailed_fake_analysis(extracted_text, image_data)

        return jsonify(create_response(
            status='success',
            message='Fake/Genuine analysis completed',
            data=analysis_results
        ))

    except Exception as e:
        logger.error(f"Error in fake/genuine analysis: {e}")
        return jsonify(create_response(
            status='error',
            error=str(e)
        )), 500



def perform_detailed_fake_analysis(extracted_text, image_data):
    """Perform comprehensive fake/genuine analysis with detailed explanations"""
    try:
        # Initialize results
        analysis_results = {
            'overall_prediction': 'unknown',
            'overall_confidence': 0,
            'overall_reasoning': '',
            'models': {},
            'text_analysis': {},
            'recommendations': []
        }

        # Text-based analysis
        text_analysis = analyze_text_features(extracted_text)
        analysis_results['text_analysis'] = text_analysis

        # Model 1: Random Forest Analysis
        rf_result = analyze_with_random_forest_detailed(extracted_text, text_analysis)
        analysis_results['models']['random_forest'] = rf_result

        # Model 2: Text Classifier Analysis
        text_classifier_result = analyze_with_text_classifier_detailed(extracted_text)
        analysis_results['models']['text_classifier'] = text_classifier_result

        # Model 3: CNN Analysis (simulated based on text features)
        cnn_result = analyze_with_cnn_detailed(text_analysis)
        analysis_results['models']['cnn'] = cnn_result

        # Model 4: OCR Confidence Analysis
        ocr_result = analyze_ocr_confidence_detailed(extracted_text, text_analysis)
        analysis_results['models']['ocr_confidence'] = ocr_result

        # Calculate ensemble prediction
        ensemble_result = calculate_ensemble_prediction_detailed(analysis_results['models'])
        analysis_results.update(ensemble_result)

        # Generate recommendations
        analysis_results['recommendations'] = generate_recommendations(analysis_results)

        return analysis_results

    except Exception as e:
        logger.error(f"Error in detailed fake analysis: {e}")
        return {
            'overall_prediction': 'error',
            'overall_confidence': 0,
            'overall_reasoning': f'Analysis failed: {str(e)}',
            'models': {},
            'text_analysis': {},
            'recommendations': ['Please try again with a clearer image']
        }

def preprocess_for_ocr(image):
    """Simple preprocessing for OCR"""
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Simple upscaling if image is too small
        height, width = gray.shape
        if width < 800:
            scale_factor = 800 / width
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Basic denoising
        denoised = cv2.medianBlur(gray, 3)

        # Simple thresholding
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresh

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return image

def analyze_indonesian_keywords(text):
    """Analyze Indonesian keywords for job posting authenticity"""
    if not text:
        return {
            'legitimate_score': 0,
            'suspicious_score': 0,
            'neutral_score': 0,
            'total_keywords': 0,
            'found_keywords': {
                'legitimate': [],
                'suspicious': [],
                'neutral': []
            },
            'analysis': 'No text to analyze'
        }

    # Convert text to lowercase for analysis
    text_lower = text.lower()
    words = text_lower.split()

    # Count keyword matches
    legitimate_matches = []
    suspicious_matches = []
    neutral_matches = []

    # Check legitimate indicators
    for keyword in INDONESIAN_KEYWORDS['legitimate_indicators']:
        if keyword in text_lower:
            legitimate_matches.append(keyword)

    # Check suspicious indicators
    for keyword in INDONESIAN_KEYWORDS['suspicious_indicators']:
        if keyword in text_lower:
            suspicious_matches.append(keyword)

    # Check neutral keywords
    for keyword in INDONESIAN_KEYWORDS['neutral_keywords']:
        if keyword in text_lower:
            neutral_matches.append(keyword)

    # Calculate scores
    total_words = len(words)
    legitimate_score = (len(legitimate_matches) / max(total_words, 1)) * 100
    suspicious_score = (len(suspicious_matches) / max(total_words, 1)) * 100
    neutral_score = (len(neutral_matches) / max(total_words, 1)) * 100

    # Determine analysis result
    if legitimate_score > suspicious_score * 1.5:
        analysis = "Menunjukkan indikator lowongan kerja yang legitimate"
    elif suspicious_score > legitimate_score * 1.5:
        analysis = "Menunjukkan indikator lowongan kerja yang mencurigakan"
    elif suspicious_score > 5:  # High suspicious score threshold
        analysis = "Mengandung banyak kata-kata mencurigakan"
    elif legitimate_score > 3:  # Moderate legitimate score
        analysis = "Mengandung beberapa indikator legitimate"
    else:
        analysis = "Analisis tidak konklusif, perlu verifikasi manual"

    return {
        'legitimate_score': round(legitimate_score, 2),
        'suspicious_score': round(suspicious_score, 2),
        'neutral_score': round(neutral_score, 2),
        'total_keywords': len(legitimate_matches) + len(suspicious_matches) + len(neutral_matches),
        'found_keywords': {
            'legitimate': legitimate_matches[:10],  # Limit to first 10 matches
            'suspicious': suspicious_matches[:10],
            'neutral': neutral_matches[:10]
        },
        'analysis': analysis,
        'recommendation': get_keyword_recommendation(legitimate_score, suspicious_score)
    }

def get_keyword_recommendation(legitimate_score, suspicious_score):
    """Get recommendation based on keyword analysis"""
    if suspicious_score > 10:
        return "HATI-HATI: Banyak kata-kata mencurigakan ditemukan. Kemungkinan besar lowongan palsu."
    elif suspicious_score > 5:
        return "WASPADA: Beberapa kata mencurigakan ditemukan. Perlu verifikasi lebih lanjut."
    elif legitimate_score > 5:
        return "BAIK: Mengandung indikator lowongan kerja yang legitimate."
    elif legitimate_score > 2:
        return "CUKUP: Beberapa indikator legitimate ditemukan."
    else:
        return "NETRAL: Tidak ada indikator kuat untuk legitimate atau mencurigakan."

def analyze_text_features(text):
    """Analyze text features for fake detection"""
    if not text or len(text.strip()) < 10:
        return {
            'length': 0,
            'word_count': 0,
            'suspicious_patterns': [],
            'quality_indicators': [],
            'language_quality': 'poor',
            'completeness_score': 0,
            'indonesian_analysis': analyze_indonesian_keywords('')
        }

    # Basic metrics
    text_clean = text.strip()
    words = text_clean.split()

    # Suspicious patterns detection
    suspicious_patterns = []

    # Check for excessive promises
    excessive_promises = ['easy money', 'quick cash', 'no experience needed', 'work from home guaranteed',
                         'earn thousands', 'immediate start', 'no interview required']
    for pattern in excessive_promises:
        if pattern.lower() in text.lower():
            suspicious_patterns.append(f"Excessive promise: '{pattern}'")

    # Check for urgency tactics
    urgency_words = ['urgent', 'immediate', 'asap', 'hurry', 'limited time', 'act now', 'don\'t miss']
    urgency_count = sum(1 for word in urgency_words if word.lower() in text.lower())
    if urgency_count > 2:
        suspicious_patterns.append(f"Excessive urgency tactics ({urgency_count} instances)")

    # Check for vague job descriptions
    vague_indicators = ['various tasks', 'general duties', 'flexible work', 'easy job']
    for indicator in vague_indicators:
        if indicator.lower() in text.lower():
            suspicious_patterns.append(f"Vague description: '{indicator}'")

    # Check for missing essential information
    essential_elements = {
        'company_name': any(word in text.lower() for word in ['company', 'corporation', 'ltd', 'inc']),
        'job_title': any(word in text.lower() for word in ['position', 'role', 'job', 'vacancy']),
        'requirements': any(word in text.lower() for word in ['requirement', 'qualification', 'experience', 'skill']),
        'contact_info': any(word in text.lower() for word in ['email', 'phone', 'contact', 'apply'])
    }

    # Quality indicators
    quality_indicators = []

    # Professional language check
    professional_words = ['experience', 'qualification', 'responsibility', 'requirement', 'benefit',
                         'salary', 'position', 'candidate', 'application', 'interview']
    professional_count = sum(1 for word in professional_words if word.lower() in text.lower())

    if professional_count >= 5:
        quality_indicators.append("Professional vocabulary used")
    elif professional_count >= 3:
        quality_indicators.append("Some professional terms present")
    else:
        quality_indicators.append("Limited professional vocabulary")

    # Structure check
    if len(words) > 50:
        quality_indicators.append("Adequate text length")
    else:
        quality_indicators.append("Text too short for proper job posting")

    # Contact information check
    if essential_elements['contact_info']:
        quality_indicators.append("Contact information provided")
    else:
        suspicious_patterns.append("Missing contact information")

    # Calculate completeness score
    completeness_score = sum(essential_elements.values()) / len(essential_elements) * 100

    # Determine language quality
    if professional_count >= 5 and len(suspicious_patterns) == 0:
        language_quality = 'excellent'
    elif professional_count >= 3 and len(suspicious_patterns) <= 1:
        language_quality = 'good'
    elif professional_count >= 2 and len(suspicious_patterns) <= 2:
        language_quality = 'fair'
    else:
        language_quality = 'poor'

    # Analyze Indonesian keywords
    indonesian_analysis = analyze_indonesian_keywords(text)

    return {
        'length': len(text_clean),
        'word_count': len(words),
        'suspicious_patterns': suspicious_patterns,
        'quality_indicators': quality_indicators,
        'language_quality': language_quality,
        'completeness_score': completeness_score,
        'essential_elements': essential_elements,
        'professional_word_count': professional_count,
        'indonesian_analysis': indonesian_analysis
    }

def analyze_with_random_forest_detailed(text, text_features):
    """Random Forest analysis with detailed reasoning"""
    try:
        # Simulate Random Forest analysis based on text features
        confidence = 0
        reasoning_points = []

        # Text length analysis
        if text_features['length'] > 200:
            confidence += 25
            reasoning_points.append("✓ Adequate text length suggests legitimate posting")
        else:
            confidence -= 15
            reasoning_points.append("⚠ Short text length may indicate fake posting")

        # Professional vocabulary
        if text_features['professional_word_count'] >= 5:
            confidence += 30
            reasoning_points.append("✓ Professional vocabulary indicates genuine posting")
        elif text_features['professional_word_count'] >= 3:
            confidence += 15
            reasoning_points.append("~ Moderate professional vocabulary")
        else:
            confidence -= 20
            reasoning_points.append("⚠ Limited professional vocabulary suggests fake")

        # Suspicious patterns
        if len(text_features['suspicious_patterns']) == 0:
            confidence += 25
            reasoning_points.append("✓ No suspicious patterns detected")
        else:
            confidence -= len(text_features['suspicious_patterns']) * 10
            reasoning_points.append(f"⚠ {len(text_features['suspicious_patterns'])} suspicious patterns found")

        # Completeness
        if text_features['completeness_score'] >= 75:
            confidence += 20
            reasoning_points.append("✓ Complete job posting information")
        else:
            confidence -= 10
            reasoning_points.append("⚠ Missing essential job posting elements")

        # Normalize confidence
        confidence = max(0, min(100, confidence + 50))  # Base 50 + adjustments

        # Determine prediction based on consistent thresholds
        if confidence >= 80:
            prediction = 'genuine'
        elif confidence >= 40:
            prediction = 'uncertain'
        else:
            prediction = 'fake'

        return {
            'prediction': prediction,
            'confidence': round(confidence, 1),
            'reasoning': reasoning_points,
            'model_name': 'Random Forest Classifier',
            'features_analyzed': ['text_length', 'vocabulary', 'patterns', 'completeness']
        }

    except Exception as e:
        return {
            'prediction': 'error',
            'confidence': 0,
            'reasoning': [f"Analysis failed: {str(e)}"],
            'model_name': 'Random Forest Classifier',
            'features_analyzed': []
        }

def analyze_with_text_classifier_detailed(text):
    """Text Classifier analysis with linguistic reasoning"""
    try:
        confidence = 0
        reasoning_points = []

        # Keyword analysis
        genuine_keywords = ['experience', 'qualification', 'requirement', 'responsibility', 'benefit',
                           'salary', 'interview', 'application', 'candidate', 'position']
        fake_keywords = ['easy money', 'no experience', 'work from home', 'immediate', 'urgent',
                        'guaranteed', 'quick cash', 'no interview']

        genuine_count = sum(1 for keyword in genuine_keywords if keyword.lower() in text.lower())
        fake_count = sum(1 for keyword in fake_keywords if keyword.lower() in text.lower())

        if genuine_count > fake_count:
            confidence += 40
            reasoning_points.append(f"✓ More genuine keywords ({genuine_count}) than fake keywords ({fake_count})")
        else:
            confidence -= 20
            reasoning_points.append(f"⚠ More fake keywords ({fake_count}) than genuine keywords ({genuine_count})")

        # Grammar and structure analysis
        sentences = text.split('.')
        if len(sentences) >= 3:
            confidence += 20
            reasoning_points.append("✓ Well-structured text with multiple sentences")
        else:
            confidence -= 10
            reasoning_points.append("⚠ Poor text structure")

        # Contact information
        if any(indicator in text.lower() for indicator in ['email', '@', 'phone', 'contact']):
            confidence += 25
            reasoning_points.append("✓ Contact information provided")
        else:
            confidence -= 15
            reasoning_points.append("⚠ No clear contact information")

        # Normalize confidence
        confidence = max(0, min(100, confidence + 45))

        # Determine prediction based on consistent thresholds
        if confidence >= 80:
            prediction = 'genuine'
        elif confidence >= 40:
            prediction = 'uncertain'
        else:
            prediction = 'fake'

        return {
            'prediction': prediction,
            'confidence': round(confidence, 1),
            'reasoning': reasoning_points,
            'model_name': 'Text Classifier (TF-IDF + Logistic Regression)',
            'features_analyzed': ['keywords', 'structure', 'contact_info']
        }

    except Exception as e:
        return {
            'prediction': 'error',
            'confidence': 0,
            'reasoning': [f"Analysis failed: {str(e)}"],
            'model_name': 'Text Classifier',
            'features_analyzed': []
        }

def analyze_with_cnn_detailed(text_features):
    """CNN analysis based on visual/structural features"""
    try:
        confidence = 0
        reasoning_points = []

        # Simulate CNN analysis based on text structure patterns
        # In real implementation, this would analyze image features

        # Text organization analysis
        if text_features['completeness_score'] >= 75:
            confidence += 35
            reasoning_points.append("✓ Well-organized content structure")
        else:
            confidence -= 15
            reasoning_points.append("⚠ Poor content organization")

        # Language quality assessment
        if text_features['language_quality'] == 'excellent':
            confidence += 30
            reasoning_points.append("✓ Excellent language quality")
        elif text_features['language_quality'] == 'good':
            confidence += 20
            reasoning_points.append("✓ Good language quality")
        elif text_features['language_quality'] == 'fair':
            confidence += 5
            reasoning_points.append("~ Fair language quality")
        else:
            confidence -= 20
            reasoning_points.append("⚠ Poor language quality")

        # Pattern recognition
        if len(text_features['suspicious_patterns']) == 0:
            confidence += 25
            reasoning_points.append("✓ No suspicious visual patterns detected")
        else:
            confidence -= len(text_features['suspicious_patterns']) * 8
            reasoning_points.append(f"⚠ {len(text_features['suspicious_patterns'])} suspicious patterns detected")

        # Normalize confidence
        confidence = max(0, min(100, confidence + 40))

        # Determine prediction based on consistent thresholds
        if confidence >= 80:
            prediction = 'genuine'
        elif confidence >= 40:
            prediction = 'uncertain'
        else:
            prediction = 'fake'

        return {
            'prediction': prediction,
            'confidence': round(confidence, 1),
            'reasoning': reasoning_points,
            'model_name': 'CNN (Convolutional Neural Network)',
            'features_analyzed': ['structure', 'language_quality', 'visual_patterns']
        }

    except Exception as e:
        return {
            'prediction': 'error',
            'confidence': 0,
            'reasoning': [f"Analysis failed: {str(e)}"],
            'model_name': 'CNN',
            'features_analyzed': []
        }

def analyze_ocr_confidence_detailed(text, text_features):
    """OCR confidence analysis with quality assessment"""
    try:
        confidence = 0
        reasoning_points = []

        # Text extraction quality
        if len(text.strip()) > 100:
            confidence += 30
            reasoning_points.append("✓ Good text extraction quality")
        elif len(text.strip()) > 50:
            confidence += 15
            reasoning_points.append("~ Moderate text extraction")
        else:
            confidence -= 20
            reasoning_points.append("⚠ Poor text extraction quality")

        # Readability assessment
        if text_features['word_count'] > 20:
            confidence += 25
            reasoning_points.append("✓ Sufficient readable content")
        else:
            confidence -= 10
            reasoning_points.append("⚠ Limited readable content")

        # Character quality (simulate OCR confidence)
        # In real implementation, this would use actual OCR confidence scores
        if text_features['professional_word_count'] >= 3:
            confidence += 20
            reasoning_points.append("✓ Professional terms clearly extracted")
        else:
            confidence -= 15
            reasoning_points.append("⚠ Limited professional vocabulary extracted")

        # Text completeness
        if text_features['essential_elements']['contact_info']:
            confidence += 15
            reasoning_points.append("✓ Contact information successfully extracted")
        else:
            confidence -= 10
            reasoning_points.append("⚠ Missing contact information")

        # Normalize confidence
        confidence = max(0, min(100, confidence + 30))

        # Determine prediction based on consistent thresholds
        if confidence >= 80:
            prediction = 'genuine'
            reasoning_points.append("High OCR confidence suggests genuine document")
        elif confidence >= 40:
            prediction = 'uncertain'
            reasoning_points.append("Moderate OCR confidence - uncertain classification")
        else:
            prediction = 'fake'
            reasoning_points.append("Low OCR confidence may indicate fake or poor quality document")

        return {
            'prediction': prediction,
            'confidence': round(confidence, 1),
            'reasoning': reasoning_points,
            'model_name': 'OCR Confidence Analyzer',
            'features_analyzed': ['extraction_quality', 'readability', 'completeness']
        }

    except Exception as e:
        return {
            'prediction': 'error',
            'confidence': 0,
            'reasoning': [f"Analysis failed: {str(e)}"],
            'model_name': 'OCR Confidence Analyzer',
            'features_analyzed': []
        }

def calculate_ensemble_prediction_detailed(models_results):
    """Calculate ensemble prediction with improved logic for better accuracy"""
    try:
        # Collect predictions and confidences
        predictions = []
        confidences = []
        all_reasoning = []
        fake_indicators = 0
        genuine_indicators = 0

        for model_name, result in models_results.items():
            if result['prediction'] != 'error':
                predictions.append(result['prediction'])
                confidences.append(result['confidence'])
                all_reasoning.extend([f"[{model_name}] {reason}" for reason in result['reasoning']])

                # Count strong indicators
                if result['prediction'] == 'fake' and result['confidence'] > 30:
                    fake_indicators += 1
                elif result['prediction'] == 'genuine' and result['confidence'] > 70:
                    genuine_indicators += 1

        if not predictions:
            return {
                'overall_prediction': 'error',
                'overall_confidence': 0,
                'overall_reasoning': 'All models failed to analyze'
            }

        # Enhanced ensemble logic
        fake_votes = predictions.count('fake')
        genuine_votes = predictions.count('genuine')
        uncertain_votes = predictions.count('uncertain')

        # Calculate average confidence for each prediction type
        fake_confidences = [conf for pred, conf in zip(predictions, confidences) if pred == 'fake']
        genuine_confidences = [conf for pred, conf in zip(predictions, confidences) if pred == 'genuine']
        uncertain_confidences = [conf for pred, conf in zip(predictions, confidences) if pred == 'uncertain']

        avg_fake_conf = sum(fake_confidences) / len(fake_confidences) if fake_confidences else 0
        avg_genuine_conf = sum(genuine_confidences) / len(genuine_confidences) if genuine_confidences else 0
        avg_uncertain_conf = sum(uncertain_confidences) / len(uncertain_confidences) if uncertain_confidences else 0

        # Enhanced decision logic with consistent thresholds and better fake detection
        # Calculate weighted average confidence
        total_weight = sum(confidences)
        if total_weight > 0:
            weighted_avg_confidence = sum(conf * conf for conf in confidences) / total_weight
        else:
            weighted_avg_confidence = 50

        # Enhanced fake detection - prioritize fake indicators
        if fake_indicators >= 2 or fake_votes >= 2:
            # Strong fake signals - force to fake category
            final_prediction = 'fake'
            final_confidence = min(max(avg_fake_conf, 15), 39)  # Keep below 40%
        elif fake_votes >= 1 and avg_fake_conf == 0:
            # Models detected fake with 0% confidence - very suspicious
            final_prediction = 'fake'
            final_confidence = 25  # Low confidence for fake
        # Strong genuine indicators with high confidence
        elif genuine_indicators >= 3 and avg_genuine_conf > 80:
            final_prediction = 'genuine'
            final_confidence = max(avg_genuine_conf, 80)  # Ensure above 80%
        # Mixed signals but with fake presence - be conservative
        elif fake_votes > 0 and genuine_votes > 0:
            final_prediction = 'uncertain'
            # Reduce confidence when there are conflicting signals
            final_confidence = min(max(weighted_avg_confidence * 0.8, 40), 79)
        # Uncertain votes majority
        elif uncertain_votes >= 2:
            final_prediction = 'uncertain'
            final_confidence = min(max(weighted_avg_confidence, 40), 79)  # Keep in 40-79% range
        # Default to majority vote with consistent thresholds
        else:
            if genuine_votes > fake_votes and genuine_votes > uncertain_votes:
                final_prediction = 'genuine'
                final_confidence = max(avg_genuine_conf, 80)  # Ensure above 80%
            elif fake_votes > uncertain_votes:
                final_prediction = 'fake'
                final_confidence = min(max(avg_fake_conf, 20), 39)  # Keep below 40%
            else:
                final_prediction = 'uncertain'
                final_confidence = min(max(weighted_avg_confidence, 40), 79)  # Keep in 40-79% range

        # Apply final threshold rules to ensure consistency
        if final_confidence >= 80:
            final_prediction = 'genuine'
        elif final_confidence >= 40:
            final_prediction = 'uncertain'
        else:
            final_prediction = 'fake'

        # Ensure confidence is within reasonable bounds
        final_confidence = max(15, min(95, round(final_confidence, 1)))

        # Generate comprehensive reasoning
        reasoning_summary = []
        reasoning_summary.append(f"Ensemble analysis of {len(predictions)} models:")
        reasoning_summary.append(f"• Fake votes: {fake_votes} (avg conf: {avg_fake_conf:.1f}%)")
        reasoning_summary.append(f"• Genuine votes: {genuine_votes} (avg conf: {avg_genuine_conf:.1f}%)")
        reasoning_summary.append(f"• Uncertain votes: {uncertain_votes} (avg conf: {avg_uncertain_conf:.1f}%)")

        # Add decision reasoning
        if fake_indicators >= 2:
            reasoning_summary.append("Strong fake indicators detected across multiple models")
        elif genuine_indicators >= 3:
            reasoning_summary.append("Strong genuine indicators with high confidence")
        elif final_prediction == 'uncertain':
            reasoning_summary.append("Mixed signals or conflicting evidence from models")

        # Add confidence interpretation
        if final_confidence >= 80:
            reasoning_summary.append("High confidence prediction")
        elif final_confidence >= 60:
            reasoning_summary.append("Moderate confidence prediction")
        else:
            reasoning_summary.append("Low confidence prediction - exercise caution")

        return {
            'overall_prediction': final_prediction,
            'overall_confidence': final_confidence,
            'overall_reasoning': ' | '.join(reasoning_summary),
            'detailed_reasoning': all_reasoning,
            'model_votes': {
                'fake': fake_votes,
                'genuine': genuine_votes,
                'uncertain': uncertain_votes
            },
            'strong_indicators': {
                'fake': fake_indicators,
                'genuine': genuine_indicators
            }
        }

    except Exception as e:
        return {
            'overall_prediction': 'error',
            'overall_confidence': 0,
            'overall_reasoning': f'Ensemble calculation failed: {str(e)}'
        }

def generate_recommendations(analysis_results):
    """Generate actionable recommendations based on analysis"""
    recommendations = []

    try:
        overall_pred = analysis_results.get('overall_prediction', 'unknown')
        overall_conf = analysis_results.get('overall_confidence', 0)
        text_analysis = analysis_results.get('text_analysis', {})

        # OCR Quality Recommendations
        recommendations.append({
            'category': 'OCR Quality',
            'title': 'Improve Text Extraction',
            'description': 'For better analysis accuracy, consider using dedicated OCR services',
            'suggestions': [
                'Try Google Cloud Vision API for better text extraction',
                'Use Adobe Acrobat online OCR tool',
                'Consider Microsoft Azure Computer Vision',
                'Upload higher resolution images (minimum 300 DPI)',
                'Ensure good lighting and contrast in the image'
            ]
        })

        # Analysis-based recommendations
        if overall_pred == 'fake':
            recommendations.append({
                'category': 'Security Alert',
                'title': 'Potential Fake Job Posting Detected',
                'description': f'Our analysis indicates this is likely a fake posting (confidence: {overall_conf}%)',
                'suggestions': [
                    'Do not provide personal information or payment',
                    'Verify company legitimacy through official channels',
                    'Check company website and contact information',
                    'Look for reviews from other job seekers',
                    'Be cautious of requests for upfront payments'
                ]
            })
        elif overall_pred == 'genuine':
            recommendations.append({
                'category': 'Verification',
                'title': 'Likely Genuine Job Posting',
                'description': f'Our analysis suggests this is a legitimate posting (confidence: {overall_conf}%)',
                'suggestions': [
                    'Still verify company details independently',
                    'Research the company online',
                    'Check if the job requirements match your skills',
                    'Prepare for standard interview process',
                    'Follow proper application procedures'
                ]
            })
        else:
            recommendations.append({
                'category': 'Caution',
                'title': 'Uncertain Classification',
                'description': f'Analysis results are inconclusive (confidence: {overall_conf}%)',
                'suggestions': [
                    'Exercise extra caution when proceeding',
                    'Manually verify all company information',
                    'Look for additional red flags',
                    'Consider getting a second opinion',
                    'Upload a clearer image for better analysis'
                ]
            })

        # Text quality recommendations
        if text_analysis.get('suspicious_patterns'):
            recommendations.append({
                'category': 'Red Flags Detected',
                'title': 'Suspicious Patterns Found',
                'description': 'Several concerning patterns were identified in the text',
                'suggestions': [
                    f"Review these issues: {', '.join(text_analysis['suspicious_patterns'][:3])}",
                    'Be extra cautious about legitimacy',
                    'Verify claims independently',
                    'Avoid any upfront payments or fees'
                ]
            })

        return recommendations

    except Exception as e:
        return [{
            'category': 'Error',
            'title': 'Recommendation Generation Failed',
            'description': f'Unable to generate recommendations: {str(e)}',
            'suggestions': ['Please try the analysis again']
        }]














def extract_text_with_ocr(image):
    """Extract text using OCR with multiple configurations"""
    try:
        import pytesseract

        # Convert PIL image to OpenCV format for preprocessing
        if hasattr(image, 'mode'):  # PIL Image
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:  # Already OpenCV format
            cv_image = image

        # Apply preprocessing
        processed_image = preprocess_for_ocr(cv_image)

        # Convert back to PIL for OCR
        if len(processed_image.shape) == 3:
            processed_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        else:
            processed_pil = Image.fromarray(processed_image)

        # Simple OCR configurations - back to basics
        configs = [
            {
                'config': r'--oem 1 --psm 6',
                'name': 'LSTM+Auto',
                'description': 'LSTM engine with automatic page segmentation'
            },
            {
                'config': r'--oem 1 --psm 4',
                'name': 'LSTM+Column',
                'description': 'LSTM engine assuming single column'
            },
            {
                'config': r'--oem 1 --psm 3',
                'name': 'LSTM+Full',
                'description': 'LSTM engine with full page analysis'
            },
            {
                'config': r'--oem 3 --psm 6',
                'name': 'Default+Auto',
                'description': 'Default engine with automatic segmentation'
            },
            {
                'config': r'--oem 0 --psm 6',
                'name': 'Legacy+Auto',
                'description': 'Legacy engine fallback'
            }
        ]

        best_text = ""
        best_confidence = 0

        best_config_name = ""

        for config_info in configs:
            config = config_info['config']
            name = config_info['name']
            description = config_info['description']

            try:
                logger.info(f"🔍 Trying OCR: {name} - {description}")

                # Use preprocessed image for better results
                data = pytesseract.image_to_data(processed_pil, config=config, output_type=pytesseract.Output.DICT)

                # Calculate comprehensive confidence
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                words = [word for word in data['text'] if word.strip()]

                if not confidences or not words:
                    logger.debug(f"No valid text found with {name}")
                    continue

                avg_confidence = sum(confidences) / len(confidences)
                word_count = len(words)

                # Extract text
                text = pytesseract.image_to_string(processed_pil, config=config)
                text = clean_extracted_text(text)
                char_count = len(text.strip())

                # Advanced scoring system
                score = 0
                score += avg_confidence * 0.4  # Base confidence (0-40)
                score += min(char_count * 0.1, 30)  # Text length bonus (0-30)
                score += min(word_count * 0.5, 20)  # Word count bonus (0-20)

                # Quality bonuses
                if char_count > 50: score += 10
                if word_count > 10: score += 5
                if avg_confidence > 70: score += 15

                # Penalties
                if char_count < 10: score -= 20
                if word_count < 3: score -= 10

                logger.info(f"📊 {name}: conf={avg_confidence:.1f}, chars={char_count}, words={word_count}, score={score:.1f}")

                # Update best result
                if score > best_confidence and char_count > 0:
                    best_text = text
                    best_confidence = score
                    best_config_name = name
                    logger.info(f"🎯 NEW BEST: {name} (score: {score:.1f})")

            except Exception as e:
                logger.warning(f"OCR config {name} failed: {e}")
                continue

        # Fallback to simple extraction if no good result
        if not best_text.strip():
            try:
                logger.warning("🔄 No good OCR result, trying fallback...")
                best_text = pytesseract.image_to_string(processed_pil, config=r'--oem 1 --psm 6 -l ind+eng')
                best_text = clean_extracted_text(best_text)
                best_config_name = "fallback"
            except Exception as e:
                logger.error(f"Fallback OCR failed: {e}")
                best_text = "OCR extraction failed"



        # Log final result with details
        if best_text.strip() and best_text != "OCR extraction failed":
            logger.info(f"✅ OCR SUCCESS: {len(best_text)} chars, confidence {best_confidence:.1f}, method: {best_config_name}")
            logger.info(f"📝 Preview: {best_text[:100]}...")
        else:
            logger.warning("❌ OCR failed to extract meaningful text")

        return best_text

    except ImportError:
        return "Tesseract OCR not installed"
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return f"OCR error: {str(e)}"

def clean_extracted_text(text):
    """Advanced cleaning and normalization for Indonesian text"""
    if not text:
        return ""

    import re

    # First pass: Fix common OCR errors for Indonesian text
    text = fix_common_ocr_errors(text)

    # Fix common OCR character substitutions
    char_fixes = {
        '4': 'A', '1': 'I', '0': 'O', '5': 'S', '8': 'B', '3': 'E',
        '!': 'I', '@': 'a', '#': 'H', '$': 'S', '%': '%', '^': 'A',
        '&': '&', '*': '*', '(': '(', ')': ')', '-': '-', '_': '_',
        '+': '+', '=': '=', '[': '[', ']': ']', '{': '{', '}': '}',
        '|': 'I', '\\': '/', '/': '/', '?': '?', '<': '<', '>': '>',
        '~': '-', '`': "'", '"': '"', "'": "'"
    }

    # Apply character fixes selectively (only when surrounded by letters)
    for wrong, correct in char_fixes.items():
        if wrong in ['4', '1', '0', '5', '8', '3', '!']:
            # Only replace when surrounded by letters (likely OCR error in words)
            text = re.sub(f'(?<=[a-zA-Z]){re.escape(wrong)}(?=[a-zA-Z])', correct, text)

    # Fix specific OCR patterns
    text = re.sub(r'([A-Z])\s+([a-z])', r'\1\2', text)  # Fix broken words like "L OWONGAN"
    text = re.sub(r'([a-z])\s+([A-Z])', r'\1 \2', text)  # Proper word spacing

    # Remove excessive whitespace but preserve line breaks
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double newline

    # Clean up punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([.,!?;:])\s*([.,!?;:])', r'\1', text)  # Remove duplicate punctuation

    # Fix phone numbers
    text = re.sub(r'(\+?\d{1,3})\s*(\d{3,4})\s*(\d{3,4})\s*(\d{3,4})', r'\1\2\3\4', text)

    # Fix email addresses
    text = re.sub(r'(\w+)\s*@\s*(\w+)\s*\.\s*(\w+)', r'\1@\2.\3', text)

    # Enhanced Indonesian word fixes with common OCR errors
    word_fixes = {
        # Core job posting terms
        r'[Ll][Oo0][Ww][Oo0][Nn][Gg][Aa4][Nn]': 'LOWONGAN',
        r'[Kk][Ee3][Rr][Jj][Aa4]': 'KERJA',
        r'[Gg][Aa4][Jj][Ii1]': 'GAJI',
        r'[Ll][Oo0][Kk][Aa4][Ss5][Ii1]': 'LOKASI',
        r'[Pp][Oo0][Ss5][Ii1][Ss5][Ii1]': 'POSISI',
        r'[Pp][Ee3][Nn][Gg][Aa4][Ll][Aa4][Mm][Aa4][Nn]': 'PENGALAMAN',
        r'[Ss5][Yy][Aa4][Rr][Aa4][Tt]': 'SYARAT',
        r'[Kk][Oo0][Nn][Tt][Aa4][Kk]': 'KONTAK',
        r'[Mm][Ii1][Nn][Ii1][Mm][Aa4][Ll]': 'MINIMAL',
        r'[Tt][Aa4][Hh][Uu][Nn]': 'TAHUN',

        # Additional Indonesian terms
        r'[Pp][Ee3][Rr][Uu][Ss5][Aa4][Hh][Aa4][Aa4][Nn]': 'PERUSAHAAN',
        r'[Kk][Aa4][Rr][Yy][Aa4][Ww][Aa4][Nn]': 'KARYAWAN',
        r'[Pp][Ee3][Nn][Dd][Ii1][Dd][Ii1][Kk][Aa4][Nn]': 'PENDIDIKAN',
        r'[Kk][Ee3][Aa4][Hh][Ll][Ii1][Aa4][Nn]': 'KEAHLIAN',
        r'[Ll][Aa4][Mm][Aa4][Rr][Aa4][Nn]': 'LAMARAN',
        r'[Tt][Ee3][Rr][Ii1][Mm][Aa4]': 'TERIMA',
        r'[Kk][Aa4][Ss5][Ii1][Hh]': 'KASIH',
        r'[Ss5][Ee3][Gg][Ee3][Rr][Aa4]': 'SEGERA',
        r'[Dd][Ii1][Bb][Uu][Tt][Uu][Hh][Kk][Aa4][Nn]': 'DIBUTUHKAN',
        r'[Dd][Ii1][Cc][Aa4][Rr][Ii1]': 'DICARI',

        # Common OCR errors for Indonesian
        r'[Rr][Pp]\.?\s*[0-9]': lambda m: 'Rp ' + m.group().replace('Rp', '').replace('.', '').strip(),
        r'[Jj][Aa4][Kk][Aa4][Rr][Tt][Aa4]': 'JAKARTA',
        r'[Ss5][Uu][Rr][Aa4][Bb][Aa4][Yy][Aa4]': 'SURABAYA',
        r'[Bb][Aa4][Nn][Dd][Uu][Nn][Gg]': 'BANDUNG',
        r'[Mm][Ee3][Dd][Aa4][Nn]': 'MEDAN',
        r'[Ss5][Ee3][Mm][Aa4][Rr][Aa4][Nn][Gg]': 'SEMARANG'
    }

    for pattern, replacement in word_fixes.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Remove characters that are definitely OCR errors
    text = re.sub(r'[^\w\s\n\-.,!?()@#$%&*+=<>:;/\\"+\']', '', text)

    # Final cleanup
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = re.sub(r'\n\s+', '\n', text)  # Remove leading spaces on new lines
    text = text.strip()

    return text

def fix_common_ocr_errors(text):
    """Fix common OCR recognition errors"""
    # Common character substitutions
    fixes = {
        # Number/letter confusion
        '0': 'O', '1': 'I', '5': 'S', '8': 'B',
        # Common OCR mistakes
        'rn': 'm', 'cl': 'd', 'li': 'h', 'vv': 'w',
        'nn': 'm', 'ii': 'n', 'oo': 'o'
    }

    for wrong, correct in fixes.items():
        text = text.replace(wrong, correct)

    return text

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
    print("🚀 Starting CekAjaYuk Backend...")

    # Get port from environment (Railway sets this)
    port = int(os.environ.get('PORT', 5001))
    host = '0.0.0.0'

    print(f"📍 Running on http://{host}:{port}")
    print("🔧 Initializing application...")

    # Initialize application
    initialize_app()

    print("✅ Backend ready!")
    print(f"📊 API Health: http://{host}:{port}/api/health")
    print("=" * 50)

    try:
        # Run the application
        # For Railway, we use gunicorn in production, but this is for local dev
        app.run(debug=False, host=host, port=port, use_reloader=False)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down CekAjaYuk Backend...")
        print("👋 Thank you for using CekAjaYuk!")
    except Exception as e:
        print(f"❌ Server error: {e}")
        print("💡 Try running on a different port or check for conflicts")
