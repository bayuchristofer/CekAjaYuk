#!/usr/bin/env python3
"""
VPS Startup Script for CekAjaYuk
Optimized for Hostinger VPS with memory management
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import VPS configuration
from vps_config import VPSConfig, get_vps_loader

def setup_vps_environment():
    """Setup VPS environment and load essential models"""
    print("🚀 Starting CekAjaYuk VPS Setup...")
    
    # Setup VPS environment
    VPSConfig.setup_vps_environment()
    logger = logging.getLogger(__name__)
    
    # Check memory
    memory_info = VPSConfig.get_memory_info()
    if 'total' in memory_info:
        logger.info(f"💾 VPS Memory: {memory_info['total']:.1f}GB total, {memory_info['available']:.1f}GB available")
    
    # Load essential models only
    vps_loader = get_vps_loader()
    models_loaded = vps_loader.load_essential_models()
    
    if models_loaded >= 2:  # At least RF + Vectorizer
        logger.info("✅ VPS setup completed successfully")
        return True
    else:
        logger.error("❌ VPS setup failed - insufficient models loaded")
        return False

def create_vps_app():
    """Create Flask app with VPS optimizations"""
    from flask import Flask, request, jsonify, send_from_directory
    from flask_cors import CORS
    
    # Create Flask app
    app = Flask(__name__, static_folder='frontend/static', static_url_path='/static')
    CORS(app)
    
    # VPS Configuration
    app.config.update(
        SECRET_KEY=VPSConfig.SECRET_KEY,
        MAX_CONTENT_LENGTH=VPSConfig.MAX_CONTENT_LENGTH,
        UPLOAD_FOLDER=str(VPSConfig.UPLOADS_DIR),
        DEBUG=VPSConfig.DEBUG
    )
    
    # Setup logging
    logger = logging.getLogger(__name__)
    
    # Get VPS model loader
    vps_loader = get_vps_loader()
    
    @app.route('/')
    def index():
        """Serve frontend"""
        return send_from_directory('frontend', 'index.html')
    
    @app.route('/static/<path:filename>')
    def serve_static(filename):
        """Serve static files"""
        return send_from_directory('frontend/static', filename)
    
    @app.route('/api/health')
    def health_check():
        """VPS Health check"""
        memory_info = VPSConfig.get_memory_info()
        model_status = vps_loader.get_status()
        
        return jsonify({
            'status': 'healthy',
            'environment': 'VPS Production',
            'memory': memory_info,
            'models': model_status,
            'timestamp': str(datetime.now())
        })
    
    @app.route('/api/models/status')
    def models_status():
        """Get model loading status"""
        return jsonify(vps_loader.get_status())
    
    @app.route('/api/analyze', methods=['POST'])
    def analyze_vps():
        """VPS-optimized analysis endpoint"""
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Check file extension
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type'}), 400
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = VPSConfig.UPLOADS_DIR / filename
            file.save(str(filepath))
            
            # Extract text using OCR
            extracted_text = extract_text_vps(str(filepath))
            
            # Analyze with essential models
            analysis_result = analyze_with_essential_models(extracted_text)
            
            # Clean up uploaded file
            try:
                filepath.unlink()
            except:
                pass
            
            return jsonify({
                'status': 'success',
                'extracted_text': extracted_text,
                'analysis': analysis_result,
                'vps_optimized': True
            })
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return jsonify({'error': str(e)}), 500
    
    def allowed_file(filename):
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in VPSConfig.ALLOWED_EXTENSIONS
    
    def secure_filename(filename):
        """Secure filename for VPS"""
        import re
        filename = re.sub(r'[^\w\s-]', '', filename).strip()
        return re.sub(r'[-\s]+', '-', filename)
    
    def extract_text_vps(image_path):
        """VPS-optimized OCR"""
        try:
            import pytesseract
            from PIL import Image
            
            # Set Tesseract path for VPS
            pytesseract.pytesseract.tesseract_cmd = VPSConfig.TESSERACT_CMD
            
            # Open and resize image if too large
            with Image.open(image_path) as img:
                if img.size[0] > VPSConfig.OCR_MAX_IMAGE_SIZE[0] or img.size[1] > VPSConfig.OCR_MAX_IMAGE_SIZE[1]:
                    img.thumbnail(VPSConfig.OCR_MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
                
                # Extract text
                text = pytesseract.image_to_string(img, lang='eng+ind', config=VPSConfig.TESSERACT_CONFIG)
                return text.strip()
                
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""
    
    def analyze_with_essential_models(text):
        """Analyze using only essential models (RF + Vectorizer)"""
        try:
            rf_model = vps_loader.get_model('rf')
            vectorizer = vps_loader.get_model('vectorizer')
            
            if not rf_model or not vectorizer:
                return {'error': 'Essential models not loaded'}
            
            # Vectorize text
            text_vector = vectorizer.transform([text])
            
            # Predict with Random Forest
            prediction_proba = rf_model.predict_proba(text_vector)[0]
            confidence = float(prediction_proba[1])  # Probability of genuine
            
            # Determine result
            if confidence < 0.4:
                result = 'fake'
                status = '🚨 KEMUNGKINAN PALSU'
            elif confidence > 0.8:
                result = 'genuine'
                status = '✅ KEMUNGKINAN ASLI'
            else:
                result = 'caution'
                status = '⚠️ PERLU HATI-HATI'
            
            return {
                'result': result,
                'confidence': confidence,
                'status': status,
                'model_used': 'Random Forest + TF-IDF',
                'vps_optimized': True
            }
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {'error': str(e)}
    
    return app

if __name__ == '__main__':
    from datetime import datetime
    
    # Setup VPS environment
    if not setup_vps_environment():
        print("❌ VPS setup failed")
        sys.exit(1)
    
    # Create and run app
    app = create_vps_app()
    
    print("🚀 Starting CekAjaYuk VPS Server...")
    print(f"📍 Server: http://0.0.0.0:5001")
    print(f"🔧 Environment: VPS Production")
    print(f"💾 Memory optimization: Enabled")
    print(f"⏳ CNN model: Lazy loading")
    print("=" * 50)
    
    # Run with VPS settings
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5001)),
        debug=False,
        use_reloader=False
    )
