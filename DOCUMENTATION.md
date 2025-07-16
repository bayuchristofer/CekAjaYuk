# CekAjaYuk - Dokumentasi Lengkap

## Deskripsi Sistem

CekAjaYuk adalah sistem deteksi iklan lowongan kerja palsu yang menggunakan teknologi Machine Learning dan Deep Learning. Sistem ini menganalisis gambar poster lowongan kerja melalui 5 tahap proses untuk menentukan keaslian iklan tersebut.

## Fitur Utama

### 1. Analisis Gambar Multi-Model
- **Random Forest Classifier**: Menganalisis fitur tradisional dari gambar
- **Convolutional Neural Network (CNN)**: Analisis deep learning dengan transfer learning
- **Ensemble Method**: Kombinasi prediksi dari kedua model untuk akurasi maksimal

### 2. Ekstraksi Teks OCR
- **Tesseract OCR**: Ekstraksi teks dari gambar poster
- **Multi-bahasa**: Mendukung bahasa Indonesia dan Inggris
- **Preprocessing**: Optimasi gambar untuk hasil OCR yang lebih baik

### 3. Analisis Teks Lanjutan
- **Pattern Detection**: Deteksi pola mencurigakan dalam teks
- **Company Legitimacy**: Analisis kredibilitas perusahaan
- **Contact Information**: Validasi informasi kontak
- **Language Quality**: Penilaian kualitas bahasa dan struktur teks

### 4. Interface Web Responsif
- **5-Step Workflow**: Proses analisis yang terstruktur dan mudah diikuti
- **Real-time Progress**: Indikator progress untuk setiap tahap analisis
- **Interactive Results**: Tampilan hasil yang komprehensif dan mudah dipahami

## Arsitektur Sistem

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │    Models       │
│   (HTML/CSS/JS) │◄──►│   (Flask API)   │◄──►│   (ML/DL)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Upload   │    │   Image/Text    │    │   Predictions   │
│   (Drag & Drop) │    │   Processing    │    │   & Analysis    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Struktur Direktori

```
cekajayuk/
├── frontend/                 # Interface web
│   ├── index.html           # Halaman utama
│   ├── css/
│   │   └── style.css        # Styling
│   └── js/
│       └── main.js          # JavaScript logic
├── backend/                 # API Flask
│   ├── app.py              # Aplikasi utama
│   ├── models.py           # Model manager
│   ├── text_analyzer.py    # Analisis teks
│   ├── utils.py            # Utility functions
│   └── config.py           # Konfigurasi
├── notebooks/              # Jupyter notebooks
│   ├── 1_data_preparation.ipynb
│   ├── 2_random_forest_training.ipynb
│   └── 3_tensorflow_training.ipynb
├── static/                 # Static files
│   ├── css/
│   ├── js/
│   └── images/
├── models/                 # Model files
├── uploads/                # Upload directory
├── logs/                   # Log files
├── requirements.txt        # Dependencies
├── run.py                  # Application runner
├── setup.py               # Setup script
├── test_api.py            # API testing
└── README.md              # Basic documentation
```

## API Endpoints

### 1. Health Check
```
GET /
Response: {"status": "success", "message": "CekAjaYuk API is running"}
```

### 2. Models Information
```
GET /api/models/info
Response: {
  "status": "success",
  "data": {
    "models_loaded": true,
    "available_models": {...}
  }
}
```

### 3. Image Analysis
```
POST /api/analyze-image
Content-Type: multipart/form-data
Body: file (image file)

Response: {
  "status": "success",
  "data": {
    "random_forest": {
      "prediction": "genuine|fake",
      "confidence": 0.85,
      "features_used": 15
    },
    "deep_learning": {
      "prediction": "genuine|fake", 
      "confidence": 0.92,
      "model_architecture": "CNN"
    },
    "combined": {
      "prediction": "genuine|fake",
      "confidence": 0.89,
      "method": "weighted_ensemble"
    }
  }
}
```

### 4. Text Extraction (OCR)
```
POST /api/extract-text
Content-Type: multipart/form-data
Body: file (image file)

Response: {
  "status": "success",
  "data": {
    "text": "extracted text content",
    "character_count": 245,
    "word_count": 42,
    "filename": "uploaded_file.jpg"
  }
}
```

### 5. Text Analysis
```
POST /api/analyze-text
Content-Type: application/json
Body: {"text": "text to analyze"}

Response: {
  "status": "success",
  "data": {
    "prediction": "genuine|fake",
    "confidence": 0.78,
    "score": 0.65,
    "assessment": {
      "level": "low_risk|medium_risk|high_risk",
      "description": "detailed assessment"
    },
    "analysis_details": {
      "suspicious_patterns": [...],
      "positive_indicators": [...],
      "language_quality": {...},
      "contact_analysis": {...}
    }
  }
}
```

### 6. Complete Analysis
```
POST /api/analyze-complete
Content-Type: multipart/form-data
Body: file (image file)

Response: {
  "status": "success",
  "data": {
    "image_analysis": {...},
    "ocr_extraction": {...},
    "text_analysis": {...},
    "final_prediction": {
      "prediction": "genuine|fake",
      "confidence": 0.87,
      "recommendation": "detailed recommendation"
    }
  }
}
```

## Workflow Analisis

### Tahap 1: Upload Gambar
- User mengupload gambar poster lowongan kerja
- Validasi format file (JPG, PNG, PDF)
- Validasi ukuran file (maksimal 16MB)
- Preview gambar yang diupload

### Tahap 2: Analisis Gambar ML/DL
- **Random Forest**: Ekstraksi fitur tradisional (warna, tekstur, layout)
- **CNN**: Analisis deep learning dengan transfer learning
- **Ensemble**: Kombinasi prediksi dengan weighted voting

### Tahap 3: Ekstraksi Teks OCR
- Preprocessing gambar untuk optimasi OCR
- Tesseract OCR dengan konfigurasi Indonesia + English
- Post-processing untuk membersihkan hasil OCR

### Tahap 4: Koreksi Manual (Opsional)
- User dapat mengedit hasil OCR jika diperlukan
- Interface text editor yang user-friendly
- Validasi input sebelum analisis lanjutan

### Tahap 5: Analisis Teks
- **Suspicious Patterns**: Deteksi kata/frasa mencurigakan
- **Company Analysis**: Validasi nama dan kredibilitas perusahaan
- **Contact Validation**: Analisis email, telepon, alamat
- **Language Quality**: Penilaian grammar dan struktur

### Tahap 6: Hasil Final
- Kombinasi semua analisis dengan weighted scoring
- Confidence level dan rekomendasi
- Penjelasan detail untuk setiap aspek analisis

## Konfigurasi

### Environment Variables
```python
# Flask Configuration
DEBUG = True
SECRET_KEY = 'your-secret-key'

# File Upload
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

# OCR Configuration
TESSERACT_CONFIG = r'--oem 3 --psm 6 -l ind+eng'

# Model Configuration
MODELS_FOLDER = 'models'
IMAGE_SIZE = (224, 224)
```

### Dependencies
```
Flask==2.3.3
Flask-CORS==4.0.0
numpy==1.24.3
opencv-python==4.8.1.78
Pillow==10.0.1
pytesseract==0.3.10
scikit-learn==1.3.0
tensorflow==2.13.0
textblob==0.17.1
nltk==3.8.1
```

## Instalasi dan Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd cekajayuk
```

### 2. Setup Environment
```bash
python setup.py
```

### 3. Install Tesseract OCR
- **Windows**: Download dari https://github.com/UB-Mannheim/tesseract/wiki
- **Linux**: `sudo apt-get install tesseract-ocr tesseract-ocr-ind`
- **macOS**: `brew install tesseract tesseract-lang`

### 4. Run Application
```bash
python run.py
```

## Testing

### Manual Testing
```bash
python test_api.py
```

### Unit Testing
```bash
python -m pytest tests/
```

### Load Testing
```bash
python tests/load_test.py
```

## Deployment

### Development
```bash
python run.py
```

### Production
```bash
gunicorn --bind 0.0.0.0:5000 backend.app:app
```

### Docker
```bash
docker build -t cekajayuk .
docker run -p 5000:5000 cekajayuk
```

## Troubleshooting

### Common Issues

1. **Tesseract not found**
   - Install Tesseract OCR
   - Add to PATH environment variable

2. **Model files missing**
   - Train models using Jupyter notebooks
   - Place model files in `models/` directory

3. **Backend connection failed**
   - Check if Flask server is running
   - Verify API_BASE_URL in frontend

4. **File upload errors**
   - Check file size limits
   - Verify file format support

### Logs
- Application logs: `logs/cekajayuk.log`
- Error logs: Console output
- Access logs: Flask development server

## Performance

### Benchmarks
- Image analysis: ~2-5 seconds
- OCR extraction: ~3-8 seconds  
- Text analysis: ~1-2 seconds
- Total workflow: ~6-15 seconds

### Optimization
- Model caching untuk startup yang lebih cepat
- Image preprocessing untuk OCR yang lebih akurat
- Async processing untuk multiple requests
- CDN untuk static files

## Security

### File Upload Security
- File type validation
- File size limits
- Virus scanning (optional)
- Secure file storage

### API Security
- Rate limiting
- Input validation
- CORS configuration
- Error handling

## Contributing

1. Fork repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## License

Educational use only. Not for commercial distribution.

## Support

For questions and support:
- Email: support@cekajayuk.com
- Documentation: /docs
- Issues: GitHub Issues
