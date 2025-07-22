# CekAjaYuk - Sistem Deteksi Lowongan Kerja Palsu

🔍 **CekAjaYuk** adalah aplikasi web berbasis AI untuk mendeteksi lowongan kerja palsu menggunakan teknologi OCR dan Machine Learning.

## ✨ Fitur Utama

- 🖼️ **OCR (Optical Character Recognition)** - Ekstraksi teks dari gambar lowongan kerja
- 🤖 **Machine Learning** - Analisis menggunakan Random Forest dan Deep Learning CNN
- 🇮🇩 **Dukungan Bahasa Indonesia** - Optimized untuk konten Indonesia
- 📊 **Analisis Komprehensif** - Kombinasi analisis teks dan fitur visual
- 🎯 **Akurasi Tinggi** - Model terlatih dengan dataset khusus

## 🛠️ Teknologi yang Digunakan

### Backend
- **Python 3.11** - Bahasa pemrograman utama
- **Flask** - Web framework
- **TensorFlow/Keras** - Deep learning framework
- **Scikit-learn** - Machine learning library
- **OpenCV** - Computer vision
- **Tesseract OCR** - Text extraction

### Frontend
- **HTML5/CSS3** - Struktur dan styling
- **JavaScript** - Interaktivitas
- **Bootstrap** - UI framework

### Machine Learning Models
- **Random Forest** - Klasifikasi berbasis fitur
- **CNN (Convolutional Neural Network)** - Deep learning untuk analisis gambar
- **TF-IDF Vectorizer** - Text feature extraction

## 🚀 Cara Menjalankan

### Prerequisites
```bash
# Install Python 3.11
# Install Tesseract OCR
```

### Instalasi
```bash
# Clone repository
git clone https://github.com/[username]/cekajayuk.git
cd cekajayuk

# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
python backend_working.py
```

### Akses Aplikasi
- Buka browser dan kunjungi: `http://localhost:5001`

## 📁 Struktur Project

```
cekajayuk/
├── backend_working.py          # Main Flask application
├── frontend/                   # Frontend files
│   ├── index.html             # Main HTML page
│   └── static/                # CSS, JS, images
├── models/                     # Trained ML models
│   ├── cnn_production.h5      # CNN model
│   ├── random_forest_*.pkl    # Random Forest models
│   └── *.pkl                  # Other model files
├── uploads/                    # Uploaded images
├── train_models.py            # Model training script
└── *.ipynb                    # Jupyter notebooks
```

## 🎯 Cara Penggunaan

1. **Upload Gambar** - Upload gambar lowongan kerja (JPG, PNG, PDF)
2. **Analisis Otomatis** - Sistem akan melakukan OCR dan analisis ML
3. **Hasil Prediksi** - Dapatkan hasil:
   - **< 40%** = Lowongan Palsu
   - **40-80%** = Perlu Verifikasi Manual
   - **> 80%** = Lowongan Asli

## 📊 Model Performance

- **Akurasi OCR**: Optimized untuk bahasa Indonesia dan Inggris
- **Akurasi ML**: Trained dengan dataset 800+ gambar lowongan kerja
- **Response Time**: < 5 detik per analisis

## 🔧 Development

### Training Models
```bash
# Retrain models dengan data baru
python train_models.py

# Atau gunakan Jupyter notebook
jupyter notebook retrain_models.ipynb
```

### API Endpoints
- `GET /` - Main web interface
- `POST /api/analyze` - Analyze job posting image
- `GET /api/health` - Health check
- `GET /api/models/info` - Model information

## 📝 License

Project ini dibuat untuk keperluan akademik - Universitas Gunadarma

## 👥 Contributors

- **[Your Name]** - Developer

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

---

**⚠️ Disclaimer**: Sistem ini adalah alat bantu deteksi. Selalu lakukan verifikasi manual untuk keputusan penting.
