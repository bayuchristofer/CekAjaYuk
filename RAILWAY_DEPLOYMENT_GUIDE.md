# Railway Deployment Guide untuk CekAjaYuk

## Masalah yang Ditemukan

Berdasarkan log error yang Anda tunjukkan, ada beberapa masalah utama:

1. **TensorFlow Compatibility Issue**: TensorFlow 2.13.0 tidak kompatibel dengan environment Railway
2. **Missing System Dependencies**: Tesseract OCR tidak terinstall di environment
3. **Heavy Dependencies**: Beberapa dependencies terlalu berat untuk deployment cloud

## Solusi yang Diterapkan

### 1. Backend Railway (backend_railway.py)
- Versi backend yang lebih ringan
- Optional loading untuk dependencies berat
- Fallback mechanisms jika dependencies tidak tersedia
- Health check endpoint untuk monitoring

### 2. Requirements Update
- Updated TensorFlow ke versi 2.15.0 (lebih kompatibel)
- Menggunakan opencv-python-headless (tanpa GUI dependencies)
- Menghapus dependencies yang tidak essential

### 3. Nixpacks Configuration (nixpacks.toml)
- Menginstall Tesseract sebagai system dependency
- Setup proper build phases
- Download NLTK data saat build

### 4. Railway Configuration (railway.toml)
- Set Python version ke 3.11
- Proper health check configuration
- Restart policy untuk handling errors

## Langkah Deployment

### Opsi 1: Gunakan Backend Railway (Recommended)
1. Commit semua perubahan:
   ```bash
   git add .
   git commit -m "Fix Railway deployment issues"
   git push
   ```

2. Di Railway dashboard:
   - Trigger manual redeploy
   - Monitor build logs
   - Check health endpoint: `/api/health`

### Opsi 2: Gunakan Requirements Minimal
1. Rename requirements file:
   ```bash
   mv requirements.txt requirements-full.txt
   mv requirements-railway.txt requirements.txt
   ```

2. Update Procfile untuk menggunakan backend_railway:
   ```
   web: gunicorn backend_railway:app --host 0.0.0.0 --port $PORT --timeout 120 --workers 1
   ```

3. Commit dan deploy

## Fitur yang Tersedia di Deployment

### ✅ Yang Berfungsi:
- Basic web server (Flask)
- File upload dan processing
- Text analysis menggunakan keyword dictionary
- Health check endpoint
- Basic OCR (jika Tesseract tersedia)

### ⚠️ Yang Mungkin Terbatas:
- Machine Learning models (jika file model tidak ada)
- Advanced OCR (tergantung Tesseract installation)
- TensorFlow models (jika TF tidak terinstall)

### 🔧 Fallback Mechanisms:
- Jika OCR gagal: menggunakan keyword analysis saja
- Jika ML models tidak ada: menggunakan rule-based analysis
- Jika dependencies tidak ada: memberikan warning tapi tetap berjalan

## Monitoring dan Debugging

### Health Check
Akses `/api/health` untuk melihat status:
```json
{
  "status": "healthy",
  "timestamp": "2025-07-17T...",
  "models_loaded": 0,
  "dependencies": {
    "opencv": true,
    "pil": true,
    "tesseract": false,
    "sklearn": true
  }
}
```

### Log Monitoring
- Check Railway logs untuk error messages
- Monitor memory usage (Railway free tier: 512MB)
- Watch for timeout issues (120s timeout set)

## Troubleshooting

### Jika Build Gagal:
1. Check requirements.txt untuk version conflicts
2. Verify nixpacks.toml syntax
3. Ensure Python 3.11 compatibility

### Jika Runtime Error:
1. Check health endpoint
2. Review Railway logs
3. Verify file permissions untuk uploads/models

### Jika OCR Tidak Berfungsi:
- App akan tetap berjalan dengan keyword analysis
- User akan mendapat disclaimer tentang OCR limitations

## Next Steps

1. **Test Deployment**: Setelah deploy, test semua endpoints
2. **Upload Models**: Upload trained models ke folder models/ jika diperlukan
3. **Monitor Performance**: Watch memory dan CPU usage
4. **Optimize**: Jika perlu, reduce model size atau optimize code

## Catatan Penting

- Railway free tier memiliki limitasi resource
- Beberapa fitur mungkin terbatas dibanding local development
- Selalu ada fallback mechanism untuk ensure app tetap berjalan
- Health check endpoint membantu monitoring status deployment
