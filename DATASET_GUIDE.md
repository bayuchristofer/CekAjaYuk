# 📊 CekAjaYuk - Panduan Pengumpulan Dataset

## 🎯 **Mengapa Dataset Real Sangat Penting?**

Sistem CekAjaYuk saat ini menggunakan **synthetic/demo data** yang hanya untuk demonstrasi. Untuk akurasi optimal dalam mendeteksi lowongan kerja palsu, diperlukan dataset gambar poster lowongan kerja **REAL**.

### ❌ **Masalah dengan Synthetic Data:**
- Akurasi rendah (~60-70%)
- Tidak mengenali pola visual real
- Overfitting pada data random
- Tidak bisa generalisasi ke kasus nyata

### ✅ **Keuntungan dengan Real Dataset:**
- Akurasi tinggi (~85-95%)
- Mengenali pola penipuan sebenarnya
- Generalisasi baik ke data baru
- Deteksi fitur visual meaningful

## 📁 **Struktur Dataset yang Dibutuhkan**

```
dataset/
├── genuine/                    # 500-1000+ poster ASLI
│   ├── company_a_job1.jpg     # Perusahaan legitimate
│   ├── government_job2.jpg    # Lowongan pemerintah
│   ├── university_job3.jpg    # Lowongan kampus
│   ├── jobfair_poster4.jpg    # Job fair resmi
│   └── ...
├── fake/                      # 500-1000+ poster PALSU
│   ├── mlm_scam1.jpg         # Penipuan MLM
│   ├── fake_company2.jpg     # Perusahaan fiktif
│   ├── unrealistic_job3.jpg  # Gaji tidak masuk akal
│   ├── whatsapp_only4.jpg    # Kontak mencurigakan
│   └── ...
└── metadata.csv              # Informasi tambahan (opsional)
```

## 🟢 **Sumber Poster Lowongan ASLI**

### 1. **Platform Resmi**
- **JobStreet Indonesia** - Screenshot lowongan verified
- **Indeed Indonesia** - Lowongan dari perusahaan terdaftar
- **LinkedIn Jobs** - Posting dari company pages resmi
- **Glints** - Startup dan tech companies
- **Kalibrr** - Platform recruitment resmi

### 2. **Website Perusahaan**
- **BUMN** - Pertamina, PLN, BNI, Mandiri, dll
- **Multinational** - Unilever, P&G, Nestle, dll
- **Tech Companies** - Gojek, Tokopedia, Bukalapak, dll
- **Banks** - BCA, BRI, CIMB, dll
- **Consulting** - McKinsey, BCG, Deloitte, dll

### 3. **Institusi Resmi**
- **Kementerian** - BUMN, Kemenkeu, Kemenkes, dll
- **Universitas** - UI, ITB, UGM career centers
- **Job Fair** - Poster resmi dari event kampus
- **Media Massa** - Kompas Karir, Tempo Jobs

### 4. **Media Sosial Resmi**
- **Instagram** company official accounts
- **Facebook** verified business pages
- **Twitter** corporate recruitment posts
- **YouTube** company recruitment videos (screenshot)

## 🔴 **Sumber Poster Lowongan PALSU**

### 1. **Laporan Pengguna**
- **Forum Kaskus** - Thread "Hati-hati penipuan kerja"
- **Reddit r/indonesia** - User reports
- **Facebook Groups** - "Info Lowongan Kerja" (yang tidak dimoderasi)
- **Telegram Groups** - Channel lowongan kerja abal-abal

### 2. **Arsip Berita Penipuan**
- **Detik.com** - Berita expose penipuan kerja
- **Kompas.com** - Laporan investigasi
- **Tribunnews** - Kasus penipuan viral
- **Liputan6** - Warning dari kepolisian

### 3. **Screenshot dari Korban**
- **WhatsApp Groups** - Forward pesan penipuan
- **Instagram Stories** - Warning dari influencer
- **TikTok** - Video expose penipuan kerja
- **Twitter** - Thread warning penipuan

### 4. **Simulasi Realistis** (Hati-hati!)
- Buat poster palsu dengan ciri-ciri umum scam
- **JANGAN** gunakan nama perusahaan real
- Gunakan watermark "CONTOH PENIPUAN"
- Fokus pada pattern, bukan konten spesifik

## 🔍 **Karakteristik yang Harus Dipelajari Model**

### 🟢 **Ciri Visual Poster ASLI:**

#### **Layout & Design:**
- Logo perusahaan yang konsisten dan profesional
- Layout terstruktur dengan hierarchy yang jelas
- Font yang konsisten dan mudah dibaca
- Color scheme yang profesional
- White space yang cukup, tidak cramped

#### **Konten Informasi:**
- Alamat kantor lengkap dengan kode pos
- Email domain perusahaan (@company.com)
- Website resmi perusahaan
- Nomor telepon kantor (bukan HP)
- Informasi HR contact person

#### **Bahasa & Teks:**
- Bahasa formal dan profesional
- Grammar yang benar, minimal typo
- Job description yang detail dan spesifik
- Requirement yang realistis
- Benefit yang masuk akal

### 🔴 **Ciri Visual Poster PALSU:**

#### **Layout & Design:**
- Logo yang blur, tidak konsisten, atau tanpa logo
- Layout berantakan, terlalu ramai
- Font yang tidak konsisten, terlalu banyak variasi
- Warna yang mencolok berlebihan
- Terlalu banyak teks dalam satu area

#### **Konten Mencurigakan:**
- Kontak WhatsApp only, tanpa email/alamat
- Gaji yang tidak realistis (terlalu tinggi/rendah)
- "Urgent", "ASAP", "Segera" berlebihan
- Tidak ada requirement jelas
- Promise yang terlalu bagus

#### **Red Flags Bahasa:**
- Banyak typo dan grammar error
- Bahasa tidak formal, seperti chat
- Penggunaan emoji berlebihan
- Klaim "mudah", "tanpa pengalaman"
- Tekanan waktu yang berlebihan

## 🛠️ **Tools untuk Pengumpulan Dataset**

### 1. **Screenshot Tools:**
```bash
# Browser extensions
- Full Page Screen Capture (Chrome)
- FireShot (Firefox)
- Awesome Screenshot

# Desktop tools
- Snipping Tool (Windows)
- Screenshot (macOS)
- Flameshot (Linux)
```

### 2. **Batch Download:**
```python
# Python script untuk download otomatis
import requests
from selenium import webdriver

def scrape_job_postings(url_list):
    driver = webdriver.Chrome()
    for url in url_list:
        driver.get(url)
        # Screenshot logic here
        driver.save_screenshot(f"job_{i}.png")
```

### 3. **Image Processing:**
```python
# Standardize image format
from PIL import Image
import os

def standardize_images(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        img = Image.open(os.path.join(input_dir, filename))
        img = img.convert('RGB')
        img = img.resize((800, 600))  # Standard size
        img.save(os.path.join(output_dir, f"{filename}.jpg"))
```

## 📋 **Checklist Pengumpulan Dataset**

### ✅ **Persiapan:**
- [ ] Buat folder `dataset/genuine/` dan `dataset/fake/`
- [ ] Install tools screenshot
- [ ] Siapkan spreadsheet untuk tracking
- [ ] Set target minimal 500 gambar per kategori

### ✅ **Pengumpulan Genuine:**
- [ ] Screenshot dari JobStreet (100+ gambar)
- [ ] Screenshot dari LinkedIn Jobs (100+ gambar)
- [ ] Website perusahaan BUMN (50+ gambar)
- [ ] Job fair poster kampus (50+ gambar)
- [ ] Media sosial perusahaan verified (100+ gambar)

### ✅ **Pengumpulan Fake:**
- [ ] Screenshot dari laporan penipuan (200+ gambar)
- [ ] Arsip berita expose scam (100+ gambar)
- [ ] Forward WhatsApp/Telegram scam (200+ gambar)
- [ ] Simulasi poster palsu (100+ gambar)

### ✅ **Quality Control:**
- [ ] Semua gambar readable dan clear
- [ ] Format konsisten (JPG/PNG)
- [ ] Ukuran file reasonable (<5MB)
- [ ] Tidak ada duplikasi
- [ ] Label yang benar (genuine/fake)

## 🚀 **Implementasi dengan Dataset Real**

### 1. **Jalankan Notebook Persiapan:**
```bash
# Buka notebook khusus dataset real
jupyter notebook notebooks/0_real_dataset_preparation.ipynb

# Atau gunakan script otomatis
python prepare_real_dataset.py
```

### 2. **Training dengan Data Real:**
```bash
# Train semua model dengan dataset real
python train_models.py --use-real-dataset

# Atau manual per model
jupyter notebook notebooks/2_random_forest_training.ipynb
jupyter notebook notebooks/3_tensorflow_training.ipynb
```

### 3. **Evaluasi Peningkatan:**
```python
# Compare performance
print("Synthetic Data Performance:")
print(f"Accuracy: {synthetic_accuracy:.2f}")

print("Real Data Performance:")
print(f"Accuracy: {real_accuracy:.2f}")
print(f"Improvement: +{(real_accuracy - synthetic_accuracy)*100:.1f}%")
```

## 📊 **Expected Performance Improvement**

| Metric | Synthetic Data | Real Data | Improvement |
|--------|---------------|-----------|-------------|
| **Accuracy** | 60-70% | 85-95% | +25-35% |
| **Precision** | 65-75% | 88-96% | +23-31% |
| **Recall** | 55-70% | 82-94% | +27-39% |
| **F1-Score** | 60-72% | 85-95% | +25-33% |

## ⚠️ **Ethical Considerations**

### 1. **Privacy:**
- Blur personal information (nama, nomor HP pribadi)
- Jangan gunakan data pribadi tanpa izin
- Anonymize company names jika perlu

### 2. **Legal:**
- Gunakan hanya untuk research/educational purposes
- Tidak untuk komersial tanpa izin
- Respect website terms of service

### 3. **Accuracy:**
- Double-check label (genuine/fake)
- Jangan bias terhadap perusahaan tertentu
- Include diverse job types dan industries

## 🎯 **Next Steps**

1. **📊 Kumpulkan Dataset** - Target 1000+ gambar per kategori
2. **🔄 Re-train Models** - Gunakan notebook real dataset
3. **📈 Evaluate Performance** - Compare dengan baseline
4. **🚀 Deploy Improved System** - Update production models
5. **📝 Document Results** - Share improvement metrics

---

**💡 Tips:** Mulai dengan 200-300 gambar per kategori untuk proof of concept, kemudian scale up untuk production quality!

**🎯 Goal:** Mencapai akurasi 90%+ untuk melindungi masyarakat dari penipuan lowongan kerja!
