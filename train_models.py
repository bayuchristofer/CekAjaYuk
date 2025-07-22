#!/usr/bin/env python3
"""
CekAjaYuk Model Training Script
Extracted from retrain_models.ipynb for easier execution
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import OCR function from backend
sys.path.append('.')
try:
    from backend_working import extract_text_with_ocr
    print("✅ OCR function imported successfully")

    # Test OCR setup
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    print("✅ Tesseract path configured")

except ImportError as e:
    print(f"❌ Error importing OCR function: {e}")
    print("Please ensure backend_working.py is in the same directory")
    sys.exit(1)

def extract_text_from_images(image_paths, label, max_images=50):
    """Extract text from images with progress tracking"""
    from PIL import Image
    
    texts = []
    labels = []
    failed_count = 0
    
    print(f"🔍 Extracting text from {min(len(image_paths), max_images)} {label} images...")
    
    for i, img_path in enumerate(image_paths[:max_images]):
        try:
            # Load image
            image = Image.open(img_path)
            
            # Extract text
            text = extract_text_with_ocr(image)
            
            if len(text.strip()) > 10:  # Only include if meaningful text extracted
                texts.append(text.strip())
                labels.append(label)
            else:
                failed_count += 1
            
            # Progress update
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{min(len(image_paths), max_images)} images...")
                
        except Exception as e:
            print(f"   ❌ Failed to process {img_path.name}: {e}")
            failed_count += 1
    
    print(f"   ✅ Successfully extracted: {len(texts)} texts")
    print(f"   ❌ Failed extractions: {failed_count}")
    
    return texts, labels

def extract_features(texts):
    """Extract comprehensive features from texts"""
    # Enhanced keyword lists
    GENUINE_KEYWORDS = [
        'pengalaman', 'kualifikasi', 'syarat', 'tanggung jawab', 'tunjangan',
        'gaji', 'wawancara', 'lamaran', 'kandidat', 'posisi', 'lowongan',
        'kerja', 'pekerjaan', 'perusahaan', 'pt', 'cv', 'tbk', 'profesional',
        'karir', 'jabatan', 'keahlian', 'kemampuan', 'keterampilan',
        'pendidikan', 'gelar', 'ijazah', 'sertifikat', 'pelatihan',
        'interview', 'recruitment', 'hiring', 'vacancy', 'position',
        'experience', 'qualification', 'requirement', 'responsibility',
        'salary', 'benefit', 'career', 'professional', 'skill',
        'kantor', 'office', 'company', 'corporation', 'enterprise',
        'industri', 'bisnis', 'organisasi', 'institusi', 'lembaga'
    ]
    
    FAKE_KEYWORDS = [
        # Urgency/pressure words
        'mudah', 'cepat', 'instant', 'langsung', 'tanpa modal', 'gratis',
        'buruan', 'terbatas', 'deadline', 'segera', 'jangan sampai', 'terlewat',
        'kesempatan emas', 'limited time', 'sekarang juga', 'hari ini',

        # MLM/Scam indicators
        'kerja rumah', 'work from home', 'online', 'part time', 'freelance',
        'sampingan', 'tambahan', 'passive income', 'join', 'member',
        'downline', 'upline', 'bonus', 'komisi', 'reward', 'cashback',

        # Money promises
        'jutaan', 'milyar', 'unlimited', 'tak terbatas', 'penghasilan besar',
        'kaya', 'sukses', 'investasi', 'trading', 'forex', 'crypto', 'bitcoin',

        # Suspicious contact methods
        'whatsapp', 'wa', 'telegram', 'dm', 'chat', 'hubungi', 'kontak',
        'no interview', 'tanpa wawancara', 'langsung kerja', 'tanpa pengalaman'
    ]
    
    features = []
    
    for text in texts:
        text_lower = text.lower()
        
        # Basic features
        feature_dict = {
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len([s for s in text.split('.') if s.strip()]),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
        }
        
        # Keyword features
        genuine_count = sum(1 for kw in GENUINE_KEYWORDS if kw in text_lower)
        fake_count = sum(1 for kw in FAKE_KEYWORDS if kw in text_lower)
        
        feature_dict.update({
            'genuine_keywords': genuine_count,
            'fake_keywords': fake_count,
            'keyword_ratio': genuine_count / max(fake_count, 1),
        })
        
        # Structure features (ENHANCED)
        feature_dict.update({
            'has_email': '@' in text,
            'has_phone': any(char.isdigit() for char in text),
            'has_address': any(word in text_lower for word in ['jl', 'jalan', 'street', 'alamat']),
            'has_company': any(word in text_lower for word in ['pt', 'cv', 'ltd', 'inc', 'corp']),

            # Advanced fake indicators
            'has_whatsapp': any(word in text_lower for word in ['whatsapp', 'wa', 'chat']),
            'has_money_promise': any(word in text_lower for word in ['jutaan', 'milyar', 'kaya', 'sukses']),
            'has_urgency': any(word in text_lower for word in ['buruan', 'segera', 'terbatas', 'deadline']),
            'has_mlm_terms': any(word in text_lower for word in ['join', 'member', 'bonus', 'komisi']),
            'has_no_experience': any(word in text_lower for word in ['tanpa pengalaman', 'no experience', 'fresh graduate']),

            # Text quality indicators
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'number_count': sum(1 for c in text if c.isdigit()),
        })
        
        features.append(feature_dict)
    
    return pd.DataFrame(features)

def main():
    """Main training function"""
    print("🚀 STARTING MODEL TRAINING")
    print("=" * 60)
    
    # Check dataset
    DATASET_DIR = Path('dataset')
    FAKE_DIR = DATASET_DIR / 'fake'
    GENUINE_DIR = DATASET_DIR / 'genuine'
    
    fake_images = list(FAKE_DIR.glob('*.jpg')) if FAKE_DIR.exists() else []
    genuine_images = list(GENUINE_DIR.glob('*.JPG')) if GENUINE_DIR.exists() else []
    
    print(f"📊 Dataset Overview:")
    print(f"   Fake images: {len(fake_images)}")
    print(f"   Genuine images: {len(genuine_images)}")
    
    if len(fake_images) == 0 or len(genuine_images) == 0:
        print("❌ Dataset not found! Please ensure dataset/fake and dataset/genuine directories exist.")
        return
    
    # Extract text from images (INCREASED for better accuracy)
    fake_texts, fake_labels = extract_text_from_images(fake_images, 'fake', max_images=100)
    genuine_texts, genuine_labels = extract_text_from_images(genuine_images, 'genuine', max_images=100)
    
    # Combine data
    all_texts = fake_texts + genuine_texts
    all_labels = fake_labels + genuine_labels
    
    if len(all_texts) < 10:
        print("❌ Not enough text extracted for training!")
        return
    
    print(f"\n📊 Text Extraction Summary:")
    print(f"   Total texts: {len(all_texts)}")
    print(f"   Fake: {len(fake_texts)}, Genuine: {len(genuine_texts)}")
    
    # Extract features
    print("\n🔧 Extracting features...")
    features_df = extract_features(all_texts)
    features_df['label'] = all_labels
    
    # Train models
    X = features_df.drop(['label'], axis=1)
    y = features_df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n🌲 Training Random Forest with optimized parameters...")
    rf_model = RandomForestClassifier(
        n_estimators=200,           # More trees for better accuracy
        max_depth=15,               # Deeper trees
        min_samples_split=3,        # More sensitive to patterns
        min_samples_leaf=1,         # Allow smaller leaves
        max_features='sqrt',        # Feature selection
        class_weight='balanced',    # Handle class imbalance
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    print(f"✅ Random Forest Accuracy: {rf_accuracy:.3f}")
    
    # Train Text Classifier
    print(f"\n📝 Training Text Classifier...")
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    
    texts_train, texts_test, labels_train, labels_test = train_test_split(
        all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    X_tfidf_train = tfidf.fit_transform(texts_train)
    X_tfidf_test = tfidf.transform(texts_test)
    
    text_model = LogisticRegression(random_state=42, max_iter=1000)
    text_model.fit(X_tfidf_train, labels_train)
    text_pred = text_model.predict(X_tfidf_test)
    text_accuracy = accuracy_score(labels_test, text_pred)
    
    print(f"✅ Text Classifier Accuracy: {text_accuracy:.3f}")
    
    # Save models
    print(f"\n💾 Saving models...")
    joblib.dump(rf_model, 'models/random_forest_retrained.pkl')
    joblib.dump(text_model, 'models/text_classifier_retrained.pkl')
    joblib.dump(tfidf, 'models/tfidf_vectorizer_retrained.pkl')
    
    print(f"✅ Models saved successfully!")
    print(f"\n📊 Training Summary:")
    print(f"   Random Forest: {rf_accuracy:.3f} ({rf_accuracy*100:.1f}%)")
    print(f"   Text Classifier: {text_accuracy:.3f} ({text_accuracy*100:.1f}%)")

if __name__ == "__main__":
    main()
