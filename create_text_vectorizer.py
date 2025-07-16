#!/usr/bin/env python3
"""
Create text vectorizer for CekAjaYuk
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

def create_text_vectorizer():
    """Create and save text vectorizer"""
    print("🔧 Creating Text Vectorizer...")
    
    # Sample job posting texts for training the vectorizer
    sample_texts = [
        "lowongan kerja software developer jakarta gaji 8 juta",
        "dicari programmer python jakarta selatan",
        "butuh web developer remote work",
        "lowongan data scientist machine learning",
        "kerja part time online marketing",
        "full time backend developer nodejs",
        "frontend developer react vue angular",
        "mobile developer android ios flutter",
        "devops engineer kubernetes docker",
        "ui ux designer figma sketch",
        "digital marketing social media",
        "content writer copywriter",
        "graphic designer photoshop illustrator",
        "project manager scrum agile",
        "business analyst requirements",
        "quality assurance tester automation",
        "database administrator mysql postgresql",
        "system administrator linux windows",
        "network engineer cisco juniper",
        "cybersecurity analyst penetration testing",
        "gaji tinggi bonus tunjangan",
        "kerja dari rumah work from home",
        "pengalaman minimal 2 tahun",
        "fresh graduate welcome",
        "s1 teknik informatika komputer",
        "diploma d3 sistem informasi",
        "sertifikat profesional",
        "portfolio github linkedin",
        "interview online offline",
        "kontrak tetap freelance",
        "startup unicorn teknologi",
        "bank fintech e-commerce",
        "konsultan software house",
        "pemerintah bumn swasta",
        "jakarta bandung surabaya",
        "yogyakarta semarang medan",
        "bali denpasar makassar",
        "remote anywhere indonesia",
        "urgent segera dibutuhkan",
        "lowongan terbatas",
        "apply sekarang kirim cv",
        "email whatsapp telegram",
        "interview langsung",
        "training provided",
        "career development",
        "health insurance",
        "annual leave bonus",
        "flexible working hours",
        "modern office facility",
        "team building event",
        "professional growth"
    ]
    
    # Create TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        stop_words=None,  # Keep Indonesian stop words
        lowercase=True,
        min_df=1,
        max_df=0.95
    )
    
    # Fit the vectorizer
    vectorizer.fit(sample_texts)
    
    # Save the vectorizer
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    vectorizer_path = os.path.join(models_dir, "text_vectorizer.pkl")
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"✅ Text Vectorizer saved to: {vectorizer_path}")
    print(f"   Features: {len(vectorizer.get_feature_names_out())}")
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Test the vectorizer
    test_text = "lowongan kerja software developer jakarta"
    test_vector = vectorizer.transform([test_text])
    print(f"   Test vector shape: {test_vector.shape}")
    
    return True

if __name__ == "__main__":
    print("🚀 Creating Text Vectorizer for CekAjaYuk")
    print("=" * 50)
    
    try:
        success = create_text_vectorizer()
        if success:
            print("\n🎉 Text Vectorizer created successfully!")
        else:
            print("\n❌ Failed to create text vectorizer")
    except Exception as e:
        print(f"\n❌ Error: {e}")
