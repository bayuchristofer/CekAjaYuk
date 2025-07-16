"""
CekAjaYuk Debug Training
Script untuk debug dan perbaiki masalah training
"""

import os
import sys
import traceback
from pathlib import Path

def check_environment():
    """Check if environment is ready for training"""
    print("🔍 Checking Environment...")
    
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 7):
        issues.append(f"Python version too old: {sys.version}")
    else:
        print(f"✅ Python version: {sys.version}")
    
    # Check required packages
    required_packages = {
        'numpy': 'numpy',
        'opencv-python': 'cv2',
        'pillow': 'PIL',
        'scikit-learn': 'sklearn',
        'joblib': 'joblib'
    }
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name}: Available")
        except ImportError:
            issues.append(f"Missing package: {package_name}")
            print(f"❌ {package_name}: Missing")
    
    # Check TensorFlow separately
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
    except ImportError:
        print(f"⚠️ TensorFlow: Not available (will skip CNN training)")
    
    return issues

def check_dataset():
    """Check dataset structure and content"""
    print("\n📂 Checking Dataset...")
    
    dataset_dir = Path('dataset')
    genuine_dir = dataset_dir / 'genuine'
    fake_dir = dataset_dir / 'fake'
    
    issues = []
    
    # Check directories
    if not dataset_dir.exists():
        issues.append("Dataset directory not found")
        return issues
    
    if not genuine_dir.exists():
        issues.append("Genuine directory not found")
    
    if not fake_dir.exists():
        issues.append("Fake directory not found")
    
    if issues:
        return issues
    
    # Count files
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    genuine_files = [f for f in genuine_dir.iterdir() 
                    if f.suffix.lower() in supported_formats]
    fake_files = [f for f in fake_dir.iterdir()
                 if f.suffix.lower() in supported_formats]
    
    print(f"✅ Genuine images: {len(genuine_files)}")
    print(f"✅ Fake images: {len(fake_files)}")
    print(f"✅ Total images: {len(genuine_files) + len(fake_files)}")
    
    if len(genuine_files) < 50:
        issues.append("Too few genuine images (need at least 50)")
    
    if len(fake_files) < 50:
        issues.append("Too few fake images (need at least 50)")
    
    # Test loading a few images
    print("\n🖼️ Testing image loading...")
    test_files = genuine_files[:3] + fake_files[:3]
    
    for img_path in test_files:
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                print(f"✅ {img_path.name}: {img.size} {img.mode}")
        except Exception as e:
            issues.append(f"Cannot load {img_path.name}: {e}")
            print(f"❌ {img_path.name}: {e}")
    
    return issues

def simple_training():
    """Simple training with minimal dependencies"""
    print("\n🚀 Starting Simple Training...")
    
    try:
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score
        import joblib
        from PIL import Image
        import cv2
        
        # Load dataset
        print("📂 Loading images...")
        dataset_dir = Path('dataset')
        
        X = []
        y = []
        
        # Load genuine images
        genuine_dir = dataset_dir / 'genuine'
        for img_path in list(genuine_dir.glob('*.jpg'))[:100]:  # Limit to 100 for speed
            try:
                img = Image.open(img_path)
                img = img.convert('RGB')
                img = img.resize((64, 64))  # Small size for speed
                img_array = np.array(img).flatten()  # Simple features
                X.append(img_array)
                y.append(1)  # genuine
            except Exception as e:
                print(f"⚠️ Error loading {img_path}: {e}")
        
        # Load fake images
        fake_dir = dataset_dir / 'fake'
        for img_path in list(fake_dir.glob('*.jpg'))[:100]:  # Limit to 100 for speed
            try:
                img = Image.open(img_path)
                img = img.convert('RGB')
                img = img.resize((64, 64))  # Small size for speed
                img_array = np.array(img).flatten()  # Simple features
                X.append(img_array)
                y.append(0)  # fake
            except Exception as e:
                print(f"⚠️ Error loading {img_path}: {e}")
        
        if len(X) < 20:
            print("❌ Not enough images loaded for training")
            return False
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Loaded {len(X)} images for training")
        print(f"  Genuine: {np.sum(y == 1)}")
        print(f"  Fake: {np.sum(y == 0)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train simple Random Forest
        print("🌲 Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = rf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Training completed!")
        print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        
        # Save model
        os.makedirs('models', exist_ok=True)
        joblib.dump(rf, 'models/simple_rf_model.pkl')
        joblib.dump(scaler, 'models/simple_scaler.pkl')
        
        print(f"💾 Model saved to models/simple_rf_model.pkl")
        
        return True
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        traceback.print_exc()
        return False

def install_missing_packages():
    """Install missing packages"""
    print("\n📦 Installing Missing Packages...")
    
    packages = [
        'numpy',
        'opencv-python',
        'pillow',
        'scikit-learn',
        'joblib'
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            os.system(f"pip install {package}")
        except Exception as e:
            print(f"❌ Failed to install {package}: {e}")

def main():
    """Main debug function"""
    print("🔧 CekAjaYuk Training Debug")
    print("=" * 50)
    
    # Check environment
    env_issues = check_environment()
    
    if env_issues:
        print(f"\n❌ Environment Issues:")
        for issue in env_issues:
            print(f"  - {issue}")
        
        response = input("\nInstall missing packages? (y/n): ").strip().lower()
        if response == 'y':
            install_missing_packages()
        else:
            print("Please install missing packages manually:")
            print("pip install numpy opencv-python pillow scikit-learn joblib")
            return False
    
    # Check dataset
    dataset_issues = check_dataset()
    
    if dataset_issues:
        print(f"\n❌ Dataset Issues:")
        for issue in dataset_issues:
            print(f"  - {issue}")
        
        print(f"\nPlease fix dataset issues:")
        print(f"1. Ensure dataset/genuine/ folder exists with job posting images")
        print(f"2. Ensure dataset/fake/ folder exists with fake job posting images")
        print(f"3. Use supported formats: .jpg, .jpeg, .png")
        print(f"4. Have at least 50 images per category")
        return False
    
    # Try simple training
    print(f"\n🎯 All checks passed! Attempting simple training...")
    success = simple_training()
    
    if success:
        print(f"\n🎉 Simple training successful!")
        print(f"✅ Basic model created and saved")
        print(f"💡 You can now try the full training script")
        print(f"🚀 Next: python run.py to test the system")
    else:
        print(f"\n❌ Simple training failed")
        print(f"💡 Please check error messages above")
    
    return success

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
