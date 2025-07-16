"""
Minimal Training Script - Basic functionality only
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_images_simple(dataset_dir='dataset', max_per_class=200):
    """Load images with minimal processing"""
    print(f"📂 Loading images from {dataset_dir}...")
    
    X = []
    y = []
    
    # Load genuine images
    genuine_dir = Path(dataset_dir) / 'genuine'
    if genuine_dir.exists():
        genuine_files = list(genuine_dir.glob('*.jpg')) + list(genuine_dir.glob('*.png'))
        print(f"Found {len(genuine_files)} genuine images")
        
        for i, img_path in enumerate(genuine_files[:max_per_class]):
            if i % 50 == 0:
                print(f"  Loading genuine {i+1}/{min(len(genuine_files), max_per_class)}")
            
            try:
                # Simple image processing
                img = Image.open(img_path)
                img = img.convert('RGB')
                img = img.resize((32, 32))  # Very small for speed
                
                # Extract simple features
                img_array = np.array(img)
                features = [
                    np.mean(img_array[:,:,0]),  # Red mean
                    np.mean(img_array[:,:,1]),  # Green mean
                    np.mean(img_array[:,:,2]),  # Blue mean
                    np.std(img_array[:,:,0]),   # Red std
                    np.std(img_array[:,:,1]),   # Green std
                    np.std(img_array[:,:,2]),   # Blue std
                    np.mean(img_array),         # Overall brightness
                    np.std(img_array)           # Overall contrast
                ]
                
                X.append(features)
                y.append(1)  # genuine
                
            except Exception as e:
                print(f"⚠️ Error loading {img_path.name}: {e}")
    
    # Load fake images
    fake_dir = Path(dataset_dir) / 'fake'
    if fake_dir.exists():
        fake_files = list(fake_dir.glob('*.jpg')) + list(fake_dir.glob('*.png'))
        print(f"Found {len(fake_files)} fake images")
        
        for i, img_path in enumerate(fake_files[:max_per_class]):
            if i % 50 == 0:
                print(f"  Loading fake {i+1}/{min(len(fake_files), max_per_class)}")
            
            try:
                # Simple image processing
                img = Image.open(img_path)
                img = img.convert('RGB')
                img = img.resize((32, 32))  # Very small for speed
                
                # Extract simple features
                img_array = np.array(img)
                features = [
                    np.mean(img_array[:,:,0]),  # Red mean
                    np.mean(img_array[:,:,1]),  # Green mean
                    np.mean(img_array[:,:,2]),  # Blue mean
                    np.std(img_array[:,:,0]),   # Red std
                    np.std(img_array[:,:,1]),   # Green std
                    np.std(img_array[:,:,2]),   # Blue std
                    np.mean(img_array),         # Overall brightness
                    np.std(img_array)           # Overall contrast
                ]
                
                X.append(features)
                y.append(0)  # fake
                
            except Exception as e:
                print(f"⚠️ Error loading {img_path.name}: {e}")
    
    return np.array(X), np.array(y)

def train_minimal_model():
    """Train minimal model with basic features"""
    print("🚀 Starting Minimal Training...")
    
    # Load data
    X, y = load_images_simple()
    
    if len(X) < 20:
        print("❌ Not enough images loaded. Need at least 20 images.")
        return False
    
    print(f"✅ Loaded {len(X)} images")
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
    
    # Train Random Forest
    print("🌲 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Training completed!")
    print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Detailed report
    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Genuine']))
    
    # Save model
    os.makedirs('models', exist_ok=True)
    
    model_path = 'models/random_forest_classifier_latest.pkl'
    scaler_path = 'models/feature_scaler.pkl'
    
    joblib.dump(rf, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"💾 Model saved: {model_path}")
    print(f"💾 Scaler saved: {scaler_path}")
    
    # Create model info
    model_info = {
        'model_type': 'RandomForest',
        'accuracy': float(accuracy),
        'features': 8,
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    import json
    with open('models/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    return True

def main():
    """Main function"""
    print("🔧 CekAjaYuk Minimal Training")
    print("=" * 40)
    
    # Check dataset
    dataset_dir = Path('dataset')
    if not dataset_dir.exists():
        print("❌ Dataset directory not found!")
        print("Please ensure you have dataset/genuine/ and dataset/fake/ folders")
        return False
    
    genuine_dir = dataset_dir / 'genuine'
    fake_dir = dataset_dir / 'fake'
    
    if not genuine_dir.exists() or not fake_dir.exists():
        print("❌ Genuine or fake directories not found!")
        print("Please ensure you have:")
        print("  - dataset/genuine/ with genuine job posting images")
        print("  - dataset/fake/ with fake job posting images")
        return False
    
    # Count files
    genuine_count = len(list(genuine_dir.glob('*.jpg')) + list(genuine_dir.glob('*.png')))
    fake_count = len(list(fake_dir.glob('*.jpg')) + list(fake_dir.glob('*.png')))
    
    print(f"📊 Dataset Summary:")
    print(f"  Genuine images: {genuine_count}")
    print(f"  Fake images: {fake_count}")
    print(f"  Total: {genuine_count + fake_count}")
    
    if genuine_count < 10 or fake_count < 10:
        print("❌ Need at least 10 images per category")
        return False
    
    # Train model
    success = train_minimal_model()
    
    if success:
        print(f"\n🎉 Minimal training successful!")
        print(f"✅ Basic model created with your real dataset")
        print(f"🚀 Next steps:")
        print(f"  1. Run: python run.py")
        print(f"  2. Test the system with job posting images")
        print(f"  3. See improved accuracy compared to demo!")
    
    return success

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
