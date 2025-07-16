"""
CekAjaYuk Training with Real Dataset
Script untuk training model dengan dataset real 800 gambar
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow not available. Only Random Forest will be trained.")
    TF_AVAILABLE = False

class RealDatasetTrainer:
    """Trainer for real job posting dataset"""
    
    def __init__(self, dataset_dir='dataset', output_dir='models'):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Image settings
        self.img_size = (224, 224)
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Training settings
        self.test_size = 0.2
        self.val_size = 0.2
        self.random_state = 42
        
    def load_real_dataset(self):
        """Load real job posting images"""
        print("📂 Loading real dataset...")
        
        genuine_dir = self.dataset_dir / 'genuine'
        fake_dir = self.dataset_dir / 'fake'
        
        if not genuine_dir.exists() or not fake_dir.exists():
            raise FileNotFoundError(f"Dataset directories not found: {genuine_dir}, {fake_dir}")
        
        images = []
        labels = []
        filenames = []
        
        # Load genuine images
        print("  Loading genuine job postings...")
        genuine_files = [f for f in genuine_dir.iterdir() 
                        if f.suffix.lower() in self.supported_formats]
        
        for img_path in genuine_files:
            img = self._load_and_preprocess_image(img_path)
            if img is not None:
                images.append(img)
                labels.append(1)  # 1 for genuine
                filenames.append(img_path.name)
        
        # Load fake images
        print("  Loading fake job postings...")
        fake_files = [f for f in fake_dir.iterdir()
                     if f.suffix.lower() in self.supported_formats]
        
        for img_path in fake_files:
            img = self._load_and_preprocess_image(img_path)
            if img is not None:
                images.append(img)
                labels.append(0)  # 0 for fake
                filenames.append(img_path.name)
        
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"✅ Loaded {len(images)} images successfully")
        print(f"  Genuine: {np.sum(labels == 1)}")
        print(f"  Fake: {np.sum(labels == 0)}")
        print(f"  Image shape: {images.shape}")
        
        return images, labels, filenames
    
    def _load_and_preprocess_image(self, img_path):
        """Load and preprocess single image"""
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to standard size
            img = cv2.resize(img, self.img_size)
            
            # Normalize pixel values
            img = img.astype(np.float32) / 255.0
            
            return img
            
        except Exception as e:
            print(f"⚠️ Error processing {img_path}: {e}")
            return None
    
    def extract_traditional_features(self, images):
        """Extract traditional ML features"""
        print("🔍 Extracting traditional features for Random Forest...")
        
        features_list = []
        feature_names = [
            'mean_red', 'mean_green', 'mean_blue',
            'std_red', 'std_green', 'std_blue',
            'brightness', 'contrast', 'edge_density',
            'texture_variance', 'color_diversity', 'layout_symmetry'
        ]
        
        for i, img in enumerate(images):
            if i % 100 == 0:
                print(f"  Processing {i+1}/{len(images)}...")
            
            # Convert to uint8 for OpenCV operations
            img_uint8 = (img * 255).astype(np.uint8)
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            
            features = []
            
            # Color features
            features.extend(np.mean(img, axis=(0, 1)))  # Mean RGB
            features.extend(np.std(img, axis=(0, 1)))   # Std RGB
            
            # Brightness and contrast
            brightness = np.mean(gray)
            contrast = np.std(gray)
            features.extend([brightness, contrast])
            
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            features.append(edge_density)
            
            # Texture variance
            texture_var = np.var(gray)
            features.append(texture_var)
            
            # Color diversity
            unique_colors = len(np.unique(img_uint8.reshape(-1, 3), axis=0))
            color_diversity = unique_colors / (self.img_size[0] * self.img_size[1])
            features.append(color_diversity)
            
            # Layout symmetry
            left_half = gray[:, :gray.shape[1]//2]
            right_half = np.fliplr(gray[:, gray.shape[1]//2:])
            if left_half.shape == right_half.shape:
                symmetry = 1 - np.mean(np.abs(left_half - right_half)) / 255
            else:
                symmetry = 0
            features.append(symmetry)
            
            features_list.append(features)
        
        features_array = np.array(features_list)
        print(f"✅ Feature extraction completed: {features_array.shape}")
        
        return features_array, feature_names
    
    def train_random_forest(self, X_features, y_labels):
        """Train Random Forest classifier"""
        print("\n🌲 Training Random Forest Classifier...")
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_features, y_labels, test_size=self.test_size + self.val_size, 
            random_state=self.random_state, stratify=y_labels
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=self.test_size / (self.test_size + self.val_size),
            random_state=self.random_state, stratify=y_temp
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        rf.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = rf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Random Forest Training Completed!")
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Test samples: {len(X_test)}")
        
        # Save model and scaler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rf_path = self.output_dir / f'random_forest_real_{timestamp}.pkl'
        scaler_path = self.output_dir / 'feature_scaler.pkl'
        latest_rf_path = self.output_dir / 'random_forest_classifier_latest.pkl'
        
        joblib.dump(rf, rf_path)
        joblib.dump(rf, latest_rf_path)
        joblib.dump(scaler, scaler_path)
        
        print(f"💾 Model saved: {rf_path}")
        print(f"💾 Latest model: {latest_rf_path}")
        print(f"💾 Scaler saved: {scaler_path}")
        
        return rf, scaler, accuracy
    
    def train_cnn(self, X_images, y_labels):
        """Train CNN with real images"""
        if not TF_AVAILABLE:
            print("⚠️ TensorFlow not available. Skipping CNN training.")
            return None, 0
        
        print("\n🧠 Training CNN with Real Images...")
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_images, y_labels, test_size=self.test_size + self.val_size,
            random_state=self.random_state, stratify=y_labels
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=self.test_size / (self.test_size + self.val_size),
            random_state=self.random_state, stratify=y_temp
        )
        
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Test samples: {len(X_test)}")
        
        # Create model with transfer learning
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        base_model.trainable = False
        
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=str(self.output_dir / 'cnn_best_real.h5'),
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=50,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"✅ CNN Training Completed!")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cnn_path = self.output_dir / f'tensorflow_real_{timestamp}.h5'
        latest_cnn_path = self.output_dir / 'tensorflow_model_latest.h5'
        
        model.save(cnn_path)
        model.save(latest_cnn_path)
        
        print(f"💾 Model saved: {cnn_path}")
        print(f"💾 Latest model: {latest_cnn_path}")
        
        return model, test_accuracy

def main():
    """Main training function"""
    print("🚀 CekAjaYuk Training with Real Dataset")
    print("=" * 50)
    
    # Initialize trainer
    trainer = RealDatasetTrainer()
    
    try:
        # Load real dataset
        images, labels, filenames = trainer.load_real_dataset()
        
        # Extract features for Random Forest
        features, feature_names = trainer.extract_traditional_features(images)
        
        # Train Random Forest
        rf_model, scaler, rf_accuracy = trainer.train_random_forest(features, labels)
        
        # Train CNN
        cnn_model, cnn_accuracy = trainer.train_cnn(images, labels)
        
        # Summary
        print(f"\n🎉 Training Summary:")
        print(f"  Dataset: {len(images)} real job posting images")
        print(f"  Random Forest Accuracy: {rf_accuracy:.4f}")
        if TF_AVAILABLE:
            print(f"  CNN Accuracy: {cnn_accuracy:.4f}")
        
        # Save training info
        training_info = {
            'timestamp': datetime.now().isoformat(),
            'dataset_size': len(images),
            'genuine_samples': int(np.sum(labels == 1)),
            'fake_samples': int(np.sum(labels == 0)),
            'rf_accuracy': float(rf_accuracy),
            'cnn_accuracy': float(cnn_accuracy) if TF_AVAILABLE else None,
            'dataset_type': 'real',
            'feature_names': feature_names
        }
        
        with open(trainer.output_dir / 'training_info_real.json', 'w') as f:
            json.dump(training_info, f, indent=2)
        
        print(f"\n🎯 Expected Performance Improvement:")
        print(f"  Previous (Synthetic): ~70% accuracy")
        print(f"  Current (Real): ~{rf_accuracy*100:.0f}% accuracy")
        print(f"  Improvement: +{(rf_accuracy-0.7)*100:.0f}% accuracy boost!")
        
        print(f"\n✅ Training completed! Models ready for production use.")
        print(f"💡 Next step: Run 'python run.py' to test improved system")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
