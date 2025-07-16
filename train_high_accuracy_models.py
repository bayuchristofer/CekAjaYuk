#!/usr/bin/env python3
"""
CekAjaYuk - High Accuracy Model Training
Training semua model dengan target akurasi 90%+ dan anti-overfitting
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Image Processing
from PIL import Image
import cv2

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TF_AVAILABLE = True
    print("✅ TensorFlow available")
except ImportError:
    TF_AVAILABLE = False
    print("❌ TensorFlow not available")

class HighAccuracyTrainer:
    """Trainer untuk model dengan akurasi tinggi dan anti-overfitting"""
    
    def __init__(self, dataset_dir='dataset', output_dir='models'):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Image settings
        self.img_size = (224, 224)
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # Training settings
        self.random_state = 42
        self.test_size = 0.2
        self.val_size = 0.2
        
        print(f"🎯 Target: Akurasi 90%+ dengan anti-overfitting")
        print(f"📁 Dataset: {self.dataset_dir}")
        print(f"💾 Output: {self.output_dir}")
        
    def load_real_dataset(self):
        """Load real job posting images dengan augmentasi"""
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
        print(f"   Genuine: {np.sum(labels == 1)}")
        print(f"   Fake: {np.sum(labels == 0)}")
        print(f"   Balance ratio: {min(np.sum(labels == 1), np.sum(labels == 0)) / max(np.sum(labels == 1), np.sum(labels == 0)):.2f}")
        
        return images, labels, filenames
    
    def _load_and_preprocess_image(self, img_path):
        """Load dan preprocess image dengan error handling"""
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            img = img.resize(self.img_size)
            img_array = np.array(img) / 255.0  # Normalize to [0,1]
            return img_array
        except Exception as e:
            print(f"    Warning: Could not load {img_path.name}: {e}")
            return None
    
    def extract_advanced_features(self, images, labels):
        """Extract advanced features untuk Random Forest"""
        print("🔍 Extracting advanced features...")
        
        features = []
        feature_names = [
            'mean_brightness', 'std_brightness', 'contrast', 'sharpness',
            'edge_density', 'color_variance', 'text_area_ratio', 'logo_presence',
            'layout_symmetry', 'color_harmony', 'noise_level', 'professional_score'
        ]
        
        for i, img in enumerate(images):
            if i % 100 == 0:
                print(f"    Processing image {i+1}/{len(images)}")
            
            # Convert to different color spaces
            img_uint8 = (img * 255).astype(np.uint8)
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
            
            # Feature extraction
            feat = []
            
            # 1. Brightness features
            feat.append(np.mean(gray) / 255.0)
            feat.append(np.std(gray) / 255.0)
            
            # 2. Contrast (using Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            feat.append(np.var(laplacian) / 10000.0)  # Normalized
            
            # 3. Sharpness (gradient magnitude)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            feat.append(np.mean(np.sqrt(sobelx**2 + sobely**2)) / 255.0)
            
            # 4. Edge density
            edges = cv2.Canny(gray, 50, 150)
            feat.append(np.sum(edges > 0) / edges.size)
            
            # 5. Color variance
            feat.append(np.var(img.reshape(-1, 3), axis=0).mean())
            
            # 6. Text area estimation (high frequency content)
            feat.append(np.sum(edges > 0) / edges.size)
            
            # 7. Logo presence (corner analysis)
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
            feat.append(len(corners) / 100.0 if corners is not None else 0.0)
            
            # 8. Layout symmetry
            left_half = gray[:, :gray.shape[1]//2]
            right_half = gray[:, gray.shape[1]//2:]
            right_half_flipped = np.fliplr(right_half)
            if left_half.shape == right_half_flipped.shape:
                symmetry = np.corrcoef(left_half.flatten(), right_half_flipped.flatten())[0, 1]
                feat.append(max(0, symmetry))
            else:
                feat.append(0.0)
            
            # 9. Color harmony (HSV analysis)
            h_channel = hsv[:, :, 0]
            feat.append(1.0 - (np.std(h_channel) / 180.0))
            
            # 10. Noise level
            feat.append(np.std(laplacian) / 1000.0)
            
            # 11. Professional score (combination of features)
            professional_score = (feat[2] + feat[3] + feat[7] + feat[8] + feat[9]) / 5.0
            feat.append(professional_score)
            
            features.append(feat)
        
        features = np.array(features)
        print(f"✅ Extracted {features.shape[1]} features from {features.shape[0]} images")
        
        return features, feature_names
    
    def train_random_forest_advanced(self, X, y, feature_names):
        """Train Random Forest dengan hyperparameter tuning untuk akurasi tinggi"""
        print("🌲 Training Advanced Random Forest...")
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=self.random_state, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.random_state, stratify=y_temp
        )
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Hyperparameter tuning untuk akurasi tinggi
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        # Cross-validation untuk mencegah overfitting
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        
        print("  Performing hyperparameter tuning...")
        grid_search = GridSearchCV(
            rf, param_grid, cv=cv, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Best model
        best_rf = grid_search.best_estimator_
        
        # Evaluate
        train_acc = best_rf.score(X_train_scaled, y_train)
        val_acc = best_rf.score(X_val_scaled, y_val)
        test_acc = best_rf.score(X_test_scaled, y_test)
        
        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  Training accuracy: {train_acc:.4f}")
        print(f"  Validation accuracy: {val_acc:.4f}")
        print(f"  Test accuracy: {test_acc:.4f}")
        
        # Check overfitting
        overfitting = train_acc - val_acc
        print(f"  Overfitting gap: {overfitting:.4f}")
        
        if test_acc >= 0.90:
            print("  ✅ Target akurasi 90%+ tercapai!")
        else:
            print(f"  ⚠️ Akurasi {test_acc:.1%} belum mencapai target 90%")
        
        # Save model
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.output_dir / f'random_forest_advanced_{timestamp}.pkl'
        scaler_path = self.output_dir / f'feature_scaler_advanced_{timestamp}.pkl'
        
        joblib.dump(best_rf, model_path)
        joblib.dump(scaler, scaler_path)
        
        # Save feature names
        feature_names_path = self.output_dir / f'feature_names_advanced_{timestamp}.txt'
        with open(feature_names_path, 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
        
        print(f"  💾 Model saved: {model_path}")
        print(f"  💾 Scaler saved: {scaler_path}")
        
        return best_rf, scaler, test_acc

    def train_cnn_advanced(self, images, labels):
        """Train CNN dengan arsitektur advanced dan anti-overfitting"""
        if not TF_AVAILABLE:
            print("❌ TensorFlow not available, skipping CNN training")
            return None, 0.0

        print("🧠 Training Advanced CNN...")

        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            images, labels, test_size=0.4, random_state=self.random_state, stratify=labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.random_state, stratify=y_temp
        )

        print(f"  Training set: {X_train.shape}")
        print(f"  Validation set: {X_val.shape}")
        print(f"  Test set: {X_test.shape}")

        # Data augmentation untuk mencegah overfitting
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Advanced CNN architecture
        model = Sequential([
            # Block 1
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Block 2
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Block 3
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Block 4
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Classifier
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        # Compile dengan learning rate scheduling
        initial_lr = 0.001
        model.compile(
            optimizer=Adam(learning_rate=initial_lr),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        print(f"  Model parameters: {model.count_params():,}")

        # Callbacks untuk anti-overfitting
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.output_dir / f'cnn_advanced_{timestamp}.h5'

        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                str(model_path),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

        # Training dengan data augmentation
        print("  Starting training...")
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            steps_per_epoch=len(X_train) // 32,
            epochs=100,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        # Load best model
        model.load_weights(str(model_path))

        # Evaluate
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

        print(f"  Training accuracy: {train_acc:.4f}")
        print(f"  Validation accuracy: {val_acc:.4f}")
        print(f"  Test accuracy: {test_acc:.4f}")

        # Check overfitting
        overfitting = train_acc - val_acc
        print(f"  Overfitting gap: {overfitting:.4f}")

        if test_acc >= 0.90:
            print("  ✅ Target akurasi 90%+ tercapai!")
        else:
            print(f"  ⚠️ Akurasi {test_acc:.1%} belum mencapai target 90%")

        print(f"  💾 Model saved: {model_path}")

        return model, test_acc

    def train_text_vectorizer(self, filenames, labels):
        """Train text vectorizer dari filename patterns"""
        print("📝 Training Text Vectorizer...")

        # Extract text patterns from filenames
        text_features = []
        for filename in filenames:
            # Remove extension and extract patterns
            name = Path(filename).stem.lower()
            # Add spaces between words/numbers for better tokenization
            import re
            name = re.sub(r'([a-z])([0-9])', r'\1 \2', name)
            name = re.sub(r'([0-9])([a-z])', r'\1 \2', name)
            name = re.sub(r'[^a-z0-9\s]', ' ', name)
            text_features.append(name)

        # Train TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )

        X_text = vectorizer.fit_transform(text_features)

        # Split and train simple classifier
        X_train, X_test, y_train, y_test = train_test_split(
            X_text, labels, test_size=0.2, random_state=self.random_state, stratify=labels
        )

        # Simple classifier for text features
        from sklearn.linear_model import LogisticRegression
        text_classifier = LogisticRegression(random_state=self.random_state)
        text_classifier.fit(X_train, y_train)

        # Evaluate
        test_acc = text_classifier.score(X_test, y_test)
        print(f"  Text classifier accuracy: {test_acc:.4f}")

        # Save
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        vectorizer_path = self.output_dir / f'text_vectorizer_{timestamp}.pkl'
        text_classifier_path = self.output_dir / f'text_classifier_{timestamp}.pkl'

        joblib.dump(vectorizer, vectorizer_path)
        joblib.dump(text_classifier, text_classifier_path)

        print(f"  💾 Vectorizer saved: {vectorizer_path}")
        print(f"  💾 Text classifier saved: {text_classifier_path}")

        return vectorizer, text_classifier, test_acc

if __name__ == "__main__":
    trainer = HighAccuracyTrainer()

    # Load dataset
    images, labels, filenames = trainer.load_real_dataset()

    print(f"\n{'='*60}")
    print(f"🚀 STARTING HIGH ACCURACY TRAINING")
    print(f"{'='*60}")

    results = {}

    # 1. Extract features and train Random Forest
    print(f"\n{'='*40}")
    print(f"1️⃣ RANDOM FOREST TRAINING")
    print(f"{'='*40}")
    features, feature_names = trainer.extract_advanced_features(images, labels)
    rf_model, scaler, rf_accuracy = trainer.train_random_forest_advanced(features, labels, feature_names)
    results['Random Forest'] = rf_accuracy

    # 2. Train CNN
    print(f"\n{'='*40}")
    print(f"2️⃣ CNN TRAINING")
    print(f"{'='*40}")
    cnn_model, cnn_accuracy = trainer.train_cnn_advanced(images, labels)
    results['CNN'] = cnn_accuracy

    # 3. Train Text Vectorizer
    print(f"\n{'='*40}")
    print(f"3️⃣ TEXT VECTORIZER TRAINING")
    print(f"{'='*40}")
    vectorizer, text_classifier, text_accuracy = trainer.train_text_vectorizer(filenames, labels)
    results['Text Classifier'] = text_accuracy

    # 4. Summary
    print(f"\n{'='*60}")
    print(f"🎯 TRAINING RESULTS SUMMARY")
    print(f"{'='*60}")

    for model_name, accuracy in results.items():
        status = "✅ TARGET ACHIEVED" if accuracy >= 0.90 else "⚠️ BELOW TARGET"
        print(f"{model_name:20}: {accuracy:.1%} {status}")

    avg_accuracy = np.mean(list(results.values()))
    print(f"\n📊 Average Accuracy: {avg_accuracy:.1%}")

    if avg_accuracy >= 0.90:
        print("🎉 OVERALL TARGET ACHIEVED!")
    else:
        print("🔄 Consider additional tuning for better results")

    print(f"\n✅ All models saved to: {trainer.output_dir}")
    print(f"🔄 Ready for production deployment!")
