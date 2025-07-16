"""
CekAjaYuk Model Training Script
Script untuk menjalankan training semua model ML/DL
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_notebook(notebook_path):
    """Run a Jupyter notebook"""
    try:
        print(f"Running {notebook_path}...")
        
        # Convert notebook to Python script and execute
        cmd = [
            sys.executable, '-m', 'jupyter', 'nbconvert',
            '--to', 'python',
            '--execute',
            str(notebook_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ {notebook_path} completed successfully")
            return True
        else:
            print(f"✗ {notebook_path} failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ Error running {notebook_path}: {e}")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'jupyter', 'numpy', 'pandas', 'scikit-learn', 
        'tensorflow', 'matplotlib', 'seaborn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def create_sample_data():
    """Create sample data for training if not exists"""
    print("Creating sample training data...")
    
    try:
        import numpy as np
        import os
        
        # Create data directory
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        
        # Create sample features and labels
        np.random.seed(42)
        
        # Generate synthetic features for 1000 samples
        n_samples = 1000
        n_features = 8
        
        # Generate features for genuine jobs (first 500)
        genuine_features = np.random.normal(0.6, 0.2, (n_samples//2, n_features))
        genuine_labels = np.ones(n_samples//2)
        
        # Generate features for fake jobs (last 500)
        fake_features = np.random.normal(0.4, 0.2, (n_samples//2, n_features))
        fake_labels = np.zeros(n_samples//2)
        
        # Combine data
        X = np.vstack([genuine_features, fake_features])
        y = np.hstack([genuine_labels, fake_labels])
        
        # Split data
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Save data
        np.save(data_dir / 'X_train.npy', X_train_scaled)
        np.save(data_dir / 'X_val.npy', X_val_scaled)
        np.save(data_dir / 'X_test.npy', X_test_scaled)
        np.save(data_dir / 'y_train.npy', y_train)
        np.save(data_dir / 'y_val.npy', y_val)
        np.save(data_dir / 'y_test.npy', y_test)
        
        # Save scaler
        import joblib
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        joblib.dump(scaler, models_dir / 'feature_scaler.pkl')
        
        # Save feature names
        feature_names = [
            'color_scheme_quality',
            'text_density', 
            'logo_presence',
            'contact_completeness',
            'layout_quality',
            'urgency_indicators',
            'spelling_accuracy',
            'image_quality'
        ]
        
        with open(data_dir / 'feature_names.txt', 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
        
        print(f"✓ Sample data created: {n_samples} samples")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples") 
        print(f"  Test: {len(X_test)} samples")
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating sample data: {e}")
        return False

def main():
    """Main training function"""
    print("=" * 60)
    print("CekAjaYuk Model Training")
    print("=" * 60)
    
    # Check dependencies
    print("1. Checking dependencies...")
    if not check_dependencies():
        return
    
    # Create directories
    print("\n2. Creating directories...")
    directories = ['data', 'models', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  Created: {directory}/")
    
    # Create sample data
    print("\n3. Preparing training data...")
    if not create_sample_data():
        return
    
    # Define notebook paths
    notebooks_dir = Path('notebooks')
    notebooks = [
        notebooks_dir / '1_data_preparation.ipynb',
        notebooks_dir / '2_random_forest_training.ipynb',
        notebooks_dir / '3_tensorflow_training.ipynb'
    ]
    
    # Check if notebooks exist
    missing_notebooks = [nb for nb in notebooks if not nb.exists()]
    if missing_notebooks:
        print(f"\n⚠ Missing notebooks: {[str(nb) for nb in missing_notebooks]}")
        print("Creating basic training scripts instead...")
        
        # Create basic Random Forest training
        create_basic_rf_training()
        
        # Create basic TensorFlow training  
        create_basic_tf_training()
        
        return
    
    # Run training notebooks
    print(f"\n4. Running training notebooks...")
    
    success_count = 0
    for i, notebook in enumerate(notebooks, 1):
        print(f"\n4.{i} Running {notebook.name}...")
        
        if run_notebook(notebook):
            success_count += 1
            print(f"✓ Completed {notebook.name}")
        else:
            print(f"✗ Failed {notebook.name}")
            
        # Add delay between notebooks
        if i < len(notebooks):
            print("Waiting 5 seconds before next notebook...")
            time.sleep(5)
    
    # Summary
    print(f"\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Notebooks completed: {success_count}/{len(notebooks)}")
    
    if success_count == len(notebooks):
        print("✓ All training completed successfully!")
        print("\nNext steps:")
        print("1. Check models/ directory for trained models")
        print("2. Run the application: python run.py")
        print("3. Test the API: python test_api.py")
    else:
        print("⚠ Some training failed. Check error messages above.")
        print("You can still run the application with demo data.")

def create_basic_rf_training():
    """Create basic Random Forest training script"""
    script_content = '''
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path

# Load data
data_dir = Path('data')
X_train = np.load(data_dir / 'X_train.npy')
X_test = np.load(data_dir / 'X_test.npy')
y_train = np.load(data_dir / 'y_train.npy')
y_test = np.load(data_dir / 'y_test.npy')

print("Training Random Forest...")

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
models_dir = Path('models')
models_dir.mkdir(exist_ok=True)
joblib.dump(rf, models_dir / 'random_forest_classifier_latest.pkl')

print("Random Forest model saved!")
'''
    
    with open('train_rf.py', 'w') as f:
        f.write(script_content)
    
    # Run the script
    subprocess.run([sys.executable, 'train_rf.py'])

def create_basic_tf_training():
    """Create basic TensorFlow training script"""
    script_content = '''
import numpy as np
import tensorflow as tf
from pathlib import Path

# Create synthetic image data
print("Creating synthetic image data...")
np.random.seed(42)

n_samples = 1000
X_images = np.random.rand(n_samples, 224, 224, 3)
y_labels = np.random.randint(0, 2, n_samples)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_images, y_labels, test_size=0.2, random_state=42
)

print("Training simple CNN...")

# Create simple model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=5,
    validation_data=(X_test, y_test),
    verbose=1
)

# Save model
models_dir = Path('models')
models_dir.mkdir(exist_ok=True)
model.save(models_dir / 'tensorflow_model_latest.h5')

print("TensorFlow model saved!")
'''
    
    with open('train_tf.py', 'w') as f:
        f.write(script_content)
    
    # Run the script
    subprocess.run([sys.executable, 'train_tf.py'])

if __name__ == '__main__':
    main()
