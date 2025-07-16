"""
Setup script for CekAjaYuk
Installs dependencies and sets up the environment
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        return True
    except subprocess.CalledProcessError:
        return False

def install_requirements():
    """Install all requirements from requirements.txt"""
    requirements_file = Path(__file__).parent / 'requirements.txt'
    
    if not requirements_file.exists():
        print("requirements.txt not found!")
        return False
    
    try:
        print("Installing Python dependencies...")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)
        ])
        print("Python dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def setup_tesseract():
    """Provide instructions for Tesseract installation"""
    system = platform.system().lower()
    
    print("\n" + "="*50)
    print("TESSERACT OCR INSTALLATION INSTRUCTIONS")
    print("="*50)
    
    if system == 'windows':
        print("For Windows:")
        print("1. Download Tesseract installer from:")
        print("   https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. Install with Indonesian language support")
        print("3. Add Tesseract to your PATH environment variable")
        print("4. Restart your command prompt/IDE")
        
    elif system == 'linux':
        print("For Linux (Ubuntu/Debian):")
        print("Run these commands:")
        print("   sudo apt-get update")
        print("   sudo apt-get install tesseract-ocr")
        print("   sudo apt-get install tesseract-ocr-ind")
        print("   sudo apt-get install tesseract-ocr-eng")
        
    elif system == 'darwin':  # macOS
        print("For macOS:")
        print("Using Homebrew:")
        print("   brew install tesseract")
        print("   brew install tesseract-lang")
        
    else:
        print("Please install Tesseract OCR for your operating system")
        print("Visit: https://tesseract-ocr.github.io/tessdoc/Installation.html")

def create_directories():
    """Create necessary directories"""
    directories = [
        'uploads',
        'models', 
        'logs',
        'data',
        'static/uploads'
    ]
    
    base_path = Path(__file__).parent
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def setup_nltk():
    """Download required NLTK data"""
    try:
        import nltk
        print("Downloading NLTK data...")
        
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        
        print("NLTK data downloaded successfully!")
        return True
        
    except ImportError:
        print("NLTK not installed. Will be installed with other dependencies.")
        return False
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        return False

def create_config_file():
    """Create configuration file"""
    config_content = """# CekAjaYuk Configuration
# Copy this file to config.local.py and modify as needed

# Flask Configuration
DEBUG = True
SECRET_KEY = 'your-secret-key-here'

# File Upload Configuration
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

# Model Configuration
MODELS_FOLDER = 'models'
IMAGE_SIZE = (224, 224)

# OCR Configuration
TESSERACT_CONFIG = r'--oem 3 --psm 6 -l ind+eng'

# API Configuration
API_RATE_LIMIT = "100 per hour"

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FILE = 'logs/cekajayuk.log'
"""
    
    config_file = Path(__file__).parent / 'config.example.py'
    
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("Created example configuration file: config.example.py")

def create_readme():
    """Create README file with instructions"""
    readme_content = """# CekAjaYuk - Fake Job Posting Detection System

## Overview
CekAjaYuk adalah sistem deteksi iklan lowongan kerja palsu menggunakan Machine Learning dan Deep Learning.

## Features
- Analisis gambar poster menggunakan Random Forest dan CNN
- Ekstraksi teks menggunakan Tesseract OCR (Indonesia & English)
- Analisis teks untuk deteksi pola mencurigakan
- Interface web yang user-friendly

## Installation

### 1. Install Python Dependencies
```bash
python setup.py
```

### 2. Install Tesseract OCR
Follow the instructions provided by the setup script for your operating system.

### 3. Run the Application
```bash
python run.py
```

## Usage

### Web Interface
1. Open frontend/index.html in your browser
2. Upload a job posting image
3. Follow the 5-step analysis process

### API Endpoints
- `POST /api/analyze-image` - Analyze image with ML/DL models
- `POST /api/extract-text` - Extract text using OCR
- `POST /api/analyze-text` - Analyze text content
- `POST /api/analyze-complete` - Complete analysis pipeline

## Project Structure
```
cekajayuk/
├── frontend/           # Web interface
├── backend/           # Flask API
├── notebooks/         # Jupyter notebooks for training
├── models/           # Trained ML/DL models
├── uploads/          # Uploaded files
├── requirements.txt  # Python dependencies
├── run.py           # Application runner
└── setup.py         # Setup script
```

## Training Models
1. Open notebooks/1_data_preparation.ipynb
2. Run notebooks/2_random_forest_training.ipynb
3. Run notebooks/3_tensorflow_training.ipynb

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License
This project is for educational purposes.
"""
    
    readme_file = Path(__file__).parent / 'README.md'
    
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("Created README.md file")

def main():
    """Main setup function"""
    print("=" * 60)
    print("CekAjaYuk Setup Script")
    print("Setting up Fake Job Posting Detection System")
    print("=" * 60)
    
    # Create directories
    print("\n1. Creating directories...")
    create_directories()
    
    # Install Python dependencies
    print("\n2. Installing Python dependencies...")
    if install_requirements():
        print("✓ Python dependencies installed successfully")
    else:
        print("✗ Failed to install Python dependencies")
        return
    
    # Setup NLTK
    print("\n3. Setting up NLTK...")
    if setup_nltk():
        print("✓ NLTK setup completed")
    else:
        print("⚠ NLTK setup skipped (will be done later)")
    
    # Create configuration files
    print("\n4. Creating configuration files...")
    create_config_file()
    create_readme()
    
    # Tesseract instructions
    print("\n5. Tesseract OCR setup...")
    setup_tesseract()
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETED!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Install Tesseract OCR (see instructions above)")
    print("2. Train models using Jupyter notebooks (optional)")
    print("3. Run the application: python run.py")
    print("\nFor more information, see README.md")

if __name__ == '__main__':
    main()
