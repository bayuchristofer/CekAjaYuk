"""
CekAjaYuk Application Runner
Script untuk menjalankan aplikasi CekAjaYuk
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        return False
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'flask', 'flask-cors', 'numpy', 'opencv-python', 
        'pillow', 'pytesseract', 'scikit-learn', 'tensorflow',
        'textblob', 'nltk'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def setup_directories():
    """Create necessary directories"""
    directories = [
        'uploads',
        'models',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def download_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk
        print("Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("NLTK data downloaded successfully")
    except Exception as e:
        print(f"Warning: Could not download NLTK data: {e}")

def check_tesseract():
    """Check if Tesseract OCR is available"""
    try:
        import pytesseract
        # Try to get Tesseract version
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract OCR found: {version}")
        return True
    except Exception as e:
        print(f"Warning: Tesseract OCR not found: {e}")
        print("Please install Tesseract OCR:")
        print("- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        print("- Linux: sudo apt-get install tesseract-ocr tesseract-ocr-ind")
        print("- macOS: brew install tesseract tesseract-lang")
        return False

def run_backend():
    """Run the Flask backend"""
    print("Starting CekAjaYuk backend...")
    
    # Change to backend directory
    backend_dir = Path(__file__).parent / 'backend'
    os.chdir(backend_dir)
    
    # Set environment variables
    os.environ['FLASK_APP'] = 'app.py'
    os.environ['FLASK_ENV'] = 'development'
    
    try:
        # Run Flask app
        subprocess.run([sys.executable, 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\nShutting down backend...")
    except subprocess.CalledProcessError as e:
        print(f"Error running backend: {e}")

def open_frontend():
    """Open frontend in browser"""
    frontend_path = Path(__file__).parent / 'frontend' / 'index.html'
    
    if frontend_path.exists():
        print("Opening frontend in browser...")
        webbrowser.open(f'file://{frontend_path.absolute()}')
    else:
        print("Frontend file not found!")

def main():
    """Main function"""
    print("=" * 50)
    print("CekAjaYuk - Fake Job Posting Detection System")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Setup directories
    setup_directories()
    
    # Download NLTK data
    download_nltk_data()
    
    # Check Tesseract
    tesseract_available = check_tesseract()
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        return
    
    print("\nAll checks passed!")
    
    # Ask user what to run
    print("\nWhat would you like to do?")
    print("1. Run backend only")
    print("2. Open frontend only")
    print("3. Run backend and open frontend")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        run_backend()
    elif choice == '2':
        open_frontend()
    elif choice == '3':
        # Open frontend first
        open_frontend()
        print("Frontend opened in browser.")
        print("Starting backend in 3 seconds...")
        time.sleep(3)
        run_backend()
    elif choice == '4':
        print("Goodbye!")
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == '__main__':
    main()
