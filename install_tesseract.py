#!/usr/bin/env python3
"""
Script to install and configure Tesseract OCR for Windows
"""
import os
import sys
import subprocess
import urllib.request
import zipfile
from pathlib import Path

def download_tesseract():
    """Download and install Tesseract OCR for Windows"""
    print("📥 Downloading Tesseract OCR...")
    
    # Tesseract installer URL for Windows
    tesseract_url = "https://github.com/UB-Mannheim/tesseract/releases/download/v5.3.0.20221214/tesseract-ocr-w64-setup-5.3.0.20221214.exe"
    
    try:
        # Download installer
        installer_path = "tesseract_installer.exe"
        urllib.request.urlretrieve(tesseract_url, installer_path)
        
        print("✅ Tesseract installer downloaded")
        print("🔧 Please run the installer manually:")
        print(f"   {os.path.abspath(installer_path)}")
        print("\n📋 Installation instructions:")
        print("1. Run the installer as Administrator")
        print("2. Install to default location: C:\\Program Files\\Tesseract-OCR")
        print("3. Make sure to install additional language packs (Indonesian)")
        print("4. Add to PATH if prompted")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading Tesseract: {e}")
        return False

def configure_tesseract():
    """Configure Tesseract path"""
    print("🔧 Configuring Tesseract...")
    
    # Common Tesseract installation paths
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe'.format(os.environ.get('USERNAME', '')),
    ]
    
    tesseract_path = None
    for path in possible_paths:
        if os.path.exists(path):
            tesseract_path = path
            break
    
    if tesseract_path:
        print(f"✅ Tesseract found at: {tesseract_path}")
        
        # Create configuration file
        config_content = f'''
# Tesseract Configuration for CekAjaYuk
import pytesseract

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'{tesseract_path}'

# Test configuration
def test_tesseract():
    try:
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {{version}}")
        return True
    except Exception as e:
        print(f"❌ Tesseract test failed: {{e}}")
        return False

if __name__ == "__main__":
    test_tesseract()
'''
        
        with open('tesseract_config.py', 'w') as f:
            f.write(config_content)
        
        print("✅ Configuration file created: tesseract_config.py")
        return True
    else:
        print("❌ Tesseract not found in common locations")
        print("💡 Please install Tesseract manually:")
        print("   1. Download from: https://github.com/UB-Mannheim/tesseract/releases")
        print("   2. Install to C:\\Program Files\\Tesseract-OCR")
        print("   3. Add to system PATH")
        return False

def install_python_packages():
    """Install required Python packages"""
    print("📦 Installing Python packages...")
    
    packages = [
        'pytesseract',
        'pillow',
        'opencv-python',
        'numpy'
    ]
    
    for package in packages:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                         check=True, capture_output=True)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")

def main():
    """Main installation function"""
    print("🚀 CekAjaYuk Tesseract OCR Setup")
    print("=" * 40)
    
    # Install Python packages
    install_python_packages()
    
    # Check if Tesseract is already installed
    if configure_tesseract():
        print("\n✅ Tesseract is already installed and configured!")
    else:
        print("\n📥 Tesseract not found. Starting download...")
        if download_tesseract():
            print("\n⚠️ Please install Tesseract manually and run this script again")
        else:
            print("\n❌ Download failed. Please install Tesseract manually")
    
    print("\n🔧 Manual Installation Steps:")
    print("1. Download Tesseract: https://github.com/UB-Mannheim/tesseract/releases")
    print("2. Install with Indonesian language pack")
    print("3. Add to system PATH")
    print("4. Restart your terminal/IDE")
    
    print("\n🧪 Test OCR after installation:")
    print("python tesseract_config.py")

if __name__ == "__main__":
    main()
