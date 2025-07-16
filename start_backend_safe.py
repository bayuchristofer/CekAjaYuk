#!/usr/bin/env python3
"""
Safe backend starter for CekAjaYuk - handles TensorFlow issues gracefully
"""

import os
import sys
import subprocess
from pathlib import Path

def check_and_install_tensorflow():
    """Check and install compatible TensorFlow version"""
    print("🔍 Checking TensorFlow compatibility...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} found")
        return True
    except ImportError:
        print("❌ TensorFlow not found")
    except Exception as e:
        print(f"⚠️ TensorFlow error: {e}")
    
    # Try to install compatible version
    print("📥 Installing TensorFlow compatible with Python 3.11...")
    
    try:
        # Try TensorFlow 2.12.0 (compatible with Python 3.11)
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', 
            'tensorflow==2.12.0', '--upgrade'
        ], check=True, capture_output=True)
        
        print("✅ TensorFlow 2.12.0 installed successfully")
        return True
        
    except subprocess.CalledProcessError:
        print("⚠️ TensorFlow 2.12.0 failed, trying CPU version...")
        
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', 
                'tensorflow-cpu==2.12.0', '--upgrade'
            ], check=True, capture_output=True)
            
            print("✅ TensorFlow CPU 2.12.0 installed successfully")
            return True
            
        except subprocess.CalledProcessError:
            print("❌ Failed to install TensorFlow")
            print("💡 App will run without deep learning features")
            return False

def install_other_dependencies():
    """Install other required dependencies"""
    print("📥 Installing other dependencies...")
    
    dependencies = [
        'flask', 'flask-cors', 'numpy', 'opencv-python', 
        'scikit-learn', 'pillow', 'pytesseract', 'textblob', 'nltk'
    ]
    
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install'
        ] + dependencies, check=True, capture_output=True)
        
        print("✅ All dependencies installed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Some dependencies failed to install: {e}")
        return False

def start_backend():
    """Start the backend server"""
    try:
        # Change to backend directory
        backend_dir = Path(__file__).parent / 'backend'
        original_dir = os.getcwd()
        os.chdir(backend_dir)
        
        # Add backend to Python path
        sys.path.insert(0, str(backend_dir))
        
        print("🚀 Starting CekAjaYuk Backend...")
        print(f"📍 Backend directory: {backend_dir}")
        print(f"🐍 Python version: {sys.version}")
        
        # Import and initialize app
        from app import app, initialize_app
        
        print("⚙️ Initializing application...")
        initialize_app()
        
        print("✅ Application initialized successfully")
        print("🌐 Starting Flask server...")
        print("🔗 Backend URL: http://localhost:5000")
        print("🔄 Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Run the Flask app
        app.run(
            debug=True, 
            host='0.0.0.0', 
            port=5000,
            use_reloader=False  # Disable reloader to avoid issues
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped by user")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure you're in the correct directory")
        print("2. Try: pip install -r requirements.txt")
        print("3. Check if port 5000 is already in use")
        sys.exit(1)
    finally:
        # Restore original directory
        os.chdir(original_dir)

def main():
    """Main function"""
    print("🎯 CekAjaYuk Backend Launcher")
    print("=" * 40)
    print(f"🐍 Python version: {sys.version}")
    print("=" * 40)
    
    # Check and install TensorFlow
    tf_success = check_and_install_tensorflow()
    
    # Install other dependencies
    deps_success = install_other_dependencies()
    
    if not deps_success:
        print("⚠️ Some dependencies failed, but continuing...")
    
    print("=" * 40)
    
    # Start backend
    start_backend()

if __name__ == '__main__':
    main()
