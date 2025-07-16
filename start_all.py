#!/usr/bin/env python3
"""
Start both backend and frontend servers for CekAjaYuk
"""

import os
import sys
import time
import subprocess
import threading
import webbrowser
from pathlib import Path

def start_backend():
    """Start the backend server"""
    try:
        # Check dependencies first
        print("🔍 Checking dependencies...")

        try:
            import tensorflow
            print("✅ TensorFlow found")
        except ImportError:
            print("❌ TensorFlow not found!")
            print("📥 Installing TensorFlow...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'tensorflow'], check=True)
            print("✅ TensorFlow installed")

        try:
            import flask
            import flask_cors
            import numpy
            import cv2
            import sklearn
            print("✅ All dependencies found")
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            print("📥 Installing all requirements...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)

        backend_dir = Path(__file__).parent / 'backend'
        os.chdir(backend_dir)

        # Import and run the app
        sys.path.insert(0, str(backend_dir))
        from app import app, initialize_app

        print("🚀 Starting Backend...")
        initialize_app()
        print("✅ Backend initialized")

        app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)

    except Exception as e:
        print(f"❌ Backend error: {e}")
        print("💡 Try running: pip install -r requirements.txt")

def start_frontend():
    """Start the frontend server"""
    try:
        frontend_dir = Path(__file__).parent / 'frontend'
        os.chdir(frontend_dir)
        
        print("🌐 Starting Frontend...")
        
        # Use Python's built-in HTTP server
        subprocess.run([
            sys.executable, '-m', 'http.server', '8000'
        ], check=True)
        
    except Exception as e:
        print(f"❌ Frontend error: {e}")

def main():
    """Main function to start both servers"""
    print("🎯 CekAjaYuk - Starting All Services")
    print("=" * 50)
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    # Wait a bit for backend to start
    time.sleep(3)
    
    print("🌐 Backend should be running on http://localhost:5000")
    print("🌐 Starting frontend on http://localhost:8000")
    print("=" * 50)
    
    # Open browser
    time.sleep(2)
    webbrowser.open('http://localhost:8000')
    
    # Start frontend (this will block)
    start_frontend()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 All servers stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
