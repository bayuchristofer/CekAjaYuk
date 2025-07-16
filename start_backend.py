#!/usr/bin/env python3
"""
Simple script to start the CekAjaYuk backend
"""

import os
import sys
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent / 'backend'
sys.path.insert(0, str(backend_dir))

# Change to backend directory
os.chdir(backend_dir)

# Import and run the app
try:
    from app import app, initialize_app
    
    print("🚀 Starting CekAjaYuk Backend...")
    print("📍 Backend directory:", backend_dir)
    
    # Initialize the application
    initialize_app()
    
    print("✅ Application initialized successfully")
    print("🌐 Starting Flask server on http://localhost:5000")
    print("🔄 Press Ctrl+C to stop the server")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
    
except KeyboardInterrupt:
    print("\n🛑 Server stopped by user")
except Exception as e:
    print(f"❌ Error starting backend: {e}")
    sys.exit(1)
