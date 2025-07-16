#!/usr/bin/env python3
"""
Backend runner for CekAjaYuk with proper initialization
"""

import os
import sys
from pathlib import Path

def main():
    """Run the backend with proper initialization"""
    print("🚀 Starting CekAjaYuk Backend...")
    
    # Change to backend directory
    backend_dir = Path(__file__).parent / 'backend'
    original_dir = os.getcwd()
    
    try:
        os.chdir(backend_dir)
        sys.path.insert(0, str(backend_dir))
        
        print(f"📍 Backend directory: {backend_dir}")
        print(f"🐍 Python version: {sys.version}")
        
        # Import and run
        from app import app, initialize_app
        
        print("⚙️ Initializing application...")
        initialize_app()
        print("✅ Application initialized")
        
        print("🌐 Starting Flask server on http://localhost:5000")
        print("🔄 Press Ctrl+C to stop")
        print("=" * 50)
        
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
        
    except KeyboardInterrupt:
        print("\n🛑 Backend stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        os.chdir(original_dir)

if __name__ == '__main__':
    main()
