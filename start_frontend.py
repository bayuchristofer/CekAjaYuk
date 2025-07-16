#!/usr/bin/env python3
"""
Simple HTTP server for CekAjaYuk frontend
"""

import os
import sys
import http.server
import socketserver
from pathlib import Path

def start_frontend_server():
    """Start a simple HTTP server for the frontend"""
    
    # Change to frontend directory
    frontend_dir = Path(__file__).parent / 'frontend'
    if not frontend_dir.exists():
        print(f"❌ Frontend directory not found: {frontend_dir}")
        return
    
    os.chdir(frontend_dir)
    
    # Configuration
    PORT = 8000
    HOST = 'localhost'
    
    # Create server
    Handler = http.server.SimpleHTTPRequestHandler
    
    # Add CORS headers
    class CORSRequestHandler(Handler):
        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            super().end_headers()
    
    try:
        with socketserver.TCPServer((HOST, PORT), CORSRequestHandler) as httpd:
            print("🌐 CekAjaYuk Frontend Server")
            print("=" * 40)
            print(f"📂 Serving from: {frontend_dir}")
            print(f"🔗 Frontend URL: http://{HOST}:{PORT}")
            print(f"🔗 Backend URL: http://localhost:5000")
            print("=" * 40)
            print("🚀 Server started successfully!")
            print("📱 Open http://localhost:8000 in your browser")
            print("🔄 Press Ctrl+C to stop the server")
            print()
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🛑 Frontend server stopped by user")
    except Exception as e:
        print(f"❌ Error starting frontend server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    start_frontend_server()
