@echo off
title CekAjaYuk - Fake Job Detector
color 0A

echo ========================================
echo    CekAjaYuk - Fake Job Detector
echo    Python 3.11 Compatible Version
echo ========================================
echo.

REM Check Python version
echo 🔍 Checking Python version...
python --version
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.11
    pause
    exit /b 1
)

echo.
echo 📥 Installing/Updating dependencies...
echo This may take a few minutes...

REM Install TensorFlow compatible with Python 3.11
echo Installing TensorFlow 2.12.0 (Python 3.11 compatible)...
pip install tensorflow==2.12.0 --upgrade --quiet

REM Install other dependencies
echo Installing other dependencies...
pip install flask flask-cors numpy opencv-python scikit-learn pillow pytesseract textblob nltk --upgrade --quiet

echo.
echo ✅ Dependencies installed
echo.

REM Kill any existing processes on ports 5000 and 8000
echo 🔧 Checking for existing servers...
netstat -ano | findstr :5000 >nul
if not errorlevel 1 (
    echo Stopping existing backend server...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000') do taskkill /PID %%a /F >nul 2>&1
)

netstat -ano | findstr :8000 >nul
if not errorlevel 1 (
    echo Stopping existing frontend server...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do taskkill /PID %%a /F >nul 2>&1
)

echo.
echo 🚀 Starting CekAjaYuk servers...
echo.

REM Start backend server
echo 🔧 Starting Backend Server (Port 5000)...
start "CekAjaYuk Backend" /MIN cmd /c "python start_backend_safe.py & pause"

REM Wait for backend to start
echo Waiting for backend to initialize...
timeout /t 8 /nobreak >nul

REM Start frontend server
echo 🌐 Starting Frontend Server (Port 8000)...
cd frontend
start "CekAjaYuk Frontend" /MIN cmd /c "python -m http.server 8000 & pause"

REM Wait for frontend to start
echo Waiting for frontend to start...
timeout /t 3 /nobreak >nul

REM Open browser
echo 🌍 Opening CekAjaYuk in browser...
start http://localhost:8000

echo.
echo ========================================
echo ✅ CekAjaYuk is now running!
echo.
echo 🌐 Frontend: http://localhost:8000
echo ⚙️  Backend:  http://localhost:5000
echo.
echo 📱 The application should open in your browser
echo 🔄 Both servers are running in background
echo.
echo To stop the servers:
echo 1. Close this window, or
echo 2. Press Ctrl+C in the server windows
echo ========================================
echo.
echo Press any key to close this launcher...
echo (Servers will continue running)
pause >nul
