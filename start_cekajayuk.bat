@echo off
echo ========================================
echo    CekAjaYuk - Fake Job Detector
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python first.
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Install dependencies
echo 📥 Installing dependencies...
pip install tensorflow flask flask-cors numpy opencv-python scikit-learn pillow pytesseract textblob nltk

echo.
echo 🚀 Starting CekAjaYuk...
echo.

REM Start backend in background
echo 🔧 Starting Backend Server...
start "CekAjaYuk Backend" cmd /k "python start_backend.py"

REM Wait a bit for backend to start
timeout /t 5 /nobreak >nul

REM Start frontend
echo 🌐 Starting Frontend Server...
cd frontend
start "CekAjaYuk Frontend" cmd /k "python -m http.server 8000"

REM Wait a bit for frontend to start
timeout /t 3 /nobreak >nul

REM Open browser
echo 🌍 Opening browser...
start http://localhost:8000

echo.
echo ========================================
echo ✅ CekAjaYuk is now running!
echo.
echo 🌐 Frontend: http://localhost:8000
echo ⚙️  Backend:  http://localhost:5000
echo.
echo Press any key to close this window...
echo (Backend and Frontend will keep running)
echo ========================================
pause
