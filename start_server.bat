@echo off
REM Startup script for Dental Voice Intelligence System
echo ======================================================================
echo   Dental Voice Intelligence System - Startup
echo ======================================================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo.
    echo Please run setup first:
    echo   1. python -m venv venv
    echo   2. venv\Scripts\activate
    echo   3. pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo [1/4] Activating virtual environment...
call venv\Scripts\activate.bat
echo      Done!
echo.

REM Check Python
echo [2/4] Checking Python...
python --version
echo      Done!
echo.

REM Verify setup
echo [3/4] Verifying system setup...
python verify_setup.py
if errorlevel 1 (
    echo.
    echo [ERROR] Setup verification failed!
    echo Please install dependencies: pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)
echo.

REM Start server
echo [4/4] Starting server...
echo.
echo ======================================================================
echo   Server will start on: http://localhost:8000
echo   Open your browser and go to: http://localhost:8000
echo ======================================================================
echo.
echo Press Ctrl+C to stop the server
echo.

python run_server_simple.py

pause
