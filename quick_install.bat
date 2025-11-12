@echo off
REM Quick installation script for Dental Voice Intelligence System
echo ======================================================================
echo   Dental Voice Intelligence System - Quick Install
echo ======================================================================
echo.
echo This will install all required dependencies.
echo Installation may take 5-10 minutes.
echo.
pause

REM Check Python
echo [1/5] Checking Python version...
python --version
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.10 or higher from python.org
    pause
    exit /b 1
)
echo      Done!
echo.

REM Create virtual environment
echo [2/5] Creating virtual environment...
if exist "venv" (
    echo      Virtual environment already exists, skipping...
) else (
    python -m venv venv
    echo      Done!
)
echo.

REM Activate virtual environment
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat
echo      Done!
echo.

REM Upgrade pip
echo [4/5] Upgrading pip...
python -m pip install --upgrade pip
echo      Done!
echo.

REM Install dependencies
echo [5/5] Installing dependencies (this may take a while)...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [WARNING] Some packages failed to install.
    echo You may need to install them manually.
    echo.
)
echo      Done!
echo.

REM Create directories
echo Creating directory structure...
if not exist "models\enrollments" mkdir "models\enrollments"
if not exist "pretrained_models\asr" mkdir "pretrained_models\asr"
if not exist "pretrained_models\spkrec" mkdir "pretrained_models\spkrec"
if not exist "logs" mkdir "logs"
echo      Done!
echo.

echo ======================================================================
echo   Installation Complete!
echo ======================================================================
echo.
echo Next steps:
echo   1. Enroll speakers (optional but recommended):
echo      python scripts\enroll_speaker.py --interactive
echo.
echo   2. Start the server:
echo      start_server.bat
echo.
echo   3. Open browser:
echo      http://localhost:8000
echo.
echo ======================================================================
echo.
pause
