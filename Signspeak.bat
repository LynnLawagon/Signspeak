
@echo off
title SignSpeak - Sign Language Recognition
color 0A

echo ==========================================
echo          SignSpeak Launcher
echo ==========================================
echo.

cd /d "%~dp0"

echo [1/4] Checking Python...

py -3.10 --version
if errorlevel 1 (
    echo ERROR: Python 3.10 is not installed.
    echo Please install Python 3.10 first.
    pause
    exit /b 1
)

echo.
echo [2/4] Setting up virtual environment...

if not exist "venv\Scripts\activate.bat" (
    py -3.10 -m venv venv
)

call venv\Scripts\activate.bat

echo.
echo [3/4] Installing requirements...

python -m pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install requirements.
    pause
    exit /b 1
)

echo.
echo [4/4] Starting SignSpeak...
echo.
echo Website: http://127.0.0.1:5000
echo Press CTRL+C to stop the server.
echo.

python web_app.py

pause