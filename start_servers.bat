@echo off
echo ============================================================
echo Starting PDF OCR Backend and Voice API...
echo Ensuring the correct virtual environment is used!
echo ============================================================

set VENV_PYTHON=".\.venv\Scripts\python.exe"

if not exist %VENV_PYTHON% (
    echo [ERROR] Virtual environment Python not found at %VENV_PYTHON%
    echo Please make sure your .venv exists in this directory.
    pause
    exit /b 1
)

echo Starting Voice API (Port 8003) in a new window...
start "Voice API Microservice" cmd /k "%VENV_PYTHON% services\voice\voice_api.py"

echo Starting Embedding API (Port 8002) in a new window...
start "Embedding API Microservice" cmd /k "%VENV_PYTHON% services\embedding\embedding_api.py"

echo Starting Main Backend (Port 8000)...
%VENV_PYTHON% main.py
