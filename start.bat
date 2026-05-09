@echo off
title Fraud Detection System Startup
echo ==============================================
echo Starting Fraud Detection System...
echo ==============================================

:: Check if virtual environment exists
if not exist .venv\Scripts\activate.bat (
    echo Error: Virtual environment not found. Please create it and install requirements.
    pause
    exit /b
)

echo Activating virtual environment...
call .venv\Scripts\activate

echo Starting FastAPI Backend...
start "FastAPI Backend" cmd /k "call .venv\Scripts\activate && python -m api.main"

echo Starting Streamlit Dashboard...
start "Streamlit Dashboard" cmd /k "call .venv\Scripts\activate && streamlit run dashboard/app.py"

echo ==============================================
echo Application is starting!
echo - API will be available at: http://localhost:8002/docs
echo - Dashboard will be available at: http://localhost:8501
echo ==============================================
