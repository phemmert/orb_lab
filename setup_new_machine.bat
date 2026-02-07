@echo off
REM =====================================================================
REM ORB Lab Setup Script
REM =====================================================================
REM Run this on a new machine after git clone to set up the environment.
REM Prerequisites: Python 3.10+, pip, Git
REM =====================================================================

echo.
echo ===============================================
echo   ORB Lab - New Machine Setup
echo ===============================================
echo.

REM Check Python
python --version 2>nul
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ first.
    pause
    exit /b 1
)

REM Install Python dependencies
echo Installing Python dependencies...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed. Check requirements.txt
    pause
    exit /b 1
)

REM Create required directories
echo Creating directories...
if not exist "results" mkdir results
if not exist "results\batch_logs" mkdir results\batch_logs
if not exist "data" mkdir data

REM Verify hmm_trading directory
if not exist "C:\hmm_trading" (
    echo.
    echo WARNING: C:\hmm_trading does not exist.
    echo Creating it and copying AHK scripts...
    mkdir "C:\hmm_trading"
)

REM Copy AHK scripts to hmm_trading if they exist in repo
if exist "hmm_trading_scripts\PresetUpdater_v4.ahk" (
    echo Copying AHK scripts to C:\hmm_trading\
    copy /Y "hmm_trading_scripts\PresetUpdater_v4.ahk" "C:\hmm_trading\"
)

REM Verify key files exist
echo.
echo Verifying installation...
set OK=1

if not exist "optimizer_app.py" (
    echo   MISSING: optimizer_app.py
    set OK=0
)
if not exist "batch_worker.py" (
    echo   MISSING: batch_worker.py
    set OK=0
)
if not exist "src\orb_optimizer_v3.py" (
    echo   MISSING: src\orb_optimizer_v3.py
    set OK=0
)
if not exist "src\orb_backtester.py" (
    echo   MISSING: src\orb_backtester.py
    set OK=0
)
if not exist "src\orb_settings_export.py" (
    echo   MISSING: src\orb_settings_export.py
    set OK=0
)

if "%OK%"=="1" (
    echo   All core files present.
) else (
    echo   Some files missing - check your git clone.
)

REM Quick Python import test
echo.
echo Testing Python imports...
python -c "import streamlit; import optuna; import numpy; import pandas; print('All imports OK')"
if errorlevel 1 (
    echo ERROR: Some Python packages failed to import.
    pause
    exit /b 1
)

echo.
echo ===============================================
echo   Setup Complete!
echo ===============================================
echo.
echo   To start ORB Lab:
echo     streamlit run optimizer_app.py
echo.
echo   Or use your existing startup .bat files.
echo.
pause
