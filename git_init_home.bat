@echo off
REM =====================================================================
REM ORB Lab - Git Repository Setup (Run on HOME machine first)
REM =====================================================================
REM This initializes the Git repo and makes the first commit.
REM 
REM BEFORE RUNNING:
REM   1. Install Git if not already: https://git-scm.com/download/win
REM   2. Create a PRIVATE repo on GitHub: https://github.com/new
REM      - Name: orb-lab (or whatever you prefer)
REM      - Set to PRIVATE (important - this has your trading code!)
REM      - Do NOT initialize with README
REM   3. Update the REPO_URL below with your repo URL
REM =====================================================================

echo.
echo ===============================================
echo   ORB Lab - Git Repository Setup
echo ===============================================
echo.

cd /d C:\Users\phemm\orb_lab

REM Initialize git
echo Initializing git repository...
git init

REM Add the .gitignore
echo Adding .gitignore...
git add .gitignore

REM Add all tracked files
echo Adding project files...
git add *.py
git add *.bat
git add *.txt
git add src\*.py
git add results\*.json 2>nul
git add data\*.csv 2>nul
git add data\*.parquet 2>nul

REM Add AHK scripts subfolder (copy from C:\hmm_trading first)
if not exist "hmm_trading_scripts" mkdir hmm_trading_scripts
echo Copying AHK scripts into repo...
copy /Y "C:\hmm_trading\PresetUpdater_v4.ahk" "hmm_trading_scripts\" 2>nul
copy /Y "C:\hmm_trading\*.ahk" "hmm_trading_scripts\" 2>nul
git add hmm_trading_scripts\*.ahk 2>nul

REM First commit
echo.
echo Making initial commit...
git commit -m "Initial commit - ORB Lab v2 with batch optimizer and preset pipeline"

REM Instructions for remote
echo.
echo ===============================================
echo   NEXT STEPS:
echo ===============================================
echo.
echo   1. Create a PRIVATE repo on GitHub
echo      https://github.com/new
echo.
echo   2. Run these commands:
echo      git remote add origin https://github.com/YOUR_USERNAME/orb-lab.git
echo      git branch -M main
echo      git push -u origin main
echo.
echo   3. On your RV laptop:
echo      cd C:\Users\phemm
echo      git clone https://github.com/YOUR_USERNAME/orb-lab.git orb_lab
echo      cd orb_lab
echo      setup_new_machine.bat
echo.
echo   4. Daily sync workflow:
echo      HOME:  git add -A ^&^& git commit -m "update" ^&^& git push
echo      RV:    git pull
echo.
pause
