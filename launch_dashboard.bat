@echo off
cd /d C:\Users\phemm\orb_lab
call C:\Users\phemm\anaconda3\Scripts\activate.bat
echo Starting Optuna Dashboard...
echo.
echo Dashboard will open at: http://127.0.0.1:8080
echo Press Ctrl+C to stop
echo.
optuna-dashboard sqlite:///orb_optuna.db
pause
