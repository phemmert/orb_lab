@echo off
cd /d C:\Users\phemm\orb_lab
call C:\Users\phemm\anaconda3\Scripts\activate.bat

echo.
echo Starting Optuna Dashboard (all symbols)
echo Dashboard: http://127.0.0.1:8080
echo Press Ctrl+C to stop
echo.

if exist orb_optuna_shared.db (
    optuna-dashboard sqlite:///orb_optuna_shared.db
) else if exist orb_optuna_v3.db (
    echo Shared DB not found, using legacy DB...
    optuna-dashboard sqlite:///orb_optuna_v3.db
) else (
    echo No Optuna database found. Run the optimizer first.
)
pause