@echo off
REM =====================================================================
REM ORB Lab - Quick Sync
REM =====================================================================
REM Run this to sync changes between machines.
REM Detects if there are local changes or remote changes and acts accordingly.
REM =====================================================================

cd /d C:\Users\phemm\orb_lab

echo.
echo ===============================================
echo   ORB Lab Sync
echo ===============================================
echo.

REM Refresh AHK scripts from C:\hmm_trading
if exist "C:\hmm_trading\PresetUpdater_v4.ahk" (
    copy /Y "C:\hmm_trading\*.ahk" "hmm_trading_scripts\" >nul 2>nul
)

REM Check for local changes
git diff --quiet --exit-code 2>nul
set DIRTY=%errorlevel%

git diff --cached --quiet --exit-code 2>nul  
set STAGED=%errorlevel%

REM Check for untracked files
for /f %%i in ('git ls-files --others --exclude-standard ^| find /c /v ""') do set UNTRACKED=%%i

echo Checking status...
echo.

if %DIRTY%==1 (
    echo   Local changes detected.
    set HAS_LOCAL=1
) else if %STAGED%==1 (
    echo   Staged changes detected.
    set HAS_LOCAL=1
) else if %UNTRACKED% GTR 0 (
    echo   New untracked files detected.
    set HAS_LOCAL=1
) else (
    echo   No local changes.
    set HAS_LOCAL=0
)

REM Fetch remote
echo   Fetching remote...
git fetch origin 2>nul

REM Check for remote changes
for /f %%i in ('git rev-list HEAD..origin/main --count 2^>nul') do set BEHIND=%%i
if not defined BEHIND set BEHIND=0

if %BEHIND% GTR 0 (
    echo   Remote has %BEHIND% new commit(s).
    set HAS_REMOTE=1
) else (
    echo   Remote is up to date.
    set HAS_REMOTE=0
)

echo.

REM Decision logic
if "%HAS_LOCAL%"=="1" if "%HAS_REMOTE%"=="1" (
    echo Both local and remote have changes.
    echo Pulling remote first, then pushing local...
    git add -A
    git stash
    git pull --rebase origin main
    git stash pop
    git add -A
    git commit -m "Sync: %date% %time:~0,5%"
    git push origin main
    echo.
    echo DONE: Merged and pushed.
    goto :end
)

if "%HAS_LOCAL%"=="1" (
    echo Pushing local changes...
    git add -A
    git commit -m "Sync: %date% %time:~0,5%"
    git push origin main
    echo.
    echo DONE: Pushed to remote.
    goto :end
)

if "%HAS_REMOTE%"=="1" (
    echo Pulling remote changes...
    git pull origin main
    echo.
    echo DONE: Pulled from remote.
    REM Copy AHK scripts back to C:\hmm_trading
    if exist "hmm_trading_scripts\PresetUpdater_v4.ahk" (
        copy /Y "hmm_trading_scripts\*.ahk" "C:\hmm_trading\" >nul 2>nul
        echo   AHK scripts updated in C:\hmm_trading
    )
    goto :end
)

echo Already in sync - nothing to do!

:end
echo.
pause
