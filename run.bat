@echo off
setlocal

rem Change to the directory that contains this script
cd /d "%~dp0"

set "PORT=8000"

rem Prefer a project-local virtual environment when available
set "PYTHON="
if exist .venv\Scripts\python.exe set "PYTHON=.venv\Scripts\python.exe"
if exist venv\Scripts\python.exe set "PYTHON=venv\Scripts\python.exe"

rem Fall back to the system interpreter
if not defined PYTHON (
    for %%P in (py.exe python.exe) do (
        where %%P >nul 2>nul
        if not errorlevel 1 (
            set "PYTHON=%%P"
            goto :run
        )
    )
    echo Could not find a Python interpreter. Install Python 3.10+ or create .venv\.\Scripts\python.exe
    exit /b 1
)

:run
echo Opening webcam preview in the browser...
start "webcam-preview" http://localhost:%PORT%/

echo Starting red scarf monitor in web mode on port %PORT%...
"%PYTHON%" red_scarf_monitor.py --config config\config.yaml --mode web --host 0.0.0.0 --port %PORT% %*

if errorlevel 1 (
    echo.
    echo The application exited with errors.
) else (
    echo.
    echo The application finished successfully.
)

echo.
pause

