
@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3 -m venv .venv
)

call ".venv\Scripts\activate.bat"

echo.
echo ========================================
echo   WILDLIFE CONFLICT AI MONITOR
echo ========================================
echo.
echo Installing requirements...
python -m pip install -r requirements.txt

echo.
echo ========================================
echo   SERVER STARTING
echo ========================================
echo Local URL: http://127.0.0.1:5000
echo Network URL: http://192.168.4.2:5000
echo Dashboard: http://localhost:5000
echo ========================================
echo.

python app.py

pause
