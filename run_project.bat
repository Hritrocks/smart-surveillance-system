@echo off

echo Starting Smart Surveillance System...

REM Go to backend folder
cd backend

REM Activate virtual environment
call .\.venv\Scripts\activate

REM Start Flask backend in new window
start cmd /k python app.py

REM Wait 3 seconds for backend to start
timeout /t 3 >nul

REM Open frontend page in browser
start "" "%CD%\..\frontend\live.html"

echo System started successfully!
pause