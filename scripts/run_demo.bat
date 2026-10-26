@echo off
REM Run StorePulse AI demo (Windows)
REM Task 9: Easy demo startup script

echo Starting StorePulse AI...
echo.

REM Activate virtual environment
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo Error: Virtual environment not found at .venv\Scripts\activate.bat
    echo Please run: python -m venv .venv
    echo Then: .venv\Scripts\activate.bat
    echo Then: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Run Streamlit app
streamlit run app.py

pause
