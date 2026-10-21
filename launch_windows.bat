@echo off
title Polymer Series Explorer
echo.
echo ============================================
echo   Polymer Series Explorer
echo ============================================
echo.
echo Checking required Python packages...
py -3.12 -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Could not install the required packages.
    echo Make sure Python 3.12 is installed.
    pause
    exit /b 1
)
echo.
echo Launching application...
py -3.12 -m streamlit run app.py
pause
