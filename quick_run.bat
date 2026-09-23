@echo off
chcp 65001 >nul
title Face Detection - Quick Run

echo.
echo ============================================
echo    FACE DETECTION PROJECT
echo    Quick Setup & Run
echo ============================================
echo.

cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [1/5] Creating Virtual Environment...
    echo.
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Cannot create virtual environment
        echo Make sure Python is installed
        pause
        exit /b 1
    )
) else (
    echo [1/5] Virtual Environment already exists
)

echo.
echo [2/5] Activating Virtual Environment...
call venv\Scripts\activate.bat

echo.
echo [3/5] Upgrading pip...
python -m pip install --upgrade pip --quiet

echo.
echo [4/5] Installing Required Packages...
echo    - PyTorch (this may take a few minutes)...

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet

echo.
echo    - Other packages...
pip install opencv-python-headless tqdm PyYAML numpy pillow tensorboard matplotlib --quiet

echo.
echo ============================================
echo [5/5] Running Model Test...
echo ============================================
echo.

python scripts/kaggle/end_to_end_smoke_test.py

echo.
echo ============================================
echo.
echo Done! Press any key to exit...
echo ============================================
pause >nul
