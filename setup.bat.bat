@echo off
REM Setup script for Face Detection & Face Segmentation project
REM This script creates a virtual environment and installs dependencies

echo ============================================================
echo    FACE DETECTION & FACE SEGMENTATION - SETUP
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo Virtual environment created successfully!
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo ============================================================
echo    INSTALLING DEPENDENCIES
echo ============================================================
echo.

REM Install requirements
if exist requirements.txt (
    echo Installing from requirements.txt...
    pip install -r requirements.txt
) else (
    echo requirements.txt not found, installing core packages...
    pip install torch torchvision numpy opencv-python Pillow matplotlib scikit-learn scipy
    pip install tensorboard tqdm PyYAML albumentations colorlog
)

echo.
echo ============================================================
echo    CHECKING CUDA
echo ============================================================
echo.

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
if %errorlevel% equ 0 (
    python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
)

echo.
echo ============================================================
echo    CREATING DIRECTORIES
echo ============================================================
echo.

if not exist "outputs\logs\detection" mkdir "outputs\logs\detection"
if not exist "outputs\logs\segmentation" mkdir "outputs\logs\segmentation"
if not exist "outputs\metrics" mkdir "outputs\metrics"
if not exist "outputs\visualizations" mkdir "outputs\visualizations"
if not exist "models\checkpoints" mkdir "models\checkpoints"
if not exist "weights\pretrained" mkdir "weights\pretrained"
if not exist "weights\trained" mkdir "weights\trained"

echo Directories created!
echo.

echo ============================================================
echo    TESTING INSTALLATION
echo ============================================================
echo.

python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torchvision; print('TorchVision:', torchvision.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"

echo.
echo ============================================================
echo    SETUP COMPLETE!
echo ============================================================
echo.
echo To activate the virtual environment, run:
echo.
echo     venv\Scripts\activate
echo.
echo To train the detection model:
echo     python src\training\train_detection.py --data_root data\processed\wider_face
echo.
echo To train the segmentation model:
echo     python src\training\train_segmentation.py --data_root data\processed\celebamask_hq
echo.
echo ============================================================

pause
