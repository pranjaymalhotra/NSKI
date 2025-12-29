@echo off
REM ============================================================================
REM NSKI Setup Script for Windows
REM ============================================================================
REM This script sets up the complete NSKI environment:
REM 1. Creates Python virtual environment
REM 2. Installs PyTorch with CUDA support
REM 3. Installs all dependencies
REM 4. Downloads benchmark datasets
REM 5. Verifies installation
REM
REM Requirements: Python 3.9+, NVIDIA GPU with CUDA support
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo    NSKI: Neural Surgical Key-Value Intervention
echo    Setup Script for Windows + CUDA
echo ============================================================================
echo.

REM Check Python version
echo [1/8] Checking Python installation...
python --version 2>nul
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.9 or higher.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo       Found Python %PYVER%

REM Check for NVIDIA GPU
echo.
echo [2/8] Checking NVIDIA GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo WARNING: nvidia-smi not found. CUDA may not be available.
    echo          Will continue with CPU-only setup.
    set CUDA_AVAILABLE=0
) else (
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    set CUDA_AVAILABLE=1
)

REM Create virtual environment
echo.
echo [3/8] Creating virtual environment...
if exist "venv" (
    echo       Virtual environment already exists.
    choice /c YN /m "       Recreate it?"
    if errorlevel 2 goto skip_venv
    rmdir /s /q venv
)
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)
echo       Virtual environment created: venv/

:skip_venv

REM Activate virtual environment
echo.
echo [4/8] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)
echo       Activated!

REM Upgrade pip
echo.
echo [5/8] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo       pip upgraded!

REM Install PyTorch with CUDA
echo.
echo [6/8] Installing PyTorch with CUDA support...
echo       This may take several minutes...

if "%CUDA_AVAILABLE%"=="1" (
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    pip install torch torchvision torchaudio
)

if errorlevel 1 (
    echo ERROR: Failed to install PyTorch.
    pause
    exit /b 1
)

REM Verify PyTorch CUDA
python -c "import torch; print(f'       PyTorch {torch.__version__}'); print(f'       CUDA available: {torch.cuda.is_available()}')"

REM Install requirements
echo.
echo [7/8] Installing dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo WARNING: Some packages may have failed. Continuing...
)

REM Install NSKI package in development mode
echo       Installing NSKI package...
pip install -e . --quiet
echo       Dependencies installed!

REM Download datasets
echo.
echo [8/8] Downloading benchmark datasets...
echo       Downloading AdvBench (520 harmful prompts)...
echo       Downloading Alpaca (utility evaluation)...
echo       Downloading HarmBench (adversarial test)...

python -c "
import sys
sys.path.insert(0, '.')
from nski.data import DatasetDownloader

print('       Starting downloads...')
downloader = DatasetDownloader(cache_dir='./data_cache')

try:
    path = downloader.download_advbench()
    print(f'       ✓ AdvBench: {path}')
except Exception as e:
    print(f'       ✗ AdvBench failed: {e}')

try:
    path = downloader.download_alpaca()
    print(f'       ✓ Alpaca: {path}')
except Exception as e:
    print(f'       ✗ Alpaca failed: {e}')

try:
    path = downloader.download_harmbench()
    print(f'       ✓ HarmBench: {path}')
except Exception as e:
    print(f'       ✗ HarmBench download note: {e}')

print('       Downloads complete!')
"

REM Final verification
echo.
echo ============================================================================
echo    Verifying Installation
echo ============================================================================

python -c "
import sys
sys.path.insert(0, '.')

print()
print('Checking imports...')

errors = []

try:
    from nski.core import KVCacheHook, RefusalDirectionExtractor, NSKISurgeon
    print('  ✓ Core modules')
except Exception as e:
    errors.append(f'Core: {e}')
    print(f'  ✗ Core modules: {e}')

try:
    from nski.models import ModelLoader, get_model_config, SUPPORTED_MODELS
    print('  ✓ Model modules')
except Exception as e:
    errors.append(f'Models: {e}')
    print(f'  ✗ Model modules: {e}')

try:
    from nski.data import DatasetDownloader, AdvBenchDataset, AlpacaDataset
    print('  ✓ Data modules')
except Exception as e:
    errors.append(f'Data: {e}')
    print(f'  ✗ Data modules: {e}')

try:
    from nski.evaluation import compute_asr, compute_perplexity, KeywordRefusalJudge
    print('  ✓ Evaluation modules')
except Exception as e:
    errors.append(f'Evaluation: {e}')
    print(f'  ✗ Evaluation modules: {e}')

try:
    from nski.baselines import ArditiSteering, BelitskyModulation, JBShield
    print('  ✓ Baseline modules')
except Exception as e:
    errors.append(f'Baselines: {e}')
    print(f'  ✗ Baseline modules: {e}')

try:
    import torch
    print(f'  ✓ PyTorch {torch.__version__}')
    if torch.cuda.is_available():
        print(f'  ✓ CUDA: {torch.cuda.get_device_name(0)}')
        print(f'  ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    else:
        print('  ⚠ CUDA not available (CPU mode)')
except Exception as e:
    errors.append(f'PyTorch: {e}')
    print(f'  ✗ PyTorch: {e}')

print()
if errors:
    print('⚠ Setup completed with warnings. Some features may not work.')
else:
    print('✓ All checks passed! NSKI is ready.')
"

echo.
echo ============================================================================
echo    Setup Complete!
echo ============================================================================
echo.
echo To activate the environment in the future:
echo     venv\Scripts\activate
echo.
echo To run the full experiment pipeline:
echo     run_full_pipeline.bat
echo.
echo To run experiments manually:
echo     python -m nski.experiments.run_all --output results
echo.
pause
