@echo off
REM ============================================================================
REM NSKI Full Pipeline Script for Windows
REM ============================================================================
REM This script runs the COMPLETE NSKI experiment pipeline:
REM 1. Validates environment and dependencies
REM 2. Downloads/verifies datasets
REM 3. Runs NSKI on full AdvBench (520 prompts) - REAL, NOT SIMULATED
REM 4. Runs baseline comparisons (Arditi, Belitsky, JBShield)
REM 5. Runs ablation studies
REM 6. Computes perplexity measurements
REM 7. Generates publication-quality figures
REM 8. Creates final results folder
REM
REM IMPORTANT: This uses REAL KV-cache hooks, NOT simulation!
REM ============================================================================

setlocal enabledelayedexpansion

set TIMESTAMP=%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set RESULTS_DIR=results_%TIMESTAMP%

echo.
echo ============================================================================
echo    NSKI: Full Experiment Pipeline
echo    Started: %date% %time%
echo    Results Directory: %RESULTS_DIR%
echo ============================================================================
echo.
echo ⚠ IMPORTANT: This runs REAL KV-cache interventions on actual models.
echo              NOT a simulation. This may take several hours.
echo.
echo Press Ctrl+C to cancel, or
pause

REM Activate virtual environment
echo.
echo [STEP 1/10] Activating virtual environment...
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found. Run setup_windows.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
echo            ✓ Environment activated

REM Create results directory
echo.
echo [STEP 2/10] Creating results directory...
mkdir "%RESULTS_DIR%" 2>nul
mkdir "%RESULTS_DIR%\logs" 2>nul
mkdir "%RESULTS_DIR%\figures" 2>nul
mkdir "%RESULTS_DIR%\data" 2>nul
mkdir "%RESULTS_DIR%\checkpoints" 2>nul
echo            ✓ Directory created: %RESULTS_DIR%

REM Run the full pipeline Python script
echo.
echo [STEP 3/10] Starting Python pipeline...
echo            Running: full_pipeline.py
echo.

python run_full_pipeline.py --output "%RESULTS_DIR%"

if errorlevel 1 (
    echo.
    echo ⚠ Pipeline completed with errors. Check logs for details.
) else (
    echo.
    echo ✓ Pipeline completed successfully!
)

echo.
echo ============================================================================
echo    Pipeline Complete!
echo    Results saved to: %RESULTS_DIR%
echo    Finished: %date% %time%
echo ============================================================================
echo.
echo Contents:
dir /b "%RESULTS_DIR%"
echo.
pause
