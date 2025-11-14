@echo off
setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo   ComfyUI-TransparencyBackgroundRemover Dependency Installer
echo ============================================================================
echo.
echo This script will install scikit-learn for the background remover node.
echo.

:: Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
set "REQUIREMENTS=%SCRIPT_DIR%requirements.txt"

:: Check if requirements.txt exists
if not exist "%REQUIREMENTS%" (
    echo ERROR: requirements.txt not found at: %REQUIREMENTS%
    echo Please ensure you're running this script from the custom node directory.
    pause
    exit /b 1
)

:: Function to try installing with a given Python executable
set "PYTHON_FOUND=0"
set "PYTHON_PATHS="

echo Searching for Python in ComfyUI installation...
echo.

:: Try to find Python in common ComfyUI locations
:: 1. Portable ComfyUI (python_embeded)
if exist "%SCRIPT_DIR%..\..\..\..\python_embeded\python.exe" (
    set "PYTHON_PATHS=!PYTHON_PATHS!;%SCRIPT_DIR%..\..\..\..\python_embeded\python.exe"
    echo Found: Portable ComfyUI Python (python_embeded^)
)

:: 2. Standalone ComfyUI with venv
if exist "%SCRIPT_DIR%..\..\..\venv\Scripts\python.exe" (
    set "PYTHON_PATHS=!PYTHON_PATHS!;%SCRIPT_DIR%..\..\..\venv\Scripts\python.exe"
    echo Found: ComfyUI venv Python
)

:: 3. ComfyUI in resources directory (ComfyUI App)
if exist "%SCRIPT_DIR%..\..\..\..\resources\ComfyUI\python_embeded\python.exe" (
    set "PYTHON_PATHS=!PYTHON_PATHS!;%SCRIPT_DIR%..\..\..\..\resources\ComfyUI\python_embeded\python.exe"
    echo Found: ComfyUI App Python (resources^)
)

:: 4. Try system Python as fallback
where python >nul 2>&1
if !errorlevel! equ 0 (
    set "PYTHON_PATHS=!PYTHON_PATHS!;python"
    echo Found: System Python
)

:: 5. Try py launcher (Windows)
where py >nul 2>&1
if !errorlevel! equ 0 (
    set "PYTHON_PATHS=!PYTHON_PATHS!;py"
    echo Found: Python launcher (py^)
)

echo.

:: If no Python found, show error
if "!PYTHON_PATHS!"==";" (
    echo ============================================================================
    echo ERROR: Could not find Python installation!
    echo ============================================================================
    echo.
    echo Please install dependencies manually using:
    echo   pip install -r "%REQUIREMENTS%"
    echo.
    echo Or ensure Python is in your PATH.
    echo ============================================================================
    pause
    exit /b 1
)

:: Try each Python path until one works
for %%P in (%PYTHON_PATHS:;= %) do (
    if "%%P" neq "" (
        echo.
        echo ----------------------------------------------------------------------------
        echo Trying: %%P
        echo ----------------------------------------------------------------------------

        :: Test if Python executable works
        "%%P" --version >nul 2>&1
        if !errorlevel! equ 0 (
            echo Python executable is valid. Installing dependencies...
            echo.

            :: Install dependencies
            "%%P" -m pip install -r "%REQUIREMENTS%"

            if !errorlevel! equ 0 (
                echo.
                echo ========================================================================
                echo SUCCESS! Dependencies installed successfully.
                echo ========================================================================
                echo.
                echo The TransparencyBackgroundRemover node is now fully functional.
                echo Please restart ComfyUI to use the updated node.
                echo.
                echo ========================================================================
                set "PYTHON_FOUND=1"
                goto :success
            ) else (
                echo.
                echo WARNING: Installation failed with this Python. Trying next option...
            )
        )
    )
)

:: If we get here, all Python attempts failed
if "!PYTHON_FOUND!"=="0" (
    echo.
    echo ============================================================================
    echo ERROR: Could not install dependencies with any available Python
    echo ============================================================================
    echo.
    echo Please try manual installation:
    echo   1. Open your ComfyUI Python environment
    echo   2. Run: pip install scikit-learn
    echo.
    echo Or contact support if you continue experiencing issues.
    echo ============================================================================
    pause
    exit /b 1
)

:success
pause
exit /b 0
