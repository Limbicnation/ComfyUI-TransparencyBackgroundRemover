#!/bin/bash

echo ""
echo "============================================================================"
echo "  ComfyUI-TransparencyBackgroundRemover Dependency Installer"
echo "============================================================================"
echo ""
echo "This script will install scikit-learn for the background remover node."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"

# Check if requirements.txt exists
if [ ! -f "$REQUIREMENTS" ]; then
    echo "ERROR: requirements.txt not found at: $REQUIREMENTS"
    echo "Please ensure you're running this script from the custom node directory."
    exit 1
fi

PYTHON_FOUND=0
PYTHON_PATHS=()

echo "Searching for Python in ComfyUI installation..."
echo ""

# Try to find Python in common ComfyUI locations
# 1. Portable ComfyUI (python_embeded) - Linux
if [ -f "$SCRIPT_DIR/../../../../python/bin/python" ]; then
    PYTHON_PATHS+=("$SCRIPT_DIR/../../../../python/bin/python")
    echo "Found: Portable ComfyUI Python (python/bin)"
fi

# 2. Standalone ComfyUI with venv
if [ -f "$SCRIPT_DIR/../../../venv/bin/python" ]; then
    PYTHON_PATHS+=("$SCRIPT_DIR/../../../venv/bin/python")
    echo "Found: ComfyUI venv Python"
fi

# 3. Python3 from PATH
if command -v python3 &> /dev/null; then
    PYTHON_PATHS+=("python3")
    echo "Found: System Python3"
fi

# 4. Python from PATH (fallback)
if command -v python &> /dev/null; then
    PYTHON_PATHS+=("python")
    echo "Found: System Python"
fi

echo ""

# If no Python found, show error
if [ ${#PYTHON_PATHS[@]} -eq 0 ]; then
    echo "============================================================================"
    echo "ERROR: Could not find Python installation!"
    echo "============================================================================"
    echo ""
    echo "Please install dependencies manually using:"
    echo "  pip install -r $REQUIREMENTS"
    echo ""
    echo "Or ensure Python is in your PATH."
    echo "============================================================================"
    exit 1
fi

# Try each Python path until one works
for PYTHON_CMD in "${PYTHON_PATHS[@]}"; do
    echo ""
    echo "----------------------------------------------------------------------------"
    echo "Trying: $PYTHON_CMD"
    echo "----------------------------------------------------------------------------"

    # Test if Python executable works
    if "$PYTHON_CMD" --version &> /dev/null; then
        echo "Python executable is valid. Installing dependencies..."
        echo ""

        # Install dependencies
        "$PYTHON_CMD" -m pip install -r "$REQUIREMENTS"

        if [ $? -eq 0 ]; then
            echo ""
            echo "========================================================================"
            echo "SUCCESS! Dependencies installed successfully."
            echo "========================================================================"
            echo ""
            echo "The TransparencyBackgroundRemover node is now fully functional."
            echo "Please restart ComfyUI to use the updated node."
            echo ""
            echo "========================================================================"
            PYTHON_FOUND=1
            exit 0
        else
            echo ""
            echo "WARNING: Installation failed with this Python. Trying next option..."
        fi
    fi
done

# If we get here, all Python attempts failed
if [ $PYTHON_FOUND -eq 0 ]; then
    echo ""
    echo "============================================================================"
    echo "ERROR: Could not install dependencies with any available Python"
    echo "============================================================================"
    echo ""
    echo "Please try manual installation:"
    echo "  1. Activate your ComfyUI Python environment"
    echo "  2. Run: pip install scikit-learn"
    echo ""
    echo "Or contact support if you continue experiencing issues."
    echo "============================================================================"
    exit 1
fi
