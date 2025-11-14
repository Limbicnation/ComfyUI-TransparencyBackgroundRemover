#!/usr/bin/env python3
"""
ComfyUI-TransparencyBackgroundRemover Dependency Installer

This script installs required dependencies for the TransparencyBackgroundRemover
custom node. It can be called programmatically or run directly.
"""

import subprocess
import sys
import os


def install_dependencies(requirements_path=None):
    """
    Install required dependencies for TransparencyBackgroundRemover.

    Args:
        requirements_path: Optional path to requirements.txt. If None, uses
                          the requirements.txt in the same directory as this script.

    Returns:
        bool: True if installation was successful, False otherwise
    """
    # Determine requirements.txt path
    if requirements_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        requirements_path = os.path.join(script_dir, 'requirements.txt')

    # Check if requirements.txt exists
    if not os.path.exists(requirements_path):
        print(f"ERROR: requirements.txt not found at: {requirements_path}")
        return False

    print("=" * 70)
    print("  ComfyUI-TransparencyBackgroundRemover Dependency Installer")
    print("=" * 70)
    print()
    print(f"Installing dependencies from: {requirements_path}")
    print(f"Using Python: {sys.executable}")
    print()

    try:
        # Install dependencies using pip
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            requirements_path,
            "--upgrade"  # Ensure we get the latest compatible versions
        ])

        print()
        print("=" * 70)
        print("SUCCESS! Dependencies installed successfully.")
        print("=" * 70)
        print()
        print("The TransparencyBackgroundRemover node is now fully functional.")
        print("Please restart ComfyUI to use the updated node.")
        print()
        print("=" * 70)
        return True

    except subprocess.CalledProcessError as e:
        print()
        print("=" * 70)
        print("ERROR: Failed to install dependencies")
        print("=" * 70)
        print()
        print(f"Error details: {e}")
        print()
        print("Please try manual installation:")
        print("  1. Open your ComfyUI Python environment")
        print("  2. Run: pip install scikit-learn")
        print()
        print("=" * 70)
        return False

    except Exception as e:
        print()
        print("=" * 70)
        print("ERROR: Unexpected error during installation")
        print("=" * 70)
        print()
        print(f"Error details: {e}")
        print()
        print("=" * 70)
        return False


def main():
    """Main entry point when script is run directly."""
    success = install_dependencies()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
