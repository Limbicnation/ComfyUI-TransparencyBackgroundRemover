# Check for optional dependencies and provide helpful warnings
try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("\n" + "="*70)
    print("⚠️  TransparencyBackgroundRemover: scikit-learn not installed!")
    print("="*70)
    print("The node will work with reduced accuracy (color clustering disabled).")
    print("\nTo install dependencies for best results, run ONE of:")
    print("  • Windows (ComfyUI App):      install.bat")
    print("  • Linux/Mac:                  ./install.sh")
    print("  • Manual:                     pip install scikit-learn")
    print("\nInstallation files are in: custom_nodes/ComfyUI-TransparencyBackgroundRemover/")
    print("="*70 + "\n")

from .nodes import NODE_CLASS_MAPPINGS as NODES_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as NODES_DISPLAY_MAPPINGS
from .grabcut_nodes import NODE_CLASS_MAPPINGS as GRABCUT_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as GRABCUT_DISPLAY_MAPPINGS

# Combine node mappings from both modules
NODE_CLASS_MAPPINGS = {**NODES_MAPPINGS, **GRABCUT_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**NODES_DISPLAY_MAPPINGS, **GRABCUT_DISPLAY_MAPPINGS}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
