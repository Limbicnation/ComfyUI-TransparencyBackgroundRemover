# Check for optional dependencies and provide helpful warnings
try:
    import sklearn  # noqa: F401
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

# Import main transparency background remover nodes.
# Try relative import first (ComfyUI load path); fall back to absolute so
# the module also works when loaded outside a package context (e.g. pytest
# discovering files under a repo dir whose hyphenated name isn't a valid
# Python identifier).
try:
    from .nodes import NODE_CLASS_MAPPINGS as NODES_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as NODES_DISPLAY_MAPPINGS
except ImportError:
    from nodes import NODE_CLASS_MAPPINGS as NODES_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as NODES_DISPLAY_MAPPINGS

# Import standalone RemoveBackgroundAndResizeNode
try:
    from .RemoveBackgroundAndResizeNode import (
        NODE_CLASS_MAPPINGS as REMBG_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as REMBG_DISPLAY_MAPPINGS,
    )
    REMBG_AVAILABLE = True
except ImportError as e:
    REMBG_MAPPINGS = {}
    REMBG_DISPLAY_MAPPINGS = {}
    REMBG_AVAILABLE = False
    print("\n" + "="*70)
    print("ℹ️  RemoveBackgroundAndResizeNode unavailable:", str(e))
    print("="*70 + "\n")

# Try to import GrabCut nodes (optional - requires ultralytics)
try:
    try:
        from .grabcut_nodes import NODE_CLASS_MAPPINGS as GRABCUT_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as GRABCUT_DISPLAY_MAPPINGS
    except ImportError:
        from grabcut_nodes import NODE_CLASS_MAPPINGS as GRABCUT_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as GRABCUT_DISPLAY_MAPPINGS
    GRABCUT_AVAILABLE = True
except ImportError as e:
    # GrabCut nodes require additional dependencies (ultralytics, etc.)
    GRABCUT_MAPPINGS = {}
    GRABCUT_DISPLAY_MAPPINGS = {}
    GRABCUT_AVAILABLE = False
    print("\n" + "="*70)
    print("ℹ️  TransparencyBackgroundRemover: GrabCut nodes disabled")
    print("="*70)
    print(f"Reason: {str(e)}")
    print("\nGrabCut nodes require additional dependencies. To enable them:")
    print("  • Install: pip install ultralytics")
    print("  • Or run: pip install -r requirements.txt")
    print("\nMain background remover nodes are still available!")
    print("="*70 + "\n")

# Combine node mappings from all modules
NODE_CLASS_MAPPINGS = {**NODES_MAPPINGS, **GRABCUT_MAPPINGS, **REMBG_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**NODES_DISPLAY_MAPPINGS, **GRABCUT_DISPLAY_MAPPINGS, **REMBG_DISPLAY_MAPPINGS}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
