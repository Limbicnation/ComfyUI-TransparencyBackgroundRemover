# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ComfyUI-TransparencyBackgroundRemover is a custom node for ComfyUI that provides AI-powered background removal with transparency generation. The project focuses on preserving fine edges and details while generating high-quality transparency masks, with specialized support for pixel art and dithered images.

## Installation & Dependencies

```bash
# Install dependencies
pip install -r requirements.txt

# Dependencies include:
# - torch (PyTorch for tensor operations)
# - numpy (numerical computing)
# - Pillow (image processing) 
# - opencv-python (computer vision)
# - scikit-learn (K-means clustering)
```

## Testing Commands

```bash
# Test node imports
python -c "from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS; print('✓ Node import successful'); print(f'Found {len(NODE_CLASS_MAPPINGS)} node classes')"

# Test background remover import
python -c "from background_remover import EnhancedPixelArtProcessor; print('✓ Background remover import successful')"

# Run scaling tests
python test_scaling.py
python test_power_of_8_scaling.py
python test_standalone.py
python test_power_of_8_standalone.py

# Lint code (matches CI configuration)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=examples
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics --exclude=examples

# Validate configuration
python -c "import toml; config=toml.load('pyproject.toml'); print('✓ pyproject.toml is valid')"
```

## Architecture Overview

### Core Components

1. **`background_remover.py`** - `EnhancedPixelArtProcessor` class
   - Core image processing engine with multiple background detection algorithms
   - Edge-based detection using Canny edge detection
   - Color clustering with K-means (2-20 clusters)
   - Corner sampling for background color estimation
   - Dither pattern detection for pixel art support
   - Performance optimizations for large images (downscaling during processing)

2. **`nodes.py`** - ComfyUI node interface
   - `TransparencyBackgroundRemover` - Single image processing
   - `TransparencyBackgroundRemoverBatch` - Batch processing with auto-adjustment
   - ComfyUI tensor format handling (4D tensors: batch, height, width, channels)
   - Graceful fallback when ComfyUI modules unavailable (for testing)

3. **`__init__.py`** - Package initialization
   - Exports `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` for ComfyUI

### Processing Pipeline

1. **Input Validation**: Minimum 64x64 pixels, 4D tensor format
2. **Multi-Algorithm Detection**: Combines edge, clustering, corner, and dither detection
3. **Mask Combination**: Weighted voting system (0.3, 0.3, 0.25, 0.15)
4. **Edge Refinement**: Morphological operations and Gaussian blur
5. **Foreground Bias**: Complexity-based foreground preservation
6. **Binary Thresholding**: Eliminates semi-transparency
7. **Scaling**: Power-of-8 optimized scaling with NEAREST neighbor interpolation

### Key Features

- **Power-of-8 Scaling**: Optimized dimensions (64x64, 256x256, 512x512, etc.) for pixel-perfect results
- **Auto-Parameter Adjustment**: Analyzes edge density, color variance, and contrast
- **Batch Processing**: Sequential processing with detailed reporting
- **Output Formats**: RGBA (embedded alpha) or RGB+mask (separate channels)
- **Performance Optimization**: Large image downscaling during processing, upscaling final mask

## Development Patterns

### Parameter Configuration
- All processing parameters are configurable via ComfyUI interface
- Ranges: tolerance (0-255), edge_sensitivity (0.0-1.0), foreground_bias (0.0-1.0)
- Auto-adjustment based on image analysis (edge density, color variance, contrast)

### Error Handling
- Comprehensive try-catch blocks in main processing functions
- Specific error types: cv2.error, MemoryError, ValueError
- Graceful fallback for failed batch items (empty results with error reporting)

### Testing Strategy
- Standalone test scripts for development outside ComfyUI environment
- Mock ComfyUI modules when dependencies unavailable
- CI testing across Python 3.8-3.11
- Import validation and scaling functionality tests

### ComfyUI Integration
- Follows ComfyUI node conventions (INPUT_TYPES, RETURN_TYPES, FUNCTION)
- Category: "image/processing"
- Tensor format: PyTorch tensors with values 0.0-1.0 (converted from 0-255 numpy arrays)
- Tooltip documentation for all parameters

## File Structure

```
.
├── __init__.py              # ComfyUI node registration
├── nodes.py                 # ComfyUI node interface classes  
├── background_remover.py    # Core processing engine
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Project configuration
├── test_*.py               # Test scripts
├── examples/               # Example images and workflows
└── .github/workflows/      # CI configuration
```

## Common Development Tasks

When modifying the background removal algorithm:
1. Update `EnhancedPixelArtProcessor` methods in `background_remover.py`
2. Test changes with standalone test scripts
3. Verify ComfyUI integration via import tests
4. Run linting before committing changes

When adding new node parameters:
1. Add to `INPUT_TYPES` in appropriate node class
2. Update function signature and processing logic
3. Add parameter documentation (tooltip)
4. Test with various parameter combinations