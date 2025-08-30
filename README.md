# ComfyUI-TransparencyBackgroundRemover

## Intelligent Background Removal Node for ComfyUI

**ComfyUI-TransparencyBackgroundRemover** is a powerful custom node that automatically removes backgrounds from images using advanced AI-powered detection algorithms. Designed for seamless integration with ComfyUI workflows, this node excels at preserving fine edges and details while generating high-quality transparency masks.

### ✨ Key Features

- 🆕 **Content-Aware Edge Detection** - Automatically adapts processing for **Pixel Art** or **Photographic** images to achieve the best results.
- 🎯 **Multi-Method Algorithm** - Combines Roberts Cross, Sobel, and Canny edge detection for superior accuracy and detail preservation.
- 🖼️ **Advanced Edge Refinement** - Specialized algorithms for crisp, pixel-perfect boundaries in pixel art and smooth, clean edges in photos.
- 🔄 **Batch Processing Support** - Process multiple images efficiently in a single operation.
- 📐 **Power-of-8 Scaling** - Optimized scaling with NEAREST neighbor interpolation for pixel-perfect results.
- 🎨 **Multiple Output Formats** - RGBA with embedded alpha or RGB with separate mask.
- 🖥️ **Dither Pattern Handling** - Specialized processing for pixel art and dithered images.
- ⚙️ **Highly Customizable** - Fine-tune parameters for different image types and requirements.

![ComfyUI-TransparencyBackgroundRemover](examples/ComfyUI-TransparencyBackgroundRemover.jpg)
![ComfyUI-TransparencyBackgroundRemover](examples/ComfyUI-TransparencyBackgroundRemover1.jpg)
![ComfyUI-TransparencyBackgroundRemover](examples/ComfyUI-TransparencyBackgroundRemover-batch.jpg)

<!-- Auto GrabCut Examples -->
<p align="center">
  <img src="examples/grabcut_remover.png" width="45%" alt="Auto GrabCut Background Remover Node">
  <img src="examples/grabcut_remover_result.png" width="45%" alt="Auto GrabCut Result">
</p>

---

## 📦 Installation

### Method 1: ComfyUI Manager (Recommended)

1. Open ComfyUI and navigate to **Manager** → **Install via Git URL**
2. Enter the repository URL:
   ```
   https://github.com/Limbicnation/ComfyUI-TransparencyBackgroundRemover
   ```
3. Click **Install** and restart ComfyUI
4. The node will appear under **image/processing** category

### Method 2: Manual Installation

1. **Clone the repository** to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/Limbicnation/ComfyUI-TransparencyBackgroundRemover.git
   ```

2. **Install dependencies**:
   ```bash
   cd ComfyUI-TransparencyBackgroundRemover
   pip install -r requirements.txt
   ```

3. **Restart ComfyUI** to load the new node

### Required Dependencies

- `torch` - PyTorch for tensor operations
- `numpy` - Numerical computing
- `Pillow` - Image processing
- `opencv-python` - Computer vision operations
- `scikit-learn` - Machine learning algorithms for clustering

---

## 🎛️ Node Parameters

### Core Processing Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| **tolerance** | INT | 0-255 | 30 | Color similarity threshold for background detection. Lower values = more selective background detection |
| **edge_sensitivity** | FLOAT | 0.0-1.0 | 0.8 | Edge detection sensitivity. Higher values = more edge detail preservation |
| **foreground_bias** | FLOAT | 0.0-1.0 | 0.7 | Bias towards preserving foreground elements. Higher values = stronger foreground protection |
| **color_clusters** | INT | 2-20 | 8 | Number of color clusters for background analysis. More clusters = finer color distinction |
| **binary_threshold** | INT | 0-255 | 128 | Threshold for generating binary alpha masks. Higher values = more opaque areas |

### Output & Scaling Options

| Parameter | Type | Options | Default | Description |
|-----------|------|---------|---------|-------------|
| **output_size** | DROPDOWN | ORIGINAL, 64x64, 96x96, 128x128, 256x256, 512x512, 768x768, 1024x1024, 1280x1280, 1536x1536, 1792x1792, 2048x2048 | ORIGINAL | Target output dimensions (power-of-8 for optimal scaling) |
| **scaling_method** | DROPDOWN | NEAREST | NEAREST | Interpolation method. NEAREST preserves pixel-perfect detail for pixel art |
| **output_format** | DROPDOWN | RGBA, RGB_WITH_MASK | RGBA | Output format: RGBA (transparency embedded) or RGB with separate mask |

### Advanced Options

| Parameter | Type | Options | Default | Description |
|-----------|------|---------|---------|-------------|
| **edge_detection_mode** | DROPDOWN | AUTO, PIXEL_ART, PHOTOGRAPHIC | AUTO | Selects the edge detection pipeline. AUTO intelligently detects content type. |
| **edge_refinement** | BOOLEAN | | True | Apply post-processing edge refinement for smoother boundaries |
| **dither_handling** | BOOLEAN | | True | Enable specialized processing for dithered patterns and pixel art |
| **batch_processing** | BOOLEAN | | True | Process all images in batch (True) or only first image (False) |

---

## 🚀 Usage Examples

### Basic Background Removal

1. **Load your image** using any ComfyUI image loader node
2. **Connect the image output** to the `image` input of the TransparencyBackgroundRemover node
3. **Adjust parameters** based on your image type:
   - **For most images**: Leave `edge_detection_mode` on `AUTO`.
   - **For specific needs**: Manually select `PIXEL_ART` or `PHOTOGRAPHIC` to override the automatic detection.
4. **Connect the outputs** to preview or save nodes

### Batch Processing Workflow

```
Load Images (Batch) → TransparencyBackgroundRemover → Save Images
                                  ↓
                            (Set batch_processing = True)
```

### Pixel Art Optimization

**Recommended settings for pixel art:**
- `edge_detection_mode`: `PIXEL_ART` (or `AUTO`)
- `tolerance`: 10-20
- `edge_sensitivity`: 0.9-1.0
- `color_clusters`: 4-8
- `dither_handling`: True
- `scaling_method`: NEAREST
- `output_size`: Power-of-8 dimensions (256x256, 512x512, etc.)

### High-Quality Photo Processing

**Recommended settings for photographs:**
- `edge_detection_mode`: `PHOTOGRAPHIC` (or `AUTO`)
- `tolerance`: 25-40
- `edge_sensitivity`: 0.7-0.8
- `foreground_bias`: 0.8-0.9
- `color_clusters`: 10-16
- `edge_refinement`: True

---

## 📋 Workflow Integration

### Example Workflow JSON
```json
{
  "nodes": [
    {
      "type": "LoadImage",
      "pos": [100, 100]
    },
    {
      "type": "TransparencyBackgroundRemover",
      "pos": [400, 100],
      "inputs": {
        "edge_detection_mode": "AUTO",
        "tolerance": 30,
        "edge_sensitivity": 0.8,
        "output_format": "RGBA"
      }
    },
    {
      "type": "PreviewImage", 
      "pos": [700, 100]
    }
  ]
}
```

### Node Connections
- **Input**: Connect any IMAGE output to the `image` input
- **Outputs**: 
  - `image` → Connect to preview, save, or further processing nodes
  - `mask` → Use for compositing, masking, or additional processing

---

## 🎯 Auto GrabCut Background Remover

### Overview
The **Auto GrabCut Background Remover** node provides advanced object detection and segmentation using YOLO and GrabCut algorithms. It can automatically detect objects in images and remove backgrounds with high precision, or refine existing masks for better quality.

### Key Features
- **Automatic Object Detection**: Uses YOLO to identify objects (person, product, vehicle, animal, furniture, electronics)
- **GrabCut Refinement**: Advanced segmentation algorithm for precise edge detection
- **Resize Functionality**: Scale output to preset or custom dimensions
- **Multiple Scaling Methods**: NEAREST (pixel-perfect), BILINEAR, BICUBIC, LANCZOS
- **Mask Refinement**: Improve existing masks from other background removal tools

### Node Parameters

#### Auto GrabCut Remover

| Parameter | Type | Range/Options | Default | Description |
|-----------|------|---------------|---------|-------------|
| **object_class** | DROPDOWN | auto, person, product, vehicle, animal, furniture, electronics | auto | Target object class for detection |
| **confidence_threshold** | FLOAT | 0.3-0.9 | 0.5 | Minimum confidence for object detection |
| **grabcut_iterations** | INT | 1-10 | 5 | Number of GrabCut algorithm iterations |
| **margin_pixels** | INT | 0-50 | 20 | Pixel margin around detected object |
| **edge_refinement** | FLOAT | 0.0-1.0 | 0.7 | Edge refinement strength (0=none, 1=maximum) |
| **binary_threshold** | INT | 128-250 | 200 | Threshold for binary mask conversion |
| **output_size** | DROPDOWN | ORIGINAL, 512x512, 1024x1024, 2048x2048, custom | ORIGINAL | Target output dimensions |
| **scaling_method** | DROPDOWN | NEAREST, BILINEAR, BICUBIC, LANCZOS | NEAREST | Interpolation method for scaling |
| **output_format** | DROPDOWN | RGBA, MASK | RGBA | Output format type |

#### Optional Parameters
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| **initial_mask** | MASK | - | - | Initial mask from previous processing |
| **custom_width** | INT | 64-4096 | 512 | Custom width (when output_size is 'custom') |
| **custom_height** | INT | 64-4096 | 512 | Custom height (when output_size is 'custom') |

### Usage Examples

#### Basic Object Removal
1. Connect your image to the Auto GrabCut node
2. Select the appropriate `object_class` (or leave as "auto")
3. Adjust `confidence_threshold` if needed
4. Choose your desired `output_size` and `scaling_method`
5. Run the workflow

#### Mask Refinement
Use the **GrabCut Refinement** node to improve masks from other sources:
1. Connect an image and its existing mask
2. Adjust `grabcut_iterations` for refinement quality
3. Set `edge_refinement` for smoothing
4. Apply resize options if needed

---

## 🔧 Technical Details

### Supported Image Formats
- **Input**: RGB/RGBA images as ComfyUI tensors
- **Output**: RGBA images with transparency or RGB + separate mask
- **Batch Format**: 4D tensors `[batch, height, width, channels]`

### Performance Considerations
- **Memory Usage**: ~2-4x input image size during processing
- **Batch Processing**: Processes images sequentially with progress indicators
- **Minimum Size**: 64x64 pixels required
- **Recommended**: Use power-of-8 dimensions for optimal scaling performance

### Algorithm Overview
1. **Content-Aware Analysis**: Detects if the image is pixel art or photographic to select the best pipeline.
2. **Multi-Method Edge Detection**: Combines Roberts Cross, Sobel, and Canny algorithms for a robust edge map.
3. **Color Analysis**: K-means clustering to identify dominant background colors.
4. **Edge Refinement**: Applies specialized, content-aware filters to preserve sharp pixel art lines or create smooth photo edges.
5. **Alpha Generation**: Creates a soft mask with configurable thresholds.
6. **Post-Processing**: Optional dither handling and final enhancements.

---

## 🐛 Troubleshooting

### Common Issues

**"Input image must be at least 64x64 pixels"**
- Ensure your input images meet the minimum size requirement
- Use an upscaling node if needed before processing

**"Insufficient memory for processing"**
- Reduce batch size or process images individually
- Set `batch_processing` to False for large images
- Close other memory-intensive applications

**Poor background detection**
- Try switching the `edge_detection_mode` between `PIXEL_ART` and `PHOTOGRAPHIC`.
- Adjust `tolerance` for similar colors.
- Modify `color_clusters` (more clusters for complex backgrounds).

**Jagged or blurry edges**
- Ensure `edge_detection_mode` is set correctly (`PIXEL_ART` for sharp edges, `PHOTOGRAPHIC` for smooth).
- Enable `edge_refinement`.
- Adjust `edge_sensitivity`.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup
```bash
git clone https://github.com/Limbicnation/ComfyUI-TransparencyBackgroundRemover.git
cd ComfyUI-TransparencyBackgroundRemover
pip install -r requirements.txt
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- ComfyUI team for the excellent framework
- Community contributors and testers
- Built with ❤️ for the AI art community

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Limbicnation/ComfyUI-TransparencyBackgroundRemover/issues)
- **Documentation**: [Project Wiki](https://github.com/Limbicnation/ComfyUI-TransparencyBackgroundRemover/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/Limbicnation/ComfyUI-TransparencyBackgroundRemover/discussions)
