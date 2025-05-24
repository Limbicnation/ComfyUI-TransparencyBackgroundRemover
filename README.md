# ComfyUI-TransparencyBackgroundRemover

## Intelligent Background Removal Node for ComfyUI

**ComfyUI-TransparencyBackgroundRemover** is a powerful custom node that automatically removes backgrounds from images using advanced AI-powered detection algorithms. Designed for seamless integration with ComfyUI workflows, this node excels at preserving fine edges and details while generating high-quality transparency masks.

### ✨ Key Features

- 🎯 **Automatic Background Detection** - Intelligent color clustering and edge analysis
- 🖼️ **Edge Preservation** - Advanced edge refinement algorithms for crisp boundaries  
- 🔄 **Batch Processing Support** - Process multiple images efficiently in a single operation
- 📐 **Power-of-8 Scaling** - Optimized scaling with NEAREST neighbor interpolation for pixel-perfect results
- 🎨 **Multiple Output Formats** - RGBA with embedded alpha or RGB with separate mask
- 🖥️ **Dither Pattern Handling** - Specialized processing for pixel art and dithered images
- ⚙️ **Highly Customizable** - Fine-tune parameters for different image types and requirements

![ComfyUI-TransparencyBackgroundRemover](examples/ComfyUI-TransparencyBackgroundRemover.jpg)
![ComfyUI-TransparencyBackgroundRemover](examples/ComfyUI-TransparencyBackgroundRemover1.jpg)

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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **edge_refinement** | BOOLEAN | True | Apply post-processing edge refinement for smoother boundaries |
| **dither_handling** | BOOLEAN | True | Enable specialized processing for dithered patterns and pixel art |
| **batch_processing** | BOOLEAN | True | Process all images in batch (True) or only first image (False) |

---

## 🚀 Usage Examples

### Basic Background Removal

1. **Load your image** using any ComfyUI image loader node
2. **Connect the image output** to the `image` input of the TransparencyBackgroundRemover node
3. **Adjust parameters** based on your image type:
   - **Photographic images**: Use default settings
   - **Pixel art**: Enable `dither_handling`, set `edge_sensitivity` to 0.9+
   - **Complex backgrounds**: Increase `color_clusters` to 12-16
4. **Connect the outputs** to preview or save nodes

### Batch Processing Workflow

```
Load Images (Batch) → TransparencyBackgroundRemover → Save Images
                                  ↓
                            (Set batch_processing = True)
```

### Pixel Art Optimization

**Recommended settings for pixel art:**
- `tolerance`: 10-20
- `edge_sensitivity`: 0.9-1.0
- `color_clusters`: 4-8
- `dither_handling`: True
- `scaling_method`: NEAREST
- `output_size`: Power-of-8 dimensions (256x256, 512x512, etc.)

### High-Quality Photo Processing

**Recommended settings for photographs:**
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
1. **Color Analysis**: K-means clustering to identify dominant colors
2. **Background Detection**: Multi-criteria analysis including edge proximity and color distribution
3. **Edge Refinement**: Gradient-based boundary smoothing
4. **Alpha Generation**: Soft masking with configurable thresholds
5. **Post-Processing**: Optional dither handling and edge enhancement

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
- Increase `tolerance` for similar colors
- Adjust `color_clusters` (more clusters for complex backgrounds)
- Try different `foreground_bias` values

**Jagged edges**
- Enable `edge_refinement`
- Increase `edge_sensitivity`
- For pixel art, ensure `dither_handling` is enabled

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
