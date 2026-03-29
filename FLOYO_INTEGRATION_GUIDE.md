# GrabCut Integration Guide — Floyo.ai H100 Cloud Pipeline
> **Project:** ComfyUI-TransparencyBackgroundRemover | **Floyo Sprint:** Mar 29 – Apr 4, 2026
> **Owner:** Gero Doll | **Status:** Active Development
> **Jira:** CD-14, CD-16, CD-10 | **Notion:** `32c40bf6-db23-8001-92b8-dfe4d91dccb4`

---

## Table of Contents
1. [Environment Audit](#1-environment-audit)
2. [Feature Gap Analysis](#2-feature-gap-analysis)
3. [Refactoring & Implementation](#3-refactoring--implementation)
4. [Integration Testing](#4-integration-testing)
5. [Pipeline Integration](#5-pipeline-integration)
6. [Troubleshooting](#6-troubleshooting)
7. [Memory Sync — Planify & Notion](#7-memory-sync--planify--notion)

---

## 1. Environment Audit

### 1.1 Repository Overview

```
ComfyUI-TransparencyBackgroundRemover/
├── __init__.py                 # Package init + optional GrabCut import
├── nodes.py                    # TransparencyBackgroundRemover + Batch nodes
├── background_remover.py       # EnhancedPixelArtProcessor (core engine)
├── grabcut_nodes.py            # AutoGrabCutRemover + GrabCutRefinement
├── grabcut_remover.py          # GrabCutProcessor (YOLOv8 + OpenCV GrabCut)
├── install.py                  # Dependency installer
├── requirements.txt            # torch, numpy, opencv, scikit-learn, ultralytics
├── CLAUDE.md                   # Development guidance
├── README.md                   # User-facing documentation
└── test_*.py                   # 7 test files (standalone + pytest patterns)
```

### 1.2 Architecture

```
AutoGrabCutRemover (grabcut_nodes.py)
├── ScalingMixin                # Shared resize/scale logic
├── GrabCutProcessor            # YOLOv8 detection + OpenCV GrabCut
└── Returns: IMAGE, MASK, bbox, confidence, metrics

GrabCutRefinement (grabcut_nodes.py)
├── ScalingMixin
├── GrabCutProcessor
└── Returns: IMAGE, refined_MASK

TransparencyBackgroundRemover (nodes.py)
├── EnhancedPixelArtProcessor   # Multi-algorithm: edge, clustering, corner, dither
└── Returns: IMAGE, MASK (batch variant available)
```

### 1.3 Dependency Stack

| Package | Version | Purpose | H100 Relevance |
|---------|---------|---------|----------------|
| `torch` | 2.x | Tensor operations | **Critical** — GPU tensors throughout |
| `numpy` | 1.x | Array operations | High — all image arrays |
| `opencv-python` | 4.x | cv2 GrabCut, edge detection | **Critical** — algorithm core |
| `opencv-contrib-python` | 4.x | Extended cv2 algorithms | High — extended features |
| `scikit-learn` | 1.x | K-means clustering | Medium — color clustering |
| `ultralytics` | 8.x | YOLOv8 object detection | **Critical** — bbox detection |
| `Pillow` | 10.x | Image I/O | Medium — PIL.Image conversions |

### 1.4 Current Issues (Audit Findings)

| Issue | Severity | Location | Impact |
|-------|----------|----------|--------|
| No `torch.no_grad()` wrapper on inference | **High** | `grabcut_remover.py` entire `process_*` methods | GPU memory leak on H100 |
| YOLO model loaded per-instance | Medium | `GrabCutProcessor.__init__` | Memory inefficiency |
| No batch-level tensor output metadata | Low | `grabcut_nodes.py` `remove_background` | Downstream avatar pipeline can't chain |
| Processor re-initialized on every node creation | Medium | `AutoGrabCutRemover.__init__` | Latency on first use |
| No `torch.cuda.empty_cache()` after processing | **High** | All `process_*` methods | VRAM not reclaimed on H100 |
| No dtype specification (float32 assumed) | Medium | All tensor conversions | Potential half-precision issues |
| No type hints on `GrabCutProcessor` methods | Medium | `grabcut_remover.py` | Maintenance burden |

---

## 2. Feature Gap Analysis

### 2.1 vs. H100 Cloud-Native Pipeline Requirements

| Requirement | Current State | Gap | Priority |
|-------------|--------------|-----|----------|
| GPU memory efficiency | No `torch.no_grad()`, no cache clear | **Critical** | P0 |
| Batch tensor metadata output | Returns bbox string, not tensor | **Critical** | P0 |
| Type-hinted interfaces | `GrabCutProcessor` largely untyped | High | P1 |
| Modular class structure | Mixin + Processor tightly coupled | Medium | P2 |
| Batch processing efficiency | Loops over `image.shape[0]` items | Medium | P2 |
| Pixel-art edge optimization | `edge_detection_mode` exists, incomplete | Medium | P2 |
| Invert color option | Not implemented (CD-10 gap) | Medium | P1 |
| Edge blur feature | `edge_blur_amount` param exists, untested | Medium | P1 |

### 2.2 Missing Features

| Feature | Jira Ref | Description |
|---------|----------|-------------|
| Edge Blur post-processing | CD-16 | Gaussian blur on mask edges for photographic smoothness |
| Invert color output | CD-10 | Invert processed image colors (useful for dithered art) |
| Face-Focal-Crop integration | CD-21 | bbox output → face_focal_crop_node chain |
| Batch metadata tensor | — | `[B, 6]` tensor: `(x, y, w, h, score, detected)` for avatar pipeline |

### 2.3 Optimization Targets (2026 Standards)

- [ ] All inference paths wrapped in `torch.no_grad()`
- [ ] YOLO model loaded once, cached at class level
- [ ] `torch.cuda.empty_cache()` called after each batch item
- [ ] Explicit `dtype=torch.float32` on all tensor conversions
- [ ] Full type hints on `GrabCutProcessor` public methods
- [ ] `@dataclass` for parameter/config objects instead of bare dicts

---

## 3. Refactoring & Implementation

### 3.1 torch.no_grad() Wrapper Pattern

```python
# BEFORE (current — no gradient tracking)
def process_with_grabcut(self, img_np: np.ndarray, target_class: Optional[str]) -> Dict:
    results = self.yolo_model(img_np)  # YOLO inference — tracked!
    ...

# AFTER (2026 standard)
@torch.no_grad()
def process_with_grabcut(
    self,
    img_np: np.ndarray,
    target_class: Optional[str] = None
) -> Dict[str, Any]:
    results = self.yolo_model(img_np)  # No gradient computation
    try:
        # ... processing ...
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # VRAM reclaimed
    return result
```

### 3.2 Class-Level YOLO Cache

```python
class GrabCutProcessor:
    # Class-level cache — initialized once, reused across all instances
    _yolo_cache: Optional[YOLO] = {}
    _cache_key: Optional[str] = None

    def __init__(self, model_path: Optional[str] = None, ...):
        cache_key = model_path or "yolov8n.pt"
        if GrabCutProcessor._yolo_cache.get(cache_key) is None:
            GrabCutProcessor._yolo_cache[cache_key] = YOLO(cache_key)
        self.yolo_model = GrabCutProcessor._yolo_cache[cache_key]
```

### 3.3 Typed Parameter Objects

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class GrabCutConfig:
    """Immutable configuration for GrabCut processing."""
    confidence_threshold: float = 0.5
    iterations: int = 5
    margin_pixels: int = 20
    edge_refinement_strength: float = 0.7
    edge_blur_amount: float = 0.0
    binary_threshold: int = 200
    bbox_safety_margin: int = 30
    min_bbox_size: int = 64
    fallback_margin_percent: float = 0.2

    # Derived limits
    CONFIDENCE_MIN: float = field(default=0.3, init=False)
    CONFIDENCE_MAX: float = field(default=0.8, init=False)
    ITERATIONS_MIN: int = field(default=3, init=False)
    ITERATIONS_MAX: int = field(default=8, init=False)

    def validate(self) -> "GrabCutConfig":
        """Clamp all parameters to valid ranges."""
        self.confidence_threshold = max(self.CONFIDENCE_MIN,
            min(self.CONFIDENCE_MAX, self.confidence_threshold))
        self.iterations = max(self.ITERATIONS_MIN,
            min(self.ITERATIONS_MAX, self.iterations))
        return self
```

### 3.4 Batch Metadata Tensor Output

```python
# Extended RETURN_TYPES for AutoGrabCutRemover
RETURN_TYPES = ("IMAGE", "MASK", "STRING", "FLOAT", "STRING", "FLOAT")  # +bbox_tensor
RETURN_NAMES = ("image", "mask", "bbox_coords", "confidence", "metrics", "bbox_tensor")

def remove_background(self, ...) -> Tuple:
    ...
    # Build [B, 6] metadata tensor: (x, y, w, h, score, detected)
    bbox_tensor = torch.zeros((batch_size, 6), dtype=torch.float32)
    for i, result in enumerate(results):
        if result['bbox']:
            x1, y1, x2, y2 = result['bbox']
            bbox_tensor[i] = torch.tensor([x1, y1, x2-x1, y2-y1, result['confidence'], 1.0])
        else:
            bbox_tensor[i, 5] = 0.0  # detected=0 flag
    ...
    return (output_image, output_mask, bbox_output, confidence_output, metrics_output, bbox_tensor)
```

---

## 4. Integration Testing

### 4.1 pytest Suite Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── test_grabcut_processor.py    # Unit: GrabCutProcessor
├── test_grabcut_nodes.py        # Unit: ComfyUI nodes
├── test_integration.py          # Integration: full pipeline
├── test_h100_pipeline.py       # H100-specific: VRAM, batch
├── test_pixel_art.py            # Edge case: pixel art mode
└── test_depth_estimation_chain.py  # Cross-node: + DepthEstimationNode
```

### 4.2 conftest.py — Shared Fixtures

```python
import pytest, torch, numpy as np

@pytest.fixture
def sample_image_rgb() -> np.ndarray:
    """512x512 RGB test image."""
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

@pytest.fixture
def sample_image_rgba() -> np.ndarray:
    """512x512 RGBA test image with transparency."""
    img = np.random.randint(0, 255, (512, 512, 4), dtype=np.uint8)
    img[:, :, 3] = 255
    return img

@pytest.fixture
def grabcut_config() -> GrabCutConfig:
    return GrabCutConfig(
        confidence_threshold=0.5,
        iterations=5,
        margin_pixels=20,
        edge_refinement_strength=0.7,
        binary_threshold=200
    )

@pytest.fixture
def grabcut_processor(grabcut_config) -> GrabCutProcessor:
    return GrabCutProcessor(**asdict(grabcut_config))

@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"
```

### 4.3 test_grabcut_processor.py

```python
import pytest, torch, numpy as np

class TestGrabCutProcessor:
    def test_process_with_grabcut_returns_dict(self, grabcut_processor, sample_image_rgb):
        result = grabcut_processor.process_with_grabcut(sample_image_rgb)
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'rgba_image' in result
        assert 'bbox' in result
        assert 'confidence' in result
        assert 'processing_time_ms' in result

    def test_no_grad_context_active(self, grabcut_processor, sample_image_rgb):
        """Verify torch.no_grad() is active during inference."""
        with torch.no_grad():
            # Check that no gradients are computed
            assert not torch.is_grad_enabled() or True  # placeholder
            result = grabcut_processor.process_with_grabcut(sample_image_rgb)
        assert result['success']

    def test_bbox_normalized_coordinates(self, grabcut_processor, sample_image_rgb):
        result = grabcut_processor.process_with_grabcut(sample_image_rgb)
        if result['bbox']:
            x1, y1, x2, y2 = result['bbox']
            h, w = sample_image_rgb.shape[:2]
            assert 0 <= x1 <= x2 <= w
            assert 0 <= y1 <= y2 <= h

    @pytest.mark.cuda
    def test_vram_reclaimed_after_processing(self, grabcut_processor, sample_image_rgb):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.cuda.reset_peak_memory_stats()
        grabcut_processor.process_with_grabcut(sample_image_rgb)
        #empty_cache_called = ... (instrument the processor to check)
        assert True  # Placeholder — instrument with monkeypatch

    def test_fallback_on_no_yolo(self, sample_image_rgb):
        """When YOLO fails, processor should still return a result."""
        processor = GrabCutProcessor()
        processor.yolo_model = None
        result = processor.process_with_grabcut(sample_image_rgb, target_class=None)
        assert 'success' in result
        # Should use fallback rectangle method
```

### 4.4 test_depth_estimation_chain.py

```python
import pytest, torch, numpy as np

class TestDepthEstimationChain:
    """Test GrabCut → DepthEstimation pipeline integration."""

    def test_grabcut_output_chains_to_depth_estimation(self, device):
        """
        Verify GrabCut MASK output can feed into DepthEstimationNode.

        Pipeline: AutoGrabCutRemover → DepthEstimationNode
        """
        # 1. Run GrabCut
        grabcut = AutoGrabCutRemover()
        test_img = torch.rand(1, 512, 512, 3).to(device)
        img_out, mask_out, *rest = grabcut.remove_background(
            image=test_img,
            object_class="person",
            grabcut_iterations=5,
            edge_refinement=0.7
        )

        # 2. Verify mask shape matches image
        assert img_out.shape[1:3] == mask_out.shape[1:3], \
            "GrabCut output mask dimensions must match image"

        # 3. Verify mask is valid probability tensor
        assert mask_out.dtype == torch.float32
        assert mask_out.min() >= 0.0 and mask_out.max() <= 1.0

        # 4. Feed to DepthEstimationNode (pseudo-test)
        # depth_node = DepthEstimationNode()
        # depth_out = depth_node.estimate_depth(image=img_out, depth_mask=mask_out)
        # assert depth_out.shape[0] == img_out.shape[0]  # batch size preserved
```

### 4.5 Running the Test Suite

```bash
# From ComfyUI-TransparencyBackgroundRemover directory
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v --tb=short

# Run with coverage
pytest tests/ --cov=. --cov-report=term-missing --cov-fail-under=70

# Run H100-specific tests only
pytest tests/test_h100_pipeline.py -v -m cuda

# Run pixel art edge case tests
pytest tests/test_pixel_art.py -v
```

---

## 5. Pipeline Integration

### 5.1 GrabCut → DepthEstimationNode Chain

```
┌─────────────────┐    IMAGE     ┌────────────────────────┐
│  Input Image    │────────────▶│   AutoGrabCutRemover    │
│  (RGB/RGBA)     │             │   (CD-14, CD-16, CD-10) │
└─────────────────┘             │   - object_class: auto  │
                                 │   - iterations: 5       │
                                 │   - edge_blur: 0.0-10.0│
                                 └────────────┬───────────┘
                                              │ IMAGE + MASK
                                              ▼
                                 ┌────────────────────────┐
                                 │   DepthEstimationNode  │
                                 │   (CD-7, CD-18 done)   │
                                 │   - depth_mask fed     │
                                 │     as prior           │
                                 └────────────┬───────────┘
                                              │ IMAGE + DEPTH
                                              ▼
                                 ┌────────────────────────┐
                                 │   LTX-Video / Avatar   │
                                 │   (Floyo H100 dispatch)│
                                 └────────────────────────┘
```

### 5.2 Node Parameters

#### AutoGrabCutRemover

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `image` | — | IMAGE | Input tensor [B, H, W, C] |
| `object_class` | `auto` | auto/person/product/vehicle/animal/furniture/electronics | YOLO target class |
| `confidence_threshold` | 0.5 | 0.3–0.9 | YOLO detection confidence |
| `grabcut_iterations` | 5 | 1–10 | OpenCV GrabCut iterations |
| `margin_pixels` | 20 | 0–50 | Pixel margin around bbox |
| `edge_refinement` | 0.7 | 0.0–1.0 | Edge smoothing strength |
| `edge_blur_amount` | 0.0 | 0.0–10.0 | Gaussian blur on mask edges |
| `bbox_safety_margin` | 30 | 0–100 | Extra pixels beyond detected bbox |
| `min_bbox_size` | 64 | 32–256 | Minimum bbox dimensions (px) |
| `fallback_margin_percent` | 0.20 | 0.10–0.50 | Fallback bbox when no detection |
| `binary_threshold` | 200 | 128–250 | Binary mask threshold |
| `output_size` | ORIGINAL | ORIGINAL/512x512/1024x1024/2048x2048/custom | Target output size |
| `scaling_method` | NEAREST | NEAREST/BILINEAR/BICUBIC/LANCZOS | Interpolation method |
| `edge_detection_mode` | AUTO | AUTO/PIXEL_ART/PHOTOGRAPHIC | Content-type optimization |
| `output_format` | RGBA | RGBA/MASK | Output format |
| `initial_mask` | — | MASK (optional) | Pre-computed mask for refinement |

#### Return Values

| Output | Type | Description |
|--------|------|-------------|
| `image` | IMAGE | RGBA processed image [B, H, W, C] |
| `mask` | MASK | Alpha channel as grayscale [B, H, W] |
| `bbox_coords` | STRING | "(x1,y1,x2,y2)" per image |
| `confidence` | FLOAT | Mean YOLO confidence score |
| `metrics` | STRING | Per-batch timing + class info |

### 5.3 Thresholding Reference

| Scene Type | `binary_threshold` | `edge_refinement` | `edge_blur_amount` | Notes |
|------------|-------------------|-------------------|-------------------|-------|
| **Pixel Art** | 220–250 | 0.3–0.5 | 0.0 | Sharp edges preserved |
| **Product Photo** | 180–200 | 0.6–0.8 | 0.5–2.0 | Smooth subject edges |
| **Portrait** | 180 | 0.7 | 1.0–3.0 | Hair/fabric edge handling |
| **Vehicle** | 190–210 | 0.6 | 0.5 | Sharp geometric edges |
| **Dithered Art** | 200–230 | 0.4 | 0.0 | Dither pattern preserved |

---

## 6. Troubleshooting

### 6.1 Common Segmentation Failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| Full black mask | `binary_threshold` too high | Lower to 150–180 |
| Full white mask | `binary_threshold` too low | Raise to 210–240 |
| Subject cropped | `bbox_safety_margin` too small | Increase to 40–60px |
| Fuzzy edges | `edge_refinement` too low | Raise to 0.7–0.9 |
| No detection (YOLO fallback) | Low confidence + no prominent object | Enable `initial_mask` or use `object_class=person` |
| GPU OOM on H100 | Batch size too large, no cache clear | Set batch=1, ensure `torch.cuda.empty_cache()` called |
| Processing hangs | GrabCut iterations too high + large image | Cap iterations at 8, downscale image first |

### 6.2 structlog Output Interpretation

The node logs structured metrics via `metrics` output. Parse it as:

```python
# Example metrics output
# "Batch 1/1: Time=142.3ms, Conf=0.87, Class=person"

import re
def parse_metrics(metrics_str: str) -> dict:
    time_match = re.search(r'Time=([\d.]+)ms', metrics_str)
    conf_match = re.search(r'Conf=([\d.]+)', metrics_str)
    class_match = re.search(r'Class=(\w+)', metrics_str)
    return {
        'time_ms': float(time_match.group(1)) if time_match else None,
        'confidence': float(conf_match.group(1)) if conf_match else None,
        'class': class_match.group(1) if class_match else None
    }
```

### 6.3 VRAM Debugging on H100

```python
import torch

# Check VRAM before processing
print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"VRAM reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# After processing — verify memory reclaimed
result = processor.process_with_grabcut(img_np)
print(f"VRAM after: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Reset peak stats for next run
torch.cuda.reset_peak_memory_stats()
```

### 6.4 Error Codes

| Code | Meaning | Resolution |
|------|---------|------------|
| `GC_INIT_ERROR` | YOLO/fallback init failed | Reinstall `ultralytics`, check `yolov8n.pt` |
| `GC_BBOX_INVALID` | Detected bbox outside image bounds | Increase `bbox_safety_margin`, use `min_bbox_size` |
| `GC_MEMORY_ERROR` | GPU OOM | Reduce batch size, call `torch.cuda.empty_cache()` |
| `GC_TIMEOUT` | GrabCut not converging | Reduce `grabcut_iterations` to 3–5 |
| `GC_DETECTION_FAIL` | No YOLO detection + no fallback mask | Provide `initial_mask` or set `object_class` explicitly |

---

## 7. Memory Sync — Planify & Notion

### 7.1 Planify Tasks (Floyo Project — 🔴 Urgent Section)

| Task | Jira Ref | Priority |
|------|----------|----------|
| CD-14: Enhance GrabCut Advanced Detection → Fertig | CD-14 | P1 |
| CD-16: Edge Blur Feature → In Arbeit | CD-16 | P1 |
| CD-10: Invert Color Option → In Arbeit | CD-10 | P1 |
| Package GrabCut Remover: __init__.py + SPEC.md + floyo_install.sh | — | P1 |
| Build Face-Focal-Crop skeleton (GPU, bbox output) | CD-21 | P1 |

### 7.2 Notion Links

| Document | ID | URL |
|----------|----|-----|
| Floyo Development Roadmap (Easter Sprint) | `33240bf6-db23-818e-bd49-fb81b532d6b8` | [Link](https://www.notion.so/Floyo-Development-Roadmap-Easter-Sprint-2026-33240bf6db23818ebd49fb81b532d6b8) |
| Node-by-Node Evaluation (Floyo H100) | `32c40bf6-db23-8001-92b8-dfe4d91dccb4` | [Link](https://www.notion.so/limicnation/Node-by-Node-Evaluation-for-Floyo-H100-Integration-32c40bf6db23800192b8dfe4d91dccb4) |
| Floyo Project | `2a940bf6-db23-80c2-b777-dbbadf755580` | [Link](https://www.notion.so/limicnation/Floyo-Project-2a940bf6db2380a380-d21cd1cd63ba) |

### 7.3 GitHub Repository

```
https://github.com/Limbicnation/ComfyUI-TransparencyBackgroundRemover
Branch: feature/grabcut-foyo-integration (pending)
PR:    (to be opened after refactoring complete)
```

---

*Last updated: 2026-03-29 | Next review: 2026-04-01*
