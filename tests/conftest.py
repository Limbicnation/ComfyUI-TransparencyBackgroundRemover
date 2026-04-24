"""Shared pytest fixtures for GrabCut and TransparencyBackgroundRemover tests."""
from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture
def sample_image_rgb() -> np.ndarray:
    """Return a 256×256 RGB image with a centred green square on white."""
    img = np.full((256, 256, 3), 255, dtype=np.uint8)
    img[80:176, 80:176] = (0, 200, 0)  # green square
    return img


@pytest.fixture
def sample_image_rgba() -> np.ndarray:
    """Return a 256×256 RGBA image."""
    img = np.full((256, 256, 4), 255, dtype=np.uint8)
    img[80:176, 80:176, :3] = (200, 0, 0)  # red square
    img[80:176, 80:176, 3] = 255
    return img


@pytest.fixture
def sample_tensor_bhw3() -> torch.Tensor:
    """Return a batch-of-2 CHW RGB tensor (2, 3, 256, 256) in [0, 1]."""
    data = np.full((2, 3, 256, 256), 0.5, dtype=np.float32)
    data[:, :, 80:176, 80:176] = 0.0  # dark patch
    return torch.from_numpy(data)


@pytest.fixture
def sample_mask_hw() -> np.ndarray:
    """Return a 256×256 binary mask with a centred filled circle."""
    y, x = np.ogrid[:256, :256]
    mask = ((x - 128) ** 2 + (y - 128) ** 2) < 40 ** 2
    return (mask * 255).astype(np.uint8)


@pytest.fixture
def sample_bbox() -> tuple[int, int, int, int]:
    """Return a typical (x1, y1, x2, y2) bounding box."""
    return (80, 80, 176, 176)


@pytest.fixture
def grabcut_processor():
    """Return a GrabCutProcessor instance with safe defaults."""
    from grabcut_remover import GrabCutProcessor
    return GrabCutProcessor(
        confidence_threshold=0.5,
        iterations=2,
        margin_pixels=10,
        edge_refinement_strength=0.0,
        edge_blur_amount=0.0,
        binary_threshold=200,
    )


@pytest.fixture
def fallback_processor():
    """Return a FallbackGrabCutProcessor (no YOLO)."""
    from grabcut_remover import create_fallback_processor
    return create_fallback_processor()(
        confidence_threshold=0.5,
        iterations=2,
        margin_pixels=10,
    )
