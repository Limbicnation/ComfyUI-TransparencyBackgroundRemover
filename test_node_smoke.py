"""Smoke tests that actually invoke each ComfyUI node's entry function.

The original CI only imported modules, which is why PR #15 shipped with
four NameError-class bugs (validate_node_params, log, _log_gpu_memory,
invert_mask) that crash at call time. These tests catch that class of
regression by constructing each node and calling its FUNCTION.

Keep the fixtures small so the suite stays fast in CI — the point is
plumbing/integration correctness, not algorithm quality.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch


# --- synthetic inputs ------------------------------------------------------

IMG_H, IMG_W = 128, 128  # node validation requires >= 64x64


@pytest.fixture
def rgb_image_tensor() -> torch.Tensor:
    """A single-image batch with a solid foreground blob on a flat background.

    Shape: (1, H, W, 3), dtype=float32, range [0, 1] — matches ComfyUI's
    IMAGE contract.
    """
    rng = np.random.default_rng(seed=42)
    img = np.full((IMG_H, IMG_W, 3), 220, dtype=np.uint8)  # light grey bg
    img[32:96, 32:96] = rng.integers(0, 80, size=(64, 64, 3), dtype=np.uint8)
    tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
    return tensor.unsqueeze(0)  # add batch dim


@pytest.fixture
def mask_tensor() -> torch.Tensor:
    """A single-image mask with a solid rectangular foreground region."""
    mask = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    mask[32:96, 32:96] = 1.0
    return torch.from_numpy(mask).unsqueeze(0)


# --- schemas module --------------------------------------------------------

def test_schemas_validate_node_params_normalizes_and_rejects():
    from schemas import validate_node_params, ValidationError

    v = validate_node_params(
        iterations=3,
        margin=10,
        confidence_threshold=0.6,
        scaling_method="lanczos",
        invert_mask=True,
    )
    assert v.iterations == 3
    assert v.scaling_method == "LANCZOS"
    assert v.invert_mask is True

    with pytest.raises(ValidationError):
        validate_node_params(iterations=999)
    with pytest.raises(ValidationError):
        validate_node_params(scaling_method="BOGUS")


# --- nodes.py smoke --------------------------------------------------------

def test_transparency_background_remover_runs(rgb_image_tensor):
    from nodes import TransparencyBackgroundRemover

    node = TransparencyBackgroundRemover()
    image_out, mask_out = node.remove_background(
        rgb_image_tensor,
        tolerance=30,
        edge_sensitivity=0.5,
        foreground_bias=0.7,
        color_clusters=4,
        binary_threshold=128,
        output_format="RGBA",
        output_size="ORIGINAL",
        scaling_method="NEAREST",
        auto_adjust=False,
        edge_detection_mode="AUTO",
    )
    assert isinstance(image_out, torch.Tensor)
    assert isinstance(mask_out, torch.Tensor)
    assert image_out.shape[0] == 1
    assert image_out.shape[1:3] == (IMG_H, IMG_W)


def test_transparency_background_remover_batch_runs(rgb_image_tensor):
    from nodes import TransparencyBackgroundRemoverBatch

    node = TransparencyBackgroundRemoverBatch()
    batch = torch.cat([rgb_image_tensor, rgb_image_tensor], dim=0)
    image_out, mask_out, report = node.batch_remove_background(
        batch,
        tolerance=30,
        edge_sensitivity=0.5,
        auto_adjust=False,
        foreground_bias=0.7,
        color_clusters=4,
        edge_refinement=True,
        dither_handling=False,
        binary_threshold=128,
        progress_reporting=True,
    )
    assert image_out.shape[0] == 2
    assert mask_out.shape[0] == 2
    assert isinstance(report, str) and "Batch" in report or "Total" in report


# --- grabcut_nodes.py smoke ------------------------------------------------

def test_auto_grabcut_remover_runs(rgb_image_tensor):
    """Smoke-test the node that PR #15's bugs primarily affected.

    If validate_node_params, log, or invert_mask aren't wired correctly,
    this will NameError — which is exactly the regression we want to catch.
    """
    from grabcut_nodes import AutoGrabCutRemover

    node = AutoGrabCutRemover()
    image_out, mask_out, bbox, conf, metrics = node.remove_background(
        rgb_image_tensor,
        object_class="auto",
        confidence_threshold=0.5,
        grabcut_iterations=2,  # keep it fast
        margin_pixels=10,
        edge_refinement=0.5,
        edge_blur_amount=0.0,
        bbox_safety_margin=10,
        min_bbox_size=64,
        fallback_margin_percent=0.20,
        binary_threshold=200,
        output_size="ORIGINAL",
        scaling_method="NEAREST",
        auto_adjust=False,
        output_format="RGBA",
        invert_mask=False,
    )
    assert isinstance(image_out, torch.Tensor)
    assert isinstance(mask_out, torch.Tensor)
    assert image_out.shape[0] == 1
    assert isinstance(bbox, str)
    assert isinstance(conf, float)
    assert isinstance(metrics, str)


def test_auto_grabcut_remover_invert_mask_changes_alpha(rgb_image_tensor):
    """invert_mask=True should flip the alpha channel."""
    from grabcut_nodes import AutoGrabCutRemover

    node = AutoGrabCutRemover()
    common = dict(
        object_class="auto",
        confidence_threshold=0.5,
        grabcut_iterations=2,
        margin_pixels=10,
        edge_refinement=0.5,
        edge_blur_amount=0.0,
        bbox_safety_margin=10,
        min_bbox_size=64,
        fallback_margin_percent=0.20,
        binary_threshold=200,
        output_size="ORIGINAL",
        scaling_method="NEAREST",
        auto_adjust=False,
        output_format="RGBA",
    )
    _, mask_normal, _, _, _ = node.remove_background(rgb_image_tensor, invert_mask=False, **common)
    _, mask_inverted, _, _, _ = node.remove_background(rgb_image_tensor, invert_mask=True, **common)

    # Sum should flip: if the total alpha under non-inverted is S, under
    # inverted it should be (H*W - S). Use a loose check since grabcut
    # outputs can be near-full or near-empty depending on the fallback.
    total = float(IMG_H * IMG_W)
    s_normal = float(mask_normal.sum().item())
    s_inverted = float(mask_inverted.sum().item())
    assert abs((s_normal + s_inverted) - total) < total * 0.02  # <2% slack


def test_auto_grabcut_remover_rejects_invalid_params(rgb_image_tensor):
    """Pydantic validation should surface bad params as ValueError, not NameError."""
    from grabcut_nodes import AutoGrabCutRemover

    node = AutoGrabCutRemover()
    with pytest.raises(ValueError, match="Invalid parameters"):
        node.remove_background(
            rgb_image_tensor,
            object_class="auto",
            confidence_threshold=0.5,
            grabcut_iterations=999,  # out of range
            margin_pixels=10,
            edge_refinement=0.5,
            edge_blur_amount=0.0,
            bbox_safety_margin=10,
            min_bbox_size=64,
            fallback_margin_percent=0.20,
            binary_threshold=200,
            output_size="ORIGINAL",
            scaling_method="NEAREST",
            auto_adjust=False,
            output_format="RGBA",
            invert_mask=False,
        )


def test_grabcut_refinement_runs(rgb_image_tensor, mask_tensor):
    from grabcut_nodes import GrabCutRefinement

    node = GrabCutRefinement()
    image_out, mask_out = node.refine_mask(
        rgb_image_tensor,
        mask_tensor,
        grabcut_iterations=2,
        edge_refinement=0.5,
        edge_blur_amount=0.0,
        expand_margin=5,
        bbox_safety_margin=10,
        min_bbox_size=64,
        output_size="ORIGINAL",
        scaling_method="NEAREST",
        invert_mask=False,
    )
    assert image_out.shape[0] == 1
    assert mask_out.shape[0] == 1
    assert image_out.shape[1:3] == (IMG_H, IMG_W)
    assert mask_out.shape[1:] == (IMG_H, IMG_W)


def test_grabcut_refinement_resize_stacks_uniformly(rgb_image_tensor, mask_tensor):
    """Regression for the pre-fix torch.stack shape mismatch.

    If any branch inside the per-item loop fell through with the
    original-sized tensor while others got resized, torch.stack would
    raise on mixed shapes. This exercises the target_size != ORIGINAL
    path end-to-end.
    """
    from grabcut_nodes import GrabCutRefinement

    node = GrabCutRefinement()
    batch = torch.cat([rgb_image_tensor, rgb_image_tensor], dim=0)
    mask_batch = torch.cat([mask_tensor, mask_tensor], dim=0)

    image_out, mask_out = node.refine_mask(
        batch,
        mask_batch,
        grabcut_iterations=2,
        edge_refinement=0.5,
        edge_blur_amount=0.0,
        expand_margin=5,
        bbox_safety_margin=10,
        min_bbox_size=64,
        output_size="512x512",
        scaling_method="NEAREST",
        invert_mask=True,
    )
    assert image_out.shape[0] == 2
    assert mask_out.shape[0] == 2
    # Aspect-preserving resize: output fits within 512 on both axes.
    assert image_out.shape[1] <= 512 and image_out.shape[2] <= 512
    assert mask_out.shape[1] <= 512 and mask_out.shape[2] <= 512


# --- module-level contract -------------------------------------------------

def test_all_nodes_declare_comfyui_metadata():
    """All 4 registered nodes must declare OUTPUT_NODE / DESCRIPTION / IS_CHANGED."""
    from nodes import NODE_CLASS_MAPPINGS as N
    from grabcut_nodes import NODE_CLASS_MAPPINGS as G

    combined = {**N, **G}
    assert len(combined) == 4, f"expected 4 node classes, got {list(combined)}"

    for name, cls in combined.items():
        assert hasattr(cls, "OUTPUT_NODE"), f"{name}: missing OUTPUT_NODE"
        assert hasattr(cls, "DESCRIPTION") and cls.DESCRIPTION, f"{name}: missing DESCRIPTION"
        assert callable(getattr(cls, "IS_CHANGED", None)), f"{name}: missing IS_CHANGED"

        # IS_CHANGED must be deterministic for the same scalar inputs.
        h1 = cls.IS_CHANGED(None, foo=1, bar="x")
        h2 = cls.IS_CHANGED(None, foo=1, bar="x")
        assert h1 == h2, f"{name}: IS_CHANGED not deterministic"
        h3 = cls.IS_CHANGED(None, foo=2, bar="x")
        assert h1 != h3, f"{name}: IS_CHANGED insensitive to inputs"
