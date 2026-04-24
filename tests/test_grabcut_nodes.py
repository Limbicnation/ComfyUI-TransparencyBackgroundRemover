"""Tests for AutoGrabCutRemover and GrabCutRefinement nodes."""
from __future__ import annotations

import numpy as np
import torch


class TestAutoGrabCutRemoverReturnTypes:
    """RETURN_TYPES and RETURN_NAMES validation."""

    def test_return_types_count_is_six(self):
        """AutoGrabCutRemover returns 6 values (IMAGE, MASK, STRING, FLOAT, STRING, FLOAT)."""
        from grabcut_nodes import AutoGrabCutRemover
        assert len(AutoGrabCutRemover.RETURN_TYPES) == 6

    def test_return_names_includes_bbox_tensor(self):
        """RETURN_NAMES includes bbox_tensor as the 6th name."""
        from grabcut_nodes import AutoGrabCutRemover
        assert "bbox_tensor" in AutoGrabCutRemover.RETURN_NAMES

    def test_bbox_tensor_is_bbox_tensor_type(self):
        """The 6th RETURN_TYPE is BBOX_TENSOR (custom type for [B,6] metadata)."""
        from grabcut_nodes import AutoGrabCutRemover
        assert "BBOX_TENSOR" in AutoGrabCutRemover.RETURN_TYPES


class TestAutoGrabCutRemoverValidation:
    """Pydantic validation at node entry point."""

    def test_valid_params_pass_validation(self):
        """Valid params do not raise during validation."""
        from grabcut_nodes import AutoGrabCutRemover
        node = AutoGrabCutRemover()
        # Should not raise
        image = torch.zeros((1, 256, 256, 3), dtype=torch.float32)
        node.remove_background(
            image,
            grabcut_iterations=5,
            margin_pixels=20,
            confidence_threshold=0.5,
            edge_blur_amount=0.5,  # float — was previously truncated to int
            output_format="RGBA",
        )

    def test_edge_blur_amount_accepts_float(self):
        """edge_blur_amount as float (e.g. 1.7) is accepted without truncation."""
        from grabcut_nodes import AutoGrabCutRemover
        node = AutoGrabCutRemover()
        image = torch.zeros((1, 256, 256, 3), dtype=torch.float32)
        result = node.remove_background(
            image,
            grabcut_iterations=2,
            margin_pixels=10,
            confidence_threshold=0.5,
            edge_blur_amount=1.7,  # was: int cast → 1
            output_format="RGBA",
        )
        assert result is not None
        assert len(result) == 6


class TestBBoxMetadataTensor:
    """Tests for the [B, 6] bbox_metadata tensor output."""

    def test_bbox_metadata_shape_batch_2(self):
        """bbox_metadata tensor has shape (batch_size, 6)."""
        from grabcut_nodes import AutoGrabCutRemover
        node = AutoGrabCutRemover()
        # Skip YOLO to avoid model loading in test
        node.processor = type("MockProc", (), {
            "confidence_threshold": 0.5,
            "iterations": 2,
            "margin_pixels": 10,
            "edge_refinement_strength": 0.0,
            "edge_blur_amount": 0.0,
            "binary_threshold": 200,
            "process_with_grabcut": lambda self, img, target=None: {
                "success": False,
                "rgba_image": np.zeros((256, 256, 4), dtype=np.uint8),
                "mask": np.zeros((256, 256), dtype=np.uint8),
                "bbox": None,
                "confidence": 0.0,
                "processing_time_ms": 0,
            },
            "_detect_pixel_art_characteristics": lambda self, img: False,
        })()

        image = torch.zeros((2, 256, 256, 3), dtype=torch.float32)
        result = node.remove_background(image, output_format="RGBA")
        bbox_tensor = result[5]  # 6th return value
        assert bbox_tensor.shape == (2, 6)

    def test_bbox_metadata_dtype_float32(self):
        """bbox_metadata tensor is dtype float32."""
        from grabcut_nodes import AutoGrabCutRemover
        node = AutoGrabCutRemover()
        node.processor = type("MockProc", (), {
            "process_with_grabcut": lambda self, img, target=None: {
                "success": False,
                "rgba_image": np.zeros((256, 256, 4), dtype=np.uint8),
                "mask": np.zeros((256, 256), dtype=np.uint8),
                "bbox": None,
                "confidence": 0.0,
                "processing_time_ms": 0,
            },
            "_detect_pixel_art_characteristics": lambda self, img: False,
        })()
        image = torch.zeros((1, 256, 256, 3), dtype=torch.float32)
        result = node.remove_background(image, output_format="RGBA")
        bbox_tensor = result[5]
        assert bbox_tensor.dtype == torch.float32

    def test_bbox_metadata_zeros_on_failure(self):
        """When detection fails, bbox_metadata is all zeros (detected=0.0)."""
        from grabcut_nodes import AutoGrabCutRemover
        node = AutoGrabCutRemover()
        node.processor = type("MockProc", (), {
            "process_with_grabcut": lambda self, img, target=None: {
                "success": False,
                "rgba_image": np.zeros((256, 256, 4), dtype=np.uint8),
                "mask": np.zeros((256, 256), dtype=np.uint8),
                "bbox": None,
                "confidence": 0.0,
                "processing_time_ms": 0,
            },
            "_detect_pixel_art_characteristics": lambda self, img: False,
        })()
        image = torch.zeros((1, 256, 256, 3), dtype=torch.float32)
        result = node.remove_background(image, output_format="RGBA")
        bbox_tensor = result[5]
        # All zeros on failure (including detected flag = 0.0)
        assert bbox_tensor.abs().sum().item() == 0.0


class TestGrabCutRefinementNode:
    """Tests for GrabCutRefinement node."""

    def test_refine_mask_invert_mask_true(self):
        """With invert_mask=True, output alpha is inverted."""
        from grabcut_nodes import GrabCutRefinement
        node = GrabCutRefinement()
        node.processor = type("MockProc", (), {
            "iterations": 2,
            "process_with_initial_mask": lambda self, img, mask, target=None: {
                "success": True,
                "rgba_image": np.zeros((256, 256, 4), dtype=np.uint8),
                "mask": np.full((256, 256), 255, dtype=np.uint8),  # full foreground
                "bbox": (50, 50, 200, 200),
                "confidence": 0.8,
                "processing_time_ms": 10,
            }
        })()

        image = torch.zeros((1, 256, 256, 3), dtype=torch.float32)
        mask = torch.ones((1, 256, 256), dtype=torch.float32)

        # invert_mask=False
        out_not_inv = node.refine_mask(image, mask, invert_mask=False)
        # invert_mask=True
        out_inv = node.refine_mask(image, mask, invert_mask=True)

        # Alpha values should differ
        alpha_not_inv = out_not_inv[1]  # refined_mask
        alpha_inv = out_inv[1]
        assert not np.allclose(alpha_not_inv.numpy(), alpha_inv.numpy())

    def test_refine_mask_no_grad(self):
        """refine_mask runs with torch.no_grad()."""
        from grabcut_nodes import GrabCutRefinement
        node = GrabCutRefinement()
        enabled = torch.is_grad_enabled()
        image = torch.zeros((1, 256, 256, 3), dtype=torch.float32)
        mask = torch.ones((1, 256, 256), dtype=torch.float32)
        node.refine_mask(image, mask)
        assert torch.is_grad_enabled() == enabled


class TestInvertMaskInputTypes:
    """Verify invert_mask is declared in INPUT_TYPES."""

    def test_auto_grabcut_has_invert_mask_input(self):
        """AutoGrabCutRemover INPUT_TYPES includes invert_mask."""
        from grabcut_nodes import AutoGrabCutRemover
        inp = AutoGrabCutRemover.INPUT_TYPES()
        assert "invert_mask" in inp.get("optional", {})

    def test_grabcut_refinement_has_invert_mask_input(self):
        """GrabCutRefinement INPUT_TYPES includes invert_mask."""
        from grabcut_nodes import GrabCutRefinement
        inp = GrabCutRefinement.INPUT_TYPES()
        assert "invert_mask" in inp.get("optional", {})
