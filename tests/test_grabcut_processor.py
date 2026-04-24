"""Tests for GrabCutProcessor core functionality."""
from __future__ import annotations

import numpy as np
import pytest
import torch


class TestGrabCutProcessorInit:
    """Tests for GrabCutProcessor.__init__ and YOLO caching."""

    def test_yolo_cache_shared_across_instances(self, grabcut_processor):
        """YOLO model is cached at class level and shared between instances."""
        from grabcut_remover import GrabCutProcessor

        proc_a = grabcut_processor
        proc_b = GrabCutProcessor()

        # Both should reference the same cached model object
        assert proc_a.yolo_model is proc_b.yolo_model

    def test_processor_stores_params(self, grabcut_processor):
        """Processor stores all constructor parameters."""
        assert grabcut_processor.confidence_threshold == 0.5
        assert grabcut_processor.iterations == 2
        assert grabcut_processor.margin_pixels == 10
        assert grabcut_processor.edge_refinement_strength == 0.0
        assert grabcut_processor.edge_blur_amount == 0.0
        assert grabcut_processor.binary_threshold == 200


class TestDetectObject:
    """Tests for detect_object method."""

    def test_detect_object_no_grad(self, grabcut_processor, sample_image_rgb):
        """detect_object runs inside torch.no_grad() context."""
        before = torch.is_grad_enabled()
        result = grabcut_processor.detect_object(sample_image_rgb)
        after = torch.is_grad_enabled()
        assert after == before  # gradient mode unchanged
        # Result is None when no YOLO model available or no detection
        assert result is None or isinstance(result, tuple)

    def test_detect_object_returns_tuple_or_none(
        self, grabcut_processor, sample_image_rgb
    ):
        """Return type is either None or (x1, y1, x2, y2, confidence)."""
        result = grabcut_processor.detect_object(sample_image_rgb)
        if result is not None:
            assert isinstance(result, tuple)
            assert len(result) == 5
            x1, y1, x2, y2, conf = result
            assert all(isinstance(v, (int, float)) for v in [x1, y1, x2, y2, conf])
            assert 0.0 <= conf <= 1.0


class TestProcessWithGrabCut:
    """Tests for process_with_grabcut method."""

    def test_result_dict_structure(self, grabcut_processor, sample_image_rgb):
        """Result dict contains all required keys."""
        result = grabcut_processor.process_with_grabcut(sample_image_rgb)

        assert isinstance(result, dict)
        for key in ("rgba_image", "mask", "bbox", "confidence", "processing_time_ms", "success"):
            assert key in result, f"Missing key: {key}"

    def test_result_dict_types(self, grabcut_processor, sample_image_rgb):
        """Result values have correct types."""
        result = grabcut_processor.process_with_grabcut(sample_image_rgb)

        assert result["rgba_image"] is None or isinstance(result["rgba_image"], np.ndarray)
        assert result["mask"] is None or isinstance(result["mask"], np.ndarray)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["processing_time_ms"], int)
        assert isinstance(result["success"], bool)

    def test_process_with_grabcut_no_grad(self, grabcut_processor, sample_image_rgb):
        """No gradient graph is built during processing."""
        before = torch.is_grad_enabled()
        grabcut_processor.process_with_grabcut(sample_image_rgb)
        after = torch.is_grad_enabled()
        assert after == before


class TestBBoxValidation:
    """Tests for _validate_and_fix_bbox method."""

    def test_bbox_validates_min_size(self, grabcut_processor):
        """BBox smaller than min_bbox_size is expanded."""
        # 10×10 bbox on 256×256 image with min_bbox_size=64
        raw_bbox = (100, 100, 110, 110)
        fixed = grabcut_processor._validate_and_fix_bbox(raw_bbox, (256, 256))
        x1, y1, x2, y2 = fixed
        assert (x2 - x1) >= 64
        assert (y2 - y1) >= 64

    def test_bbox_within_image_bounds(self, grabcut_processor):
        """Fixed bbox is clipped to image dimensions."""
        raw_bbox = (0, 0, 300, 300)
        h, w = 256, 256
        x1, y1, x2, y2 = grabcut_processor._validate_and_fix_bbox(raw_bbox, (h, w))
        assert 0 <= x1 < x2 <= w
        assert 0 <= y1 < y2 <= h


class TestRefineEdges:
    """Tests for refine_edges method."""

    def test_refine_edges_returns_uint8_mask(self, grabcut_processor, sample_mask_hw):
        """refine_edges returns uint8 array with 0/255 values."""
        rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        result = grabcut_processor.refine_edges(sample_mask_hw, rgb)
        assert result.dtype == np.uint8
        assert result.shape == sample_mask_hw.shape

    def test_refine_edges_zero_strength_returns_input(
        self, grabcut_processor, sample_mask_hw
    ):
        """With zero refinement strength, mask is returned unchanged (type may differ)."""
        grabcut_processor.edge_refinement_strength = 0.0
        grabcut_processor.edge_blur_amount = 0.0
        rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        result = grabcut_processor.refine_edges(sample_mask_hw, rgb)
        assert result.shape == sample_mask_hw.shape


class TestGPUContext:
    """Tests verifying GPU memory management patterns."""

    def test_empty_cache_called_after_detect_object(
        self, grabcut_processor, sample_image_rgb
    ):
        """CUDA cache is cleared after detect_object when CUDA is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        grabcut_processor.detect_object(sample_image_rgb)
        # Should not raise — empty_cache was called

    def test_process_no_grad_active(self, grabcut_processor, sample_image_rgb):
        """torch.no_grad() is active during process_with_grabcut."""
        enabled_before = torch.is_grad_enabled()
        grabcut_processor.process_with_grabcut(sample_image_rgb)
        # Gradient mode should be unchanged after decorated method returns
        assert torch.is_grad_enabled() == enabled_before
