"""Tests for Pydantic validation models in src/validation.py."""
from __future__ import annotations

import pytest
from pydantic import ValidationError


class TestMaskParamsEdgeBlur:
    """edge_blur_amount must accept float values without truncation."""

    def test_edge_blur_amount_stores_float(self):
        """MaskParams.edge_blur_amount preserves float (e.g. 1.7)."""
        from src.validation import MaskParams
        p = MaskParams(edge_blur_amount=1.7)
        assert isinstance(p.edge_blur_amount, float)
        assert p.edge_blur_amount == 1.7

    def test_edge_blur_amount_float_half_step(self):
        """MaskParams accepts 0.5 (was previously truncated to 0 by int() cast)."""
        from src.validation import MaskParams
        p = MaskParams(edge_blur_amount=0.5)
        assert p.edge_blur_amount == 0.5

    def test_edge_blur_amount_boundary_zero(self):
        """edge_blur_amount accepts 0.0."""
        from src.validation import MaskParams
        p = MaskParams(edge_blur_amount=0.0)
        assert p.edge_blur_amount == 0.0

    def test_edge_blur_amount_boundary_max(self):
        """edge_blur_amount accepts 20.0 (upper bound)."""
        from src.validation import MaskParams
        p = MaskParams(edge_blur_amount=20.0)
        assert p.edge_blur_amount == 20.0

    def test_edge_blur_amount_rejects_above_max(self):
        """edge_blur_amount > 20.0 raises ValidationError."""
        from src.validation import MaskParams
        with pytest.raises(ValidationError):
            MaskParams(edge_blur_amount=20.1)

    def test_edge_blur_amount_rejects_negative(self):
        """edge_blur_amount < 0.0 raises ValidationError."""
        from src.validation import MaskParams
        with pytest.raises(ValidationError):
            MaskParams(edge_blur_amount=-0.1)


class TestGrabCutParams:
    """Tests for GrabCutParams model."""

    def test_iterations_min(self):
        """iterations=1 is the minimum accepted value."""
        from src.validation import GrabCutParams
        p = GrabCutParams(iterations=1)
        assert p.iterations == 1

    def test_iterations_max(self):
        """iterations=100 is the maximum accepted value."""
        from src.validation import GrabCutParams
        p = GrabCutParams(iterations=100)
        assert p.iterations == 100

    def test_iterations_rejects_zero(self):
        """iterations=0 is rejected."""
        from src.validation import GrabCutParams
        with pytest.raises(ValidationError):
            GrabCutParams(iterations=0)

    def test_confidence_threshold_range(self):
        """confidence_threshold is clamped to [0.0, 1.0]."""
        from src.validation import GrabCutParams
        p = GrabCutParams(confidence_threshold=0.99)
        assert p.confidence_threshold == 0.99
        with pytest.raises(ValidationError):
            GrabCutParams(confidence_threshold=1.01)


class TestBBoxParams:
    """Tests for BBoxParams model."""

    def test_bbox_safety_margin_range(self):
        """bbox_safety_margin accepts 0-200."""
        from src.validation import BBoxParams
        p = BBoxParams(bbox_safety_margin=150)
        assert p.bbox_safety_margin == 150
        with pytest.raises(ValidationError):
            BBoxParams(bbox_safety_margin=201)

    def test_fallback_margin_percent_range(self):
        """fallback_margin_percent must be between 0.01 and 0.5."""
        from src.validation import BBoxParams
        p = BBoxParams(fallback_margin_percent=0.3)
        assert p.fallback_margin_percent == 0.3
        with pytest.raises(ValidationError):
            BBoxParams(fallback_margin_percent=0.6)


class TestScalingParams:
    """Tests for ScalingParams model."""

    def test_target_long_edge_range(self):
        """target_long_edge accepts 256-4096."""
        from src.validation import ScalingParams
        p = ScalingParams(target_long_edge=2048)
        assert p.target_long_edge == 2048
        with pytest.raises(ValidationError):
            ScalingParams(target_long_edge=128)  # below 256
        with pytest.raises(ValidationError):
            ScalingParams(target_long_edge=8192)  # above 4096

    def test_scaling_method_enum(self):
        """scaling_method must be one of the valid set."""
        from src.validation import ScalingParams
        for method in ("auto", "nearest", "bilinear", "bicubic", "lanczos", "power-of-8"):
            p = ScalingParams(scaling_method=method)
            assert p.scaling_method == method
        with pytest.raises(ValidationError):
            ScalingParams(scaling_method="cubic")  # not in enum

    def test_scaling_method_case_insensitive(self):
        """scaling_method accepts uppercase (node-supplied) values."""
        from src.validation import ScalingParams
        p = ScalingParams(scaling_method="NEAREST")
        assert p.scaling_method == "nearest"
        p2 = ScalingParams(scaling_method="BILINEAR")
        assert p2.scaling_method == "bilinear"


class TestGrabCutNodeParamsCombined:
    """Tests for the combined GrabCutNodeParams model."""

    def test_full_params_combined_model(self):
        """All params pass validation together."""
        from src.validation import GrabCutNodeParams
        p = GrabCutNodeParams(
            iterations=5,
            margin=20,
            edge_threshold=0.5,
            confidence_threshold=0.5,
            target_long_edge=1024,
            maintain_aspect=True,
            scaling_method="auto",
            edge_blur_amount=1.5,
            invert_mask=False,
            edge_refinement_strength=0.7,
            bbox_safety_margin=30,
            min_bbox_size=64,
            fallback_margin_percent=0.2,
            binary_threshold=200,
            output_format="RGBA",
            auto_adjust=False,
        )
        assert p.iterations == 5
        assert p.edge_blur_amount == 1.5
        assert isinstance(p.edge_blur_amount, float)

    def test_output_format_enum(self):
        """output_format must be RGBA or MASK."""
        from src.validation import GrabCutNodeParams
        p = GrabCutNodeParams(output_format="MASK")
        assert p.output_format == "MASK"
        with pytest.raises(ValidationError):
            GrabCutNodeParams(output_format="RGB")

    def test_invert_mask_bool(self):
        """invert_mask is stored as bool."""
        from src.validation import GrabCutNodeParams
        p_true = GrabCutNodeParams(invert_mask=True)
        p_false = GrabCutNodeParams(invert_mask=False)
        assert p_true.invert_mask is True
        assert p_false.invert_mask is False


class TestConvenienceValidators:
    """Tests for the convenience validator functions."""

    def test_validate_grabcut_params(self):
        """validate_grabcut_params returns GrabCutParams."""
        from src.validation import validate_grabcut_params
        result = validate_grabcut_params(iterations=3, margin=15)
        assert result.iterations == 3
        assert result.margin == 15

    def test_validate_node_params_raises_on_bad_input(self):
        """validate_node_params raises ValidationError on invalid params."""
        from src.validation import validate_node_params
        with pytest.raises(ValidationError):
            validate_node_params(iterations=0)  # iterations must be >= 1
