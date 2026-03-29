"""Pydantic validation models for all user-facing GrabCut node parameters.

Security-hardened: sanitizes and validates all inputs before GPU execution.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class GrabCutParams(BaseModel):
    """Parameters for the core GrabCut algorithm."""

    iterations: int = Field(default=5, ge=1, le=100, description="GrabCut iterations (1-100)")
    margin: int = Field(default=20, ge=0, le=500, description="Margin around detected object in pixels")
    edge_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Edge detection threshold for auto-adjustment"
    )
    confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="YOLO confidence threshold"
    )

    model_config = {"str_strip_whitespace": True}


class ScalingParams(BaseModel):
    """Parameters for image scaling / resize preprocessing."""

    target_long_edge: int = Field(
        default=1024, ge=256, le=4096,
        description="Target size for longest image edge"
    )
    maintain_aspect: bool = Field(
        default=True,
        description="Preserve aspect ratio during scaling"
    )
    scaling_method: str = Field(
        default="auto",
        description="Scaling method: auto, nearest, bilinear, lanczos, power-of-8"
    )

    @model_validator(mode="after")
    def check_scaling_method(self) -> "ScalingParams":
        valid = {"auto", "nearest", "bilinear", "bicubic", "lanczos", "power-of-8"}
        normalised = self.scaling_method.lower()
        if normalised not in valid:
            raise ValueError(
                f"Invalid scaling_method '{self.scaling_method}'. "
                f"Must be one of: {', '.join(sorted(valid))}"
            )
        self.scaling_method = normalised
        return self


class MaskParams(BaseModel):
    """Parameters for mask post-processing."""

    edge_blur_amount: float = Field(
        default=0.0, ge=0.0, le=20.0,
        description="Gaussian blur sigma for mask edge softening (0=off, 0.1-20.0)"
    )
    invert_mask: bool = Field(
        default=False,
        description="Invert the output mask (foreground becomes background)"
    )
    edge_refinement_strength: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Strength of edge refinement pass"
    )

    model_config = {"str_strip_whitespace": True}


class BBoxParams(BaseModel):
    """Bounding-box / detection parameters."""

    bbox_safety_margin: int = Field(
        default=30, ge=0, le=200,
        description="Safety margin added to detected bounding boxes in pixels"
    )
    min_bbox_size: int = Field(
        default=64, ge=8, le=1024,
        description="Minimum bounding box side length in pixels"
    )
    fallback_margin_percent: float = Field(
        default=0.2, ge=0.01, le=0.5,
        description="When no object is detected, use this fraction of image size as margin"
    )


class GrabCutNodeParams(GrabCutParams, ScalingParams, MaskParams, BBoxParams):
    """Combined validation model for AutoGrabCutRemover node parameters.

    Used to validate ALL user inputs in a single pass before any GPU work.
    """

    binary_threshold: int = Field(
        default=200, ge=0, le=255,
        description="Threshold for binary mask generation"
    )
    output_format: str = Field(
        default="RGBA",
        description="Output format: RGBA or MASK"
    )
    auto_adjust: bool = Field(
        default=True,
        description="Enable automatic parameter adjustment based on image analysis"
    )

    @model_validator(mode="after")
    def check_output_format(self) -> "GrabCutNodeParams":
        valid = {"RGBA", "MASK"}
        if self.output_format not in valid:
            raise ValueError(
                f"Invalid output_format '{self.output_format}'. Must be one of: {', '.join(sorted(valid))}"
            )
        return self


# ---------------------------------------------------------------------------
# Convenience validator functions (call these at the top of each execute())
# ---------------------------------------------------------------------------

def validate_grabcut_params(**kwargs: Any) -> GrabCutParams:
    """Validate and return GrabCutParams. Raises ValueError on failure."""
    return GrabCutParams(**kwargs)


def validate_scaling_params(**kwargs: Any) -> ScalingParams:
    """Validate and return ScalingParams. Raises ValueError on failure."""
    return ScalingParams(**kwargs)


def validate_mask_params(**kwargs: Any) -> MaskParams:
    """Validate and return MaskParams. Raises ValueError on failure."""
    return MaskParams(**kwargs)


def validate_bbox_params(**kwargs: Any) -> BBoxParams:
    """Validate and return BBoxParams. Raises ValueError on failure."""
    return BBoxParams(**kwargs)


def validate_node_params(**kwargs: Any) -> GrabCutNodeParams:
    """Validate ALL parameters for a GrabCut node.

    Call this at the START of every execute() method before any processing.
    """
    return GrabCutNodeParams(**kwargs)
