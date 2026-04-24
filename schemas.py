"""Node parameter validation schemas.

Uses Pydantic v2 when available for strict validation; falls back to a
lightweight dataclass-based normalizer when Pydantic is not installed so
the nodes still work in minimal environments.
"""
from __future__ import annotations

from typing import Any

try:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
    _PYDANTIC_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in minimal envs
    _PYDANTIC_AVAILABLE = False
    ValidationError = ValueError  # type: ignore[misc,assignment]


_VALID_SCALING = {"NEAREST", "BILINEAR", "BICUBIC", "LANCZOS"}
_VALID_OUTPUT_FORMAT = {"RGBA", "MASK"}


if _PYDANTIC_AVAILABLE:

    class NodeParams(BaseModel):
        """Validated parameters shared by GrabCut-family nodes.

        Ranges mirror the min/max declared in each node's INPUT_TYPES so
        validation failures match what the ComfyUI UI already enforces.
        """

        model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

        iterations: int = Field(default=5, ge=1, le=10)
        margin: int = Field(default=20, ge=0, le=50)
        confidence_threshold: float = Field(default=0.5, ge=0.3, le=0.9)
        scaling_method: str = Field(default="NEAREST")
        edge_blur_amount: float = Field(default=0.0, ge=0.0, le=10.0)
        invert_mask: bool = False
        edge_refinement_strength: float = Field(default=0.7, ge=0.0, le=1.0)
        bbox_safety_margin: int = Field(default=30, ge=0, le=100)
        min_bbox_size: int = Field(default=64, ge=32, le=256)
        fallback_margin_percent: float = Field(default=0.20, ge=0.10, le=0.50)
        binary_threshold: int = Field(default=200, ge=128, le=250)
        output_format: str = Field(default="RGBA")
        auto_adjust: bool = False

        @field_validator("scaling_method", mode="before")
        @classmethod
        def _normalize_scaling_method(cls, v: Any) -> Any:
            if isinstance(v, str):
                v = v.strip().upper()
                if v not in _VALID_SCALING:
                    raise ValueError(
                        f"scaling_method must be one of {sorted(_VALID_SCALING)}, got {v!r}"
                    )
            return v

        @field_validator("output_format", mode="before")
        @classmethod
        def _normalize_output_format(cls, v: Any) -> Any:
            if isinstance(v, str):
                v = v.strip().upper()
                if v not in _VALID_OUTPUT_FORMAT:
                    raise ValueError(
                        f"output_format must be one of {sorted(_VALID_OUTPUT_FORMAT)}, got {v!r}"
                    )
            return v

    def validate_node_params(**kwargs: Any) -> NodeParams:
        """Validate & normalise keyword args, returning a NodeParams instance."""
        return NodeParams(**kwargs)

else:
    from dataclasses import dataclass, fields

    @dataclass
    class NodeParams:  # type: ignore[no-redef]
        iterations: int = 5
        margin: int = 20
        confidence_threshold: float = 0.5
        scaling_method: str = "NEAREST"
        edge_blur_amount: float = 0.0
        invert_mask: bool = False
        edge_refinement_strength: float = 0.7
        bbox_safety_margin: int = 30
        min_bbox_size: int = 64
        fallback_margin_percent: float = 0.20
        binary_threshold: int = 200
        output_format: str = "RGBA"
        auto_adjust: bool = False

    _RANGES = {
        "iterations": (1, 10),
        "margin": (0, 50),
        "confidence_threshold": (0.3, 0.9),
        "edge_blur_amount": (0.0, 10.0),
        "edge_refinement_strength": (0.0, 1.0),
        "bbox_safety_margin": (0, 100),
        "min_bbox_size": (32, 256),
        "fallback_margin_percent": (0.10, 0.50),
        "binary_threshold": (128, 250),
    }

    def validate_node_params(**kwargs: Any) -> NodeParams:
        """Lightweight fallback validator used when Pydantic is absent.

        Performs the same range clamps and string normalisation Pydantic
        would apply, raising ValueError on out-of-range or unknown enum
        values. Unknown kwargs are silently ignored (matches
        ConfigDict(extra="ignore") on the Pydantic path).
        """
        known = {f.name for f in fields(NodeParams)}
        cleaned = {k: v for k, v in kwargs.items() if k in known}

        scaling = cleaned.get("scaling_method")
        if isinstance(scaling, str):
            scaling = scaling.strip().upper()
            if scaling not in _VALID_SCALING:
                raise ValueError(
                    f"scaling_method must be one of {sorted(_VALID_SCALING)}, got {scaling!r}"
                )
            cleaned["scaling_method"] = scaling

        fmt = cleaned.get("output_format")
        if isinstance(fmt, str):
            fmt = fmt.strip().upper()
            if fmt not in _VALID_OUTPUT_FORMAT:
                raise ValueError(
                    f"output_format must be one of {sorted(_VALID_OUTPUT_FORMAT)}, got {fmt!r}"
                )
            cleaned["output_format"] = fmt

        for name, (lo, hi) in _RANGES.items():
            if name in cleaned:
                v = cleaned[name]
                if v < lo or v > hi:
                    raise ValueError(f"{name} must be in [{lo}, {hi}], got {v}")

        return NodeParams(**cleaned)


__all__ = ["NodeParams", "ValidationError", "validate_node_params"]
