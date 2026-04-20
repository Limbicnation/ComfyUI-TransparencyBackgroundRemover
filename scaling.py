"""
scaling
=======
Shared resize logic for ComfyUI custom nodes.

Provides ``ScalingMixin`` — a mixin class with deterministic resize helpers
and power-of-8 size support.  Used by both ``RemoveBackgroundAndResizeNode``
and ``grabcut_nodes.py`` to avoid code duplication.

Author:  Gero Doll / Limbicnation
License: Apache-2.0
"""

from __future__ import annotations

from typing import Optional, Tuple

from PIL import Image


class ScalingMixin:
    """
    Provides deterministic resize helpers with power-of-8 size support.
    Mix into any node class that needs resize.
    """

    _RESAMPLING_MAP = {
        "NEAREST": Image.Resampling.NEAREST,
        "BILINEAR": Image.Resampling.BILINEAR,
        "BICUBIC": Image.Resampling.BICUBIC,
        "LANCZOS": Image.Resampling.LANCZOS,
    }

    @staticmethod
    def _parse_size(size_str: str) -> Optional[Tuple[int, int]]:
        """Parse ``WIDTHxHEIGHT`` string → ``(w, h)`` or ``None`` for ORIGINAL."""
        if size_str == "ORIGINAL":
            return None
        try:
            w, h = size_str.lower().split("x")
            return int(w), int(h)
        except (ValueError, AttributeError):
            raise ValueError(f"Invalid size format: {size_str!r}. Use WxH or ORIGINAL.")

    @staticmethod
    def _scale_factor(current: Tuple[int, int], target: Tuple[int, int]) -> float:
        """Smallest scale that fits image inside target, preserving aspect."""
        sw = target[0] / current[0]
        sh = target[1] / current[1]
        return min(sw, sh)

    # ------------------------------------------------------------------
    # public helpers (override in node class if you need custom behaviour)
    # ------------------------------------------------------------------
    def resize_image(
        self,
        image: Image.Image,
        size_str: str,
        method: str = "LANCZOS",
    ) -> Image.Image:
        """
        Resize ``image`` to fit inside the dimensions described by ``size_str``.

        Uses ``Image.resize`` with the selected interpolation method.
        When ``size_str == "ORIGINAL"`` the image is returned unchanged.
        """
        if size_str == "ORIGINAL":
            return image

        target = self._parse_size(size_str)
        if target is None:
            return image  # should not reach

        cur = (image.width, image.height)
        if cur == target:
            return image

        factor = self._scale_factor(cur, target)
        new_w = max(1, int(image.width * factor))
        new_h = max(1, int(image.height * factor))

        resample = self._RESAMPLING_MAP.get(method.upper(), Image.Resampling.LANCZOS)
        return image.resize((new_w, new_h), resample)
