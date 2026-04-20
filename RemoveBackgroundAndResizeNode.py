"""
RemoveBackgroundAndResizeNode
============================
Single-file ComfyUI node for background removal with integrated resize.

Based on ``TransparencyBackgroundRemover`` and ``ScalingMixin`` from the
ComfyUI-TransparencyBackgroundRemover package.  Adapted for standalone use
with minimal dependencies — PIL only for background removal, no hard
OpenCV requirement.

Author:  Gero Doll / Limbicnation
License: Apache-2.0
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

try:
    import structlog
    log = structlog.get_logger(__name__)
except ImportError:
    log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ScalingMixin — shared resize logic (see scaling.py)
# ---------------------------------------------------------------------------

try:
    from .scaling import ScalingMixin  # noqa: F401 — re-exported for convenience
except ImportError:
    from scaling import ScalingMixin


# ---------------------------------------------------------------------------
# PIL-only background removal
# ---------------------------------------------------------------------------

def _kmeans_colors(
    arr: np.ndarray, n_clusters: int, tol: float = 0.01, max_iter: int = 20,
    max_samples: int = 50000,
) -> np.ndarray:
    """
    Simple k-means on pixel array [H*W, channels].
    Returns ``(n_clusters, channels)`` cluster centres.
    Fallback when scikit-learn is unavailable.

    Parameters
    ----------
    arr : np.ndarray
        Input image array (H, W, C).
    n_clusters : int
        Number of k-means clusters.
    tol : float
        Convergence tolerance for centre shift.
    max_iter : int
        Maximum k-means iterations.
    max_samples : int
        Cap on number of pixels fed to k-means to avoid OOM on large images.
        Subsampled uniformly when pixel count exceeds this value.
    """
    pixels = arr.reshape(-1, arr.shape[-1]).astype(np.float32)

    # Subsample to avoid OOM on large images (e.g. 4K → 8.3M pixels)
    if len(pixels) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(pixels), size=max_samples, replace=False)
        pixels = pixels[idx]

    indices = np.random.RandomState(42).choice(
        len(pixels), size=n_clusters, replace=False
    )
    centres = pixels[indices].copy()

    for _ in range(max_iter):
        dists = np.linalg.norm(pixels[:, None] - centres[None], axis=-1)
        labels = dists.argmin(axis=1)
        new_centres = np.stack([pixels[labels == i].mean(axis=0) for i in range(n_clusters)])
        if np.abs(new_centres - centres).max() < tol:
            break
        centres = new_centres

    return centres


def _dominant_bg(arr: np.ndarray, samples: int = 5000) -> np.ndarray:
    """Sample corners + border to estimate background colour."""
    h, w = arr.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    bw = max(1, min(h, w) // 10)

    # top / bottom strips and left / right strips (intersection gives the four corners)
    mask[:bw, :] = True
    mask[-bw:, :] = True
    mask[:, :bw] = True
    mask[:, -bw:] = True

    bg_pixels = arr[mask]
    if len(bg_pixels) > samples:
        idx = np.random.RandomState(42).choice(len(bg_pixels), samples, replace=False)
        bg_pixels = bg_pixels[idx]

    return bg_pixels.mean(axis=0).astype(np.uint8)


def remove_background_pil(
    image: Image.Image,
    tolerance: int = 30,
    color_clusters: int = 8,
    foreground_bias: float = 0.7,
    binary_threshold: int = 128,
    dither_handling: bool = True,
) -> Image.Image:
    """
    Remove background from a PIL Image using colour clustering.

    Pipeline
    --------
    1. Flatten to ``[H*W, channels]`` and run k-means to find dominant colours.
    2. Identify background as the colour most similar to image edges / corners.
    3. Compute per-pixel distance to foreground centres; keep pixels that are
       closer than ``tolerance`` (adjusted by ``foreground_bias``).
    4. Apply simple edge refinement via binary thresholding.
    5. Return ``RGBA`` PIL Image.

    Parameters
    ----------
    image : PIL.Image
        Input image (RGB or RGBA; RGBA ``A`` is ignored).
    tolerance : int
        Colour distance threshold (0-255). Higher = more pixels accepted as foreground.
    color_clusters : int
        K-means centres used to identify distinct colours.
    foreground_bias : float
        Bias towards keeping pixels (0 = discard more, 1 = keep more).
    binary_threshold : int
        Threshold for alpha mask binarisation (0-255).
    dither_handling : bool
        When True, apply a 1-pixel border erosion to reduce dither artefacts.

    Returns
    -------
    PIL.Image
        RGBA image with transparent background.
    """
    rgb = image.convert("RGB")
    arr = np.array(rgb, dtype=np.float32)  # [H, W, 3]

    # 1. k-means colour clustering
    if color_clusters < 2:
        color_clusters = 2
    centres = _kmeans_colors(arr, color_clusters)

    # 2. background colour from image perimeter
    bg_colour = _dominant_bg(arr)

    # 3. distance of each pixel to background vs foreground centres
    h, w = arr.shape[:2]
    flat = arr.reshape(-1, 3)

    bg_dist = np.linalg.norm(flat - bg_colour.astype(np.float32), axis=1)
    fg_dist = np.linalg.norm(
        flat[:, None] - centres.astype(np.float32), axis=-1
    ).min(axis=1)

    # effective tolerance — foreground_bias shifts the acceptance boundary
    effective_tol = tolerance * (1.5 - foreground_bias)
    mask = (fg_dist + effective_tol) < bg_dist

    alpha = mask.reshape(h, w).astype(np.uint8) * 255

    # 4. edge refinement
    if dither_handling:
        from PIL import ImageFilter

        alpha_pil = Image.fromarray(alpha, mode="L")
        alpha_pil = alpha_pil.filter(ImageFilter.MinFilter(3))
        alpha = np.array(alpha_pil)

    # 5. binarise
    alpha = (alpha > binary_threshold).astype(np.uint8) * 255

    # 6. assemble RGBA
    result = np.dstack([arr.astype(np.uint8), alpha])
    return Image.fromarray(result, mode="RGBA")


# ---------------------------------------------------------------------------
# ComfyUI node
# ---------------------------------------------------------------------------

class RemoveBackgroundAndResizeNode(ScalingMixin):
    """
    Combines background removal and resize in a single node.

    ``INPUT_TYPES`` exposes tolerance, edge sensitivity, colour-cluster, and
    output-size controls so the node can be dropped into any workflow without
    separate background-removal and resize chains.

    Outputs
    -------
    IMAGE : RGBA tensor (N, H, W, 4), values 0-1
    MASK  : greyscale alpha tensor (N, H, W),    values 0-1
    """

    # ------------------------------------------------------------------
    # ComfyUI contract
    # ------------------------------------------------------------------
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "process"
    CATEGORY = "image/processing"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image tensor from upstream node"}),
                "tolerance": (
                    "INT",
                    {
                        "default": 30,
                        "min": 0,
                        "max": 255,
                        "step": 1,
                        "display": "number",
                        "tooltip": (
                            "Colour distance threshold. Higher = more aggressive "
                            "foreground extraction (0-255)."
                        ),
                    },
                ),
                "foreground_bias": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": (
                            "Bias toward keeping pixels as foreground. "
                            "Higher = more pixels preserved."
                        ),
                    },
                ),
                "color_clusters": (
                    "INT",
                    {
                        "default": 8,
                        "min": 2,
                        "max": 20,
                        "step": 1,
                        "display": "number",
                        "tooltip": "Number of k-means centres for colour segmentation.",
                    },
                ),
                "output_size": (
                    [
                        "ORIGINAL",
                        "512x512",
                        "768x768",
                        "1024x1024",
                        "1280x720",
                        "1920x1080",
                    ],
                    {
                        "default": "ORIGINAL",
                        "tooltip": (
                            "Resize output to this size. "
                            "ORIGINAL keeps source dimensions."
                        ),
                    },
                ),
                "scaling_method": (
                    ["NEAREST", "BILINEAR", "BICUBIC", "LANCZOS"],
                    {
                        "default": "LANCZOS",
                        "tooltip": (
                            "Interpolation method. "
                            "LANCZOS = best quality; "
                            "NEAREST = pixel-perfect for pixel art."
                        ),
                    },
                ),
            },
            "optional": {
                "dither_handling": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Apply edge erosion to suppress dither artefacts.",
                    },
                ),
                "binary_threshold": (
                    "INT",
                    {
                        "default": 128,
                        "min": 0,
                        "max": 255,
                        "step": 1,
                        "display": "number",
                        "tooltip": "Alpha mask binarisation cutoff (0-255).",
                    },
                ),
            },
        }

    # ------------------------------------------------------------------
    # processing
    # ------------------------------------------------------------------

    @torch.no_grad()
    def process(
        self,
        image: torch.Tensor,
        tolerance: int,
        foreground_bias: float,
        color_clusters: int,
        output_size: str,
        scaling_method: str,
        dither_handling: bool = True,
        binary_threshold: int = 128,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        ``image`` — ComfyUI IMAGE tensor, shape (N, H, W, C), float32 0-1.
        Returns (IMAGE tensor, MASK tensor).
        """
        # ------------------------------------------------------------------
        # validation
        # ------------------------------------------------------------------
        if image is None or image.numel() == 0:
            raise ValueError("RemoveBackgroundAndResizeNode: received empty image tensor.")
        if image.ndim != 4:
            raise ValueError(
                f"RemoveBackgroundAndResizeNode: expected 4-D IMAGE tensor "
                f"(N,H,W,C), got {image.ndim}-D."
            )
        if image.shape[-1] not in (3, 4):
            raise ValueError(
                f"RemoveBackgroundAndResizeNode: IMAGE must be RGB or RGBA; "
                f"got channels={image.shape[-1]}."
            )

        # ------------------------------------------------------------------
        # per-batch processing
        # ------------------------------------------------------------------
        rgba_tensors: list[torch.Tensor] = []
        mask_tensors: list[torch.Tensor] = []

        for i in range(image.shape[0]):
            img_np = (image[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

            # handle RGB input
            if img_np.shape[-1] == 3:
                pil_in = Image.fromarray(img_np, mode="RGB")
            else:
                pil_in = Image.fromarray(img_np[:, :, :3], mode="RGB")

            # ---- remove background ----
            pil_rgba = remove_background_pil(
                pil_in,
                tolerance=tolerance,
                foreground_bias=foreground_bias,
                color_clusters=color_clusters,
                binary_threshold=binary_threshold,
                dither_handling=dither_handling,
            )

            # ---- resize ----
            pil_rgba = self.resize_image(pil_rgba, output_size, scaling_method)

            # ---- back to tensor ----
            arr = np.array(pil_rgba, dtype=np.float32) / 255.0
            rgba_tensors.append(torch.from_numpy(arr))

            # extract alpha channel as mask
            mask_np = (np.array(pil_rgba.split()[3], dtype=np.float32) / 255.0)
            mask_tensors.append(torch.from_numpy(mask_np))

        # ------------------------------------------------------------------
        # stack and return in ComfyUI tensor format
        # ------------------------------------------------------------------
        stacked_image: torch.Tensor = torch.stack(rgba_tensors).to(dtype=torch.float32)
        stacked_mask: torch.Tensor = torch.stack(mask_tensors).to(dtype=torch.float32)

        # ComfyUI MASK convention: (N, H, W)
        if stacked_mask.ndim == 4 and stacked_mask.shape[-1] == 1:
            stacked_mask = stacked_mask.squeeze(-1)

        log.debug(
            "RemoveBackgroundAndResizeNode.done",
            batch=image.shape[0],
            output_size=output_size,
            scaling_method=scaling_method,
        )

        return (stacked_image, stacked_mask)


# ComfyUI registration dict
NODE_CLASS_MAPPINGS = {"RemoveBackgroundAndResize": RemoveBackgroundAndResizeNode}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoveBackgroundAndResize": "Remove Background & Resize"
}
