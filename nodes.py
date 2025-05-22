import os
import numpy as np
import torch
from PIL import Image
import cv2
from sklearn.cluster import KMeans
import colorsys
import folder_paths
import comfy.utils

class TransparencyBackgroundRemover:
    """
    ComfyUI node for automatic background removal with transparency generation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tolerance": ("INT", {
                    "default": 30,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Color similarity threshold for background detection (0-255)"
                }),
                "edge_sensitivity": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Edge detection sensitivity (0-1)"
                }),
                "foreground_bias": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Bias towards foreground preservation (0-1)"
                }),
                "color_clusters": ("INT", {
                    "default": 8,
                    "min": 2,
                    "max": 20,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of color clusters for background detection"
                }),
            },
            "optional": {
                "edge_refinement": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply edge refinement post-processing"
                }),
                "dither_handling": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable dithered pattern detection and handling"
                }),
                "output_format": (["RGBA", "RGB_WITH_MASK"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "remove_background"
    CATEGORY = "image/processing"

    def __init__(self):
        self.processor = None

    def remove_background(self, image, tolerance=30, edge_sensitivity=0.8,
                         foreground_bias=0.7, color_clusters=8,
                         edge_refinement=True, dither_handling=True,
                         output_format="RGBA"):
        """
        Main processing function for background removal.
        """
        # Convert from ComfyUI tensor format to numpy
        batch_size, height, width, channels = image.shape
        results = []
        masks = []

        for i in range(batch_size):
            # Convert single image to numpy array
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)

            # Initialize processor with parameters
            from .background_remover import EnhancedPixelArtProcessor
            processor = EnhancedPixelArtProcessor(
                tolerance=tolerance,
                edge_sensitivity=edge_sensitivity,
                color_clusters=color_clusters,
                foreground_bias=foreground_bias,
                edge_refinement=edge_refinement,
                dither_handling=dither_handling
            )

            # Process image
            rgba_result = processor.remove_background_advanced(img_np)

            # Extract alpha channel as mask
            alpha_channel = rgba_result[:, :, 3]
            masks.append(alpha_channel)

            # Handle output format
            if output_format == "RGBA":
                results.append(rgba_result)
            else:  # RGB_WITH_MASK
                rgb_result = rgba_result[:, :, :3]
                results.append(rgb_result)

        # Convert back to ComfyUI tensor format
        result_tensor = torch.from_numpy(np.array(results)).float() / 255.0
        mask_tensor = torch.from_numpy(np.array(masks)).float() / 255.0

        return (result_tensor, mask_tensor)

class TransparencyBackgroundRemoverBatch:
    """
    Batch processing version with additional options.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "tolerance": ("INT", {"default": 30, "min": 0, "max": 255}),
                "edge_sensitivity": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0}),
                "auto_adjust": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically adjust parameters based on image content"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("images", "masks", "report")
    FUNCTION = "batch_remove_background"
    CATEGORY = "image/processing"

# Node registration
NODE_CLASS_MAPPINGS = {
    "TransparencyBackgroundRemover": TransparencyBackgroundRemover,
    "TransparencyBackgroundRemoverBatch": TransparencyBackgroundRemoverBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TransparencyBackgroundRemover": "Transparency Background Remover",
    "TransparencyBackgroundRemoverBatch": "Transparency Background Remover (Batch)",
}