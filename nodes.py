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
                "binary_threshold": ("INT", {
                    "default": 128,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Threshold for binary alpha mask (0-255)"
                }),
                "output_size": (["ORIGINAL", "64x64", "96x96", "128x128", "256x256", "512x512", "768x768", "1024x1024", "1280x1280", "1536x1536", "1792x1792", "2048x2048"], {
                    "default": "ORIGINAL",
                    "tooltip": "Target output size (power-of-8 dimensions for optimal scaling)"
                }),
                "scaling_method": (["NEAREST"], {
                    "default": "NEAREST",
                    "tooltip": "Nearest neighbor interpolation for pixel-perfect scaling"
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
                "output_format": (["RGBA", "RGB_WITH_MASK"], {
                    "default": "RGBA"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "remove_background"
    CATEGORY = "image/processing"

    def __init__(self):
        self.processor = None
    
    def parse_output_size(self, size_string):
        """
        Parse output size string to width, height tuple.
        
        Args:
            size_string: String like "512x512" or "ORIGINAL"
            
        Returns:
            Tuple (width, height) or None for ORIGINAL
        """
        if size_string == "ORIGINAL":
            return None
        
        try:
            width, height = size_string.split('x')
            return (int(width), int(height))
        except ValueError:
            raise ValueError(f"Invalid output size format: {size_string}")
    
    def calculate_scaling_factor(self, current_size, target_size):
        """
        Calculate optimal integer scaling factor for NEAREST interpolation.
        
        Args:
            current_size: Tuple (width, height) of current image
            target_size: Tuple (width, height) of target size
            
        Returns:
            Scaling factor that achieves target size
        """
        current_w, current_h = current_size
        target_w, target_h = target_size
        
        # Calculate scale factors for width and height
        scale_w = target_w / current_w
        scale_h = target_h / current_h
        
        # Use the same scale for both dimensions to maintain aspect ratio
        # Choose the smaller scale to ensure we don't exceed target dimensions
        scale_factor = min(scale_w, scale_h)
        
        return scale_factor
    
    def intelligent_scale(self, image_pil, target_size):
        """
        Scale image to target dimensions using intelligent NEAREST scaling.
        
        Args:
            image_pil: PIL Image object
            target_size: Tuple (width, height) for target dimensions
            
        Returns:
            Scaled PIL Image
        """
        if target_size is None:
            return image_pil
            
        current_size = (image_pil.width, image_pil.height)
        target_w, target_h = target_size
        
        # If already at target size, return as-is
        if current_size == target_size:
            return image_pil
        
        # Calculate scaling factor
        scale_factor = self.calculate_scaling_factor(current_size, target_size)
        
        # Apply scaling
        new_width = int(image_pil.width * scale_factor)
        new_height = int(image_pil.height * scale_factor)
        
        # If calculated size matches target exactly, use target dimensions
        if abs(new_width - target_w) <= 1 and abs(new_height - target_h) <= 1:
            new_width, new_height = target_w, target_h
        
        return image_pil.resize(
            (new_width, new_height),
            Image.Resampling.NEAREST
        )

    def remove_background(self, image, tolerance=30, edge_sensitivity=0.8,
                         foreground_bias=0.7, color_clusters=8, binary_threshold=128,
                         edge_refinement=True, dither_handling=True, output_format="RGBA",
                         output_size="ORIGINAL", scaling_method="NEAREST"):
        """
        Main processing function for background removal with error handling.
        """
        try:
            # Validate input
            if image is None or image.shape[0] == 0:
                raise ValueError("No input image provided")

            # Check image dimensions
            if len(image.shape) != 4:
                raise ValueError(f"Expected 4D tensor, got {len(image.shape)}D")
            
            # Validate minimum input size (64x64 pixels)
            _, height, width, _ = image.shape
            if height < 64 or width < 64:
                raise ValueError(f"Input image must be at least 64x64 pixels, got {width}x{height}")

            # Process with error catching
            results, masks = self._process_images(
                image=image, 
                tolerance=tolerance,
                edge_sensitivity=edge_sensitivity,
                foreground_bias=foreground_bias,
                color_clusters=color_clusters,
                edge_refinement=edge_refinement,
                dither_handling=dither_handling,
                binary_threshold=binary_threshold,
                output_format=output_format,
                output_size=output_size,
                scaling_method=scaling_method
            )

            return (results, masks)

        except cv2.error as e:
            raise RuntimeError(f"OpenCV processing error: {str(e)}")
        except MemoryError:
            raise RuntimeError("Insufficient memory for processing. Try reducing batch size.")
        except Exception as e:
            raise RuntimeError(f"Background removal failed: {str(e)}")

    def _process_images(self, image, tolerance=30, edge_sensitivity=0.8,
                       foreground_bias=0.7, color_clusters=8, binary_threshold=128,
                       edge_refinement=True, dither_handling=True, output_format="RGBA",
                       output_size="ORIGINAL", scaling_method="NEAREST"):
        """
        Internal method for processing images without error handling wrapper.
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
                dither_handling=dither_handling,
                binary_threshold=binary_threshold
            )

            # Process image
            rgba_result = processor.remove_background_advanced(img_np)
            
            # Apply scaling if requested and scaling method is NEAREST
            if output_size != "ORIGINAL" and scaling_method == "NEAREST":
                # Parse target dimensions
                target_dimensions = self.parse_output_size(output_size)
                
                if target_dimensions is not None:
                    # Convert to PIL Image for scaling
                    rgba_pil = Image.fromarray(rgba_result, 'RGBA')
                    
                    # Apply intelligent nearest neighbor scaling
                    rgba_scaled = self.intelligent_scale(rgba_pil, target_dimensions)
                    
                    # Convert back to numpy
                    rgba_result = np.array(rgba_scaled)

            # Extract alpha channel as mask (invert: 0=transparent, 255=opaque)
            alpha_channel = rgba_result[:, :, 3]
            masks.append(alpha_channel)

            # Handle output format
            if output_format == "RGBA":
                results.append(rgba_result)
            else:  # RGB_WITH_MASK
                rgb_result = rgba_result[:, :, :3]
                results.append(rgb_result)

        # Convert back to ComfyUI tensor format
        if output_format == "RGBA":
            # For RGBA, maintain 4 channels
            result_tensor = torch.from_numpy(np.array(results)).float() / 255.0
        else:
            # For RGB_WITH_MASK, use 3 channels
            result_tensor = torch.from_numpy(np.array(results)).float() / 255.0
        
        mask_tensor = torch.from_numpy(np.array(masks)).float() / 255.0

        return result_tensor, mask_tensor

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