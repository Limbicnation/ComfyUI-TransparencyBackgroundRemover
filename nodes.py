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
                "scaling_method": (["NEAREST", "BILINEAR", "BICUBIC", "LANCZOS"], {
                    "default": "NEAREST",
                    "tooltip": "Interpolation method: NEAREST (pixel-perfect), BILINEAR (smooth), BICUBIC (high-quality), LANCZOS (best quality)"
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
                    "default": "RGBA",
                    "tooltip": "Output format: RGBA with alpha channel or RGB with separate mask"
                }),
                "auto_adjust": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Automatically adjust parameters based on image content analysis"
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
    
    def intelligent_scale(self, image_pil, target_size, scaling_method="NEAREST"):
        """
        Scale image to target dimensions using specified interpolation method.
        
        Args:
            image_pil: PIL Image object
            target_size: Tuple (width, height) for target dimensions
            scaling_method: Interpolation method ("NEAREST", "BILINEAR", "BICUBIC", "LANCZOS")
            
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
        
        # Select resampling method based on scaling_method
        resampling_map = {
            "NEAREST": Image.Resampling.NEAREST,
            "BILINEAR": Image.Resampling.BILINEAR, 
            "BICUBIC": Image.Resampling.BICUBIC,
            "LANCZOS": Image.Resampling.LANCZOS
        }
        
        resampling_method = resampling_map.get(scaling_method, Image.Resampling.NEAREST)
        
        return image_pil.resize(
            (new_width, new_height),
            resampling_method
        )

    def remove_background(self, image, tolerance=30, edge_sensitivity=0.8,
                         foreground_bias=0.7, color_clusters=8, binary_threshold=128,
                         edge_refinement=True, dither_handling=True, output_format="RGBA",
                         output_size="ORIGINAL", scaling_method="NEAREST", auto_adjust=False):
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
                scaling_method=scaling_method,
                auto_adjust=auto_adjust
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
                       output_size="ORIGINAL", scaling_method="NEAREST", auto_adjust=False):
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

            # Auto-adjust parameters if enabled
            if auto_adjust:
                adjustments = processor.auto_adjust_parameters(img_np)
                if adjustments:
                    # Apply adjustments
                    if 'tolerance' in adjustments:
                        processor.tolerance = adjustments['tolerance']
                    if 'edge_sensitivity' in adjustments:
                        processor.edge_sensitivity = adjustments['edge_sensitivity']
                    if 'foreground_bias' in adjustments:
                        processor.foreground_bias = adjustments['foreground_bias']

            # Process image
            rgba_result = processor.remove_background_advanced(img_np)
            
            # Apply scaling if requested
            if output_size != "ORIGINAL":
                # Parse target dimensions
                target_dimensions = self.parse_output_size(output_size)
                
                if target_dimensions is not None:
                    # Convert to PIL Image for scaling
                    rgba_pil = Image.fromarray(rgba_result, 'RGBA')
                    
                    # Apply intelligent scaling with specified method
                    rgba_scaled = self.intelligent_scale(rgba_pil, target_dimensions, scaling_method)
                    
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
    Batch processing version with additional options and auto-adjustment.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "tolerance": ("INT", {
                    "default": 30,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Base color similarity threshold for background detection (0-255)"
                }),
                "edge_sensitivity": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Base edge detection sensitivity (0-1)"
                }),
                "auto_adjust": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically adjust parameters based on image content"
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
                "binary_threshold": ("INT", {
                    "default": 128,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Threshold for binary alpha mask (0-255)"
                }),
                "progress_reporting": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Generate detailed processing report"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("images", "masks", "report")
    FUNCTION = "batch_remove_background"
    CATEGORY = "image/processing"

    def batch_remove_background(self, images, tolerance=30, edge_sensitivity=0.8, 
                               auto_adjust=True, foreground_bias=0.7, color_clusters=8,
                               edge_refinement=True, dither_handling=True, binary_threshold=128,
                               progress_reporting=True):
        """
        Batch process multiple images with optional auto-adjustment.
        """
        try:
            if images is None or images.shape[0] == 0:
                raise ValueError("No input images provided")

            batch_size, height, width, channels = images.shape
            
            # Validate input dimensions
            if len(images.shape) != 4:
                raise ValueError(f"Expected 4D tensor (batch, height, width, channels), got {len(images.shape)}D")
            
            if channels not in [3, 4]:
                raise ValueError(f"Expected 3 or 4 channels (RGB or RGBA), got {channels}")
            
            # Validate minimum input size for all images
            if height < 64 or width < 64:
                raise ValueError(f"All input images must be at least 64x64 pixels, got {width}x{height}")

            results = []
            masks = []
            reports = []
            processing_stats = {
                'total_images': batch_size,
                'successful': 0,
                'failed': 0,
                'auto_adjustments': 0,
                'avg_processing_time': 0.0,
                'adjustments_made': []
            }

            import time
            total_processing_time = 0.0

            for i in range(batch_size):
                start_time = time.time()
                
                try:
                    # Convert single image to numpy array
                    img_np = (images[i].cpu().numpy() * 255).astype(np.uint8)

                    # Initialize processor with base parameters
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

                    # Auto-adjust parameters if enabled
                    adjustments = {}
                    if auto_adjust:
                        adjustments = processor.auto_adjust_parameters(img_np)
                        if adjustments:
                            processing_stats['auto_adjustments'] += 1
                            processing_stats['adjustments_made'].append({
                                'image_index': i,
                                'adjustments': adjustments
                            })
                            
                            # Apply adjustments
                            if 'tolerance' in adjustments:
                                processor.tolerance = adjustments['tolerance']
                            if 'edge_sensitivity' in adjustments:
                                processor.edge_sensitivity = adjustments['edge_sensitivity']
                            if 'foreground_bias' in adjustments:
                                processor.foreground_bias = adjustments['foreground_bias']

                    # Process image
                    rgba_result = processor.remove_background_advanced(img_np)

                    # Extract alpha channel as mask
                    alpha_channel = rgba_result[:, :, 3]
                    masks.append(alpha_channel)
                    results.append(rgba_result)

                    processing_stats['successful'] += 1
                    
                    processing_time = time.time() - start_time
                    total_processing_time += processing_time

                    if progress_reporting:
                        report = f"Image {i+1}: Processed successfully"
                        if adjustments:
                            report += f" (auto-adjusted: {list(adjustments.keys())})"
                        report += f" in {processing_time:.3f}s"
                        reports.append(report)

                except Exception as e:
                    processing_stats['failed'] += 1
                    error_msg = f"Image {i+1}: Failed - {str(e)}"
                    reports.append(error_msg)
                    
                    # Create empty result for failed image
                    empty_result = np.zeros_like(img_np if 'img_np' in locals() else (height, width, 4), dtype=np.uint8)
                    if len(empty_result.shape) == 3 and empty_result.shape[2] == 3:
                        empty_rgba = np.zeros((empty_result.shape[0], empty_result.shape[1], 4), dtype=np.uint8)
                        empty_rgba[:, :, :3] = empty_result
                        empty_result = empty_rgba
                    results.append(empty_result)
                    masks.append(np.zeros((height, width), dtype=np.uint8))

            # Calculate statistics
            processing_stats['avg_processing_time'] = total_processing_time / batch_size if batch_size > 0 else 0.0

            # Convert results to tensors
            result_tensor = torch.from_numpy(np.array(results)).float() / 255.0
            mask_tensor = torch.from_numpy(np.array(masks)).float() / 255.0

            # Generate summary report
            summary_report = self._generate_summary_report(processing_stats, reports if progress_reporting else [])

            return (result_tensor, mask_tensor, summary_report)

        except Exception as e:
            error_report = f"Batch processing failed: {str(e)}"
            # Return empty tensors in case of complete failure
            empty_images = torch.zeros_like(images)
            empty_masks = torch.zeros((images.shape[0], images.shape[1], images.shape[2]))
            return (empty_images, empty_masks, error_report)

    def _generate_summary_report(self, stats, detailed_reports):
        """Generate a summary report of batch processing results."""
        lines = [
            "=== Batch Processing Report ===",
            f"Total Images: {stats['total_images']}",
            f"Successful: {stats['successful']}",
            f"Failed: {stats['failed']}",
            f"Success Rate: {(stats['successful']/stats['total_images']*100):.1f}%" if stats['total_images'] > 0 else "N/A",
            f"Auto-adjustments Applied: {stats['auto_adjustments']}",
            f"Average Processing Time: {stats['avg_processing_time']:.3f}s per image",
            ""
        ]

        if stats['adjustments_made']:
            lines.append("Auto-adjustment Details:")
            for adj in stats['adjustments_made']:
                lines.append(f"  Image {adj['image_index']+1}: {adj['adjustments']}")
            lines.append("")

        if detailed_reports:
            lines.append("Detailed Processing Log:")
            lines.extend(detailed_reports)

        return "\n".join(lines)

# Node registration
NODE_CLASS_MAPPINGS = {
    "TransparencyBackgroundRemover": TransparencyBackgroundRemover,
    "TransparencyBackgroundRemoverBatch": TransparencyBackgroundRemoverBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TransparencyBackgroundRemover": "Transparency Background Remover",
    "TransparencyBackgroundRemoverBatch": "Transparency Background Remover (Batch)",
}