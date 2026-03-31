from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
import structlog
import torch
from PIL import Image

log = structlog.get_logger(__name__)

# Try to import ComfyUI modules
try:
    import folder_paths
    import comfy.utils
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False
    log.info("comfyui_modules_unavailable", mode="standalone")

# Import our GrabCut processor
try:
    from .grabcut_remover import GrabCutProcessor, create_fallback_processor, _log_gpu_memory
except ImportError:
    from grabcut_remover import GrabCutProcessor, create_fallback_processor, _log_gpu_memory

# Pydantic validation — security-hardened parameter sanitisation before GPU execution
import os.path as _path
import sys as _sys
_parent = _path.dirname(__file__)
if _parent not in _sys.path:
    _sys.path.insert(0, _parent)
try:
    from src.validation import validate_node_params, GrabCutParams, MaskParams
except ImportError:
    # Graceful degradation — warn and continue without validation
    log.warning("grabcut_node.validation_unavailable",
                hint="pydantic not installed — parameter validation disabled")
    validate_node_params = GrabCutParams = MaskParams = None


class ScalingMixin:
    """
    Mixin class providing shared scaling functionality for GrabCut nodes.
    Eliminates code duplication and improves performance.
    """
    
    # Class-level constants
    
    # Class-level resampling map to avoid recreation on every call
    _RESAMPLING_MAP = {
        "NEAREST": Image.Resampling.NEAREST,
        "BILINEAR": Image.Resampling.BILINEAR,
        "BICUBIC": Image.Resampling.BICUBIC,
        "LANCZOS": Image.Resampling.LANCZOS
    }
    
    @staticmethod
    def parse_output_size(size_string: str) -> Optional[Tuple[int, int]]:
        """
        Parse output size string to width, height tuple.
        
        Args:
            size_string: String like "512x512" or "ORIGINAL" or "custom"
            
        Returns:
            Tuple (width, height) or None for ORIGINAL
        """
        if size_string == "ORIGINAL":
            return None
        elif size_string == "custom":
            return None  # Will be handled with custom_width/custom_height
        
        try:
            width, height = size_string.split('x')
            return (int(width), int(height))
        except ValueError:
            raise ValueError(f"Invalid output size format: {size_string}")
    
    @staticmethod
    def calculate_scaling_factor(current_size: Tuple[int, int], target_size: Tuple[int, int]) -> float:
        """
        Calculate optimal scaling factor while preserving aspect ratio.
        
        Uses the smaller of width/height scale factors to ensure the scaled image
        fits within target dimensions without exceeding them. This means the final
        image may be smaller than target_size in one dimension to maintain the
        original aspect ratio.
        
        Args:
            current_size: Tuple (width, height) of current image
            target_size: Tuple (width, height) of target size
            
        Returns:
            Scaling factor that preserves aspect ratio and fits within target_size
            
        Example:
            current_size=(100, 200), target_size=(150, 150)
            -> scale_w=1.5, scale_h=0.75, returns min(1.5, 0.75) = 0.75
            -> final size would be (75, 150) to preserve aspect ratio
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
    
    def intelligent_scale(self, image_pil: Image.Image, target_size: Tuple[int, int], scaling_method: str = "NEAREST") -> Image.Image:
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
        
        # If already at target size, return as-is
        if current_size == target_size:
            return image_pil
        
        # Calculate scaling factor
        scale_factor = self.calculate_scaling_factor(current_size, target_size)
        
        # Apply scaling
        new_width = int(image_pil.width * scale_factor)
        new_height = int(image_pil.height * scale_factor)
        
        # Note: Removed dimension snapping to preserve perfect aspect ratio
        # The scaled image will fit within target dimensions, potentially smaller on one axis
        
        # Use class-level resampling map for better performance
        resampling_method = self._RESAMPLING_MAP.get(scaling_method.upper(), Image.Resampling.NEAREST)
        
        return image_pil.resize(
            (new_width, new_height),
            resampling_method
        )
    
    def _apply_resize(self, image_np: np.ndarray, output_size: str, scaling_method: str, 
                     custom_width: int, custom_height: int) -> np.ndarray:
        """
        Apply resize operation to image array.
        
        Args:
            image_np: Image as numpy array (RGBA format)
            output_size: Size specification string
            scaling_method: Scaling method to use
            custom_width: Custom width for 'custom' size option
            custom_height: Custom height for 'custom' size option
            
        Returns:
            Resized image as numpy array
        """
        if output_size == "ORIGINAL":
            return image_np
        
        # Parse target dimensions
        if output_size == "custom":
            target_dimensions = (custom_width, custom_height)
        else:
            target_dimensions = self.parse_output_size(output_size)
        
        if target_dimensions is None:
            return image_np
        
        # Convert to PIL Image for scaling
        image_pil = Image.fromarray(image_np, 'RGBA')
        
        # Apply intelligent scaling with specified method
        scaled_pil = self.intelligent_scale(image_pil, target_dimensions, scaling_method)
        
        # Convert back to numpy
        return np.array(scaled_pil)


class AutoGrabCutRemover(ScalingMixin):
    """
    ComfyUI node for automated GrabCut background removal with object detection.
    Refines existing background removal or processes raw images.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "object_class": (["auto", "person", "product", "vehicle", "animal", "furniture", "electronics"],),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.3,
                    "max": 0.9,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Minimum confidence for object detection"
                }),
                "grabcut_iterations": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of GrabCut algorithm iterations"
                }),
                "margin_pixels": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 50,
                    "step": 5,
                    "tooltip": "Pixel margin around detected object"
                }),
                "edge_refinement": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Edge refinement strength (0=none, 1=maximum)"
                }),
                "edge_blur_amount": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Amount of Gaussian blur to apply to mask edges (0=none, 10=maximum)"
                }),
                "bbox_safety_margin": ("INT", {
                    "default": 30,
                    "min": 0,
                    "max": 100,
                    "step": 5,
                    "tooltip": "Extra pixels beyond detected bounding box for safety"
                }),
                "min_bbox_size": ("INT", {
                    "default": 64,
                    "min": 32,
                    "max": 256,
                    "step": 16,
                    "tooltip": "Minimum bounding box dimensions to prevent over-cropping"
                }),
                "fallback_margin_percent": ("FLOAT", {
                    "default": 0.20,
                    "min": 0.10,
                    "max": 0.50,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Margin percentage for fallback bbox when no object detected"
                }),
                "binary_threshold": ("INT", {
                    "default": 200,
                    "min": 128,
                    "max": 250,
                    "step": 10,
                    "tooltip": "Threshold for binary mask conversion"
                }),
                "output_size": (["ORIGINAL", "512x512", "1024x1024", "2048x2048", "custom"], {
                    "default": "ORIGINAL",
                    "tooltip": "Target output size for the processed image and mask"
                }),
                "scaling_method": (["NEAREST", "BILINEAR", "BICUBIC", "LANCZOS"], {
                    "default": "NEAREST",
                    "tooltip": "Interpolation method for scaling: NEAREST (pixel-perfect), BILINEAR (smooth), BICUBIC (high-quality), LANCZOS (best quality)"
                }),
                "auto_adjust": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Automatically adjust parameters based on image content analysis"
                }),
            },
            "optional": {
                "initial_mask": ("MASK",),
                "output_format": (["RGBA", "MASK"], {
                    "default": "RGBA",
                    "tooltip": "Output format: RGBA with alpha channel or binary MASK (0=background, 255=foreground)"
                }),
                "invert_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert the output mask (foreground becomes background and vice versa)"
                }),
                "custom_width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Custom width (used when output_size is 'custom')"
                }),
                "custom_height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Custom height (used when output_size is 'custom')"
                }),
                "edge_detection_mode": (["AUTO", "PIXEL_ART", "PHOTOGRAPHIC"], {
                    "default": "AUTO",
                    "tooltip": "Edge detection optimization: AUTO (detect content type), PIXEL_ART (sharp edges), PHOTOGRAPHIC (smooth edges)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "FLOAT", "STRING", "BBOX_TENSOR")
    RETURN_NAMES = ("image", "mask", "bbox_coords", "confidence", "metrics", "bbox_tensor")
    FUNCTION = "remove_background"
    CATEGORY = "image/processing"
    
    def __init__(self):
        """Initialize the node with GrabCut processor."""
        self.processor = None
        self._initialize_processor()
    
    def _initialize_processor(self):
        """Initialize the GrabCut processor, with fallback if YOLO unavailable."""
        try:
            self.processor = GrabCutProcessor()
        except Exception as e:
            log.warning("grabcut_node.yolo_init_failed", error=str(e),
                        fallback="FallbackGrabCutProcessor")
            self.processor = create_fallback_processor()()
    
    def _map_object_class(self, object_class: str) -> Optional[str]:
        """Map UI object class to YOLO class names."""
        mapping = {
            "auto": None,
            "person": "person",
            "product": "bottle",  # Generic product detection
            "vehicle": "car",
            "animal": "dog",  # Generic animal detection
            "furniture": "chair",
            "electronics": "laptop"
        }
        return mapping.get(object_class, None)
    
    @torch.no_grad()
    def remove_background(self, image: torch.Tensor,
                         initial_mask: Optional[torch.Tensor] = None,
                         object_class: str = "auto",
                         confidence_threshold: float = 0.5,
                         grabcut_iterations: int = 5,
                         margin_pixels: int = 20,
                         edge_refinement: float = 0.7,
                         edge_blur_amount: float = 0.0,
                         bbox_safety_margin: int = 30,
                         min_bbox_size: int = 64,
                         fallback_margin_percent: float = 0.20,
                         binary_threshold: int = 200,
                         output_size: str = "ORIGINAL",
                         scaling_method: str = "NEAREST",
                         auto_adjust: bool = False,
                         output_format: str = "RGBA",
                         invert_mask: bool = False,
                         custom_width: int = 512,
                         custom_height: int = 512,
                         edge_detection_mode: str = "AUTO") -> Tuple:
        """
        Process image with automated GrabCut background removal.
        
        Args:
            image: Input image tensor from ComfyUI
            object_class: Target object class for detection
            confidence_threshold: Minimum detection confidence
            grabcut_iterations: Number of GrabCut iterations
            margin_pixels: Margin around detected object
            edge_refinement: Edge refinement strength
            binary_threshold: Binary mask threshold
            edge_detection_mode: Edge detection optimization mode
            initial_mask: Optional initial mask from previous processing
            
        Returns:
            Tuple of (processed_image, mask, bbox_string, confidence, metrics)
        """
        # --- Pydantic validation: sanitise ALL user params before GPU execution ---
        if validate_node_params is not None:
            try:
                validated = validate_node_params(
                    iterations=grabcut_iterations,
                    margin=margin_pixels,
                    confidence_threshold=confidence_threshold,
                    scaling_method=scaling_method,
                    edge_blur_amount=edge_blur_amount,
                    invert_mask=invert_mask,
                    edge_refinement_strength=edge_refinement,
                    bbox_safety_margin=bbox_safety_margin,
                    min_bbox_size=min_bbox_size,
                    fallback_margin_percent=fallback_margin_percent,
                    binary_threshold=binary_threshold,
                    output_format=output_format,
                    auto_adjust=auto_adjust,
                )
                # Use normalised values from Pydantic (e.g. scaling_method lowercased)
                grabcut_iterations = validated.iterations
                margin_pixels = validated.margin
                confidence_threshold = validated.confidence_threshold
                scaling_method = validated.scaling_method
                edge_blur_amount = validated.edge_blur_amount
                invert_mask = validated.invert_mask
                edge_refinement = validated.edge_refinement_strength
                bbox_safety_margin = validated.bbox_safety_margin
                min_bbox_size = validated.min_bbox_size
                fallback_margin_percent = validated.fallback_margin_percent
                binary_threshold = validated.binary_threshold
                output_format = validated.output_format
                auto_adjust = validated.auto_adjust
            except Exception as exc:
                log.error("grabcut_node.validation_failed", node="AutoGrabCutRemover", error=str(exc))
                raise ValueError(f"[AutoGrabCutRemover] Invalid parameters: {exc}") from exc

        log.info("grabcut_node.remove_background.start",
                 batch_size=image.shape[0] if len(image.shape) == 4 else 1,
                 output_format=output_format)
        if torch.cuda.is_available():
            log.debug("gpu_memory.remove_background.start",
                      allocated_gb=round(torch.cuda.memory_allocated() / 1e9, 3))

        # Ensure processor is initialized
        if self.processor is None:
            self._initialize_processor()

        # Update processor parameters
        self.processor.confidence_threshold = confidence_threshold
        self.processor.iterations = grabcut_iterations
        self.processor.margin_pixels = margin_pixels
        self.processor.edge_refinement_strength = edge_refinement
        self.processor.edge_blur_amount = edge_blur_amount
        self.processor.bbox_safety_margin = bbox_safety_margin
        self.processor.min_bbox_size = min_bbox_size
        self.processor.fallback_margin_percent = fallback_margin_percent
        self.processor.binary_threshold = binary_threshold
        
        # Apply edge detection mode optimizations
        if edge_detection_mode != "AUTO":
            if edge_detection_mode == "PIXEL_ART":
                # Optimize for pixel art: precise edges, minimal smoothing
                self.processor.edge_refinement_strength = min(0.4, edge_refinement)
                self.processor.binary_threshold = max(220, binary_threshold)
                self.processor.margin_pixels = max(5, min(15, margin_pixels))
            elif edge_detection_mode == "PHOTOGRAPHIC":
                # Optimize for photos: smooth edges, more refinement
                self.processor.edge_refinement_strength = min(0.9, edge_refinement + 0.2)
                self.processor.binary_threshold = max(180, binary_threshold - 20)
                self.processor.margin_pixels = max(15, min(35, margin_pixels + 10))
        else:
            # AUTO mode: detect content type for first image
            first_image = image[0] if len(image.shape) == 4 else image
            img_np = (first_image.cpu().numpy() * 255).astype(np.uint8)
            
            # Ensure channels last format (H, W, C)
            if img_np.shape[0] == 3 or img_np.shape[0] == 4:
                img_np = np.transpose(img_np, (1, 2, 0))
            
            # Convert to RGB if needed
            if img_np.shape[2] == 4:
                img_np = img_np[:, :, :3]
            
            # Detect if pixel art
            is_pixel_art = self.processor._detect_pixel_art_characteristics(img_np)
            
            if is_pixel_art:
                # Apply pixel art optimizations
                self.processor.edge_refinement_strength = min(0.4, edge_refinement)
                self.processor.binary_threshold = max(220, binary_threshold)
                self.processor.margin_pixels = max(5, min(15, margin_pixels))
            else:
                # Apply photographic optimizations
                self.processor.edge_refinement_strength = min(0.9, edge_refinement + 0.1)
                self.processor.binary_threshold = max(180, binary_threshold - 10)
                self.processor.margin_pixels = max(10, min(30, margin_pixels + 5))
        
        # Auto-adjust parameters if enabled (applied to first image for batch processing)
        if auto_adjust:
            # Get first image for analysis
            first_image = image[0] if len(image.shape) == 4 else image
            
            # Convert to numpy for analysis
            img_np = (first_image.cpu().numpy() * 255).astype(np.uint8)
            
            # Ensure channels last format (H, W, C)
            if img_np.shape[0] == 3 or img_np.shape[0] == 4:
                img_np = np.transpose(img_np, (1, 2, 0))
            
            # Convert to RGB if needed
            if img_np.shape[2] == 4:
                img_np = img_np[:, :, :3]
            
            # Get parameter adjustments
            adjustments = self.processor.auto_adjust_parameters(img_np)
            
            # Apply adjustments to processor
            if adjustments:
                for param, value in adjustments.items():
                    if hasattr(self.processor, param):
                        setattr(self.processor, param, value)
        
        # Handle batch dimension
        if len(image.shape) == 4:
            batch_size = image.shape[0]
        else:
            batch_size = 1
            image = image.unsqueeze(0)

        # Ensure initial_mask has batch dimension regardless of image path
        if initial_mask is not None and len(initial_mask.shape) == 2:
            initial_mask = initial_mask.unsqueeze(0)

        processed_images = []
        masks = []
        all_bboxes = []
        all_confidences = []
        all_metrics = []
        bbox_metadata = torch.zeros((batch_size, 6), dtype=torch.float32)
        
        for i in range(batch_size):
            _log_gpu_memory(f"batch_item_{i}.start")
            try:
                # Convert from ComfyUI tensor format to numpy
                img_tensor = image[i]

                # Validate input tensor — append fallback on failure to keep
                # batch alignment with bbox_metadata and avoid empty torch.stack
                if len(img_tensor.shape) != 3:
                    log.warning("grabcut_node.unexpected_shape", shape=list(img_tensor.shape))
                    # image[i] from a 4D tensor is always 3D — direct index is safe
                    h = img_tensor.shape[-3]
                    w = img_tensor.shape[-2]
                    processed_images.append(torch.zeros((h, w, 4 if output_format == "RGBA" else 1), dtype=torch.float32))
                    masks.append(torch.zeros((h, w), dtype=torch.float32))
                    all_bboxes.append("(0,0,0,0)")
                    all_confidences.append(0.0)
                    all_metrics.append(f"Batch {i+1}/{batch_size}: Invalid tensor shape")
                    continue

                img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)

                # Ensure channels last format (H, W, C)
                if img_np.shape[0] == 3 or img_np.shape[0] == 4:
                    img_np = np.transpose(img_np, (1, 2, 0))

                # Final validation
                if len(img_np.shape) != 3 or img_np.shape[2] not in [3, 4]:
                    log.warning("grabcut_node.invalid_shape", shape=list(img_np.shape))
                    h = img_np.shape[0]
                    w = img_np.shape[1]
                    processed_images.append(torch.zeros((h, w, 4 if output_format == "RGBA" else 1), dtype=torch.float32))
                    masks.append(torch.zeros((h, w), dtype=torch.float32))
                    all_bboxes.append("(0,0,0,0)")
                    all_confidences.append(0.0)
                    all_metrics.append(f"Batch {i+1}/{batch_size}: Invalid image shape")
                    continue
                    
                # Convert to RGB if needed
                if img_np.shape[2] == 4:
                    img_np = img_np[:, :, :3]
                
                # Map object class
                target_class = self._map_object_class(object_class)
                
                # Process with GrabCut
                if initial_mask is not None and i < initial_mask.shape[0]:
                    # Use initial mask if provided
                    mask_np = (initial_mask[i].cpu().numpy() * 255).astype(np.uint8)
                    if len(mask_np.shape) == 3:
                        mask_np = mask_np.squeeze()
                    result = self.processor.process_with_initial_mask(img_np, mask_np, target_class)
                else:
                    # Process without initial mask
                    result = self.processor.process_with_grabcut(img_np, target_class)
                
                if result['success']:
                    # Convert RGBA result back to tensor format
                    rgba = result['rgba_image']
                    
                    # Apply resize if requested
                    rgba = self._apply_resize(rgba, output_size, scaling_method, custom_width, custom_height)
                    
                    # Separate RGB and alpha
                    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
                    
                    # Apply mask inversion if requested (before any tensor creation)
                    if invert_mask:
                        alpha = 1.0 - alpha
                        rgba[:, :, 3] = (alpha * 255).astype(np.uint8)
                    
                    if output_format == "RGBA":
                        # For RGBA output: preserve transparency, don't premultiply alpha
                        # Create 4-channel RGBA tensor
                        rgba_tensor = torch.from_numpy(rgba.astype(np.float32) / 255.0).to(dtype=torch.float32)
                        processed_images.append(rgba_tensor)
                    else:  # output_format == "MASK"
                        # For MASK output: return binary mask as primary output
                        # Convert alpha to binary mask (0 or 255)
                        binary_mask = (alpha > 0.5).astype(np.float32)
                        mask_tensor = torch.from_numpy(binary_mask).to(dtype=torch.float32)
                        processed_images.append(mask_tensor.unsqueeze(-1))  # Add channel dimension
                    
                    # Alpha tensor for mask output (always provided)
                    alpha_tensor = torch.from_numpy(alpha).to(dtype=torch.float32)
                    masks.append(alpha_tensor)
                    
                    # Format bbox and metrics
                    if result['bbox']:
                        x1, y1, x2, y2 = result['bbox']
                        bbox_str = f"({x1},{y1},{x2},{y2})"
                        all_bboxes.append(bbox_str)
                    else:
                        all_bboxes.append("(0,0,0,0)")
                    
                    all_confidences.append(result['confidence'])
                    
                    # Populate batch metadata tensor [B, 6] = (x1, y1, w, h, confidence, detected)
                    if result['bbox']:
                        bx1, by1, bx2, by2 = result['bbox']
                        bw = bx2 - bx1
                        bh = by2 - by1
                        bbox_metadata[i] = torch.tensor(
                            [bx1, by1, bw, bh, result['confidence'], 1.0],
                            dtype=torch.float32
                        )
                    
                    metrics = (f"Batch {i+1}/{batch_size}: "
                              f"Time={result['processing_time_ms']}ms, "
                              f"Conf={result['confidence']:.2f}, "
                              f"Class={object_class}")
                    all_metrics.append(metrics)
                else:
                    # Fallback: handle based on output format
                    if output_format == "RGBA":
                        # Return original RGB with full alpha channel
                        h, w = img_np.shape[:2]
                        rgba_fallback = np.zeros((h, w, 4), dtype=np.float32)
                        rgba_fallback[:, :, :3] = img_np.astype(np.float32) / 255.0
                        rgba_fallback[:, :, 3] = 1.0  # Full opacity
                        rgb_tensor = torch.from_numpy(rgba_fallback).to(dtype=torch.float32)
                    else:  # output_format == "MASK"
                        # Return full foreground mask
                        alpha_fallback = np.ones((img_np.shape[0], img_np.shape[1], 1), dtype=np.float32)
                        rgb_tensor = torch.from_numpy(alpha_fallback).to(dtype=torch.float32)
                    
                    alpha_tensor = torch.ones((img_np.shape[0], img_np.shape[1]), dtype=torch.float32)
                    
                    processed_images.append(rgb_tensor)
                    masks.append(alpha_tensor)
                    all_bboxes.append("(0,0,0,0)")
                    all_confidences.append(0.0)
                    all_metrics.append(f"Batch {i+1}/{batch_size}: Processing failed")
                    
            except Exception as e:
                log.error("grabcut_node.batch_error", item=i, error=str(e))
                # Derive h,w from original input tensor to avoid shape mismatch
                img_shape = image[i].shape
                h = img_shape[-3] if len(img_shape) >= 3 else 512
                w = img_shape[-2] if len(img_shape) >= 2 else 512

                if output_format == "RGBA":
                    rgb_tensor = torch.zeros((h, w, 4), dtype=torch.float32)
                else:
                    rgb_tensor = torch.zeros((h, w, 1), dtype=torch.float32)

                alpha_tensor = torch.zeros((h, w), dtype=torch.float32)

                processed_images.append(rgb_tensor)
                masks.append(alpha_tensor)
                all_bboxes.append("(0,0,0,0)")
                all_confidences.append(0.0)
                all_metrics.append(f"Batch {i+1}/{batch_size}: Error occurred")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _log_gpu_memory(f"batch_item_{i}.end")

        output_image = torch.stack(processed_images)
        output_mask = torch.stack(masks)

        bbox_output = " | ".join(all_bboxes)
        confidence_output = float(np.mean(all_confidences))
        metrics_output = "\n".join(all_metrics)

        _log_gpu_memory("remove_background.end")
        log.info("grabcut_node.remove_background.done",
                 batch_size=batch_size, mean_confidence=round(confidence_output, 3))

        return (output_image, output_mask, bbox_output, confidence_output, metrics_output, bbox_metadata)


class GrabCutRefinement(ScalingMixin):
    """
    ComfyUI node for refining existing masks using GrabCut.
    Takes an image and mask, refines the mask with GrabCut.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "grabcut_iterations": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of GrabCut refinement iterations"
                }),
                "edge_refinement": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Edge smoothing strength"
                }),
                "edge_blur_amount": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Amount of Gaussian blur to apply to mask edges (0=none, 10=maximum)"
                }),
                "expand_margin": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 50,
                    "step": 5,
                    "tooltip": "Pixels to expand mask boundary for refinement"
                }),
                "bbox_safety_margin": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 100,
                    "step": 5,
                    "tooltip": "Extra pixels beyond detected bounding box for safety"
                }),
                "min_bbox_size": ("INT", {
                    "default": 64,
                    "min": 32,
                    "max": 256,
                    "step": 16,
                    "tooltip": "Minimum bounding box dimensions to prevent over-cropping"
                }),
                "output_size": (["ORIGINAL", "512x512", "1024x1024", "2048x2048", "custom"], {
                    "default": "ORIGINAL",
                    "tooltip": "Target output size for the refined image and mask"
                }),
                "scaling_method": (["NEAREST", "BILINEAR", "BICUBIC", "LANCZOS"], {
                    "default": "NEAREST",
                    "tooltip": "Interpolation method for scaling"
                }),
            },
            "optional": {
                "custom_width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Custom width (used when output_size is 'custom')"
                }),
                "custom_height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Custom height (used when output_size is 'custom')"
                }),
                "invert_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert the output mask (foreground becomes background and vice versa)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "refined_mask")
    FUNCTION = "refine_mask"
    CATEGORY = "image/processing"
    
    def __init__(self):
        """Initialize the refinement node."""
        self.processor = None
        self._initialize_processor()
    
    def _initialize_processor(self):
        """Initialize processor for refinement."""
        try:
            self.processor = GrabCutProcessor(iterations=3)
        except Exception:
            self.processor = create_fallback_processor()(iterations=3)
    
    @torch.no_grad()
    def refine_mask(self, image: torch.Tensor, mask: torch.Tensor,
                   grabcut_iterations: int = 3,
                   edge_refinement: float = 0.5,
                   edge_blur_amount: float = 0.0,
                   expand_margin: int = 10,
                   bbox_safety_margin: int = 20,
                   min_bbox_size: int = 64,
                   output_size: str = "ORIGINAL",
                   scaling_method: str = "NEAREST",
                   invert_mask: bool = False,
                   custom_width: int = 512,
                   custom_height: int = 512) -> Tuple:
        """
        Refine existing mask using GrabCut.
        
        Args:
            image: Input image tensor
            mask: Initial mask to refine
            grabcut_iterations: Number of iterations
            edge_refinement: Edge smoothing strength
            expand_margin: Margin expansion pixels
            
        Returns:
            Tuple of (image_with_refined_alpha, refined_mask)
        """
        # --- Pydantic validation: sanitise params before GPU execution ---
        if GrabCutParams is not None and MaskParams is not None:
            try:
                validated_gc = GrabCutParams(
                    iterations=grabcut_iterations,
                    margin=expand_margin,
                )
                validated_mask = MaskParams(
                    edge_blur_amount=edge_blur_amount,
                    invert_mask=invert_mask,
                    edge_refinement_strength=edge_refinement,
                )
                # Use validated/clamped values
                grabcut_iterations = validated_gc.iterations
                expand_margin = validated_gc.margin
                edge_blur_amount = validated_mask.edge_blur_amount
                invert_mask = validated_mask.invert_mask
                edge_refinement = validated_mask.edge_refinement_strength
            except Exception as exc:
                log.error("grabcut_node.validation_failed",
                          node="GrabCutRefinement", error=str(exc))
                raise ValueError(f"[GrabCutRefinement] Invalid parameters: {exc}") from exc

        log.info("grabcut_node.refine_mask.start",
                 batch_size=image.shape[0] if len(image.shape) == 4 else 1)
        _log_gpu_memory("refine_mask.start")

        if self.processor is None:
            self._initialize_processor()

        # Update parameters
        self.processor.iterations = grabcut_iterations
        self.processor.edge_refinement_strength = edge_refinement
        self.processor.edge_blur_amount = edge_blur_amount
        self.processor.margin_pixels = expand_margin
        self.processor.bbox_safety_margin = bbox_safety_margin
        self.processor.min_bbox_size = min_bbox_size
        self.processor.fallback_margin_percent = 0.15

        # Handle batch
        if len(image.shape) == 4:
            batch_size = image.shape[0]
        else:
            batch_size = 1
            image = image.unsqueeze(0)

        # Ensure mask has batch dimension regardless of image path
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        refined_images = []
        refined_masks = []

        for i in range(batch_size):
            _log_gpu_memory(f"refine_batch_{i}.start")
            try:
                img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
                if img_np.shape[0] == 3:
                    img_np = np.transpose(img_np, (1, 2, 0))
                mask_np = (mask[i].cpu().numpy() * 255).astype(np.uint8)
                if len(mask_np.shape) == 3:
                    mask_np = mask_np.squeeze()
                result = self.processor.process_with_initial_mask(img_np, mask_np, None)
                if result['success']:
                    rgba = result['rgba_image']
                    rgba = self._apply_resize(rgba, output_size, scaling_method, custom_width, custom_height)
                    rgb = rgba[:, :, :3].astype(np.float32) / 255.0
                    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
                    if invert_mask:
                        alpha = 1.0 - alpha
                    rgb_tensor = torch.from_numpy(rgb).to(dtype=torch.float32)
                    alpha_tensor = torch.from_numpy(alpha).to(dtype=torch.float32)
                    refined_images.append(rgb_tensor)
                    refined_masks.append(alpha_tensor)
                else:
                    refined_images.append(image[i])
                    refined_masks.append(mask[i])
            except Exception as e:
                log.error("grabcut_node.refine_error", item=i, error=str(e))
                refined_images.append(image[i])
                refined_masks.append(mask[i])
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _log_gpu_memory(f"refine_batch_{i}.end")

        output_image = torch.stack(refined_images)
        output_mask = torch.stack(refined_masks)
        _log_gpu_memory("refine_mask.end")
        log.info("grabcut_node.refine_mask.done", batch_size=batch_size)
        return (output_image, output_mask)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AutoGrabCutRemover": AutoGrabCutRemover,
    "GrabCutRefinement": GrabCutRefinement,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoGrabCutRemover": "Auto GrabCut Background Remover",
    "GrabCutRefinement": "GrabCut Mask Refinement",
}