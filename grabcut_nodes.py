import numpy as np
import torch
from typing import Tuple, Optional
import time
from PIL import Image

# Try to import ComfyUI modules
try:
    import folder_paths
    import comfy.utils
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False
    print("ComfyUI modules not available. Running in standalone mode.")

# Import our GrabCut processor
try:
    from .grabcut_remover import GrabCutProcessor, create_fallback_processor
except ImportError:
    from grabcut_remover import GrabCutProcessor, create_fallback_processor


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
        resampling_method = self._RESAMPLING_MAP.get(scaling_method, Image.Resampling.NEAREST)
        
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
    
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("image", "mask", "bbox_coords", "confidence", "metrics")
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
            print(f"Warning: Could not initialize YOLO-based processor: {e}")
            print("Using fallback processor without YOLO")
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
    
    def remove_background(self, image: torch.Tensor, 
                         initial_mask: Optional[torch.Tensor] = None,
                         object_class: str = "auto",
                         confidence_threshold: float = 0.5,
                         grabcut_iterations: int = 5,
                         margin_pixels: int = 20,
                         edge_refinement: float = 0.7,
                         edge_blur_amount: float = 0.0,
                         binary_threshold: int = 200,
                         output_size: str = "ORIGINAL",
                         scaling_method: str = "NEAREST",
                         auto_adjust: bool = False,
                         output_format: str = "RGBA",
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
        # Ensure processor is initialized
        if self.processor is None:
            self._initialize_processor()
        
        # Update processor parameters
        self.processor.confidence_threshold = confidence_threshold
        self.processor.iterations = grabcut_iterations
        self.processor.margin_pixels = margin_pixels
        self.processor.edge_refinement_strength = edge_refinement
        self.processor.edge_blur_amount = edge_blur_amount
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
        
        processed_images = []
        masks = []
        all_bboxes = []
        all_confidences = []
        all_metrics = []
        
        for i in range(batch_size):
            try:
                # Convert from ComfyUI tensor format to numpy
                img_tensor = image[i]
                
                # Validate input tensor
                if len(img_tensor.shape) != 3:
                    print(f"Warning: Unexpected tensor shape {img_tensor.shape}, expected 3D tensor")
                    continue
                
                img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                
                # Ensure channels last format (H, W, C)
                if img_np.shape[0] == 3 or img_np.shape[0] == 4:
                    img_np = np.transpose(img_np, (1, 2, 0))
                
                # Final validation
                if len(img_np.shape) != 3 or img_np.shape[2] not in [3, 4]:
                    print(f"Warning: Invalid image shape {img_np.shape} after conversion")
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
                    
                    if output_format == "RGBA":
                        # For RGBA output: preserve transparency, don't premultiply alpha
                        # Create 4-channel RGBA tensor
                        rgba_tensor = torch.from_numpy(rgba.astype(np.float32) / 255.0).float()
                        processed_images.append(rgba_tensor)
                    else:  # output_format == "MASK"
                        # For MASK output: return binary mask as primary output
                        # Convert alpha to binary mask (0 or 255)
                        binary_mask = (alpha > 0.5).astype(np.float32)
                        mask_tensor = torch.from_numpy(binary_mask).float()
                        processed_images.append(mask_tensor.unsqueeze(-1))  # Add channel dimension
                    
                    # Alpha tensor for mask output (always provided)
                    alpha_tensor = torch.from_numpy(alpha).float()
                    masks.append(alpha_tensor)
                    
                    # Format bbox and metrics
                    if result['bbox']:
                        x1, y1, x2, y2 = result['bbox']
                        bbox_str = f"({x1},{y1},{x2},{y2})"
                        all_bboxes.append(bbox_str)
                    else:
                        all_bboxes.append("(0,0,0,0)")
                    
                    all_confidences.append(result['confidence'])
                    
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
                        rgb_tensor = torch.from_numpy(rgba_fallback).float()
                    else:  # output_format == "MASK"
                        # Return full foreground mask
                        alpha_fallback = np.ones((img_np.shape[0], img_np.shape[1], 1), dtype=np.float32)
                        rgb_tensor = torch.from_numpy(alpha_fallback).float()
                    
                    alpha_tensor = torch.ones((img_np.shape[0], img_np.shape[1]), dtype=torch.float32)
                    
                    processed_images.append(rgb_tensor)
                    masks.append(alpha_tensor)
                    all_bboxes.append("(0,0,0,0)")
                    all_confidences.append(0.0)
                    all_metrics.append(f"Batch {i+1}/{batch_size}: Processing failed")
                    
            except Exception as e:
                print(f"Error processing batch item {i+1}: {e}")
                # Add fallback empty tensors to maintain batch consistency
                h, w = 512, 512  # Default dimensions
                if len(processed_images) > 0:
                    # Use dimensions from previous successful processing
                    h, w = processed_images[0].shape[:2]
                
                if output_format == "RGBA":
                    # Empty RGBA tensor
                    rgb_tensor = torch.zeros((h, w, 4), dtype=torch.float32)
                else:  # output_format == "MASK"
                    # Empty mask tensor (single channel)
                    rgb_tensor = torch.zeros((h, w, 1), dtype=torch.float32)
                
                alpha_tensor = torch.zeros((h, w), dtype=torch.float32)
                
                processed_images.append(rgb_tensor)
                masks.append(alpha_tensor)
                all_bboxes.append("(0,0,0,0)")
                all_confidences.append(0.0)
                all_metrics.append(f"Batch {i+1}/{batch_size}: Error occurred")
        
        # Stack results
        output_image = torch.stack(processed_images)
        output_mask = torch.stack(masks)
        
        # Format outputs
        bbox_output = " | ".join(all_bboxes)
        confidence_output = float(np.mean(all_confidences))
        metrics_output = "\n".join(all_metrics)
        
        return (output_image, output_mask, bbox_output, confidence_output, metrics_output)


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
                    "tooltip": "Amount of Gaussian blur to apply to mask edges (0=none, 5=maximum)"
                }),
                "expand_margin": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 30,
                    "step": 5,
                    "tooltip": "Pixels to expand mask boundary"
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
    
    def refine_mask(self, image: torch.Tensor, mask: torch.Tensor,
                   grabcut_iterations: int = 3,
                   edge_refinement: float = 0.5,
                   edge_blur_amount: float = 0.0,
                   expand_margin: int = 10,
                   output_size: str = "ORIGINAL",
                   scaling_method: str = "NEAREST",
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
        if self.processor is None:
            self._initialize_processor()
        
        # Update parameters
        self.processor.iterations = grabcut_iterations
        self.processor.edge_refinement_strength = edge_refinement
        self.processor.edge_blur_amount = edge_blur_amount
        self.processor.margin_pixels = expand_margin
        
        # Handle batch
        if len(image.shape) == 4:
            batch_size = image.shape[0]
        else:
            batch_size = 1
            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0)
        
        refined_images = []
        refined_masks = []
        
        for i in range(batch_size):
            # Convert to numpy
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            if img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            
            mask_np = (mask[i].cpu().numpy() * 255).astype(np.uint8)
            if len(mask_np.shape) == 3:
                mask_np = mask_np.squeeze()
            
            # Refine with GrabCut
            result = self.processor.process_with_initial_mask(img_np, mask_np, None)
            
            if result['success']:
                rgba = result['rgba_image']
                
                # Apply resize if requested
                rgba = self._apply_resize(rgba, output_size, scaling_method, custom_width, custom_height)
                
                rgb = rgba[:, :, :3].astype(np.float32) / 255.0
                alpha = rgba[:, :, 3].astype(np.float32) / 255.0
                
                # Apply refined alpha
                for c in range(3):
                    rgb[:, :, c] *= alpha
                
                rgb_tensor = torch.from_numpy(rgb).float()
                alpha_tensor = torch.from_numpy(alpha).float()
                
                refined_images.append(rgb_tensor)
                refined_masks.append(alpha_tensor)
            else:
                # Return original if refinement fails
                refined_images.append(image[i])
                refined_masks.append(mask[i])
        
        output_image = torch.stack(refined_images)
        output_mask = torch.stack(refined_masks)
        
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