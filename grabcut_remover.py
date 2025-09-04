import numpy as np
import cv2
import time
from typing import Tuple, Dict, Optional, List
import torch
from ultralytics import YOLO
import os


# Parameter Adjustment Thresholds and Constants
# These constants define the thresholds used in auto_adjust_parameters()
# for intelligent parameter tuning based on image characteristics

# Contrast Analysis Thresholds
_CONTRAST_HIGH = 40          # High contrast threshold - allows lower confidence
_CONTRAST_LOW = 25           # Low contrast threshold - requires higher confidence

# Edge Density Thresholds  
_EDGE_DENSITY_HIGH = 0.08    # High edge density - clear edges detected
_EDGE_DENSITY_LOW = 0.04     # Low edge density - few edges detected
_EDGE_DENSITY_SHARP = 0.1    # Very sharp edges - can use smaller margin
_EDGE_DENSITY_SOFT = 0.05    # Soft edges - need larger margin

# Complexity Score Calculation Constants
_EDGE_DENSITY_MULTIPLIER = 10      # Weight for edge density in complexity score
_COLOR_VARIANCE_DIVISOR = 1000     # Divisor for color variance normalization

# Complexity Score Thresholds
_COMPLEXITY_HIGH = 1.5       # High complexity - more iterations needed
_COMPLEXITY_LOW = 0.5        # Low complexity - fewer iterations sufficient

# Laplacian Variance Thresholds (noise/sharpness detection)
_LAPLACIAN_HIGH_NOISE = 1000      # High noise level - reduce edge refinement
_LAPLACIAN_SHARP_EDGES = 500      # Sharp, well-defined edges
_LAPLACIAN_SOFT_EDGES = 100       # Soft or unclear edges
_LAPLACIAN_LOW_NOISE = 200        # Low noise level - can use stronger refinement

# Brightness Thresholds
_BRIGHTNESS_DARK = 80        # Dark image threshold - lower binary threshold
_BRIGHTNESS_BRIGHT = 180     # Bright image threshold - higher binary threshold

# Parameter Adjustment Values
_CONFIDENCE_ADJUSTMENT_DOWN = 0.1    # Amount to decrease confidence for clear images
_CONFIDENCE_ADJUSTMENT_UP = 0.15     # Amount to increase confidence for unclear images
_ITERATIONS_ADJUSTMENT = 2           # Amount to adjust iterations
_MARGIN_ADJUSTMENT = 5              # Amount to adjust margin pixels
_REFINEMENT_ADJUSTMENT = 0.2        # Amount to adjust edge refinement strength
_BINARY_THRESHOLD_ADJUSTMENT = 30   # Amount to adjust binary threshold for dark images
_BINARY_THRESHOLD_BRIGHT_ADJUSTMENT = 20  # Amount to adjust for bright images
_EDGE_BLUR_ADJUSTMENT = 0.5         # Amount to adjust edge blur for different conditions

# Parameter Limits
_CONFIDENCE_MIN = 0.3        # Minimum confidence threshold
_CONFIDENCE_MAX = 0.8        # Maximum confidence threshold
_ITERATIONS_MIN = 3          # Minimum GrabCut iterations
_ITERATIONS_MAX = 8          # Maximum GrabCut iterations
_MARGIN_MIN = 10            # Minimum margin pixels
_MARGIN_MAX = 35            # Maximum margin pixels
_REFINEMENT_MIN = 0.4       # Minimum edge refinement strength
_REFINEMENT_MAX = 0.9       # Maximum edge refinement strength
_BINARY_THRESHOLD_MIN = 150  # Minimum binary threshold
_BINARY_THRESHOLD_MAX = 240  # Maximum binary threshold
_EDGE_BLUR_MIN = 0.0        # Minimum edge blur amount
_EDGE_BLUR_MAX = 3.0        # Maximum edge blur amount


class GrabCutProcessor:
    """
    Advanced GrabCut background removal with automated object detection.
    Combines YOLOv8 object detection with OpenCV's GrabCut algorithm for
    precise foreground extraction with zero manual intervention.
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.5,
                 iterations: int = 5,
                 margin_pixels: int = 20,
                 edge_refinement_strength: float = 0.7,
                 edge_blur_amount: float = 0.0,
                 binary_threshold: int = 200,
                 model_path: Optional[str] = None):
        """
        Initialize GrabCut processor with configuration.
        
        Args:
            confidence_threshold: Minimum confidence for object detection (0.0-1.0)
            iterations: Number of GrabCut iterations
            margin_pixels: Pixel margin around detected object
            edge_refinement_strength: Strength of edge refinement (0.0-1.0)
            edge_blur_amount: Amount of Gaussian blur to apply to mask edges (0.0-5.0)
            binary_threshold: Threshold for binary mask conversion
            model_path: Optional custom YOLO model path
        """
        self.confidence_threshold = confidence_threshold
        self.iterations = iterations
        self.margin_pixels = margin_pixels
        self.edge_refinement_strength = edge_refinement_strength
        self.edge_blur_amount = edge_blur_amount
        self.binary_threshold = binary_threshold
        
        # Initialize YOLO model
        self.yolo_model = None
        self._initialize_yolo(model_path)
        
        # Object class mapping
        self.target_classes = {
            'person': 0,
            'bicycle': 1,
            'car': 2,
            'motorcycle': 3,
            'airplane': 4,
            'bus': 5,
            'train': 6,
            'truck': 7,
            'boat': 8,
            'bird': 14,
            'cat': 15,
            'dog': 16,
            'horse': 17,
            'sheep': 18,
            'cow': 19,
            'elephant': 20,
            'bear': 21,
            'zebra': 22,
            'giraffe': 23,
            'backpack': 24,
            'umbrella': 25,
            'handbag': 26,
            'tie': 27,
            'suitcase': 28,
            'bottle': 39,
            'wine glass': 40,
            'cup': 41,
            'fork': 42,
            'knife': 43,
            'spoon': 44,
            'bowl': 45,
            'chair': 56,
            'couch': 57,
            'potted plant': 58,
            'bed': 59,
            'dining table': 60,
            'toilet': 61,
            'tv': 62,
            'laptop': 63,
            'mouse': 64,
            'remote': 65,
            'keyboard': 66,
            'cell phone': 67,
            'book': 73,
            'clock': 74,
            'vase': 75,
            'teddy bear': 77,
        }
    
    def _initialize_yolo(self, model_path: Optional[str] = None):
        """Initialize YOLO model for object detection."""
        try:
            if model_path and os.path.exists(model_path):
                self.yolo_model = YOLO(model_path)
            else:
                # Use YOLOv8 nano model for speed
                self.yolo_model = YOLO('yolov8n.pt')
        except Exception as e:
            print(f"Warning: Could not initialize YOLO model: {e}")
            print("Falling back to manual rectangle mode")
            self.yolo_model = None
    
    def detect_object(self, image: np.ndarray, target_class: Optional[str] = None) -> Optional[Tuple[int, int, int, int, float]]:
        """
        Detect primary object in image using YOLO.
        
        Args:
            image: Input image (RGB)
            target_class: Specific object class to detect, None for auto-detect
            
        Returns:
            Tuple of (x1, y1, x2, y2, confidence) or None if no object detected
        """
        if self.yolo_model is None:
            return None
        
        try:
            # Run YOLO detection
            results = self.yolo_model(image, conf=self.confidence_threshold, verbose=False)
            
            if len(results) == 0 or len(results[0].boxes) == 0:
                return None
            
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            
            # Filter by target class if specified
            if target_class and target_class != 'auto':
                if target_class in self.target_classes:
                    target_class_id = self.target_classes[target_class]
                    valid_indices = np.where(classes == target_class_id)[0]
                    
                    if len(valid_indices) == 0:
                        return None
                    
                    # Get highest confidence match for target class
                    best_idx = valid_indices[np.argmax(confidences[valid_indices])]
                else:
                    # Unknown class, use largest detection
                    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                    best_idx = np.argmax(areas)
            else:
                # Auto-detect: use highest confidence detection
                best_idx = np.argmax(confidences)
            
            # Get bounding box
            x1, y1, x2, y2 = xyxy[best_idx].astype(int)
            confidence = float(confidences[best_idx])
            
            return (x1, y1, x2, y2, confidence)
            
        except Exception as e:
            print(f"Error in object detection: {e}")
            return None
    
    def apply_grabcut(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Apply GrabCut algorithm with given bounding box.
        
        Args:
            image: Input image (RGB)
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Binary mask (0=background, 255=foreground)
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Add margin to bounding box
        x1 = max(0, x1 - self.margin_pixels)
        y1 = max(0, y1 - self.margin_pixels)
        x2 = min(w, x2 + self.margin_pixels)
        y2 = min(h, y2 + self.margin_pixels)
        
        # Convert to GrabCut rectangle format (x, y, width, height)
        rect = (x1, y1, x2 - x1, y2 - y1)
        
        # Initialize mask
        mask = np.zeros((h, w), np.uint8)
        
        # Initialize foreground and background models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            # Apply GrabCut
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 
                       self.iterations, cv2.GC_INIT_WITH_RECT)
            
            # Convert mask to binary
            # GrabCut mask values: 0=BG, 1=FG, 2=PR_BG, 3=PR_FG
            output_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
            
            return output_mask
            
        except Exception as e:
            print(f"Error in GrabCut: {e}")
            # Return rectangle mask as fallback
            fallback_mask = np.zeros((h, w), dtype=np.uint8)
            fallback_mask[y1:y2, x1:x2] = 255
            return fallback_mask
    
    def refine_edges(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Apply edge refinement to improve mask quality.
        
        Args:
            mask: Binary mask
            image: Original image for guided filtering
            
        Returns:
            Refined mask
        """
        if self.edge_refinement_strength <= 0:
            return mask
        
        # Convert to float for processing
        mask_float = mask.astype(np.float32) / 255.0
        
        # Apply bilateral filter for edge-preserving smoothing
        refined = cv2.bilateralFilter(
            mask_float,
            d=9,
            sigmaColor=0.1,
            sigmaSpace=7
        )
        
        # Apply guided filter using original image
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Ensure proper data types for guided filter
            gray = gray.astype(np.float32)
            refined_input = refined.astype(np.float32)
            
            refined = cv2.ximgproc.guidedFilter(
                guide=gray,
                src=refined_input,
                radius=4,
                eps=self.edge_refinement_strength * 0.01
            )
        except Exception as e:
            print(f"Guided filter failed, using bilateral filter only: {e}")
            # Continue with just the bilateral filter result
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
        
        if self.edge_blur_amount > 0:
            # Convert to 0-255 range for edge blur processing, apply blur, and return.
            refined_255 = (refined * 255).astype(np.uint8)
            return self._apply_edge_blur(refined_255)
        else:
            # Apply binary threshold to eliminate semi-transparency for sharp edges.
            _, binary = cv2.threshold(refined, self.binary_threshold / 255.0, 1.0, cv2.THRESH_BINARY)
            return (binary * 255).astype(np.uint8)
    
    def _apply_edge_blur(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur to mask edges for softer transitions.
        
        Args:
            mask: Binary mask to blur
            
        Returns:
            Blurred mask with soft edges
        """
        if self.edge_blur_amount <= 0:
            return mask
        
        # Calculate dynamic kernel size based on blur amount
        # Ensure kernel size is odd and reasonable
        kernel_size = int(self.edge_blur_amount * 2) * 2 + 1
        # With edge_blur_amount max 5.0, kernel_size max is 21.
        kernel_size = max(3, min(kernel_size, 21))
        
        # Apply Gaussian blur. Let OpenCV calculate sigma from the kernel size for a standard blur.
        blurred_mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        
        return blurred_mask
    
    def auto_adjust_parameters(self, image: np.ndarray) -> dict:
        """
        Automatically adjust parameters based on image analysis.
        Analyzes image characteristics and returns optimal parameter adjustments.
        
        Args:
            image: Input image for analysis (RGB format)
            
        Returns:
            Dictionary of adjusted parameters
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        total_pixels = h * w
        
        # Analyze image characteristics
        # 1. Edge density analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / total_pixels
        
        # 2. Contrast analysis
        contrast = gray.std()
        
        # 3. Brightness distribution
        brightness = gray.mean()
        
        # 4. Color variance analysis
        color_variance = np.var(image.reshape(-1, 3), axis=0).mean()
        
        # 5. Noise level estimation (using Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Initialize adjustments dictionary
        adjustments = {}
        
        # Adjust confidence_threshold based on contrast and edge clarity
        base_confidence = self.confidence_threshold
        if contrast > _CONTRAST_HIGH and edge_density > _EDGE_DENSITY_HIGH:
            # High contrast, clear edges -> can lower confidence threshold
            adjustments['confidence_threshold'] = max(_CONFIDENCE_MIN, base_confidence - _CONFIDENCE_ADJUSTMENT_DOWN)
        elif contrast < _CONTRAST_LOW or edge_density < _EDGE_DENSITY_LOW:
            # Low contrast or few edges -> need higher confidence threshold
            adjustments['confidence_threshold'] = min(_CONFIDENCE_MAX, base_confidence + _CONFIDENCE_ADJUSTMENT_UP)
        
        # Adjust iterations based on image complexity
        base_iterations = self.iterations
        complexity_score = (edge_density * _EDGE_DENSITY_MULTIPLIER) + (color_variance / _COLOR_VARIANCE_DIVISOR)
        if complexity_score > _COMPLEXITY_HIGH:
            # High complexity -> more iterations needed
            adjustments['iterations'] = min(_ITERATIONS_MAX, base_iterations + _ITERATIONS_ADJUSTMENT)
        elif complexity_score < _COMPLEXITY_LOW:
            # Low complexity -> fewer iterations sufficient
            adjustments['iterations'] = max(_ITERATIONS_MIN, base_iterations - _ITERATIONS_ADJUSTMENT)
        
        # Adjust margin_pixels based on edge sharpness and object size estimation
        base_margin = self.margin_pixels
        if edge_density > _EDGE_DENSITY_SHARP and laplacian_var > _LAPLACIAN_SHARP_EDGES:
            # Sharp, well-defined edges -> can use smaller margin
            adjustments['margin_pixels'] = max(_MARGIN_MIN, base_margin - _MARGIN_ADJUSTMENT)
        elif edge_density < _EDGE_DENSITY_SOFT or laplacian_var < _LAPLACIAN_SOFT_EDGES:
            # Soft or unclear edges -> need larger margin
            adjustments['margin_pixels'] = min(_MARGIN_MAX, base_margin + _MARGIN_ADJUSTMENT)
        
        # Adjust edge_refinement_strength based on noise level
        base_refinement = self.edge_refinement_strength
        if laplacian_var > _LAPLACIAN_HIGH_NOISE:
            # High noise -> reduce edge refinement to avoid artifacts
            adjustments['edge_refinement_strength'] = max(_REFINEMENT_MIN, base_refinement - _REFINEMENT_ADJUSTMENT)
        elif laplacian_var < _LAPLACIAN_LOW_NOISE:
            # Low noise -> can use stronger edge refinement
            adjustments['edge_refinement_strength'] = min(_REFINEMENT_MAX, base_refinement + _REFINEMENT_ADJUSTMENT)
        
        # Adjust binary_threshold based on brightness distribution
        base_threshold = self.binary_threshold
        if brightness < _BRIGHTNESS_DARK:
            # Dark image -> lower threshold
            adjustments['binary_threshold'] = max(_BINARY_THRESHOLD_MIN, base_threshold - _BINARY_THRESHOLD_ADJUSTMENT)
        elif brightness > _BRIGHTNESS_BRIGHT:
            # Bright image -> higher threshold
            adjustments['binary_threshold'] = min(_BINARY_THRESHOLD_MAX, base_threshold + _BINARY_THRESHOLD_BRIGHT_ADJUSTMENT)
        
        # Adjust edge_blur_amount based on edge characteristics and noise level
        base_blur = self.edge_blur_amount
        if laplacian_var > _LAPLACIAN_HIGH_NOISE or edge_density < _EDGE_DENSITY_SOFT:
            # Noisy or soft edges -> apply blur to smooth transitions
            adjustments['edge_blur_amount'] = min(_EDGE_BLUR_MAX, base_blur + _EDGE_BLUR_ADJUSTMENT)
        elif edge_density > _EDGE_DENSITY_SHARP and laplacian_var > _LAPLACIAN_SHARP_EDGES:
            # Sharp, clean edges -> minimal blur to preserve detail
            adjustments['edge_blur_amount'] = max(_EDGE_BLUR_MIN, base_blur - _EDGE_BLUR_ADJUSTMENT)
        elif contrast < _CONTRAST_LOW:
            # Low contrast images benefit from edge blur for smoother results
            adjustments['edge_blur_amount'] = min(_EDGE_BLUR_MAX, base_blur + (_EDGE_BLUR_ADJUSTMENT * 0.5))
        
        return adjustments
    
    def _detect_pixel_art_characteristics(self, image: np.ndarray) -> bool:
        """
        Detect if image has pixel art characteristics.
        Analyzes color count, edge sharpness, and dithering patterns.
        
        Args:
            image: Input image (RGB format)
            
        Returns:
            Boolean indicating if image appears to be pixel art
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        total_pixels = h * w
        
        # 1. Color count analysis - pixel art typically has limited colors
        unique_colors = len(np.unique(image.reshape(-1, 3), axis=0))
        color_density = unique_colors / total_pixels
        
        # 2. Edge sharpness analysis - pixel art has very sharp edges
        # Calculate edge sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 3. Dither pattern detection - look for regular patterns
        # Use FFT to detect repeating patterns
        fft = np.fft.fft2(gray)
        fft_magnitude = np.abs(fft)
        
        # Look for peaks in frequency domain that suggest regular patterns
        # Remove DC component and low frequencies
        fft_magnitude[0:5, 0:5] = 0
        peak_ratio = np.max(fft_magnitude) / np.mean(fft_magnitude)
        
        # 4. Small dimensions often indicate pixel art
        is_small = w <= 512 or h <= 512
        
        # Decision logic based on multiple factors
        pixel_art_score = 0
        
        # Low color density suggests pixel art
        if color_density < 0.01:  # Very limited colors
            pixel_art_score += 3
        elif color_density < 0.05:  # Limited colors
            pixel_art_score += 2
        elif color_density < 0.1:  # Somewhat limited colors
            pixel_art_score += 1
        
        # High edge sharpness suggests pixel art
        if laplacian_var > 1000:  # Very sharp edges
            pixel_art_score += 2
        elif laplacian_var > 500:  # Sharp edges
            pixel_art_score += 1
        
        # Regular patterns suggest dithering/pixel art
        if peak_ratio > 50:  # Strong regular patterns
            pixel_art_score += 2
        elif peak_ratio > 20:  # Moderate patterns
            pixel_art_score += 1
        
        # Small dimensions bonus
        if is_small:
            pixel_art_score += 1
        
        # Threshold for pixel art detection
        return pixel_art_score >= 3
    
    def process_with_grabcut(self, image: np.ndarray, target_class: Optional[str] = None) -> Dict:
        """
        Complete GrabCut processing pipeline with automated object detection.
        
        Args:
            image: Input image (RGB)
            target_class: Target object class or 'auto' for automatic
            
        Returns:
            Dictionary containing:
                - rgba_image: RGBA image with transparent background
                - mask: Alpha mask
                - bbox: Detected bounding box (x1, y1, x2, y2)
                - confidence: Detection confidence
                - processing_time_ms: Processing time in milliseconds
                - success: Whether processing succeeded
        """
        start_time = time.time()
        h, w = image.shape[:2]
        
        # Initialize result
        result = {
            'rgba_image': None,
            'mask': None,
            'bbox': None,
            'confidence': 0.0,
            'processing_time_ms': 0,
            'success': False
        }
        
        # Ensure RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        # Step 1: Detect object
        detection = self.detect_object(image, target_class)
        
        if detection is None:
            # Fallback: use entire image with margin
            margin = int(min(h, w) * 0.1)
            detection = (margin, margin, w - margin, h - margin, 0.5)
            print("No object detected, using fallback rectangle")
        
        x1, y1, x2, y2, confidence = detection
        result['bbox'] = (x1, y1, x2, y2)
        result['confidence'] = confidence
        
        # Step 2: Apply GrabCut
        mask = self.apply_grabcut(image, (x1, y1, x2, y2))
        
        # Step 3: Refine edges
        try:
            mask = self.refine_edges(mask, image)
        except Exception as e:
            print(f"Edge refinement skipped: {e}")
        
        # Step 4: Create RGBA output
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = image
        rgba[:, :, 3] = mask
        
        # Update result
        result['rgba_image'] = rgba
        result['mask'] = mask
        result['success'] = True
        result['processing_time_ms'] = int((time.time() - start_time) * 1000)
        
        return result
    
    def process_with_initial_mask(self, image: np.ndarray, initial_mask: np.ndarray, 
                                  target_class: Optional[str] = None) -> Dict:
        """
        Process image with an initial mask from previous processing.
        Useful for refining results from other background removal methods.
        
        Args:
            image: Input image (RGB)
            initial_mask: Initial mask from previous processing
            target_class: Target object class for detection
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        h, w = image.shape[:2]
        
        result = {
            'rgba_image': None,
            'mask': None,
            'bbox': None,
            'confidence': 0.0,
            'processing_time_ms': 0,
            'success': False
        }
        
        # Ensure proper formats
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        # Get bounding box from initial mask if no detection
        detection = self.detect_object(image, target_class)
        
        if detection is None:
            # Find bounding box from initial mask
            if initial_mask.max() > 0:
                coords = np.where(initial_mask > 127)
                y1, y2 = coords[0].min(), coords[0].max()
                x1, x2 = coords[1].min(), coords[1].max()
                detection = (x1, y1, x2, y2, 0.8)
            else:
                # Fallback to full image
                margin = int(min(h, w) * 0.1)
                detection = (margin, margin, w - margin, h - margin, 0.5)
        
        x1, y1, x2, y2, confidence = detection
        result['bbox'] = (x1, y1, x2, y2)
        result['confidence'] = confidence
        
        # Initialize GrabCut mask from initial mask
        grabcut_mask = np.zeros((h, w), np.uint8)
        
        # Convert initial mask to GrabCut format
        # 0=BG, 1=FG, 2=PR_BG, 3=PR_FG
        grabcut_mask[initial_mask > 200] = cv2.GC_FGD  # Definite foreground
        grabcut_mask[(initial_mask > 50) & (initial_mask <= 200)] = cv2.GC_PR_FGD  # Probable foreground
        grabcut_mask[(initial_mask > 0) & (initial_mask <= 50)] = cv2.GC_PR_BGD  # Probable background
        # grabcut_mask[initial_mask == 0] remains cv2.GC_BGD (0)
        
        # Apply GrabCut with mask initialization
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            cv2.grabCut(image, grabcut_mask, None, bgd_model, fgd_model,
                       self.iterations, cv2.GC_INIT_WITH_MASK)
            
            # Convert to binary mask
            output_mask = np.where((grabcut_mask == 2) | (grabcut_mask == 0), 0, 255).astype('uint8')
            
            # Refine edges
            output_mask = self.refine_edges(output_mask, image)
            
        except Exception as e:
            print(f"Error in GrabCut with initial mask: {e}")
            output_mask = initial_mask
        
        # Create RGBA output
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = image
        rgba[:, :, 3] = output_mask
        
        result['rgba_image'] = rgba
        result['mask'] = output_mask
        result['success'] = True
        result['processing_time_ms'] = int((time.time() - start_time) * 1000)
        
        return result


def create_fallback_processor():
    """
    Create a fallback processor that works without YOLO.
    Uses simple image analysis to find the main subject.
    """
    
    class FallbackGrabCutProcessor(GrabCutProcessor):
        def __init__(self, **kwargs):
            # Initialize without YOLO
            super().__init__(**kwargs)
            self.yolo_model = None
        
        def detect_object(self, image: np.ndarray, target_class: Optional[str] = None) -> Optional[Tuple[int, int, int, int, float]]:
            """
            Fallback object detection using image analysis.
            Finds the largest connected component that's not background.
            """
            h, w = image.shape[:2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                # Return center region
                margin = int(min(h, w) * 0.2)
                return (margin, margin, w - margin, h - margin, 0.5)
            
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, cw, ch = cv2.boundingRect(largest_contour)
            
            # Add some margin
            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w, x + cw + margin)
            y2 = min(h, y + ch + margin)
            
            return (x1, y1, x2, y2, 0.7)
    
    return FallbackGrabCutProcessor