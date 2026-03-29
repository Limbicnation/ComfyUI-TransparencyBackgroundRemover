from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import structlog
import torch
from ultralytics import YOLO

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Parameter Adjustment Thresholds and Constants
# ---------------------------------------------------------------------------

# Contrast Analysis Thresholds
_CONTRAST_HIGH = 40
_CONTRAST_LOW = 25

# Edge Density Thresholds
_EDGE_DENSITY_HIGH = 0.08
_EDGE_DENSITY_LOW = 0.04
_EDGE_DENSITY_SHARP = 0.1
_EDGE_DENSITY_SOFT = 0.05

# Complexity Score Calculation Constants
_EDGE_DENSITY_MULTIPLIER = 10
_COLOR_VARIANCE_DIVISOR = 1000

# Complexity Score Thresholds
_COMPLEXITY_HIGH = 1.5
_COMPLEXITY_LOW = 0.5

# Laplacian Variance Thresholds
_LAPLACIAN_HIGH_NOISE = 1000
_LAPLACIAN_SHARP_EDGES = 500
_LAPLACIAN_SOFT_EDGES = 100
_LAPLACIAN_LOW_NOISE = 200

# Brightness Thresholds
_BRIGHTNESS_DARK = 80
_BRIGHTNESS_BRIGHT = 180

# Parameter Adjustment Values
_CONFIDENCE_ADJUSTMENT_DOWN = 0.1
_CONFIDENCE_ADJUSTMENT_UP = 0.15
_ITERATIONS_ADJUSTMENT = 2
_MARGIN_ADJUSTMENT = 5
_REFINEMENT_ADJUSTMENT = 0.2
_BINARY_THRESHOLD_ADJUSTMENT = 30
_BINARY_THRESHOLD_BRIGHT_ADJUSTMENT = 20
_EDGE_BLUR_ADJUSTMENT = 0.5

# Edge Blur Processing Constants
_SHARP_EDGE_BLUR_THRESHOLD = 0.5
_KERNEL_SIZE_SCALAR = 4
_MAX_KERNEL_SIZE = 31
_SIGMA_SCALAR = 0.5

# Parameter Limits
_CONFIDENCE_MIN = 0.3
_CONFIDENCE_MAX = 0.8
_ITERATIONS_MIN = 3
_ITERATIONS_MAX = 8
_MARGIN_MIN = 10
_MARGIN_MAX = 35
_REFINEMENT_MIN = 0.4
_REFINEMENT_MAX = 0.9
_BINARY_THRESHOLD_MIN = 150
_BINARY_THRESHOLD_MAX = 240
_EDGE_BLUR_MIN = 0.0
_EDGE_BLUR_MAX = 3.0


class GrabCutProcessor:
    """
    Advanced GrabCut background removal with automated object detection.

    Combines YOLOv8 object detection with OpenCV's GrabCut algorithm for
    precise foreground extraction with zero manual intervention.
    """

    # Class-level YOLO model cache — shared across all instances
    _yolo_cache: Dict[str, YOLO] = {}

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        iterations: int = 5,
        margin_pixels: int = 20,
        edge_refinement_strength: float = 0.7,
        edge_blur_amount: float = 0.0,
        binary_threshold: int = 200,
        model_path: Optional[str] = None,
        bbox_safety_margin: int = 30,
        min_bbox_size: int = 64,
        fallback_margin_percent: float = 0.2,
    ) -> None:
        """Initialize GrabCut processor with configuration."""
        self.confidence_threshold = confidence_threshold
        self.iterations = iterations
        self.margin_pixels = margin_pixels
        self.edge_refinement_strength = edge_refinement_strength
        self.edge_blur_amount = edge_blur_amount
        self.binary_threshold = binary_threshold
        self.bbox_safety_margin = bbox_safety_margin
        self.min_bbox_size = min_bbox_size
        self.fallback_margin_percent = max(0.1, min(0.5, fallback_margin_percent))

        # Initialize YOLO model (cached at class level)
        self.yolo_model: Optional[YOLO] = None
        self._initialize_yolo(model_path)

        # Object class mapping
        self.target_classes: Dict[str, int] = {
            'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3,
            'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7, 'boat': 8,
            'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18,
            'cow': 19, 'elephant': 20, 'bear': 21, 'zebra': 22, 'giraffe': 23,
            'backpack': 24, 'umbrella': 25, 'handbag': 26, 'tie': 27,
            'suitcase': 28, 'bottle': 39, 'wine glass': 40, 'cup': 41,
            'fork': 42, 'knife': 43, 'spoon': 44, 'bowl': 45,
            'chair': 56, 'couch': 57, 'potted plant': 58, 'bed': 59,
            'dining table': 60, 'toilet': 61, 'tv': 62, 'laptop': 63,
            'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell phone': 67,
            'book': 73, 'clock': 74, 'vase': 75, 'teddy bear': 77,
        }

        log.info("grabcut_processor.initialized",
                 confidence=self.confidence_threshold,
                 iterations=self.iterations,
                 yolo_available=self.yolo_model is not None)

    # ------------------------------------------------------------------
    # YOLO initialization
    # ------------------------------------------------------------------

    def _initialize_yolo(self, model_path: Optional[str] = None) -> None:
        """Initialize YOLO model, using class-level cache for efficiency."""
        cache_key = model_path if model_path and os.path.exists(model_path) else "yolov8n.pt"

        try:
            if cache_key not in GrabCutProcessor._yolo_cache:
                log.info("grabcut_processor.yolo_loading", model=cache_key)
                GrabCutProcessor._yolo_cache[cache_key] = YOLO(cache_key)
                # Attempt torch.compile for H100 acceleration
                try:
                    GrabCutProcessor._yolo_cache[cache_key].model = torch.compile(
                        GrabCutProcessor._yolo_cache[cache_key].model,
                    )
                    log.info("grabcut_processor.torch_compile.success", model=cache_key)
                except Exception as compile_err:
                    log.info("grabcut_processor.torch_compile.skipped",
                             reason=str(compile_err))
                log.info("grabcut_processor.yolo_loaded", model=cache_key)
            self.yolo_model = GrabCutProcessor._yolo_cache[cache_key]
        except Exception as e:
            log.warning("grabcut_processor.yolo_init_failed",
                        error=str(e), fallback="manual_rectangle")
            self.yolo_model = None

    # ------------------------------------------------------------------
    # Bounding box validation
    # ------------------------------------------------------------------

    def _validate_and_fix_bbox(
        self,
        bbox: tuple[int, int, int, int],
        image_shape: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        """Validate and fix bounding box to prevent cropping and ensure minimum size."""
        h, w = image_shape
        x1, y1, x2, y2 = bbox

        # Ensure coordinates are in correct order
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Ensure minimum size
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        if bbox_w < self.min_bbox_size:
            center_x = (x1 + x2) // 2
            x1 = center_x - self.min_bbox_size // 2
            x2 = x1 + self.min_bbox_size
            if x2 > w:
                x2 = w
                x1 = max(0, w - self.min_bbox_size)
            elif x1 < 0:
                x1 = 0
                x2 = min(w, self.min_bbox_size)

        if bbox_h < self.min_bbox_size:
            center_y = (y1 + y2) // 2
            y1 = center_y - self.min_bbox_size // 2
            y2 = y1 + self.min_bbox_size
            if y2 > h:
                y2 = h
                y1 = max(0, h - self.min_bbox_size)
            elif y1 < 0:
                y1 = 0
                y2 = min(h, self.min_bbox_size)

        # Add safety margin
        x1 = max(0, x1 - self.bbox_safety_margin)
        y1 = max(0, y1 - self.bbox_safety_margin)
        x2 = min(w, x2 + self.bbox_safety_margin)
        y2 = min(h, y2 + self.bbox_safety_margin)

        # Final bounds check
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        # Ensure we still have a valid bbox
        if x2 <= x1 or y2 <= y1:
            margin = int(min(h, w) * self.fallback_margin_percent)
            x1, y1 = margin, margin
            x2, y2 = w - margin, h - margin

        return (x1, y1, x2, y2)

    # ------------------------------------------------------------------
    # Object detection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def detect_object(
        self,
        image: np.ndarray,
        target_class: Optional[str] = None,
    ) -> Optional[tuple[int, int, int, int, float]]:
        """Detect primary object in image using YOLO."""
        if self.yolo_model is None:
            return None

        _log_gpu_memory("detect_object.start")

        try:
            results = self.yolo_model(image, conf=self.confidence_threshold, verbose=False)

            if len(results) == 0 or len(results[0].boxes) == 0:
                log.debug("grabcut_processor.no_detection")
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

                    best_idx = valid_indices[np.argmax(confidences[valid_indices])]
                else:
                    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                    best_idx = np.argmax(areas)
            else:
                best_idx = np.argmax(confidences)

            x1, y1, x2, y2 = xyxy[best_idx].astype(int)
            confidence = float(confidences[best_idx])

            log.info("grabcut_processor.detected",
                     bbox=(int(x1), int(y1), int(x2), int(y2)),
                     confidence=round(confidence, 3))
            return (x1, y1, x2, y2, confidence)

        except Exception as e:
            log.error("grabcut_processor.detection_error", error=str(e))
            return None
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _log_gpu_memory("detect_object.end")

    # ------------------------------------------------------------------
    # GrabCut core
    # ------------------------------------------------------------------

    def apply_grabcut(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Apply GrabCut algorithm with given bounding box."""
        h, w = image.shape[:2]
        log.debug("grabcut_processor.apply_grabcut", image_size=(w, h))

        # Validate and fix bounding box
        x1, y1, x2, y2 = self._validate_and_fix_bbox(bbox, (h, w))

        # Add additional margin
        x1 = max(0, x1 - self.margin_pixels)
        y1 = max(0, y1 - self.margin_pixels)
        x2 = min(w, x2 + self.margin_pixels)
        y2 = min(h, y2 + self.margin_pixels)

        rect = (x1, y1, x2 - x1, y2 - y1)
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model,
                        self.iterations, cv2.GC_INIT_WITH_RECT)
            output_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
            return output_mask
        except Exception as e:
            log.error("grabcut_processor.grabcut_error", error=str(e))
            fallback_mask = np.zeros((h, w), dtype=np.uint8)
            fallback_mask[y1:y2, x1:x2] = 255
            return fallback_mask

    # ------------------------------------------------------------------
    # Edge refinement
    # ------------------------------------------------------------------

    def refine_edges(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Apply edge refinement to improve mask quality with artifact-free edge blur."""
        if self.edge_refinement_strength <= 0 and self.edge_blur_amount <= 0:
            return mask

        mask_float = mask.astype(np.float32) / 255.0

        # Step 1: Bilateral filter for edge-preserving smoothing
        if self.edge_refinement_strength > 0:
            refined = cv2.bilateralFilter(mask_float, d=9, sigmaColor=0.1, sigmaSpace=7)

            # Step 2: Guided filter
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
                refined_input = refined.astype(np.float32)
                refined = cv2.ximgproc.guidedFilter(
                    guide=gray, src=refined_input, radius=4,
                    eps=self.edge_refinement_strength * 0.01,
                )
            except Exception as e:
                log.warning("grabcut_processor.guided_filter_failed", error=str(e))

            # Step 3: Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
            refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
        else:
            refined = mask_float

        # Step 4: Edge blur BEFORE binary thresholding
        if self.edge_blur_amount > 0:
            refined_255 = (refined * 255).astype(np.uint8)
            blurred = self._apply_edge_blur(refined_255)
            refined = blurred.astype(np.float32) / 255.0

        # Step 5: Binary threshold
        if self.edge_blur_amount <= _SHARP_EDGE_BLUR_THRESHOLD:
            _, binary = cv2.threshold(
                refined, self.binary_threshold / 255.0, 1.0, cv2.THRESH_BINARY,
            )
            return (binary * 255).astype(np.uint8)
        else:
            return (refined * 255).astype(np.uint8)

    def _apply_edge_blur(self, mask: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to mask edges for softer transitions."""
        if self.edge_blur_amount <= 0:
            return mask

        kernel_size = max(3, int(self.edge_blur_amount * _KERNEL_SIZE_SCALAR) | 1)
        kernel_size = min(kernel_size, _MAX_KERNEL_SIZE)
        sigma = self.edge_blur_amount * _SIGMA_SCALAR

        return cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma)

    # ------------------------------------------------------------------
    # Auto parameter adjustment
    # ------------------------------------------------------------------

    def auto_adjust_parameters(self, image: np.ndarray) -> dict[str, Any]:
        """Automatically adjust parameters based on image analysis."""
        log.debug("grabcut_processor.auto_adjust.start")

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        total_pixels = h * w

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / total_pixels
        contrast = gray.std()
        brightness = gray.mean()
        color_variance = np.var(image.reshape(-1, 3), axis=0).mean()
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        adjustments: dict[str, Any] = {}

        # Confidence threshold
        base_confidence = self.confidence_threshold
        if contrast > _CONTRAST_HIGH and edge_density > _EDGE_DENSITY_HIGH:
            adjustments['confidence_threshold'] = max(
                _CONFIDENCE_MIN, base_confidence - _CONFIDENCE_ADJUSTMENT_DOWN)
        elif contrast < _CONTRAST_LOW or edge_density < _EDGE_DENSITY_LOW:
            adjustments['confidence_threshold'] = min(
                _CONFIDENCE_MAX, base_confidence + _CONFIDENCE_ADJUSTMENT_UP)

        # Iterations
        base_iterations = self.iterations
        complexity_score = (edge_density * _EDGE_DENSITY_MULTIPLIER) + (
            color_variance / _COLOR_VARIANCE_DIVISOR)
        if complexity_score > _COMPLEXITY_HIGH:
            adjustments['iterations'] = min(
                _ITERATIONS_MAX, base_iterations + _ITERATIONS_ADJUSTMENT)
        elif complexity_score < _COMPLEXITY_LOW:
            adjustments['iterations'] = max(
                _ITERATIONS_MIN, base_iterations - _ITERATIONS_ADJUSTMENT)

        # Margin pixels
        base_margin = self.margin_pixels
        if edge_density > _EDGE_DENSITY_SHARP and laplacian_var > _LAPLACIAN_SHARP_EDGES:
            adjustments['margin_pixels'] = max(
                _MARGIN_MIN, base_margin - _MARGIN_ADJUSTMENT)
        elif edge_density < _EDGE_DENSITY_SOFT or laplacian_var < _LAPLACIAN_SOFT_EDGES:
            adjustments['margin_pixels'] = min(
                _MARGIN_MAX, base_margin + _MARGIN_ADJUSTMENT)

        # Edge refinement strength
        base_refinement = self.edge_refinement_strength
        if laplacian_var > _LAPLACIAN_HIGH_NOISE:
            adjustments['edge_refinement_strength'] = max(
                _REFINEMENT_MIN, base_refinement - _REFINEMENT_ADJUSTMENT)
        elif laplacian_var < _LAPLACIAN_LOW_NOISE:
            adjustments['edge_refinement_strength'] = min(
                _REFINEMENT_MAX, base_refinement + _REFINEMENT_ADJUSTMENT)

        # Binary threshold
        base_threshold = self.binary_threshold
        if brightness < _BRIGHTNESS_DARK:
            adjustments['binary_threshold'] = max(
                _BINARY_THRESHOLD_MIN, base_threshold - _BINARY_THRESHOLD_ADJUSTMENT)
        elif brightness > _BRIGHTNESS_BRIGHT:
            adjustments['binary_threshold'] = min(
                _BINARY_THRESHOLD_MAX, base_threshold + _BINARY_THRESHOLD_BRIGHT_ADJUSTMENT)

        # Edge blur amount
        base_blur = self.edge_blur_amount
        if laplacian_var > _LAPLACIAN_HIGH_NOISE or edge_density < _EDGE_DENSITY_SOFT:
            adjustments['edge_blur_amount'] = min(
                _EDGE_BLUR_MAX, base_blur + _EDGE_BLUR_ADJUSTMENT)
        elif edge_density > _EDGE_DENSITY_SHARP and laplacian_var > _LAPLACIAN_SHARP_EDGES:
            adjustments['edge_blur_amount'] = max(
                _EDGE_BLUR_MIN, base_blur - _EDGE_BLUR_ADJUSTMENT)
        elif contrast < _CONTRAST_LOW:
            adjustments['edge_blur_amount'] = min(
                _EDGE_BLUR_MAX, base_blur + (_EDGE_BLUR_ADJUSTMENT * 0.5))

        log.info("grabcut_processor.auto_adjust.done", adjustments=adjustments)
        return adjustments

    # ------------------------------------------------------------------
    # Pixel art detection
    # ------------------------------------------------------------------

    def _detect_pixel_art_characteristics(self, image: np.ndarray) -> bool:
        """Detect if image has pixel art characteristics."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        total_pixels = h * w

        unique_colors = len(np.unique(image.reshape(-1, 3), axis=0))
        color_density = unique_colors / total_pixels
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        fft = np.fft.fft2(gray)
        fft_magnitude = np.abs(fft)
        fft_magnitude[0:5, 0:5] = 0
        peak_ratio = np.max(fft_magnitude) / np.mean(fft_magnitude)

        is_small = w <= 512 or h <= 512

        pixel_art_score = 0
        if color_density < 0.01:
            pixel_art_score += 3
        elif color_density < 0.05:
            pixel_art_score += 2
        elif color_density < 0.1:
            pixel_art_score += 1

        if laplacian_var > 1000:
            pixel_art_score += 2
        elif laplacian_var > 500:
            pixel_art_score += 1

        if peak_ratio > 50:
            pixel_art_score += 2
        elif peak_ratio > 20:
            pixel_art_score += 1

        if is_small:
            pixel_art_score += 1

        return pixel_art_score >= 3

    # ------------------------------------------------------------------
    # Main processing pipelines
    # ------------------------------------------------------------------

    @torch.no_grad()
    def process_with_grabcut(
        self,
        image: np.ndarray,
        target_class: Optional[str] = None,
    ) -> dict[str, Any]:
        """Complete GrabCut processing pipeline with automated object detection."""
        start_time = time.time()
        h, w = image.shape[:2]

        log.info("grabcut_processor.process.start", size=(w, h), target_class=target_class)
        _log_gpu_memory("process_with_grabcut.start")

        result: dict[str, Any] = {
            'rgba_image': None,
            'mask': None,
            'bbox': None,
            'confidence': 0.0,
            'processing_time_ms': 0,
            'success': False,
        }

        # Ensure RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        # Step 1: Detect object
        detection = self.detect_object(image, target_class)

        if detection is None:
            margin = int(min(h, w) * self.fallback_margin_percent)
            detection = (margin, margin, w - margin, h - margin, 0.5)
            log.info("grabcut_processor.using_fallback",
                     margin_pct=self.fallback_margin_percent)

        x1, y1, x2, y2, confidence = detection
        result['bbox'] = (x1, y1, x2, y2)
        result['confidence'] = confidence

        # Step 2: Apply GrabCut
        mask = self.apply_grabcut(image, (x1, y1, x2, y2))

        # Step 3: Refine edges
        try:
            mask = self.refine_edges(mask, image)
        except Exception as e:
            log.warning("grabcut_processor.edge_refinement_skipped", error=str(e))

        # Step 4: Create RGBA output
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = image
        rgba[:, :, 3] = mask

        result['rgba_image'] = rgba
        result['mask'] = mask
        result['success'] = True
        result['processing_time_ms'] = int((time.time() - start_time) * 1000)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _log_gpu_memory("process_with_grabcut.end")

        log.info("grabcut_processor.process.done",
                 time_ms=result['processing_time_ms'],
                 confidence=round(confidence, 3))
        return result

    @torch.no_grad()
    def process_with_initial_mask(
        self,
        image: np.ndarray,
        initial_mask: np.ndarray,
        target_class: Optional[str] = None,
    ) -> dict[str, Any]:
        """Process image with an initial mask from previous processing."""
        start_time = time.time()
        h, w = image.shape[:2]

        log.info("grabcut_processor.refine.start", size=(w, h))
        _log_gpu_memory("process_with_initial_mask.start")

        result: dict[str, Any] = {
            'rgba_image': None,
            'mask': None,
            'bbox': None,
            'confidence': 0.0,
            'processing_time_ms': 0,
            'success': False,
        }

        # Ensure proper formats
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        # Get bounding box
        detection = self.detect_object(image, target_class)

        if detection is None:
            if initial_mask.max() > 0:
                coords = np.where(initial_mask > 127)
                y1, y2 = coords[0].min(), coords[0].max()
                x1, x2 = coords[1].min(), coords[1].max()
                raw_bbox = (x1, y1, x2, y2)
                x1, y1, x2, y2 = self._validate_and_fix_bbox(raw_bbox, (h, w))
                detection = (x1, y1, x2, y2, 0.8)
            else:
                margin = int(min(h, w) * self.fallback_margin_percent)
                detection = (margin, margin, w - margin, h - margin, 0.5)

        x1, y1, x2, y2, confidence = detection
        result['bbox'] = (x1, y1, x2, y2)
        result['confidence'] = confidence

        # Initialize GrabCut mask from initial mask
        grabcut_mask = np.zeros((h, w), np.uint8)
        grabcut_mask[initial_mask > 200] = cv2.GC_FGD
        grabcut_mask[(initial_mask > 50) & (initial_mask <= 200)] = cv2.GC_PR_FGD
        grabcut_mask[(initial_mask > 0) & (initial_mask <= 50)] = cv2.GC_PR_BGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(image, grabcut_mask, None, bgd_model, fgd_model,
                        self.iterations, cv2.GC_INIT_WITH_MASK)
            output_mask = np.where(
                (grabcut_mask == 2) | (grabcut_mask == 0), 0, 255,
            ).astype('uint8')
            output_mask = self.refine_edges(output_mask, image)
        except Exception as e:
            log.error("grabcut_processor.mask_refinement_error", error=str(e))
            output_mask = initial_mask

        # Create RGBA output
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = image
        rgba[:, :, 3] = output_mask

        result['rgba_image'] = rgba
        result['mask'] = output_mask
        result['success'] = True
        result['processing_time_ms'] = int((time.time() - start_time) * 1000)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _log_gpu_memory("process_with_initial_mask.end")

        log.info("grabcut_processor.refine.done",
                 time_ms=result['processing_time_ms'])
        return result


# ---------------------------------------------------------------------------
# Helper: GPU memory logging
# ---------------------------------------------------------------------------

def _log_gpu_memory(tag: str) -> None:
    """Log GPU memory statistics if CUDA is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        log.debug("gpu_memory", tag=tag,
                  allocated_gb=round(allocated, 3),
                  reserved_gb=round(reserved, 3))


# ---------------------------------------------------------------------------
# Fallback processor (no YOLO)
# ---------------------------------------------------------------------------

def create_fallback_processor() -> type[GrabCutProcessor]:
    """Create a fallback processor class that works without YOLO."""

    class FallbackGrabCutProcessor(GrabCutProcessor):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.yolo_model = None

        def detect_object(
            self,
            image: np.ndarray,
            target_class: Optional[str] = None,
        ) -> Optional[tuple[int, int, int, int, float]]:
            """Fallback object detection using image analysis."""
            h, w = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                margin = int(min(h, w) * self.fallback_margin_percent)
                return (margin, margin, w - margin, h - margin, 0.5)

            largest_contour = max(contours, key=cv2.contourArea)
            x, y, cw, ch = cv2.boundingRect(largest_contour)

            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w, x + cw + margin)
            y2 = min(h, y + ch + margin)

            return (x1, y1, x2, y2, 0.7)

    return FallbackGrabCutProcessor
