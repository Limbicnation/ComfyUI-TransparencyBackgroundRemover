import numpy as np
import cv2
from sklearn.cluster import KMeans
import colorsys
from typing import Tuple, Optional, List


class EnhancedPixelArtProcessor:
    """
    Enhanced pixel art background removal processor with advanced algorithms
    for detecting and removing backgrounds while preserving foreground details.
    """
    
    def __init__(self, tolerance: int = 30, edge_sensitivity: float = 0.8,
                 color_clusters: int = 8, foreground_bias: float = 0.7,
                 edge_refinement: bool = True, dither_handling: bool = True,
                 binary_threshold: int = 128):
        """
        Initialize the processor with configuration parameters.
        
        Args:
            tolerance: Color similarity threshold (0-255)
            edge_sensitivity: Edge detection sensitivity (0.0-1.0)
            color_clusters: Number of K-means clusters for analysis
            foreground_bias: Bias towards foreground preservation (0.0-1.0)
            edge_refinement: Enable edge refinement post-processing
            dither_handling: Enable dithered pattern detection
            binary_threshold: Threshold for binary alpha mask (0-255)
        """
        self.tolerance = tolerance
        self.edge_sensitivity = edge_sensitivity
        self.color_clusters = color_clusters
        self.foreground_bias = foreground_bias
        self.edge_refinement = edge_refinement
        self.dither_handling = dither_handling
        self.binary_threshold = binary_threshold
        
    def remove_background_advanced(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced background removal with multiple detection methods.
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            RGBA image with transparent background
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
            
        # Create RGBA output
        rgba_result = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgba_result[:, :, :3] = image
        
        # Generate initial mask using multiple methods
        masks = []
        
        # Method 1: Edge-based detection
        edge_mask = self._edge_based_detection(image)
        masks.append(edge_mask)
        
        # Method 2: Color clustering
        cluster_mask = self._color_clustering_detection(image)
        masks.append(cluster_mask)
        
        # Method 3: Corner sampling
        corner_mask = self._corner_sampling_detection(image)
        masks.append(corner_mask)
        
        # Method 4: Dither detection (if enabled)
        if self.dither_handling:
            dither_mask = self._dither_pattern_detection(image)
            masks.append(dither_mask)
        
        # Combine masks with weighted voting
        final_mask = self._combine_masks(masks, image)
        
        # Apply edge refinement if enabled
        if self.edge_refinement:
            final_mask = self._refine_edges(final_mask, image)
        
        # Apply foreground bias
        final_mask = self._apply_foreground_bias(final_mask, image)
        
        # Convert to binary mask to eliminate semi-transparency
        final_mask = self._make_binary_mask(final_mask, self.binary_threshold)
        
        # Invert mask: background=0 (transparent), foreground=255 (opaque)
        final_mask = 255 - final_mask
        
        rgba_result[:, :, 3] = final_mask
        return rgba_result
    
    def _edge_based_detection(self, image: np.ndarray) -> np.ndarray:
        """Detect background using edge analysis."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Edge detection with adaptive threshold
        threshold_val = int(255 * (1.0 - self.edge_sensitivity))
        edges = cv2.Canny(blurred, threshold_val // 2, threshold_val)
        
        # Dilate edges to create regions
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours and create mask
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.zeros(gray.shape, dtype=np.uint8)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small contours
                cv2.fillPoly(mask, [contour], 255)
        
        return mask
    
    def _color_clustering_detection(self, image: np.ndarray) -> np.ndarray:
        """Detect background using K-means color clustering."""
        # Reshape image for clustering
        pixels = image.reshape(-1, 3)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=self.color_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_
        
        # Find background clusters (assume corners are background)
        h, w = image.shape[:2]
        corner_pixels = np.concatenate([
            pixels[:w],  # Top row
            pixels[-w:],  # Bottom row
            pixels[::w],  # Left column
            pixels[w-1::w]  # Right column
        ])
        
        corner_labels = kmeans.predict(corner_pixels)
        background_clusters = np.bincount(corner_labels).argsort()[-2:]  # Top 2 most common
        
        # Create mask
        mask = np.zeros(h * w, dtype=np.uint8)
        for cluster_id in background_clusters:
            cluster_mask = (labels == cluster_id)
            mask[cluster_mask] = 255
        
        return mask.reshape(h, w)
    
    def _corner_sampling_detection(self, image: np.ndarray) -> np.ndarray:
        """Detect background by sampling corner regions."""
        h, w = image.shape[:2]
        
        # Sample corner regions (10% of image size)
        corner_size = min(h, w) // 10
        corners = [
            image[:corner_size, :corner_size],  # Top-left
            image[:corner_size, -corner_size:],  # Top-right
            image[-corner_size:, :corner_size],  # Bottom-left
            image[-corner_size:, -corner_size:]  # Bottom-right
        ]
        
        # Calculate average background color
        bg_colors = []
        for corner in corners:
            avg_color = np.mean(corner.reshape(-1, 3), axis=0)
            bg_colors.append(avg_color)
        
        bg_color = np.mean(bg_colors, axis=0)
        
        # Create mask based on color similarity
        color_diff = np.sqrt(np.sum((image.astype(float) - bg_color) ** 2, axis=2))
        mask = (color_diff <= self.tolerance).astype(np.uint8) * 255
        
        return mask
    
    def _dither_pattern_detection(self, image: np.ndarray) -> np.ndarray:
        """Detect and handle dithered patterns."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect checkerboard patterns using template matching
        h, w = gray.shape
        
        # Create checkerboard templates
        templates = []
        for size in [2, 3, 4]:
            template = np.kron([[0, 1], [1, 0]], np.ones((size, size))) * 255
            templates.append(template.astype(np.uint8))
        
        dither_mask = np.zeros((h, w), dtype=np.uint8)
        
        for template in templates:
            if template.shape[0] < h and template.shape[1] < w:
                result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                threshold = 0.6
                locations = np.where(result >= threshold)
                
                for pt in zip(*locations[::-1]):
                    dither_mask[pt[1]:pt[1]+template.shape[0], 
                               pt[0]:pt[0]+template.shape[1]] = 255
        
        return dither_mask
    
    def _combine_masks(self, masks: List[np.ndarray], image: np.ndarray) -> np.ndarray:
        """Combine multiple masks using weighted voting."""
        if not masks:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Normalize masks to 0-1 range
        normalized_masks = []
        for mask in masks:
            norm_mask = mask.astype(float) / 255.0
            normalized_masks.append(norm_mask)
        
        # Weighted combination
        weights = [0.3, 0.3, 0.25, 0.15]  # Adjust based on number of masks
        weights = weights[:len(normalized_masks)]
        weights = np.array(weights) / np.sum(weights)  # Normalize weights
        
        combined = np.zeros_like(normalized_masks[0])
        for mask, weight in zip(normalized_masks, weights):
            combined += mask * weight
        
        # Convert back to 0-255 range
        return (combined * 255).astype(np.uint8)
    
    def _refine_edges(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Apply morphological operations to refine mask edges."""
        # Remove small noise
        kernel_small = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Fill small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
        
        # Smooth edges with closing operation
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_smooth, iterations=1)
        
        # Apply Gaussian blur for softer edges
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        
        return mask
    
    def _apply_foreground_bias(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Apply bias towards preserving foreground elements."""
        if self.foreground_bias <= 0.5:
            return mask
        
        # Calculate image complexity (more complex areas are likely foreground)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Edge density as complexity measure
        edges = cv2.Canny(gray, 50, 150)
        edge_density = cv2.blur(edges.astype(float), (15, 15))
        edge_density = edge_density / edge_density.max()
        
        # Variance as complexity measure
        variance = cv2.Laplacian(gray, cv2.CV_64F)
        variance = np.abs(variance)
        variance = cv2.blur(variance, (15, 15))
        variance = variance / variance.max() if variance.max() > 0 else variance
        
        # Combine complexity measures
        complexity = (edge_density + variance) / 2.0
        
        # Apply bias: reduce background probability in complex areas
        bias_strength = (self.foreground_bias - 0.5) * 2.0  # Scale to 0-1
        mask_float = mask.astype(float) / 255.0
        
        # Reduce background mask in complex areas
        mask_float = mask_float * (1.0 - complexity * bias_strength)
        
        return (mask_float * 255).astype(np.uint8)
    
    def _make_binary_mask(self, mask: np.ndarray, threshold: int = 128) -> np.ndarray:
        """
        Convert grayscale mask to binary mask to eliminate semi-transparency.
        
        Args:
            mask: Input grayscale mask (0-255)
            threshold: Threshold value for binarization (default: 128)
            
        Returns:
            Binary mask with only 0 (transparent) or 255 (opaque) values
        """
        # Apply binary threshold
        _, binary_mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
        
        # Optional: Apply morphological operations to clean up the binary mask
        kernel = np.ones((3, 3), np.uint8)
        
        # Remove small noise (opening)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Fill small holes (closing)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return binary_mask
    
    def auto_adjust_parameters(self, image: np.ndarray) -> dict:
        """
        Automatically adjust parameters based on image analysis.
        
        Args:
            image: Input image for analysis
            
        Returns:
            Dictionary of adjusted parameters
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Analyze image characteristics
        # 1. Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        
        # 2. Color variance
        color_variance = np.var(image.reshape(-1, 3), axis=0).mean()
        
        # 3. Contrast
        contrast = gray.std()
        
        # Adjust parameters based on analysis
        adjustments = {}
        
        # High edge density -> increase edge sensitivity
        if edge_density > 0.1:
            adjustments['edge_sensitivity'] = min(0.9, self.edge_sensitivity + 0.1)
        elif edge_density < 0.05:
            adjustments['edge_sensitivity'] = max(0.6, self.edge_sensitivity - 0.1)
        
        # High color variance -> increase tolerance
        if color_variance > 1000:
            adjustments['tolerance'] = min(50, self.tolerance + 10)
        elif color_variance < 500:
            adjustments['tolerance'] = max(20, self.tolerance - 5)
        
        # Low contrast -> increase foreground bias
        if contrast < 30:
            adjustments['foreground_bias'] = min(0.8, self.foreground_bias + 0.1)
        
        return adjustments