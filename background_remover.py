import numpy as np
import cv2
import colorsys
from typing import Tuple, Optional, List

# Lazy import sklearn with graceful fallback
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    KMeans = None


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

        # Warn if sklearn is not available and color clustering is requested
        if not SKLEARN_AVAILABLE and self.color_clusters > 0:
            print("⚠️  Warning: scikit-learn not installed. Color clustering detection disabled.")
            print("   For best results, install dependencies: pip install scikit-learn")
            print("   Or run install.bat (Windows) / install.sh (Linux/Mac) in the custom node directory.")
        
    def remove_background_advanced(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced background removal with multiple detection methods and performance optimizations.
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            RGBA image with transparent background
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        # Performance optimization: Downscale for large images during processing
        h, w = image.shape[:2]
        scale_factor = 1.0
        processed_image = image
        
        # Downscale if image is very large (> 1024px on any side) for faster processing
        max_dimension = max(h, w)
        if max_dimension > 1024:
            scale_factor = 1024.0 / max_dimension
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            processed_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create RGBA output (full size)
        rgba_result = np.zeros((h, w, 4), dtype=np.uint8)
        rgba_result[:, :, :3] = image
        
        # Generate initial mask using multiple methods on processed image
        masks = []
        
        # Method 1: Edge-based detection
        edge_mask = self._edge_based_detection(processed_image)
        masks.append(edge_mask)
        
        # Method 2: Color clustering (skip for very large images to save time)
        if max_dimension <= 1536 and SKLEARN_AVAILABLE:  # Only use clustering for reasonably sized images
            cluster_mask = self._color_clustering_detection(processed_image)
            masks.append(cluster_mask)
        elif max_dimension <= 1536 and not SKLEARN_AVAILABLE:
            # Info message only shown once per session (first time processor is created)
            pass  # Already warned in __init__
        
        # Method 3: Corner sampling
        corner_mask = self._corner_sampling_detection(processed_image)
        masks.append(corner_mask)
        
        # Method 4: Dither detection (if enabled and image not too large)
        if self.dither_handling and max_dimension <= 1024:
            dither_mask = self._dither_pattern_detection(processed_image)
            masks.append(dither_mask)
        
        # Combine masks with weighted voting
        final_mask = self._combine_masks(masks, processed_image)
        
        # Upscale mask back to original size if it was downscaled
        if scale_factor < 1.0:
            final_mask = cv2.resize(final_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Apply edge refinement if enabled (on full-size mask for better quality)
        if self.edge_refinement:
            final_mask = self._refine_edges(final_mask, image)
        
        # Apply foreground bias (on full-size for accuracy)
        final_mask = self._apply_foreground_bias(final_mask, image)
        
        # Convert to binary mask to eliminate semi-transparency
        final_mask = self._make_binary_mask(final_mask, self.binary_threshold)
        
        # Debug: Ensure mask has some variation
        unique_values = np.unique(final_mask)
        if len(unique_values) == 1:
            # If mask is uniform, force some background detection using corner sampling
            h, w = final_mask.shape
            corner_size = min(h, w) // 10
            if corner_size > 0:
                # Mark corners as background
                final_mask[:corner_size, :corner_size] = 255  # Top-left
                final_mask[:corner_size, -corner_size:] = 255  # Top-right
                final_mask[-corner_size:, :corner_size] = 255  # Bottom-left
                final_mask[-corner_size:, -corner_size:] = 255  # Bottom-right
        
        # Invert mask: background detected pixels (255) -> alpha=0 (transparent)
        #              foreground pixels (0) -> alpha=255 (opaque)
        alpha_mask = 255 - final_mask
        
        rgba_result[:, :, 3] = alpha_mask
        return rgba_result
    
    def _edge_based_detection(self, image: np.ndarray) -> np.ndarray:
        """Detect background using optimized edge analysis for pixel art."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Use pixel art optimized edge detection
        if self.dither_handling:  # Assume pixel art mode when dither handling is enabled
            return self._pixel_art_edge_detection(gray)
        else:
            return self._standard_edge_detection(gray)
    
    def _pixel_art_edge_detection(self, gray: np.ndarray) -> np.ndarray:
        """Optimized edge detection specifically for pixel art images."""
        h, w = gray.shape
        
        # Multi-method edge detection for pixel art
        edge_maps = []
        
        # Method 1: Roberts Cross-Gradient (excellent for sharp pixel edges)
        roberts_edges = self._roberts_cross_edge_detection(gray)
        edge_maps.append(roberts_edges)
        
        # Method 2: Enhanced Sobel (better noise resistance)
        sobel_edges = self._enhanced_sobel_edge_detection(gray)
        edge_maps.append(sobel_edges)
        
        # Method 3: Pixel-aware Canny with adaptive thresholds
        canny_edges = self._pixel_aware_canny(gray)
        edge_maps.append(canny_edges)
        
        # Combine edge maps with weighted voting
        combined_edges = self._combine_edge_maps(edge_maps)
        
        # Apply pixel-perfect morphological operations
        refined_edges = self._pixel_perfect_morphology(combined_edges)
        
        # Smart contour analysis for character shapes
        mask = self._smart_contour_analysis(refined_edges, gray.shape)
        
        return mask
    
    def _standard_edge_detection(self, gray: np.ndarray) -> np.ndarray:
        """Enhanced standard edge detection for photographic images."""
        h, w = gray.shape
        
        # Adaptive noise reduction based on image characteristics
        noise_level = self._estimate_noise_level(gray)
        
        # Scale-aware Gaussian blur
        blur_size = max(3, min(7, int(np.sqrt(h * w) / 200)))  # Dynamic blur size
        if blur_size % 2 == 0:  # Ensure odd kernel size
            blur_size += 1
        
        # Apply noise-adaptive preprocessing
        if noise_level > 15:  # High noise
            blurred = cv2.bilateralFilter(gray, blur_size, 75, 75)
        elif noise_level > 8:  # Medium noise
            blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        else:  # Low noise - preserve details
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Multi-scale Canny edge detection
        threshold_val = int(255 * (1.0 - self.edge_sensitivity))
        
        # Fine scale edges
        edges_fine = cv2.Canny(blurred, threshold_val // 3, threshold_val // 2)
        
        # Coarse scale edges
        coarse_blurred = cv2.GaussianBlur(gray, (blur_size + 2, blur_size + 2), 0)
        edges_coarse = cv2.Canny(coarse_blurred, threshold_val // 2, threshold_val)
        
        # Combine multi-scale edges
        combined_edges = cv2.bitwise_or(edges_fine, edges_coarse)
        
        # Adaptive morphological operations
        kernel_size = max(3, min(7, int(np.sqrt(h * w) / 300)))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        edges_dilated = cv2.dilate(combined_edges, kernel, iterations=2)
        
        # Enhanced contour analysis
        mask = self._enhanced_contour_analysis(edges_dilated, gray.shape)
        
        return mask
    
    def _roberts_cross_edge_detection(self, gray: np.ndarray) -> np.ndarray:
        """Roberts Cross-Gradient operator - ideal for sharp pixel art edges."""
        # Roberts Cross kernels
        roberts_cross_v = np.array([[1, 0], [0, -1]], dtype=np.float32)
        roberts_cross_h = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        
        # Apply Roberts operators
        vertical = cv2.filter2D(gray.astype(np.float32), -1, roberts_cross_v)
        horizontal = cv2.filter2D(gray.astype(np.float32), -1, roberts_cross_h)
        
        # Compute gradient magnitude
        magnitude = np.sqrt(vertical**2 + horizontal**2)
        
        # Normalize and threshold
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
        threshold = int(255 * (1.0 - self.edge_sensitivity) * 0.3)  # Roberts is more sensitive
        
        _, binary_edges = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)
        return binary_edges
    
    def _enhanced_sobel_edge_detection(self, gray: np.ndarray) -> np.ndarray:
        """Enhanced Sobel edge detection with adaptive parameters."""
        # Apply Sobel operators
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude and direction
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Normalize
        max_magnitude = magnitude.max()
        if max_magnitude > 0:
            magnitude = np.clip(magnitude / max_magnitude * 255, 0, 255).astype(np.uint8)
        else:
            magnitude = magnitude.astype(np.uint8)
        
        # Adaptive threshold based on edge sensitivity
        threshold = int(255 * (1.0 - self.edge_sensitivity) * 0.6)
        _, binary_edges = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)
        
        return binary_edges
    
    def _pixel_aware_canny(self, gray: np.ndarray) -> np.ndarray:
        """Pixel-aware Canny edge detection with minimal blur."""
        # Minimal blur to preserve pixel boundaries
        blurred = cv2.GaussianBlur(gray, (3, 3), 0.5)  # Reduced sigma
        
        # Adaptive thresholds
        threshold_val = int(255 * (1.0 - self.edge_sensitivity))
        low_threshold = max(30, threshold_val // 3)  # Ensure minimum threshold
        high_threshold = min(200, threshold_val)     # Cap maximum threshold
        
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        return edges
    
    def _combine_edge_maps(self, edge_maps: List[np.ndarray]) -> np.ndarray:
        """Combine multiple edge maps using weighted voting."""
        if not edge_maps:
            raise ValueError("Cannot combine an empty list of edge maps.")
        
        # Weights: Roberts (sharp edges), Sobel (noise resistance), Canny (completeness)
        weights = [0.4, 0.35, 0.25]
        weights = weights[:len(edge_maps)]
        
        # Normalize edge maps and combine
        combined = np.zeros_like(edge_maps[0], dtype=np.float32)
        for edge_map, weight in zip(edge_maps, weights):
            normalized = edge_map.astype(np.float32) / 255.0
            combined += normalized * weight
        
        # Threshold combined result
        _, binary_combined = cv2.threshold((combined * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
        return binary_combined
    
    def _pixel_perfect_morphology(self, edges: np.ndarray) -> np.ndarray:
        """Apply pixel-perfect morphological operations preserving sharp edges."""
        # Use minimal kernels to preserve pixel boundaries
        kernel_small = np.ones((2, 2), np.uint8)  # Smaller than standard 3x3
        
        # Light closing to connect nearby edges without over-smoothing
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        
        # Remove single-pixel noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        return opened
    
    def _smart_contour_analysis(self, edges: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """Smart contour analysis optimized for character shapes."""
        h, w = shape
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Dynamic area threshold based on image size
        min_area = max(50, (h * w) // 1000)  # Adaptive minimum area
        max_area = (h * w) * 0.8  # Max 80% of image
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Area filtering
            if area < min_area or area > max_area:
                continue
            
            # Aspect ratio filtering (reasonable for character shapes)
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect_ratio = float(cw) / ch if ch > 0 else 0
            
            # Allow wide range of aspect ratios but filter extreme cases
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                continue
            
            # Solidity filtering (shape complexity)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Keep reasonably solid shapes (not too fragmented)
            if solidity > 0.3:  # Allow some complexity for character details
                cv2.fillPoly(mask, [contour], 255)
        
        return mask
    
    def _enhanced_contour_analysis(self, edges: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """Enhanced contour analysis for photographic images."""
        h, w = shape
        
        # Dilate edges slightly for better contour detection
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours with hierarchy for nested shapes
        contours, hierarchy = cv2.findContours(edges_dilated, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # More sophisticated area thresholding
        image_area = h * w
        min_area = max(100, image_area // 2000)
        max_area = image_area * 0.7
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if min_area <= area <= max_area:
                # Check if this is an outer contour (not a hole)
                if hierarchy[0][i][3] == -1:  # No parent (outer contour)
                    cv2.fillPoly(mask, [contour], 255)
        
        return mask
    
    def _estimate_noise_level(self, gray: np.ndarray) -> float:
        """Estimate noise level in the image for adaptive preprocessing."""
        # Use Laplacian variance as noise estimate
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_level = laplacian.var()
        
        # Normalize to 0-100 scale
        return min(100, max(0, noise_level / 10))
    
    def _color_clustering_detection(self, image: np.ndarray) -> np.ndarray:
        """Detect background using K-means color clustering with performance optimizations."""
        # Return neutral mask if sklearn is not available
        if not SKLEARN_AVAILABLE:
            h, w = image.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

        h, w = image.shape[:2]

        # Performance optimization: Sample pixels for very large images
        if h * w > 500000:  # For images larger than ~500k pixels
            # Sample every 4th pixel in both dimensions
            sampled_image = image[::4, ::4]
            pixels = sampled_image.reshape(-1, 3)
        else:
            pixels = image.reshape(-1, 3)
        
        # Apply K-means clustering with optimized parameters
        kmeans = KMeans(
            n_clusters=self.color_clusters, 
            random_state=42, 
            n_init=5,  # Reduced from 10 for speed
            max_iter=50,  # Limit iterations for speed
            tol=1e-3  # Slightly looser convergence tolerance
        )
        
        if h * w > 500000:
            # Fit on sampled data, then predict on full image
            kmeans.fit(pixels)
            full_pixels = image.reshape(-1, 3)
            labels = kmeans.predict(full_pixels)
        else:
            labels = kmeans.fit_predict(pixels)
        
        # Find background clusters (assume corners are background)
        # Use original image dimensions for corner sampling
        full_pixels = image.reshape(-1, 3)
        corner_pixels = np.concatenate([
            full_pixels[:w],  # Top row
            full_pixels[-w:],  # Bottom row
            full_pixels[::w],  # Left column
            full_pixels[w-1::w]  # Right column
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
        
        # Apply conservative threshold (> 0.6 for background detection)
        binary_combined = (combined > 0.6).astype(float)
        
        # Convert back to 0-255 range
        return (binary_combined * 255).astype(np.uint8)
    
    def _refine_edges(self, mask: np.ndarray, image: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply optimized edge refinement based on content type."""
        if image is not None:
            # Determine if this looks like pixel art
            is_pixel_art = self._detect_pixel_art_characteristics(image)
            
            if is_pixel_art and self.dither_handling:
                return self._pixel_art_edge_refinement(mask, image)
            else:
                return self._photographic_edge_refinement(mask, image)
        else:
            # Fallback to conservative refinement
            return self._conservative_edge_refinement(mask)
    
    def _detect_pixel_art_characteristics(self, image: np.ndarray) -> bool:
        """Detect if image has pixel art characteristics."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        
        # Check for low resolution (common in pixel art)
        if h <= 128 or w <= 128:
            return True
        
        # Check for limited color palette
        unique_colors = len(np.unique(image.reshape(-1, image.shape[-1] if len(image.shape) == 3 else 1), axis=0))
        total_pixels = h * w
        color_density = unique_colors / total_pixels
        
        # Pixel art typically has low color density
        if color_density < 0.1:  # Less than 10% unique colors
            return True
        
        # Check for sharp edges (no anti-aliasing)
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges > 0)
        
        if edge_pixels > 0:
            # Check gradient sharpness around edges
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # High gradient values suggest sharp, non-antialiased edges
            avg_gradient = np.mean(gradient_magnitude[edges > 0]) if edge_pixels > 0 else 0
            
            if avg_gradient > 30:  # Sharp edges threshold
                return True
        
        return False
    
    def _pixel_art_edge_refinement(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Pixel art specific edge refinement that preserves sharp boundaries."""
        # Use minimal kernels to preserve pixel-perfect edges
        kernel_tiny = np.ones((2, 2), np.uint8)
        kernel_small = np.ones((3, 3), np.uint8)
        
        # Very light morphological operations
        # Remove single-pixel noise without affecting shape
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_tiny, iterations=1)
        
        # Connect very close pixel groups (1-pixel gaps)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_tiny, iterations=1)
        
        # Final light closing to solidify shapes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        
        # NO Gaussian blur for pixel art - preserves sharp edges
        
        return mask
    
    def _photographic_edge_refinement(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Enhanced edge refinement for photographic content."""
        h, w = mask.shape
        image_area = h * w
        
        # Scale-adaptive kernel sizes
        small_kernel_size = max(3, min(5, int(np.sqrt(image_area) / 300)))
        large_kernel_size = max(5, min(9, int(np.sqrt(image_area) / 200)))
        
        # Ensure odd kernel sizes
        if small_kernel_size % 2 == 0:
            small_kernel_size += 1
        if large_kernel_size % 2 == 0:
            large_kernel_size += 1
        
        kernel_small = np.ones((small_kernel_size, small_kernel_size), np.uint8)
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (large_kernel_size, large_kernel_size))
        
        # Progressive refinement
        # 1. Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # 2. Fill small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
        
        # 3. Smooth edges with larger kernel
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=1)
        
        # 4. Apply bilateral filtering for edge-preserving smoothing
        # Convert to float for bilateral filter
        mask_float = mask.astype(np.float32) / 255.0
        
        # Bilateral filter preserves edges while smoothing noise
        bilateral_filtered = cv2.bilateralFilter(
            mask_float, 
            d=5,  # Neighborhood diameter
            sigmaColor=0.1,  # Color similarity threshold
            sigmaSpace=5    # Coordinate space threshold
        )
        
        # Convert back and apply final light Gaussian blur
        mask = (bilateral_filtered * 255).astype(np.uint8)
        mask = cv2.GaussianBlur(mask, (3, 3), 0.5)  # Light blur
        
        return mask
    
    def _conservative_edge_refinement(self, mask: np.ndarray) -> np.ndarray:
        """Conservative edge refinement when image data is not available."""
        # Minimal processing to avoid assumptions about content type
        kernel_small = np.ones((3, 3), np.uint8)
        
        # Basic noise removal and hole filling
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
        
        # Very light smoothing
        mask = cv2.GaussianBlur(mask, (3, 3), 0.5)
        
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
        max_edge = edge_density.max()
        edge_density = edge_density / max_edge if max_edge > 0 else edge_density
        
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
        
        # Ensure valid values
        mask_float = np.clip(mask_float, 0.0, 1.0)
        
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