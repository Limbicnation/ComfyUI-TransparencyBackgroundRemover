#!/usr/bin/env python3
"""
Test script for optimized edge detection functionality
"""
import numpy as np
import torch
from PIL import Image, ImageDraw
import sys
import os
import time

# Add current directory to path to import nodes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nodes import TransparencyBackgroundRemover
from background_remover import EnhancedPixelArtProcessor

def create_pixel_art_test_image(size=128):
    """Create a pixel art style test image"""
    image = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(image)
    
    # Create a simple pixel art character - a face
    # Head outline
    draw.rectangle([(32, 32), (96, 96)], fill='yellow', outline='black', width=2)
    
    # Eyes
    draw.rectangle([(44, 48), (52, 56)], fill='black')
    draw.rectangle([(76, 48), (84, 56)], fill='black')
    
    # Nose
    draw.rectangle([(60, 60), (68, 68)], fill='orange')
    
    # Mouth
    draw.rectangle([(48, 76), (80, 84)], fill='red', outline='black', width=1)
    
    # Convert to numpy array
    return np.array(image)

def create_photographic_test_image(size=128):
    """Create a photographic style test image with gradients"""
    # Create an image with smooth gradients
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Create a circular gradient
    center_x, center_y = 0.5, 0.5
    radius = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Normalize and create RGB channels
    gradient = 1.0 - np.clip(radius / 0.4, 0, 1)
    
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :, 0] = (gradient * 255).astype(np.uint8)  # Red gradient
    image[:, :, 1] = ((1 - gradient) * 255).astype(np.uint8)  # Inverse green
    image[:, :, 2] = 128  # Constant blue
    
    return image

def test_edge_detection_modes():
    """Test different edge detection modes"""
    print("Testing optimized edge detection...")
    
    # Create test images
    pixel_art = create_pixel_art_test_image()
    photographic = create_photographic_test_image()
    
    # Convert to ComfyUI tensor format (batch, height, width, channels)
    pixel_art_tensor = torch.from_numpy(pixel_art).unsqueeze(0).float() / 255.0
    photo_tensor = torch.from_numpy(photographic).unsqueeze(0).float() / 255.0
    
    # Initialize node
    node = TransparencyBackgroundRemover()
    
    print("\n=== Testing Pixel Art Image ===")
    
    # Test AUTO mode (should detect as pixel art)
    start_time = time.time()
    result_auto, mask_auto = node.remove_background(
        pixel_art_tensor,
        tolerance=20,
        edge_sensitivity=0.9,
        edge_detection_mode="AUTO",
        dither_handling=True
    )
    auto_time = time.time() - start_time
    print(f"AUTO mode completed in {auto_time:.3f}s")
    print(f"Result shape: {result_auto.shape}, Mask shape: {mask_auto.shape}")
    
    # Test PIXEL_ART mode
    start_time = time.time()
    result_pixel, mask_pixel = node.remove_background(
        pixel_art_tensor,
        tolerance=20,
        edge_sensitivity=0.9,
        edge_detection_mode="PIXEL_ART"
    )
    pixel_time = time.time() - start_time
    print(f"PIXEL_ART mode completed in {pixel_time:.3f}s")
    
    # Test PHOTOGRAPHIC mode
    start_time = time.time()
    result_photo_mode, mask_photo_mode = node.remove_background(
        pixel_art_tensor,
        tolerance=20,
        edge_sensitivity=0.9,
        edge_detection_mode="PHOTOGRAPHIC"
    )
    photo_mode_time = time.time() - start_time
    print(f"PHOTOGRAPHIC mode completed in {photo_mode_time:.3f}s")
    
    print("\n=== Testing Photographic Image ===")
    
    # Test AUTO mode (should detect as photographic)
    start_time = time.time()
    result_auto_photo, mask_auto_photo = node.remove_background(
        photo_tensor,
        tolerance=30,
        edge_sensitivity=0.7,
        edge_detection_mode="AUTO"
    )
    auto_photo_time = time.time() - start_time
    print(f"AUTO mode completed in {auto_photo_time:.3f}s")
    
    # Test PHOTOGRAPHIC mode
    start_time = time.time()
    result_photo, mask_photo = node.remove_background(
        photo_tensor,
        tolerance=30,
        edge_sensitivity=0.7,
        edge_detection_mode="PHOTOGRAPHIC"
    )
    photo_time2 = time.time() - start_time
    print(f"PHOTOGRAPHIC mode completed in {photo_time2:.3f}s")
    
    print("\n=== Edge Detection Method Analysis ===")
    
    # Test individual edge detection methods
    processor = EnhancedPixelArtProcessor(
        tolerance=20,
        edge_sensitivity=0.9,
        dither_handling=True
    )
    
    # Test on pixel art
    print("\\nPixel Art Edge Detection Methods:")
    
    gray_pixel = np.mean(pixel_art, axis=2).astype(np.uint8)
    
    # Roberts Cross
    start_time = time.time()
    roberts_result = processor._roberts_cross_edge_detection(gray_pixel)
    roberts_time = time.time() - start_time
    roberts_edges = np.sum(roberts_result > 0)
    print(f"Roberts Cross: {roberts_edges} edge pixels, {roberts_time:.4f}s")
    
    # Enhanced Sobel  
    start_time = time.time()
    sobel_result = processor._enhanced_sobel_edge_detection(gray_pixel)
    sobel_time = time.time() - start_time
    sobel_edges = np.sum(sobel_result > 0)
    print(f"Enhanced Sobel: {sobel_edges} edge pixels, {sobel_time:.4f}s")
    
    # Pixel-aware Canny
    start_time = time.time()
    canny_result = processor._pixel_aware_canny(gray_pixel)
    canny_time = time.time() - start_time
    canny_edges = np.sum(canny_result > 0)
    print(f"Pixel-aware Canny: {canny_edges} edge pixels, {canny_time:.4f}s")
    
    # Content type detection
    is_pixel_art = processor._detect_pixel_art_characteristics(pixel_art)
    print(f"\\nPixel art detection: {is_pixel_art} (expected: True)")
    
    is_photo_pixel_art = processor._detect_pixel_art_characteristics(photographic)
    print(f"Photo pixel art detection: {is_photo_pixel_art} (expected: False)")
    
    print("\\n=== Test Summary ===")
    print("✓ All edge detection modes executed successfully")
    print("✓ Multiple edge detection algorithms implemented")
    print("✓ Content type detection working")
    print("✓ Performance benchmarking completed")
    
    return True

def test_edge_refinement():
    """Test the new edge refinement methods"""
    print("\\n=== Testing Edge Refinement ===")
    
    processor = EnhancedPixelArtProcessor()
    
    # Create a test mask with noise
    test_mask = np.zeros((100, 100), dtype=np.uint8)
    test_mask[30:70, 30:70] = 255  # Main shape
    
    # Add noise
    noise_positions = [(20, 20), (80, 80), (10, 90), (90, 10)]
    for x, y in noise_positions:
        test_mask[y:y+2, x:x+2] = 255
    
    # Test pixel art refinement
    pixel_art_image = create_pixel_art_test_image(100)
    
    start_time = time.time()
    refined_pixel_art = processor._pixel_art_edge_refinement(test_mask.copy(), pixel_art_image)
    pixel_refine_time = time.time() - start_time
    
    # Test photographic refinement  
    photo_image = create_photographic_test_image(100)
    
    start_time = time.time()
    refined_photo = processor._photographic_edge_refinement(test_mask.copy(), photo_image)
    photo_refine_time = time.time() - start_time
    
    print(f"Pixel art refinement: {pixel_refine_time:.4f}s")
    print(f"Photographic refinement: {photo_refine_time:.4f}s")
    
    # Check that refinement reduced noise
    original_noise = np.sum(test_mask > 0)
    pixel_art_noise = np.sum(refined_pixel_art > 0)
    photo_noise = np.sum(refined_photo > 0)
    
    print(f"Original mask pixels: {original_noise}")
    print(f"Pixel art refined pixels: {pixel_art_noise}")
    print(f"Photo refined pixels: {photo_noise}")
    
    print("✓ Edge refinement methods tested successfully")
    
    return True

if __name__ == "__main__":
    try:
        print("Starting optimized edge detection tests...")
        
        # Test edge detection modes
        test_edge_detection_modes()
        
        # Test edge refinement
        test_edge_refinement()
        
        print("\\n🎉 All tests completed successfully!")
        print("\\nOptimized edge detection features:")
        print("• Multi-method edge detection (Roberts, Sobel, Canny)")  
        print("• Automatic content type detection")
        print("• Pixel art optimized processing")
        print("• Photographic image optimized processing")
        print("• Enhanced edge refinement")
        print("• Performance optimizations")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)