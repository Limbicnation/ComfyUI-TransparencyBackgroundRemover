#!/usr/bin/env python3
"""
Standalone test script for the background remover functionality
without ComfyUI dependencies.
"""
import sys
import os
import numpy as np
from PIL import Image

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_image(width=64, height=64):
    """Create a simple test image with distinct patterns"""
    # Create a simple pattern with clear foreground/background
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add background (blue)
    image[:, :] = [50, 100, 200]
    
    # Add foreground object (red square in center)
    center_size = min(width, height) // 3
    start_x = (width - center_size) // 2
    start_y = (height - center_size) // 2
    image[start_y:start_y+center_size, start_x:start_x+center_size] = [200, 50, 50]
    
    return image

def test_background_remover():
    """Test the EnhancedPixelArtProcessor directly"""
    print("Testing EnhancedPixelArtProcessor...")
    
    try:
        from background_remover import EnhancedPixelArtProcessor
        
        # Create test image
        test_image = create_test_image(128, 128)
        print(f"✅ Created test image: {test_image.shape}")
        
        # Initialize processor
        processor = EnhancedPixelArtProcessor(
            tolerance=30,
            edge_sensitivity=0.8,
            color_clusters=8,
            foreground_bias=0.7,
            edge_refinement=True,
            dither_handling=True,
            binary_threshold=128
        )
        print("✅ Initialized processor")
        
        # Process image
        result = processor.remove_background_advanced(test_image)
        print(f"✅ Processed image: {result.shape}")
        
        # Verify result has alpha channel
        if result.shape[2] == 4:
            print("✅ Result has alpha channel")
        else:
            print(f"❌ Expected 4 channels, got {result.shape[2]}")
            return False
        
        # Check that some pixels are transparent
        alpha_channel = result[:, :, 3]
        transparent_pixels = np.sum(alpha_channel == 0)
        opaque_pixels = np.sum(alpha_channel == 255)
        
        print(f"✅ Transparent pixels: {transparent_pixels}")
        print(f"✅ Opaque pixels: {opaque_pixels}")
        
        if transparent_pixels > 0 and opaque_pixels > 0:
            print("✅ Background removal successful - has both transparent and opaque pixels")
        else:
            print("❌ Background removal may not be working correctly")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_auto_adjustment():
    """Test auto-adjustment feature"""
    print("Testing auto-adjustment...")
    
    try:
        from background_remover import EnhancedPixelArtProcessor
        
        # Create test image
        test_image = create_test_image(64, 64)
        
        # Initialize processor
        processor = EnhancedPixelArtProcessor()
        
        # Test auto-adjustment
        adjustments = processor.auto_adjust_parameters(test_image)
        
        if adjustments:
            print(f"✅ Auto-adjustments made: {adjustments}")
        else:
            print("✅ No auto-adjustments needed")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_scaling_methods():
    """Test different scaling methods"""
    print("Testing scaling methods...")
    
    try:
        # Test PIL resampling methods
        test_image = create_test_image(64, 64)
        pil_image = Image.fromarray(test_image)
        
        methods = {
            "NEAREST": Image.Resampling.NEAREST,
            "BILINEAR": Image.Resampling.BILINEAR,
            "BICUBIC": Image.Resampling.BICUBIC,
            "LANCZOS": Image.Resampling.LANCZOS
        }
        
        for method_name, method in methods.items():
            scaled = pil_image.resize((128, 128), method)
            print(f"✅ {method_name} scaling: {scaled.size}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_performance_large_image():
    """Test performance optimizations with large image"""
    print("Testing performance with large image...")
    
    try:
        from background_remover import EnhancedPixelArtProcessor
        import time
        
        # Create large test image
        large_image = create_test_image(1200, 1200)
        print(f"✅ Created large test image: {large_image.shape}")
        
        # Initialize processor
        processor = EnhancedPixelArtProcessor()
        
        # Process with timing
        start_time = time.time()
        result = processor.remove_background_advanced(large_image)
        processing_time = time.time() - start_time
        
        print(f"✅ Processed large image in {processing_time:.3f}s")
        print(f"✅ Result shape: {result.shape}")
        
        # Performance should be reasonable (< 5 seconds for 1200x1200)
        if processing_time < 5.0:
            print("✅ Performance is acceptable")
        else:
            print(f"⚠️  Performance slower than expected: {processing_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def main():
    """Run all standalone tests"""
    print("🧪 Running Standalone Background Remover Tests")
    print("=" * 50)
    
    tests = [
        test_background_remover,
        test_auto_adjustment,
        test_scaling_methods,
        test_performance_large_image
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print()
        if test():
            passed += 1
        print("-" * 30)
    
    print()
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests PASSED!")
        return True
    else:
        print("❌ Some tests FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)