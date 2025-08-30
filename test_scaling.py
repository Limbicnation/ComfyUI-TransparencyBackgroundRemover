#!/usr/bin/env python3
"""
Test script for NEAREST NEIGHBOR scaling functionality
"""
import numpy as np
import torch
from PIL import Image
import sys
import os

# Add current directory to path to import nodes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nodes import TransparencyBackgroundRemover

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

def test_input_validation():
    """Test minimum size validation"""
    print("Testing input validation...")
    
    node = TransparencyBackgroundRemover()
    
    # Test with image smaller than 64x64
    small_image = create_test_image(32, 32)
    small_tensor = torch.from_numpy(small_image).unsqueeze(0).float() / 255.0
    
    try:
        node.remove_background(small_tensor)
        print("❌ FAILED: Should have raised ValueError for small image")
        return False
    except ValueError as e:
        if "64x64" in str(e):
            print("✅ PASSED: Correctly rejected small image")
        else:
            print(f"❌ FAILED: Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"❌ FAILED: Unexpected error: {e}")
        return False
    
    return True

def test_scaling_functionality():
    """Test NEAREST scaling functionality"""
    print("Testing scaling functionality...")
    
    node = TransparencyBackgroundRemover()
    
    # Test with valid 64x64 image
    test_image = create_test_image(64, 64)
    test_tensor = torch.from_numpy(test_image).unsqueeze(0).float() / 255.0
    
    try:
        # Test 1x scaling (no scaling)
        result_1x, mask_1x = node.remove_background(
            test_tensor, 
            scale_factor=1,
            scaling_method="NEAREST"
        )
        print(f"✅ 1x scaling: {result_1x.shape}")
        
        # Test 2x scaling
        result_2x, mask_2x = node.remove_background(
            test_tensor,
            scale_factor=2,
            scaling_method="NEAREST"
        )
        print(f"✅ 2x scaling: {result_2x.shape}")
        
        # Verify dimensions are doubled
        expected_height = test_tensor.shape[1] * 2
        expected_width = test_tensor.shape[2] * 2
        
        if result_2x.shape[1] == expected_height and result_2x.shape[2] == expected_width:
            print("✅ PASSED: 2x scaling dimensions correct")
        else:
            print(f"❌ FAILED: Expected {expected_height}x{expected_width}, got {result_2x.shape[1]}x{result_2x.shape[2]}")
            return False
            
        # Test 4x scaling
        result_4x, mask_4x = node.remove_background(
            test_tensor,
            scale_factor=4,
            scaling_method="NEAREST"
        )
        print(f"✅ 4x scaling: {result_4x.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: Scaling test error: {e}")
        return False

def test_nearest_scale_function():
    """Test the nearest_scale function directly"""
    print("Testing nearest_scale function...")
    
    node = TransparencyBackgroundRemover()
    
    # Create a small PIL image
    test_array = create_test_image(8, 8)
    test_pil = Image.fromarray(test_array)
    
    try:
        # Test 1x (should return same image)
        result_1x = node.nearest_scale(test_pil, 1)
        if result_1x.size == test_pil.size:
            print("✅ 1x scaling preserves size")
        else:
            print("❌ FAILED: 1x scaling changed size")
            return False
        
        # Test 2x scaling
        result_2x = node.nearest_scale(test_pil, 2)
        expected_size = (test_pil.width * 2, test_pil.height * 2)
        if result_2x.size == expected_size:
            print("✅ 2x scaling correct size")
        else:
            print(f"❌ FAILED: Expected {expected_size}, got {result_2x.size}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: nearest_scale test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing NEAREST NEIGHBOR Scaling Implementation")
    print("=" * 50)
    
    tests = [
        test_input_validation,
        test_nearest_scale_function,
        test_scaling_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
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