#!/usr/bin/env python3
"""
Standalone test script for Power-of-8 scaling functionality
without ComfyUI dependencies.
"""
import sys
import os
import numpy as np
from PIL import Image

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock ComfyUI modules for testing
class MockFolderPaths:
    pass

class MockComfyUtils:
    pass

class MockComfy:
    utils = MockComfyUtils()

# Mock the imports
sys.modules['folder_paths'] = MockFolderPaths()
sys.modules['comfy'] = MockComfy()
sys.modules['comfy.utils'] = MockComfyUtils()

def test_parsing():
    """Test output size parsing"""
    print("Testing output size parsing...")
    
    try:
        from nodes import TransparencyBackgroundRemover
        node = TransparencyBackgroundRemover()
        
        # Test valid sizes
        test_cases = [
            ("ORIGINAL", None),
            ("64x64", (64, 64)),
            ("512x512", (512, 512)),
            ("1024x1024", (1024, 1024)),
            ("2048x2048", (2048, 2048))
        ]
        
        for input_str, expected in test_cases:
            result = node.parse_output_size(input_str)
            if result == expected:
                print(f"✅ {input_str} -> {result}")
            else:
                print(f"❌ {input_str} -> {result}, expected {expected}")
                return False
        
        # Test invalid format
        try:
            node.parse_output_size("invalid")
            print("❌ Should have raised ValueError for invalid format")
            return False
        except ValueError:
            print("✅ Correctly rejected invalid format")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_scaling_calculation():
    """Test scaling factor calculation"""
    print("Testing scaling factor calculation...")
    
    try:
        from nodes import TransparencyBackgroundRemover
        node = TransparencyBackgroundRemover()
        
        test_cases = [
            # (current_size, target_size, expected_scale_approx)
            ((64, 64), (128, 128), 2.0),
            ((128, 128), (256, 256), 2.0),
            ((100, 100), (512, 512), 5.12),
            ((256, 256), (128, 128), 0.5),
            ((512, 512), (512, 512), 1.0)
        ]
        
        for current, target, expected_scale in test_cases:
            result = node.calculate_scaling_factor(current, target)
            if abs(result - expected_scale) < 0.01:
                print(f"✅ {current} -> {target}: scale={result:.2f}")
            else:
                print(f"❌ {current} -> {target}: scale={result:.2f}, expected≈{expected_scale}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_input_types():
    """Test that INPUT_TYPES includes new parameters"""
    print("Testing INPUT_TYPES configuration...")
    
    try:
        from nodes import TransparencyBackgroundRemover
        node = TransparencyBackgroundRemover()
        inputs = node.INPUT_TYPES()
        
        # Check output_size parameter
        if 'output_size' in inputs['required']:
            sizes = inputs['required']['output_size'][0]
            expected_sizes = ["ORIGINAL", "64x64", "96x96", "128x128", "256x256", 
                            "512x512", "768x768", "1024x1024", "1280x1280", 
                            "1536x1536", "1792x1792", "2048x2048"]
            
            if sizes == expected_sizes:
                print("✅ output_size parameter configured correctly")
            else:
                print(f"❌ output_size sizes mismatch")
                print(f"   Expected: {expected_sizes}")
                print(f"   Got: {sizes}")
                return False
        else:
            print("❌ output_size parameter missing")
            return False
        
        # Check scaling_method parameter
        if 'scaling_method' in inputs['required']:
            methods = inputs['required']['scaling_method'][0]
            expected_methods = ["NEAREST", "BILINEAR", "BICUBIC", "LANCZOS"]
            if methods == expected_methods:
                print("✅ scaling_method parameter configured correctly")
            else:
                print(f"❌ scaling_method wrong: {methods}")
                return False
        else:
            print("❌ scaling_method parameter missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_interpolation_methods():
    """Test interpolation methods"""
    print("Testing interpolation methods...")
    
    try:
        from nodes import TransparencyBackgroundRemover
        
        # Create test image
        test_array = np.zeros((64, 64, 3), dtype=np.uint8)
        test_array[16:48, 16:48] = [255, 0, 0]  # Red square
        test_pil = Image.fromarray(test_array)
        
        node = TransparencyBackgroundRemover()
        
        # Test all interpolation methods
        methods = ["NEAREST", "BILINEAR", "BICUBIC", "LANCZOS"]
        target_size = (128, 128)
        
        for method in methods:
            result = node.intelligent_scale(test_pil, target_size, method)
            if result.size == target_size:
                print(f"✅ {method} scaling works correctly")
            else:
                print(f"❌ {method} scaling failed: expected {target_size}, got {result.size}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_power_of_8_validation():
    """Verify all sizes are powers/multiples of 8"""
    print("Testing power-of-8 validation...")
    
    sizes = [64, 96, 128, 256, 512, 768, 1024, 1280, 1536, 1792, 2048]
    
    for size in sizes:
        if size % 8 == 0:
            print(f"✅ {size} is multiple of 8")
        else:
            print(f"❌ {size} is NOT multiple of 8")
            return False
    
    return True

def main():
    """Run all tests"""
    print("🧪 Testing Power-of-8 Scaling Implementation (Standalone)")
    print("=" * 60)
    
    tests = [
        test_power_of_8_validation,
        test_input_types,
        test_parsing,
        test_scaling_calculation,
        test_interpolation_methods
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
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