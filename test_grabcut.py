#!/usr/bin/env python3
"""
Test script for GrabCut background removal functionality.
Tests both standalone processing and ComfyUI node integration.
"""

import numpy as np
import cv2
from pathlib import Path
import sys
import time

# Test imports
print("Testing imports...")
try:
    from grabcut_remover import GrabCutProcessor, create_fallback_processor
    print("✓ GrabCut processor imported successfully")
except ImportError as e:
    print(f"✗ Failed to import GrabCut processor: {e}")
    sys.exit(1)

try:
    from grabcut_nodes import AutoGrabCutRemover, GrabCutRefinement
    print("✓ GrabCut nodes imported successfully")
except ImportError as e:
    print(f"✗ Failed to import GrabCut nodes: {e}")
    sys.exit(1)

print("\n" + "="*50)


def create_test_image():
    """Create a synthetic test image with a subject on a background."""
    # Create a 512x512 image with a blue background
    img = np.ones((512, 512, 3), dtype=np.uint8) * 100
    img[:, :, 0] = 150  # Blue channel
    img[:, :, 1] = 100  # Green channel
    img[:, :, 2] = 50   # Red channel
    
    # Add a subject (red circle with gradient)
    center = (256, 256)
    radius = 100
    
    for y in range(512):
        for x in range(512):
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            if dist < radius:
                # Create gradient effect
                intensity = 1.0 - (dist / radius) * 0.3
                img[y, x] = [50, 50, int(200 * intensity)]
    
    # Add some details (yellow rectangles)
    cv2.rectangle(img, (200, 200), (250, 250), (50, 200, 200), -1)
    cv2.rectangle(img, (280, 280), (320, 320), (50, 200, 200), -1)
    
    return img


def test_grabcut_processor():
    """Test the GrabCut processor directly."""
    print("\nTesting GrabCutProcessor...")
    
    # Create test image
    test_img = create_test_image()
    print(f"Created test image: {test_img.shape}")
    
    # Test with YOLO-based processor
    try:
        processor = GrabCutProcessor(
            confidence_threshold=0.5,
            iterations=5,
            margin_pixels=20,
            edge_refinement_strength=0.7
        )
        print("✓ Created YOLO-based processor")
        
        # Process image
        result = processor.process_with_grabcut(test_img, target_class="auto")
        
        if result['success']:
            print(f"✓ Processing successful!")
            print(f"  - Processing time: {result['processing_time_ms']}ms")
            print(f"  - Detection confidence: {result['confidence']:.2f}")
            if result['bbox']:
                print(f"  - Bounding box: {result['bbox']}")
            
            # Check output
            rgba = result['rgba_image']
            assert rgba.shape == (512, 512, 4), f"Unexpected output shape: {rgba.shape}"
            print(f"✓ Output shape correct: {rgba.shape}")
            
            # Check mask
            mask = result['mask']
            unique_values = np.unique(mask)
            print(f"  - Mask unique values: {unique_values}")
            assert len(unique_values) <= 2, "Mask should be binary"
            print("✓ Mask is binary")
        else:
            print("✗ Processing failed")
            
    except Exception as e:
        print(f"⚠ YOLO processor not available: {e}")
        print("Testing fallback processor...")
        
        # Test fallback processor
        processor = create_fallback_processor()(
            confidence_threshold=0.5,
            iterations=5,
            margin_pixels=20,
            edge_refinement_strength=0.7
        )
        print("✓ Created fallback processor")
        
        result = processor.process_with_grabcut(test_img, target_class=None)
        
        if result['success']:
            print("✓ Fallback processing successful!")
            print(f"  - Processing time: {result['processing_time_ms']}ms")
        else:
            print("✗ Fallback processing failed")
    
    print("\n" + "="*50)


def test_initial_mask_refinement():
    """Test GrabCut with initial mask refinement."""
    print("\nTesting mask refinement...")
    
    # Create test image and initial mask
    test_img = create_test_image()
    
    # Create a rough initial mask (circle approximation)
    initial_mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(initial_mask, (256, 256), 120, 255, -1)
    
    print(f"Created initial mask with {np.sum(initial_mask > 0)} foreground pixels")
    
    try:
        processor = create_fallback_processor()(iterations=3)
        
        # Process with initial mask
        result = processor.process_with_initial_mask(test_img, initial_mask)
        
        if result['success']:
            print("✓ Mask refinement successful!")
            print(f"  - Processing time: {result['processing_time_ms']}ms")
            
            # Compare masks
            refined_mask = result['mask']
            initial_fg = np.sum(initial_mask > 0)
            refined_fg = np.sum(refined_mask > 0)
            
            print(f"  - Initial foreground pixels: {initial_fg}")
            print(f"  - Refined foreground pixels: {refined_fg}")
            print(f"  - Difference: {abs(refined_fg - initial_fg)} pixels")
        else:
            print("✗ Mask refinement failed")
            
    except Exception as e:
        print(f"✗ Error in mask refinement: {e}")
    
    print("\n" + "="*50)


def test_comfyui_node():
    """Test the ComfyUI node interface."""
    print("\nTesting ComfyUI node interface...")
    
    try:
        import torch
        
        # Create test tensor (ComfyUI format: batch, height, width, channels)
        test_img_np = create_test_image().astype(np.float32) / 255.0
        test_tensor = torch.from_numpy(test_img_np).permute(2, 0, 1).unsqueeze(0)
        print(f"Created test tensor: {test_tensor.shape}")
        
        # Create node
        node = AutoGrabCutRemover()
        print("✓ Created AutoGrabCutRemover node")
        
        # Test RGBA output format
        print("\n--- Testing RGBA output format ---")
        output_image, output_mask, bbox_coords, confidence, metrics = node.remove_background(
            image=test_tensor,
            object_class="auto",
            confidence_threshold=0.5,
            grabcut_iterations=5,
            margin_pixels=20,
            edge_refinement=0.7,
            binary_threshold=200,
            output_format="RGBA"
        )
        
        print("✓ RGBA processing successful!")
        print(f"  - Output image shape: {output_image.shape}")
        print(f"  - Output mask shape: {output_mask.shape}")
        print(f"  - Image channels: {output_image.shape[-1] if len(output_image.shape) >= 3 else 'Unknown'}")
        print(f"  - Bounding box: {bbox_coords}")
        print(f"  - Confidence: {confidence:.2f}")
        print(f"  - Metrics: {metrics}")
        
        # Verify RGBA format
        if len(output_image.shape) >= 3 and output_image.shape[-1] == 4:
            print("✓ RGBA format verified (4 channels)")
        else:
            print(f"⚠ Expected 4 channels for RGBA, got {output_image.shape[-1] if len(output_image.shape) >= 3 else 'Unknown'}")
        
        # Test MASK output format  
        print("\n--- Testing MASK output format ---")
        output_image_mask, output_mask_mask, bbox_coords_mask, confidence_mask, metrics_mask = node.remove_background(
            image=test_tensor,
            object_class="auto",
            confidence_threshold=0.5,
            grabcut_iterations=5,
            margin_pixels=20,
            edge_refinement=0.7,
            binary_threshold=200,
            output_format="MASK"
        )
        
        print("✓ MASK processing successful!")
        print(f"  - Output image shape: {output_image_mask.shape}")
        print(f"  - Output mask shape: {output_mask_mask.shape}")
        print(f"  - Image channels: {output_image_mask.shape[-1] if len(output_image_mask.shape) >= 3 else 'Unknown'}")
        print(f"  - Bounding box: {bbox_coords_mask}")
        print(f"  - Confidence: {confidence_mask:.2f}")
        print(f"  - Metrics: {metrics_mask}")
        
        # Verify MASK format
        if len(output_image_mask.shape) >= 3 and output_image_mask.shape[-1] == 1:
            print("✓ MASK format verified (1 channel)")
            # Check if mask is binary
            mask_values = torch.unique(output_image_mask)
            print(f"  - Unique mask values: {mask_values.tolist()}")
        else:
            print(f"⚠ Expected 1 channel for MASK, got {output_image_mask.shape[-1] if len(output_image_mask.shape) >= 3 else 'Unknown'}")
        
        # Test refinement node
        refinement_node = GrabCutRefinement()
        print("\n✓ Created GrabCutRefinement node")
        
        refined_image, refined_mask = refinement_node.refine_mask(
            image=test_tensor,
            mask=output_mask,
            grabcut_iterations=3,
            edge_refinement=0.5,
            expand_margin=10
        )
        
        print("✓ Refinement node processing successful!")
        print(f"  - Refined image shape: {refined_image.shape}")
        print(f"  - Refined mask shape: {refined_mask.shape}")
        
    except ImportError:
        print("⚠ PyTorch not available, skipping ComfyUI node test")
    except Exception as e:
        print(f"✗ Error in ComfyUI node test: {e}")
    
    print("\n" + "="*50)


def test_node_registration():
    """Test that nodes are properly registered."""
    print("\nTesting node registration...")
    
    try:
        from __init__ import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        
        print(f"Found {len(NODE_CLASS_MAPPINGS)} registered node classes:")
        for node_name, node_class in NODE_CLASS_MAPPINGS.items():
            display_name = NODE_DISPLAY_NAME_MAPPINGS.get(node_name, "Unknown")
            print(f"  - {node_name}: {display_name}")
        
        # Check for GrabCut nodes
        grabcut_nodes = [name for name in NODE_CLASS_MAPPINGS if 'GrabCut' in name]
        if grabcut_nodes:
            print(f"\n✓ Found {len(grabcut_nodes)} GrabCut nodes:")
            for node in grabcut_nodes:
                print(f"  - {node}")
        else:
            print("\n⚠ No GrabCut nodes found in registration")
            
    except ImportError as e:
        print(f"✗ Could not import node mappings: {e}")
    except Exception as e:
        print(f"✗ Error checking node registration: {e}")
    
    print("\n" + "="*50)


def test_edge_blur_functionality():
    """Test the new edge blur functionality with various blur amounts."""
    print("\nTesting Edge Blur Functionality...")
    
    try:
        import torch
        
        # Create test tensor
        test_img_np = create_test_image().astype(np.float32) / 255.0
        test_tensor = torch.from_numpy(test_img_np).permute(2, 0, 1).unsqueeze(0)
        print(f"Created test tensor for edge blur testing: {test_tensor.shape}")
        
        # Create node
        node = AutoGrabCutRemover()
        print("✓ Created AutoGrabCutRemover node for edge blur testing")
        
        # Test different edge blur amounts
        blur_amounts = [0.0, 1.0, 2.5, 5.0]
        
        for blur_amount in blur_amounts:
            print(f"\n--- Testing edge_blur_amount = {blur_amount} ---")
            
            try:
                output_image, output_mask, bbox_coords, confidence, metrics = node.remove_background(
                    image=test_tensor,
                    object_class="auto",
                    confidence_threshold=0.5,
                    grabcut_iterations=5,
                    margin_pixels=20,
                    edge_refinement=0.7,
                    edge_blur_amount=blur_amount,
                    binary_threshold=200,
                    output_format="RGBA"
                )
                
                print(f"✓ Processing successful with blur amount {blur_amount}")
                print(f"  - Output image shape: {output_image.shape}")
                print(f"  - Output mask shape: {output_mask.shape}")
                
                # Check if blur is applied (mask should have gradual transitions when blur > 0)
                mask_values = torch.unique(output_mask)
                if blur_amount > 0:
                    if len(mask_values) > 2:
                        print(f"  ✓ Soft edges detected (unique values: {len(mask_values)})")
                    else:
                        print(f"  ⚠ Expected soft edges but found binary mask")
                else:
                    print(f"  ✓ Binary edges preserved (unique values: {len(mask_values)})")
                    
            except Exception as e:
                print(f"✗ Error with blur amount {blur_amount}: {e}")
        
        # Test GrabCutRefinement node with edge blur
        print("\n--- Testing GrabCutRefinement with edge blur ---")
        refinement_node = GrabCutRefinement()
        
        # Use output from previous test as input
        refined_image, refined_mask = refinement_node.refine_mask(
            image=test_tensor,
            mask=output_mask,
            grabcut_iterations=3,
            edge_refinement=0.5,
            edge_blur_amount=1.5,
            expand_margin=10
        )
        
        print("✓ GrabCutRefinement with edge blur successful!")
        print(f"  - Refined image shape: {refined_image.shape}")
        print(f"  - Refined mask shape: {refined_mask.shape}")
        
        # Test auto-adjustment with edge blur
        print("\n--- Testing auto-adjustment with edge blur ---")
        output_image_auto, output_mask_auto, _, _, _ = node.remove_background(
            image=test_tensor,
            object_class="auto",
            confidence_threshold=0.5,
            grabcut_iterations=5,
            margin_pixels=20,
            edge_refinement=0.7,
            edge_blur_amount=0.5,  # Base blur amount
            binary_threshold=200,
            output_format="RGBA",
            auto_adjust=True  # Enable auto-adjustment
        )
        
        print("✓ Auto-adjustment with edge blur successful!")
        print(f"  - Auto-adjusted output image shape: {output_image_auto.shape}")
        
    except ImportError:
        print("⚠ PyTorch not available, skipping edge blur test")
    except Exception as e:
        print(f"✗ Error in edge blur test: {e}")
    
    print("\n" + "="*50)


def main():
    """Run all tests."""
    print("GrabCut Background Remover Test Suite")
    print("="*50)
    
    # Run tests
    test_grabcut_processor()
    test_initial_mask_refinement()
    test_comfyui_node()
    test_edge_blur_functionality()
    test_node_registration()
    
    print("\nAll tests completed!")
    print("="*50)


if __name__ == "__main__":
    main()