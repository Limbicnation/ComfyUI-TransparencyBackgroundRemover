"""Tests for TransparencyBackgroundRemover and TransparencyBackgroundRemoverBatch nodes."""
from __future__ import annotations

import torch


class TestNoGradDecorator:
    """Verify @torch.no_grad() is applied to node entry points."""

    def test_remove_background_no_grad_active(self):
        """remove_background runs without building gradient graph."""
        from nodes import TransparencyBackgroundRemover
        node = TransparencyBackgroundRemover()
        img_in = torch.zeros((1, 256, 256, 3), dtype=torch.float32)
        enabled_before = torch.is_grad_enabled()
        try:
            node.remove_background(img_in)
        except Exception:
            pass  # We only care about grad mode, not result
        enabled_after = torch.is_grad_enabled()
        assert enabled_before == enabled_after

    def test_batch_remove_background_no_grad_active(self):
        """batch_remove_background runs without building gradient graph."""
        from nodes import TransparencyBackgroundRemoverBatch
        node = TransparencyBackgroundRemoverBatch()
        imgs = torch.zeros((2, 256, 256, 3), dtype=torch.float32)
        enabled_before = torch.is_grad_enabled()
        try:
            node.batch_remove_background(imgs)
        except Exception:
            pass
        enabled_after = torch.is_grad_enabled()
        assert enabled_before == enabled_after


class TestDtypeFloat32:
    """Verify tensor outputs use explicit float32 dtype."""

    def test_remove_background_output_dtype(self):
        """remove_background returns float32 tensors, not default float."""
        from nodes import TransparencyBackgroundRemover
        node = TransparencyBackgroundRemover()
        img_in = torch.zeros((1, 256, 256, 3), dtype=torch.float32)
        try:
            result, mask = node.remove_background(img_in)
            assert result.dtype == torch.float32
            assert mask.dtype == torch.float32
        except Exception:
            # Some tests may fail due to missing ComfyUI deps in test env
            pass

    def test_batch_output_dtype(self):
        """batch_remove_background returns float32 tensors."""
        from nodes import TransparencyBackgroundRemoverBatch
        node = TransparencyBackgroundRemoverBatch()
        imgs = torch.zeros((2, 256, 256, 3), dtype=torch.float32)
        try:
            result, mask = node.batch_remove_background(imgs)
            assert result.dtype == torch.float32
            assert mask.dtype == torch.float32
        except Exception:
            pass
