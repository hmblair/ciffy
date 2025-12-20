"""Tests for ciffy.nn.transformer module."""

import pytest
import numpy as np

from tests.utils import TORCH_AVAILABLE
from tests.testing import get_tolerances


# =============================================================================
# Test RMSNorm
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestRMSNorm:
    """Tests for RMSNorm layer."""

    def test_rmsnorm_shape(self):
        """Test RMSNorm preserves input shape."""
        import torch
        from ciffy.nn.transformer import RMSNorm

        norm = RMSNorm(dim=64)
        x = torch.randn(2, 10, 64)

        out = norm(x)
        assert out.shape == x.shape

    def test_rmsnorm_normalization(self):
        """Test RMSNorm produces unit RMS."""
        import torch
        from ciffy.nn.transformer import RMSNorm

        norm = RMSNorm(dim=64)
        # Reset weight to 1 for testing
        norm.weight.data.fill_(1.0)

        x = torch.randn(2, 10, 64) * 5  # Large values
        out = norm(x)

        # RMS of output should be approximately 1
        rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_rmsnorm_gradients(self):
        """Test RMSNorm gradients flow."""
        import torch
        from ciffy.nn.transformer import RMSNorm

        norm = RMSNorm(dim=64)
        x = torch.randn(2, 10, 64, requires_grad=True)

        out = norm(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert norm.weight.grad is not None


# =============================================================================
# Test RoPE
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestRoPE:
    """Tests for Rotary Position Embeddings."""

    def test_rope_shape(self):
        """Test RoPE preserves input shape."""
        import torch
        from ciffy.nn.transformer import RotaryPositionEmbedding

        rope = RotaryPositionEmbedding(dim=64, max_seq_len=100)

        q = torch.randn(2, 8, 50, 64)  # (batch, heads, seq, head_dim)
        k = torch.randn(2, 8, 50, 64)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rope_different_positions(self):
        """Test RoPE produces different embeddings for different positions."""
        import torch
        from ciffy.nn.transformer import RotaryPositionEmbedding

        rope = RotaryPositionEmbedding(dim=64, max_seq_len=100)

        # Same content, different positions
        q = torch.randn(1, 1, 10, 64)
        k = torch.randn(1, 1, 10, 64)

        q_rot, k_rot = rope(q, k)

        # Different positions should have different rotations
        # Check that position 0 and position 5 are different
        assert not torch.allclose(q_rot[0, 0, 0], q_rot[0, 0, 5])

    def test_rope_equivariance(self):
        """Test RoPE is translation equivariant for relative positions."""
        import torch
        from ciffy.nn.transformer import RotaryPositionEmbedding

        rope = RotaryPositionEmbedding(dim=64, max_seq_len=100)

        # Same q and k vectors
        q_orig = torch.randn(1, 1, 1, 64)
        k_orig = torch.randn(1, 1, 1, 64)

        # Create sequences at different positions but same relative offset
        # Position 0 and 2 vs Position 5 and 7 (both have offset 2)
        q1 = q_orig.expand(1, 1, 10, 64).clone()
        k1 = k_orig.expand(1, 1, 10, 64).clone()

        q_rot, k_rot = rope(q1, k1)

        # The dot product q[i] @ k[j] should depend only on (i-j), not absolute position
        # Compare: q[0] @ k[2] vs q[5] @ k[7] (both have relative position -2)
        score_02 = (q_rot[0, 0, 0] * k_rot[0, 0, 2]).sum()
        score_57 = (q_rot[0, 0, 5] * k_rot[0, 0, 7]).sum()

        # These should be approximately equal due to relative position encoding
        tol = get_tolerances()
        assert torch.allclose(score_02, score_57, atol=tol.allclose_atol)


# =============================================================================
# Test SwiGLU
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestSwiGLU:
    """Tests for SwiGLU feedforward network."""

    def test_swiglu_shape(self):
        """Test SwiGLU preserves input shape."""
        import torch
        from ciffy.nn.transformer import SwiGLU

        ffn = SwiGLU(d_model=64)
        x = torch.randn(2, 10, 64)

        out = ffn(x)
        assert out.shape == x.shape

    def test_swiglu_gradients(self):
        """Test SwiGLU gradients flow."""
        import torch
        from ciffy.nn.transformer import SwiGLU

        ffn = SwiGLU(d_model=64)
        x = torch.randn(2, 10, 64, requires_grad=True)

        out = ffn(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        for param in ffn.parameters():
            assert param.grad is not None


# =============================================================================
# Test MultiHeadAttention
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestMultiHeadAttention:
    """Tests for MultiHeadAttention."""

    def test_attention_shape(self):
        """Test attention output shape."""
        import torch
        from ciffy.nn.transformer import MultiHeadAttention

        attn = MultiHeadAttention(d_model=64, num_heads=8)
        x = torch.randn(2, 10, 64)

        out = attn(x)
        assert out.shape == x.shape

    def test_attention_with_mask(self):
        """Test attention respects mask."""
        import torch
        from ciffy.nn.transformer import MultiHeadAttention

        attn = MultiHeadAttention(d_model=64, num_heads=8)
        x = torch.randn(2, 10, 64)
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[:, 8:] = True  # Mask last 2 positions

        out = attn(x, mask=mask)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()


# =============================================================================
# Test TransformerBlock
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTransformerBlock:
    """Tests for TransformerBlock."""

    def test_block_shape(self):
        """Test block preserves input shape."""
        import torch
        from ciffy.nn.transformer import TransformerBlock

        block = TransformerBlock(d_model=64, num_heads=8)
        x = torch.randn(2, 10, 64)

        out = block(x)
        assert out.shape == x.shape

    def test_block_gradients(self):
        """Test block gradients flow."""
        import torch
        from ciffy.nn.transformer import TransformerBlock

        block = TransformerBlock(d_model=64, num_heads=8)
        x = torch.randn(2, 10, 64, requires_grad=True)

        out = block(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None


# =============================================================================
# Test Full Transformer
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTransformer:
    """Tests for full Transformer."""

    def test_transformer_shape(self):
        """Test transformer output shape."""
        import torch
        from ciffy.nn.transformer import Transformer

        model = Transformer(d_model=64, num_layers=2, num_heads=8)
        x = torch.randn(2, 10, 64)

        out = model(x)
        assert out.shape == x.shape

    def test_transformer_with_mask(self):
        """Test transformer respects mask."""
        import torch
        from ciffy.nn.transformer import Transformer

        model = Transformer(d_model=64, num_layers=2, num_heads=8)
        x = torch.randn(2, 10, 64)
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[:, 8:] = True

        out = model(x, mask=mask)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_transformer_gradients(self):
        """Test transformer gradients flow through all layers."""
        import torch
        from ciffy.nn.transformer import Transformer

        model = Transformer(d_model=64, num_layers=4, num_heads=8)
        x = torch.randn(2, 10, 64, requires_grad=True)

        out = model(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None

        # Check gradients exist for all layers
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_transformer_variable_length(self):
        """Test transformer handles different sequence lengths."""
        import torch
        from ciffy.nn.transformer import Transformer

        model = Transformer(d_model=64, num_layers=2, num_heads=8)

        for seq_len in [1, 5, 50, 200]:
            x = torch.randn(2, seq_len, 64)
            out = model(x)
            assert out.shape == (2, seq_len, 64)

    def test_transformer_device_transfer(self):
        """Test transformer can be moved to different devices via .to()."""
        import torch
        from ciffy.nn.transformer import Transformer

        model = Transformer(d_model=64, num_layers=2, num_heads=8)

        # Should not raise - this tests that _apply isn't overridden incorrectly
        model_cpu = model.to("cpu")
        assert next(model_cpu.parameters()).device == torch.device("cpu")

        # Test forward pass still works after device transfer
        x = torch.randn(2, 10, 64)
        out = model_cpu(x)
        assert out.shape == x.shape


# =============================================================================
# Test Reusability
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTransformerReusability:
    """Test that transformer components are reusable outside VAE."""

    def test_transformer_standalone_classification(self):
        """Test using transformer for a simple classification task."""
        import torch
        import torch.nn as nn
        from ciffy.nn.transformer import Transformer

        class SimpleClassifier(nn.Module):
            def __init__(self, vocab_size, d_model, num_classes):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, d_model)
                self.transformer = Transformer(
                    d_model=d_model,
                    num_layers=2,
                    num_heads=4,
                )
                self.head = nn.Linear(d_model, num_classes)

            def forward(self, x):
                h = self.embed(x)
                h = self.transformer(h)
                h = h.mean(dim=1)  # Pool
                return self.head(h)

        model = SimpleClassifier(vocab_size=100, d_model=64, num_classes=5)
        x = torch.randint(0, 100, (4, 20))

        logits = model(x)
        assert logits.shape == (4, 5)

        # Test backward
        loss = logits.sum()
        loss.backward()

    def test_transformer_components_composable(self):
        """Test that individual components can be composed."""
        import torch
        import torch.nn as nn
        from ciffy.nn.transformer import (
            RMSNorm,
            MultiHeadAttention,
            SwiGLU,
        )

        # Build a custom block
        class CustomBlock(nn.Module):
            def __init__(self, d_model, num_heads):
                super().__init__()
                self.norm1 = RMSNorm(d_model)
                self.attn = MultiHeadAttention(d_model, num_heads)
                self.norm2 = RMSNorm(d_model)
                self.ffn = SwiGLU(d_model)

            def forward(self, x):
                x = x + self.attn(self.norm1(x))
                x = x + self.ffn(self.norm2(x))
                return x

        block = CustomBlock(d_model=64, num_heads=8)
        x = torch.randn(2, 10, 64)

        out = block(x)
        assert out.shape == x.shape
