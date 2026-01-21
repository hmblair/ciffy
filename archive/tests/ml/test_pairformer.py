"""Tests for ciffy.nn.layers.pairformer module."""

import pytest

from tests.utils import TORCH_AVAILABLE


# =============================================================================
# Test TriangularMultiplicativeUpdate
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTriangularMultiplicativeUpdate:
    """Tests for TriangularMultiplicativeUpdate."""

    def test_output_shape_outgoing(self):
        """Test outgoing direction preserves shape."""
        import torch
        from ciffy.nn.layers.pairformer import TriangularMultiplicativeUpdate

        tmu = TriangularMultiplicativeUpdate(d_pair=64, direction="outgoing")
        pair = torch.randn(2, 20, 20, 64)

        out = tmu(pair)
        assert out.shape == pair.shape

    def test_output_shape_incoming(self):
        """Test incoming direction preserves shape."""
        import torch
        from ciffy.nn.layers.pairformer import TriangularMultiplicativeUpdate

        tmu = TriangularMultiplicativeUpdate(d_pair=64, direction="incoming")
        pair = torch.randn(2, 20, 20, 64)

        out = tmu(pair)
        assert out.shape == pair.shape

    def test_gradients_flow(self):
        """Test gradients propagate correctly."""
        import torch
        from ciffy.nn.layers.pairformer import TriangularMultiplicativeUpdate

        tmu = TriangularMultiplicativeUpdate(d_pair=64, direction="outgoing")
        pair = torch.randn(2, 10, 10, 64, requires_grad=True)

        out = tmu(pair)
        loss = out.sum()
        loss.backward()

        assert pair.grad is not None
        for param in tmu.parameters():
            assert param.grad is not None

    def test_with_mask(self):
        """Test masking works correctly."""
        import torch
        from ciffy.nn.layers.pairformer import TriangularMultiplicativeUpdate

        tmu = TriangularMultiplicativeUpdate(d_pair=64, direction="outgoing")
        pair = torch.randn(2, 10, 10, 64)
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[:, 8:] = True  # Mask last 2 positions

        out = tmu(pair, mask=mask)
        assert out.shape == pair.shape
        assert not torch.isnan(out).any()

    def test_no_nan_output(self):
        """Test no NaN values in output."""
        import torch
        from ciffy.nn.layers.pairformer import TriangularMultiplicativeUpdate

        tmu = TriangularMultiplicativeUpdate(d_pair=64, direction="outgoing")
        pair = torch.randn(2, 15, 15, 64)

        out = tmu(pair)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_custom_hidden_dim(self):
        """Test custom hidden dimension."""
        import torch
        from ciffy.nn.layers.pairformer import TriangularMultiplicativeUpdate

        tmu = TriangularMultiplicativeUpdate(d_pair=64, d_hidden=32, direction="outgoing")
        pair = torch.randn(2, 10, 10, 64)

        out = tmu(pair)
        assert out.shape == pair.shape

    def test_invalid_direction_raises(self):
        """Test invalid direction raises error."""
        from ciffy.nn.layers.pairformer import TriangularMultiplicativeUpdate

        with pytest.raises(ValueError, match="direction must be"):
            TriangularMultiplicativeUpdate(d_pair=64, direction="invalid")


# =============================================================================
# Test TriangularAttention
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTriangularAttention:
    """Tests for TriangularAttention."""

    def test_output_shape_starting(self):
        """Test starting direction output shape."""
        import torch
        from ciffy.nn.layers.pairformer import TriangularAttention

        attn = TriangularAttention(d_pair=64, num_heads=8, direction="starting")
        pair = torch.randn(2, 20, 20, 64)

        out = attn(pair)
        assert out.shape == pair.shape

    def test_output_shape_ending(self):
        """Test ending direction output shape."""
        import torch
        from ciffy.nn.layers.pairformer import TriangularAttention

        attn = TriangularAttention(d_pair=64, num_heads=8, direction="ending")
        pair = torch.randn(2, 20, 20, 64)

        out = attn(pair)
        assert out.shape == pair.shape

    def test_gradients_flow(self):
        """Test gradient flow through attention."""
        import torch
        from ciffy.nn.layers.pairformer import TriangularAttention

        attn = TriangularAttention(d_pair=64, num_heads=8, direction="starting")
        pair = torch.randn(2, 10, 10, 64, requires_grad=True)

        out = attn(pair)
        loss = out.sum()
        loss.backward()

        assert pair.grad is not None
        for param in attn.parameters():
            assert param.grad is not None

    def test_with_mask(self):
        """Test attention with mask."""
        import torch
        from ciffy.nn.layers.pairformer import TriangularAttention

        attn = TriangularAttention(d_pair=64, num_heads=8, direction="starting")
        pair = torch.randn(2, 10, 10, 64)
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[:, 8:] = True

        out = attn(pair, mask=mask)
        assert out.shape == pair.shape

    def test_invalid_direction_raises(self):
        """Test invalid direction raises error."""
        from ciffy.nn.layers.pairformer import TriangularAttention

        with pytest.raises(ValueError, match="direction must be"):
            TriangularAttention(d_pair=64, num_heads=8, direction="invalid")

    def test_invalid_heads_raises(self):
        """Test d_pair not divisible by num_heads raises error."""
        from ciffy.nn.layers.pairformer import TriangularAttention

        with pytest.raises(ValueError, match="must be divisible"):
            TriangularAttention(d_pair=64, num_heads=5)


# =============================================================================
# Test PairTransition
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPairTransition:
    """Tests for PairTransition."""

    def test_output_shape(self):
        """Test transition preserves shape."""
        import torch
        from ciffy.nn.layers.pairformer import PairTransition

        transition = PairTransition(d_pair=64)
        pair = torch.randn(2, 20, 20, 64)

        out = transition(pair)
        assert out.shape == pair.shape

    def test_gradients_flow(self):
        """Test gradients flow through transition."""
        import torch
        from ciffy.nn.layers.pairformer import PairTransition

        transition = PairTransition(d_pair=64)
        pair = torch.randn(2, 10, 10, 64, requires_grad=True)

        out = transition(pair)
        loss = out.sum()
        loss.backward()

        assert pair.grad is not None

    def test_custom_ffn_dim(self):
        """Test custom feedforward dimension."""
        import torch
        from ciffy.nn.layers.pairformer import PairTransition

        transition = PairTransition(d_pair=64, d_ff=128)
        pair = torch.randn(2, 10, 10, 64)

        out = transition(pair)
        assert out.shape == pair.shape


# =============================================================================
# Test OuterProductMean
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestOuterProductMean:
    """Tests for OuterProductMean."""

    def test_output_shape(self):
        """Test outer product mean output shape."""
        import torch
        from ciffy.nn.layers.pairformer import OuterProductMean

        opm = OuterProductMean(d_single=128, d_pair=64)
        single = torch.randn(2, 20, 128)

        out = opm(single)
        assert out.shape == (2, 20, 20, 64)

    def test_gradients_flow(self):
        """Test gradients flow through outer product."""
        import torch
        from ciffy.nn.layers.pairformer import OuterProductMean

        opm = OuterProductMean(d_single=128, d_pair=64)
        single = torch.randn(2, 10, 128, requires_grad=True)

        out = opm(single)
        loss = out.sum()
        loss.backward()

        assert single.grad is not None

    def test_with_mask(self):
        """Test outer product with mask."""
        import torch
        from ciffy.nn.layers.pairformer import OuterProductMean

        opm = OuterProductMean(d_single=128, d_pair=64)
        single = torch.randn(2, 10, 128)
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[:, 8:] = True

        out = opm(single, mask=mask)
        assert out.shape == (2, 10, 10, 64)


# =============================================================================
# Test PairToSingleAttention
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPairToSingleAttention:
    """Tests for PairToSingleAttention."""

    def test_output_shape(self):
        """Test pair to single attention output shape."""
        import torch
        from ciffy.nn.layers.pairformer import PairToSingleAttention

        p2s = PairToSingleAttention(d_pair=64, d_single=128, num_heads=8)
        pair = torch.randn(2, 20, 20, 64)
        single = torch.randn(2, 20, 128)

        out = p2s(pair, single)
        assert out.shape == single.shape

    def test_gradients_flow(self):
        """Test gradients flow through pair to single."""
        import torch
        from ciffy.nn.layers.pairformer import PairToSingleAttention

        p2s = PairToSingleAttention(d_pair=64, d_single=128, num_heads=8)
        pair = torch.randn(2, 10, 10, 64, requires_grad=True)
        single = torch.randn(2, 10, 128, requires_grad=True)

        out = p2s(pair, single)
        loss = out.sum()
        loss.backward()

        assert pair.grad is not None
        assert single.grad is not None


# =============================================================================
# Test PairformerBlock
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPairformerBlock:
    """Tests for PairformerBlock."""

    def test_output_shape(self):
        """Test block preserves input shape."""
        import torch
        from ciffy.nn.layers.pairformer import PairformerBlock

        block = PairformerBlock(d_pair=64, num_heads=8)
        pair = torch.randn(2, 15, 15, 64)

        out = block(pair)
        assert out.shape == pair.shape

    def test_gradients_flow(self):
        """Test gradients flow through all sublayers."""
        import torch
        from ciffy.nn.layers.pairformer import PairformerBlock

        block = PairformerBlock(d_pair=64, num_heads=8)
        pair = torch.randn(2, 10, 10, 64, requires_grad=True)

        out = block(pair)
        loss = out.sum()
        loss.backward()

        assert pair.grad is not None
        for name, param in block.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_with_mask(self):
        """Test block with mask."""
        import torch
        from ciffy.nn.layers.pairformer import PairformerBlock

        block = PairformerBlock(d_pair=64, num_heads=8)
        pair = torch.randn(2, 10, 10, 64)
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[:, 8:] = True

        out = block(pair, mask=mask)
        assert out.shape == pair.shape

    def test_residual_connections(self):
        """Verify residual connections are active (output != input)."""
        import torch
        from ciffy.nn.layers.pairformer import PairformerBlock

        block = PairformerBlock(d_pair=64, num_heads=8)
        pair = torch.randn(2, 10, 10, 64)

        out = block(pair)
        # Output should be different from input (residual + transformation)
        assert not torch.allclose(out, pair)


# =============================================================================
# Test Full Pairformer
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPairformer:
    """Tests for full Pairformer model."""

    def test_forward_shape(self):
        """Test model output shape."""
        import torch
        from ciffy.nn.layers.pairformer import Pairformer

        model = Pairformer(d_pair=64, num_layers=2, num_heads=8)
        pair = torch.randn(2, 15, 15, 64)

        out = model(pair)
        assert out.shape == pair.shape

    def test_forward_with_single(self):
        """Test model with single representation track."""
        import torch
        from ciffy.nn.layers.pairformer import Pairformer

        model = Pairformer(d_pair=64, num_layers=2, num_heads=8, d_single=128)
        pair = torch.randn(2, 15, 15, 64)
        single = torch.randn(2, 15, 128)

        pair_out, single_out = model(pair, single=single)
        assert pair_out.shape == pair.shape
        assert single_out.shape == single.shape

    def test_variable_sequence_length(self):
        """Test with different sequence lengths."""
        import torch
        from ciffy.nn.layers.pairformer import Pairformer

        model = Pairformer(d_pair=64, num_layers=2, num_heads=8)

        for seq_len in [5, 10, 20, 50]:
            pair = torch.randn(2, seq_len, seq_len, 64)
            out = model(pair)
            assert out.shape == (2, seq_len, seq_len, 64)

    def test_gradients_flow(self):
        """Test gradients flow through all layers."""
        import torch
        from ciffy.nn.layers.pairformer import Pairformer

        model = Pairformer(d_pair=64, num_layers=2, num_heads=8)
        pair = torch.randn(2, 10, 10, 64, requires_grad=True)

        out = model(pair)
        loss = out.sum()
        loss.backward()

        assert pair.grad is not None
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_gradients_flow_with_single(self):
        """Test gradients flow with single track."""
        import torch
        from ciffy.nn.layers.pairformer import Pairformer

        model = Pairformer(d_pair=64, num_layers=2, num_heads=8, d_single=128)
        pair = torch.randn(2, 10, 10, 64, requires_grad=True)
        single = torch.randn(2, 10, 128, requires_grad=True)

        pair_out, single_out = model(pair, single=single)
        loss = pair_out.sum() + single_out.sum()
        loss.backward()

        assert pair.grad is not None
        assert single.grad is not None

    def test_no_nan_output(self):
        """Test no NaN in outputs."""
        import torch
        from ciffy.nn.layers.pairformer import Pairformer

        model = Pairformer(d_pair=64, num_layers=4, num_heads=8)
        pair = torch.randn(2, 20, 20, 64)

        out = model(pair)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_with_mask(self):
        """Test model with padding mask."""
        import torch
        from ciffy.nn.layers.pairformer import Pairformer

        model = Pairformer(d_pair=64, num_layers=2, num_heads=8)
        pair = torch.randn(2, 15, 15, 64)
        mask = torch.zeros(2, 15, dtype=torch.bool)
        mask[:, 12:] = True

        out = model(pair, mask=mask)
        assert out.shape == pair.shape

    def test_single_without_d_single_raises(self):
        """Test providing single without d_single raises error."""
        import torch
        from ciffy.nn.layers.pairformer import Pairformer

        model = Pairformer(d_pair=64, num_layers=2, num_heads=8)
        pair = torch.randn(2, 10, 10, 64)
        single = torch.randn(2, 10, 128)

        with pytest.raises(ValueError, match="without d_single"):
            model(pair, single=single)

    def test_d_single_without_single_raises(self):
        """Test d_single without providing single raises error."""
        import torch
        from ciffy.nn.layers.pairformer import Pairformer

        model = Pairformer(d_pair=64, num_layers=2, num_heads=8, d_single=128)
        pair = torch.randn(2, 10, 10, 64)

        with pytest.raises(ValueError, match="single tensor not provided"):
            model(pair)


# =============================================================================
# Test Error Handling
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPairformerErrorHandling:
    """Tests for Pairformer error handling."""

    def test_non_square_pair_raises(self):
        """Test non-square pair tensor raises error."""
        import torch
        from ciffy.nn.layers.pairformer import Pairformer

        model = Pairformer(d_pair=64, num_layers=2, num_heads=8)
        pair = torch.randn(2, 10, 15, 64)  # Non-square

        with pytest.raises(ValueError, match="must be square"):
            model(pair)

    def test_wrong_d_pair_raises(self):
        """Test wrong d_pair raises error."""
        import torch
        from ciffy.nn.layers.pairformer import Pairformer

        model = Pairformer(d_pair=64, num_layers=2, num_heads=8)
        pair = torch.randn(2, 10, 10, 32)  # Wrong dimension

        with pytest.raises(ValueError, match="d_pair mismatch"):
            model(pair)

    def test_nan_input_raises(self):
        """Test NaN in input raises error."""
        import torch
        from ciffy.nn.layers.pairformer import Pairformer

        model = Pairformer(d_pair=64, num_layers=2, num_heads=8)
        pair = torch.randn(2, 10, 10, 64)
        pair[0, 0, 0, 0] = float("nan")

        with pytest.raises(RuntimeError, match="NaN"):
            model(pair)


# =============================================================================
# Test Integration
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPairformerIntegration:
    """Integration tests for Pairformer."""

    def test_components_composable(self):
        """Test individual components can be composed."""
        import torch
        import torch.nn as nn
        from ciffy.nn.layers.pairformer import (
            TriangularMultiplicativeUpdate,
            TriangularAttention,
            PairTransition,
        )

        class CustomPairBlock(nn.Module):
            def __init__(self, d_pair, num_heads):
                super().__init__()
                self.tri_mul = TriangularMultiplicativeUpdate(d_pair, direction="outgoing")
                self.tri_attn = TriangularAttention(d_pair, num_heads, direction="starting")
                self.transition = PairTransition(d_pair)

            def forward(self, pair):
                pair = pair + self.tri_mul(pair)
                pair = pair + self.tri_attn(pair)
                pair = pair + self.transition(pair)
                return pair

        block = CustomPairBlock(d_pair=64, num_heads=8)
        pair = torch.randn(2, 10, 10, 64)

        out = block(pair)
        assert out.shape == pair.shape

    def test_device_transfer(self):
        """Test model can be moved to different devices."""
        import torch
        from ciffy.nn.layers.pairformer import Pairformer

        model = Pairformer(d_pair=64, num_layers=2, num_heads=8)

        model_cpu = model.to("cpu")
        assert next(model_cpu.parameters()).device == torch.device("cpu")

        pair = torch.randn(2, 10, 10, 64)
        out = model_cpu(pair)
        assert out.shape == pair.shape
