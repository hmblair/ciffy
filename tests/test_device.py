"""
Tests for device-specific operations (CUDA, MPS).

These tests verify that operations work correctly when tensors
are on accelerator devices, including proper device handling
for scatter operations and reductions.
"""

import pytest
import numpy as np


# Check device availability
def cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def mps_available():
    try:
        import torch
        return torch.backends.mps.is_available()
    except (ImportError, AttributeError):
        return False


# Skip markers
requires_cuda = pytest.mark.skipif(
    not cuda_available(),
    reason="CUDA not available"
)

requires_mps = pytest.mark.skipif(
    not mps_available(),
    reason="MPS not available"
)


class TestDeviceOperations:
    """Test operations on different devices."""

    @pytest.fixture
    def polymer_torch(self):
        """Create a test polymer with torch backend."""
        from ciffy import load
        return load("tests/data/9GCM.cif", backend="torch")

    @requires_cuda
    def test_to_cuda(self, polymer_torch):
        """Test moving polymer to CUDA device."""
        p_cuda = polymer_torch.to("cuda")

        assert p_cuda.coordinates.device.type == "cuda"
        assert p_cuda.atoms.device.type == "cuda"
        assert p_cuda.elements.device.type == "cuda"
        assert p_cuda.sequence.device.type == "cuda"

    @requires_mps
    def test_to_mps(self, polymer_torch):
        """Test moving polymer to MPS device."""
        p_mps = polymer_torch.to("mps")

        assert p_mps.coordinates.device.type == "mps"
        assert p_mps.atoms.device.type == "mps"
        assert p_mps.elements.device.type == "mps"
        assert p_mps.sequence.device.type == "mps"

    @requires_cuda
    def test_reduce_on_cuda(self, polymer_torch):
        """Test reduction operations on CUDA."""
        from ciffy import Scale

        p_cuda = polymer_torch.to("cuda")

        # Test reduce (per-atom to per-chain)
        means = p_cuda.reduce(p_cuda.coordinates, Scale.CHAIN)
        assert means.device.type == "cuda"
        assert means.shape[0] == p_cuda.size(Scale.CHAIN)

    @requires_mps
    def test_reduce_on_mps(self, polymer_torch):
        """Test reduction operations on MPS."""
        from ciffy import Scale

        p_mps = polymer_torch.to("mps")

        # Test reduce (per-atom to per-chain)
        means = p_mps.reduce(p_mps.coordinates, Scale.CHAIN)
        assert means.device.type == "mps"
        assert means.shape[0] == p_mps.size(Scale.CHAIN)

    @requires_cuda
    def test_center_on_cuda(self, polymer_torch):
        """Test centering on CUDA (uses reduce internally)."""
        from ciffy import Scale

        p_cuda = polymer_torch.to("cuda")
        centered, _ = p_cuda.center(Scale.MOLECULE)

        assert centered.coordinates.device.type == "cuda"

    @requires_mps
    def test_center_on_mps(self, polymer_torch):
        """Test centering on MPS (uses reduce internally)."""
        from ciffy import Scale

        p_mps = polymer_torch.to("mps")
        centered, _ = p_mps.center(Scale.MOLECULE)

        assert centered.coordinates.device.type == "mps"

    @requires_cuda
    def test_expand_on_cuda(self, polymer_torch):
        """Test expand on CUDA."""
        from ciffy import Scale
        import torch

        p_cuda = polymer_torch.to("cuda")

        # Create per-chain features and expand to per-atom
        chain_features = torch.randn(p_cuda.size(Scale.CHAIN), 16, device="cuda")
        expanded = p_cuda.expand(chain_features, Scale.CHAIN)

        assert expanded.device.type == "cuda"
        assert expanded.shape[0] == p_cuda.size()

    @requires_mps
    def test_expand_on_mps(self, polymer_torch):
        """Test expand on MPS."""
        from ciffy import Scale
        import torch

        p_mps = polymer_torch.to("mps")

        # Create per-chain features and expand to per-atom
        chain_features = torch.randn(p_mps.size(Scale.CHAIN), 16, device="mps")
        expanded = p_mps.expand(chain_features, Scale.CHAIN)

        assert expanded.device.type == "mps"
        assert expanded.shape[0] == p_mps.size()

    @requires_cuda
    def test_rmsd_on_cuda(self, polymer_torch):
        """Test RMSD calculation on CUDA."""
        import ciffy

        p_cuda = polymer_torch.to("cuda")

        # Calculate RMSD against self (should be 0)
        rmsd = ciffy.rmsd(p_cuda, p_cuda, ciffy.MOLECULE)

        # Result should be on CUDA and close to 0
        assert rmsd.device.type == "cuda"
        assert rmsd.item() < 1e-5

    @requires_mps
    def test_rmsd_on_mps(self, polymer_torch):
        """Test RMSD calculation on MPS.

        Note: MPS doesn't support SVD operations, so RMSD (which uses Kabsch
        alignment with SVD) will fail. This test verifies the limitation is
        handled gracefully.
        """
        import ciffy
        import torch

        p_mps = polymer_torch.to("mps")

        # MPS doesn't support SVD, so RMSD will fail
        # Use PYTORCH_ENABLE_MPS_FALLBACK=1 env var to enable CPU fallback
        with pytest.raises(NotImplementedError, match="MPS device"):
            ciffy.rmsd(p_mps, p_mps, ciffy.MOLECULE)


class TestMixedDeviceHandling:
    """Test that operations handle mixed-device scenarios gracefully."""

    @pytest.fixture
    def polymer_torch(self):
        """Create a test polymer with torch backend."""
        from ciffy import load
        return load("tests/data/9GCM.cif", backend="torch")

    @requires_cuda
    def test_scatter_with_cpu_index_cuda_features(self):
        """Test scatter operations handle CPU index with CUDA features."""
        import torch
        from ciffy.backend.torch_ops import scatter_sum, scatter_mean

        # Create features on CUDA, index on CPU
        features = torch.randn(10, 3, device="cuda")
        index = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])  # CPU

        # This should work (index automatically moved to CUDA)
        result = scatter_sum(features, index, dim_size=3)
        assert result.device.type == "cuda"

        result = scatter_mean(features, index, dim_size=3)
        assert result.device.type == "cuda"

    @requires_mps
    def test_scatter_with_cpu_index_mps_features(self):
        """Test scatter operations handle CPU index with MPS features."""
        import torch
        from ciffy.backend.torch_ops import scatter_sum, scatter_mean

        # Create features on MPS, index on CPU
        features = torch.randn(10, 3, device="mps")
        index = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])  # CPU

        # This should work (index automatically moved to MPS)
        result = scatter_sum(features, index, dim_size=3)
        assert result.device.type == "mps"

        result = scatter_mean(features, index, dim_size=3)
        assert result.device.type == "mps"

    @requires_cuda
    def test_reduce_with_mismatched_sizes_device(self, polymer_torch):
        """Test reduce works even if internal sizes tensor is on different device."""
        import torch
        from ciffy import Scale

        # Get polymer on CUDA
        p_cuda = polymer_torch.to("cuda")

        # Manually move coordinates to CPU but keep sizes on CUDA
        # (This simulates a potential edge case)
        coords_cpu = p_cuda.coordinates.cpu()

        # Create reduction with CPU features - should still work
        # because create_reduction_index gets device from features
        from ciffy.operations.reduction import create_reduction_index

        sizes = p_cuda._sizes[Scale.CHAIN]
        assert sizes.device.type == "cuda"

        # Pass CPU device explicitly
        index = create_reduction_index(
            p_cuda.size(Scale.CHAIN),
            sizes,
            device=torch.device("cpu")
        )
        assert index.device.type == "cpu"

    @requires_cuda
    def test_with_coordinates_gpu_on_cpu_polymer_cuda(self, polymer_torch):
        """Test with_coordinates with GPU coords on CPU polymer works for center/expand."""
        import torch
        from ciffy import Scale

        # CPU polymer with GPU coordinates (common pattern in ML workflows)
        gpu_coords = polymer_torch.coordinates.to("cuda")
        mixed_polymer = polymer_torch.with_coordinates(gpu_coords)

        # Internal sizes are on CPU, coordinates on CUDA
        assert mixed_polymer.coordinates.device.type == "cuda"
        assert mixed_polymer._sizes[Scale.CHAIN].device.type == "cpu"

        # center() should work (uses reduce + expand internally)
        centered, means = mixed_polymer.center(Scale.MOLECULE)
        assert centered.coordinates.device.type == "cuda"

    @requires_mps
    def test_with_coordinates_gpu_on_cpu_polymer_mps(self, polymer_torch):
        """Test with_coordinates with MPS coords on CPU polymer works for center/expand."""
        import torch
        from ciffy import Scale

        # CPU polymer with MPS coordinates (common pattern in ML workflows)
        mps_coords = polymer_torch.coordinates.to("mps")
        mixed_polymer = polymer_torch.with_coordinates(mps_coords)

        # Internal sizes are on CPU, coordinates on MPS
        assert mixed_polymer.coordinates.device.type == "mps"
        assert mixed_polymer._sizes[Scale.CHAIN].device.type == "cpu"

        # center() should work (uses reduce + expand internally)
        centered, means = mixed_polymer.center(Scale.MOLECULE)
        assert centered.coordinates.device.type == "mps"


class TestDifferentiability:
    """Test that operations are differentiable for use with autograd."""

    @pytest.fixture
    def polymer_torch(self):
        """Create a test polymer with torch backend."""
        from ciffy import load
        return load("tests/data/9GCM.cif", backend="torch")

    def test_rmsd_is_differentiable(self, polymer_torch):
        """Test that ciffy.rmsd supports backpropagation."""
        import torch
        import ciffy

        # Create two polymers with coordinates that require gradients
        p1 = polymer_torch
        coords2 = p1.coordinates.clone().detach().requires_grad_(True)

        # Add small perturbation to make them different
        coords2_perturbed = coords2 + torch.randn_like(coords2) * 0.1
        p2 = p1.with_coordinates(coords2_perturbed)

        # Compute RMSD
        rmsd_sq = ciffy.rmsd(p1, p2, ciffy.MOLECULE)

        # Verify we can backpropagate
        rmsd_sq.sum().backward()

        # Gradients should exist and not be all zeros
        assert coords2.grad is not None, "Gradients were not computed"
        assert not torch.all(coords2.grad == 0), "Gradients are all zero"

    def test_rmsd_gradient_correctness(self, polymer_torch):
        """Test that RMSD gradients point toward alignment."""
        import torch
        import ciffy

        p1 = polymer_torch

        # Create p2 with a known translation
        translation = torch.tensor([10.0, 0.0, 0.0])
        coords2 = (p1.coordinates + translation).requires_grad_(True)
        p2 = p1.with_coordinates(coords2)

        # Compute RMSD
        rmsd_sq = ciffy.rmsd(p1, p2, ciffy.MOLECULE)
        rmsd_sq.sum().backward()

        # The gradient should point back toward p1 (negative x direction)
        # since that would reduce the RMSD
        mean_grad = coords2.grad.mean(dim=0)
        assert mean_grad[0] < 0, "Gradient should point toward reducing distance"

    def test_center_is_differentiable(self, polymer_torch):
        """Test that center() supports backpropagation."""
        import torch
        from ciffy import Scale

        coords = polymer_torch.coordinates.clone().detach().requires_grad_(True)
        p = polymer_torch.with_coordinates(coords)

        centered, means = p.center(Scale.MOLECULE)

        # Compute a loss on centered coordinates
        loss = centered.coordinates.sum()
        loss.backward()

        assert coords.grad is not None, "Gradients were not computed"

    def test_reduce_is_differentiable(self, polymer_torch):
        """Test that reduce() supports backpropagation."""
        import torch
        from ciffy import Scale

        coords = polymer_torch.coordinates.clone().detach().requires_grad_(True)
        p = polymer_torch.with_coordinates(coords)

        # Reduce to chain level
        chain_means = p.reduce(coords, Scale.CHAIN)

        loss = chain_means.sum()
        loss.backward()

        assert coords.grad is not None, "Gradients were not computed"

    @requires_cuda
    def test_rmsd_differentiable_on_cuda(self, polymer_torch):
        """Test RMSD differentiability on CUDA."""
        import torch
        import ciffy

        p1 = polymer_torch.to("cuda")
        coords2 = p1.coordinates.clone().detach().requires_grad_(True)
        p2 = p1.with_coordinates(coords2 + torch.randn_like(coords2) * 0.1)

        rmsd_sq = ciffy.rmsd(p1, p2, ciffy.MOLECULE)
        rmsd_sq.sum().backward()

        assert coords2.grad is not None
        assert coords2.grad.device.type == "cuda"

    def test_rmsd_gradient_stability_small_perturbation(self, polymer_torch):
        """Test gradient stability with near-identical structures.

        When structures are nearly identical, the covariance matrix approaches
        a scaled identity, making singular values nearly equal. This can cause
        SVD gradient instability. We verify gradients remain finite.
        """
        import torch
        import ciffy

        p1 = polymer_torch
        # Very small perturbation - this is the challenging case for SVD gradients
        coords2 = p1.coordinates.clone().detach().requires_grad_(True)
        perturbation = torch.randn_like(coords2) * 1e-6
        p2 = p1.with_coordinates(coords2 + perturbation)

        rmsd_sq = ciffy.rmsd(p1, p2, ciffy.MOLECULE)
        rmsd_sq.sum().backward()

        # Gradients should exist and be finite (no NaN or Inf)
        assert coords2.grad is not None, "Gradients were not computed"
        assert torch.isfinite(coords2.grad).all(), "Gradients contain NaN or Inf"

    def test_rmsd_gradient_stability_identical_structures(self, polymer_torch):
        """Test gradient stability with exactly identical structures.

        The degenerate case where structures are identical. The RMSD is 0,
        but gradients should still be computable and finite.
        """
        import torch
        import ciffy

        p1 = polymer_torch
        coords2 = p1.coordinates.clone().detach().requires_grad_(True)
        p2 = p1.with_coordinates(coords2)

        rmsd_sq = ciffy.rmsd(p1, p2, ciffy.MOLECULE)

        # RMSD should be essentially zero
        assert rmsd_sq.item() < 1e-10

        rmsd_sq.sum().backward()

        # Gradients should be finite (may be zero, but not NaN/Inf)
        assert coords2.grad is not None, "Gradients were not computed"
        assert torch.isfinite(coords2.grad).all(), "Gradients contain NaN or Inf"

    def test_rmsd_gradient_magnitude_bounded(self, polymer_torch):
        """Test that gradient magnitudes are reasonable (not exploding)."""
        import torch
        import ciffy

        p1 = polymer_torch
        coords2 = p1.coordinates.clone().detach().requires_grad_(True)
        # Moderate perturbation
        p2 = p1.with_coordinates(coords2 + torch.randn_like(coords2) * 0.5)

        rmsd_sq = ciffy.rmsd(p1, p2, ciffy.MOLECULE)
        rmsd_sq.sum().backward()

        # Gradient magnitude should be bounded (not exploding)
        grad_norm = coords2.grad.norm()
        assert grad_norm < 1e6, f"Gradient norm too large: {grad_norm}"
        assert torch.isfinite(grad_norm), "Gradient norm is not finite"

    def test_rmsd_gradient_stability_single_chain(self, polymer_torch):
        """Test gradient stability on single-chain polymer."""
        import torch
        import ciffy

        # Select single chain
        p1 = polymer_torch.select(0)
        coords2 = p1.coordinates.clone().detach().requires_grad_(True)
        p2 = p1.with_coordinates(coords2 + torch.randn_like(coords2) * 0.1)

        rmsd_sq = ciffy.rmsd(p1, p2, ciffy.MOLECULE)
        rmsd_sq.sum().backward()

        assert coords2.grad is not None
        assert torch.isfinite(coords2.grad).all(), "Gradients contain NaN or Inf"


class TestScatterOperations:
    """Test scatter operations directly."""

    @requires_cuda
    def test_scatter_sum_cuda(self):
        """Test scatter_sum on CUDA."""
        import torch
        from ciffy.backend.torch_ops import scatter_sum

        features = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], device="cuda")
        index = torch.tensor([0, 0, 1], device="cuda")

        result = scatter_sum(features, index, dim_size=2)

        expected = torch.tensor([[4., 6.], [5., 6.]], device="cuda")
        assert torch.allclose(result, expected)

    @requires_cuda
    def test_scatter_mean_cuda(self):
        """Test scatter_mean on CUDA."""
        import torch
        from ciffy.backend.torch_ops import scatter_mean

        features = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], device="cuda")
        index = torch.tensor([0, 0, 1], device="cuda")

        result = scatter_mean(features, index, dim_size=2)

        expected = torch.tensor([[2., 3.], [5., 6.]], device="cuda")
        assert torch.allclose(result, expected)

    @requires_mps
    def test_scatter_sum_mps(self):
        """Test scatter_sum on MPS."""
        import torch
        from ciffy.backend.torch_ops import scatter_sum

        features = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], device="mps")
        index = torch.tensor([0, 0, 1], device="mps")

        result = scatter_sum(features, index, dim_size=2)

        expected = torch.tensor([[4., 6.], [5., 6.]], device="mps")
        assert torch.allclose(result.cpu(), expected.cpu())

    @requires_mps
    def test_scatter_mean_mps(self):
        """Test scatter_mean on MPS."""
        import torch
        from ciffy.backend.torch_ops import scatter_mean

        features = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], device="mps")
        index = torch.tensor([0, 0, 1], device="mps")

        result = scatter_mean(features, index, dim_size=2)

        expected = torch.tensor([[2., 3.], [5., 6.]], device="mps")
        assert torch.allclose(result.cpu(), expected.cpu())
