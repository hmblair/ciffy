"""
Tests for device-specific operations (CUDA, MPS).

These tests verify that operations work correctly when tensors
are on accelerator devices, including proper device handling
for scatter operations and reductions.
"""

import pytest
import numpy as np

from tests.utils import (
    GPU_DEVICES,
    skip_if_no_device,
    requires_cuda,
    requires_mps,
)


class TestDeviceOperations:
    """Test operations on different devices.

    Uses parametrized any_polymer_torch fixture to run on all test PDBs.
    Tests are parameterized over GPU_DEVICES to eliminate CUDA/MPS duplication.
    """

    @pytest.mark.parametrize("device", GPU_DEVICES)
    def test_to_device(self, any_polymer_torch, device):
        """Test moving polymer to GPU device."""
        skip_if_no_device(device)
        p_gpu = any_polymer_torch.to(device)

        assert p_gpu.coordinates.device.type == device
        assert p_gpu.atoms.device.type == device
        assert p_gpu.elements.device.type == device
        assert p_gpu.sequence.device.type == device

    @pytest.mark.parametrize("device", GPU_DEVICES)
    def test_reduce_on_device(self, any_polymer_torch, device):
        """Test reduction operations on GPU."""
        skip_if_no_device(device)
        from ciffy import Scale

        p_gpu = any_polymer_torch.to(device)

        # Test reduce (per-atom to per-chain)
        means = p_gpu.reduce(p_gpu.coordinates, Scale.CHAIN)
        assert means.device.type == device
        assert means.shape[0] == p_gpu.size(Scale.CHAIN)

    @pytest.mark.parametrize("device", GPU_DEVICES)
    def test_center_on_device(self, any_polymer_torch, device):
        """Test centering on GPU (uses reduce internally)."""
        skip_if_no_device(device)
        from ciffy import Scale

        p_gpu = any_polymer_torch.to(device)
        centered, _ = p_gpu.center(Scale.MOLECULE)

        assert centered.coordinates.device.type == device

    @pytest.mark.parametrize("device", GPU_DEVICES)
    def test_expand_on_device(self, any_polymer_torch, device):
        """Test expand on GPU."""
        skip_if_no_device(device)
        from ciffy import Scale
        import torch

        p_gpu = any_polymer_torch.to(device)

        # Create per-chain features and expand to per-atom
        chain_features = torch.randn(p_gpu.size(Scale.CHAIN), 16, device=device)
        expanded = p_gpu.expand(chain_features, Scale.CHAIN)

        assert expanded.device.type == device
        assert expanded.shape[0] == p_gpu.size()

    @pytest.mark.parametrize("device", GPU_DEVICES)
    def test_rmsd_on_device(self, any_polymer_torch, device):
        """Test RMSD calculation on GPU devices.

        Note: MPS doesn't support SVD operations, so RMSD (which uses Kabsch
        alignment with SVD) will fail on MPS.
        """
        skip_if_no_device(device)
        import ciffy

        # Use .poly() to exclude ligands/water.
        p_gpu = any_polymer_torch.poly().to(device)

        if device == "mps":
            # MPS doesn't support SVD, so RMSD will fail
            with pytest.raises(NotImplementedError, match="MPS device"):
                ciffy.rmsd(p_gpu, p_gpu, ciffy.MOLECULE)
        else:
            # CUDA: Calculate RMSD against self (should be ~0)
            rmsd = ciffy.rmsd(p_gpu, p_gpu, ciffy.MOLECULE)

            # Result should be on device and close to 0
            # CUDA SVD (cuSOLVER) has lower precision than CPU
            assert rmsd.device.type == device
            n_atoms = p_gpu.coordinates.shape[0]
            tolerance = max(1e-2, (n_atoms ** 0.5) * 2e-4)
            assert rmsd.item() < tolerance, f"RMSD {rmsd.item():.6f} >= {tolerance:.6f} for {n_atoms} atoms"


class TestMixedDeviceHandling:
    """Test that operations handle mixed-device scenarios gracefully."""

    @pytest.mark.parametrize("device", GPU_DEVICES)
    def test_scatter_with_cpu_index(self, device):
        """Test scatter operations handle CPU index with GPU features."""
        skip_if_no_device(device)
        import torch
        from ciffy.backend import scatter_sum, scatter_mean

        # Create features on GPU, index on CPU
        features = torch.randn(10, 3, device=device)
        index = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])  # CPU

        # This should work (index automatically moved to GPU)
        result = scatter_sum(features, index, dim_size=3)
        assert result.device.type == device

        result = scatter_mean(features, index, dim_size=3)
        assert result.device.type == device

    @requires_cuda
    def test_reduce_with_mismatched_sizes_device(self, any_polymer_torch):
        """Test reduce works even if internal sizes tensor is on different device."""
        import torch
        from ciffy import Scale

        # Get polymer on CUDA
        p_cuda = any_polymer_torch.to("cuda")

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

    @pytest.mark.parametrize("device", GPU_DEVICES)
    def test_with_coordinates_rejects_cross_device(self, any_polymer_torch, device):
        """Test with_coordinates rejects GPU coords on CPU polymer."""
        skip_if_no_device(device)
        import torch

        # CPU polymer should reject GPU coordinates
        gpu_coords = any_polymer_torch.coordinates.to(device)
        with pytest.raises(ValueError, match="device"):
            any_polymer_torch.with_coordinates(gpu_coords)


class TestDifferentiability:
    """Test that operations are differentiable for use with autograd.

    Uses parametrized any_polymer_torch fixture to run on all test PDBs.
    """

    def test_rmsd_is_differentiable(self, any_polymer_torch):
        """Test that ciffy.rmsd supports backpropagation."""
        import torch
        import ciffy

        # Create two polymers with coordinates that require gradients
        p1 = any_polymer_torch
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

    def test_rmsd_gradient_correctness(self, any_polymer_torch):
        """Test that RMSD gradients point toward alignment."""
        import torch
        import ciffy

        p1 = any_polymer_torch

        # Use noise perturbation that can't be perfectly aligned by Kabsch
        # (pure translation would be perfectly removed, giving RMSD=0 and grad=0)
        torch.manual_seed(42)
        noise = torch.randn_like(p1.coordinates) * 1.0
        coords2 = (p1.coordinates + noise).requires_grad_(True)
        p2 = p1.with_coordinates(coords2)

        # Compute RMSD
        rmsd_sq = ciffy.rmsd(p1, p2, ciffy.MOLECULE)
        rmsd_sq.sum().backward()

        # Gradients should be non-zero and point in a consistent direction
        # (we can't predict exact direction with random noise, but they should exist)
        assert coords2.grad is not None, "Gradients were not computed"
        grad_norm = coords2.grad.norm()
        assert grad_norm > 1e-6, f"Gradient norm too small: {grad_norm}"

    def test_center_is_differentiable(self, any_polymer_torch):
        """Test that center() supports backpropagation."""
        import torch
        from ciffy import Scale

        coords = any_polymer_torch.coordinates.clone().detach().requires_grad_(True)
        p = any_polymer_torch.with_coordinates(coords)

        centered, means = p.center(Scale.MOLECULE)

        # Compute a loss on centered coordinates
        loss = centered.coordinates.sum()
        loss.backward()

        assert coords.grad is not None, "Gradients were not computed"

    def test_reduce_is_differentiable(self, any_polymer_torch):
        """Test that reduce() supports backpropagation."""
        import torch
        from ciffy import Scale

        coords = any_polymer_torch.coordinates.clone().detach().requires_grad_(True)
        p = any_polymer_torch.with_coordinates(coords)

        # Reduce to chain level
        chain_means = p.reduce(coords, Scale.CHAIN)

        loss = chain_means.sum()
        loss.backward()

        assert coords.grad is not None, "Gradients were not computed"

    @requires_cuda
    def test_rmsd_differentiable_on_cuda(self, any_polymer_torch):
        """Test RMSD differentiability on CUDA."""
        import torch
        import ciffy

        p1 = any_polymer_torch.to("cuda")
        coords2 = p1.coordinates.clone().detach().requires_grad_(True)
        p2 = p1.with_coordinates(coords2 + torch.randn_like(coords2) * 0.1)

        rmsd_sq = ciffy.rmsd(p1, p2, ciffy.MOLECULE)
        rmsd_sq.sum().backward()

        assert coords2.grad is not None
        assert coords2.grad.device.type == "cuda"

    def test_rmsd_gradient_stability_small_perturbation(self, any_polymer_torch):
        """Test gradient stability with near-identical structures.

        When structures are nearly identical, the covariance matrix approaches
        a scaled identity, making singular values nearly equal. This can cause
        SVD gradient instability. We verify gradients remain finite.
        """
        import torch
        import ciffy

        p1 = any_polymer_torch
        # Very small perturbation - this is the challenging case for SVD gradients
        coords2 = p1.coordinates.clone().detach().requires_grad_(True)
        perturbation = torch.randn_like(coords2) * 1e-6
        p2 = p1.with_coordinates(coords2 + perturbation)

        rmsd_sq = ciffy.rmsd(p1, p2, ciffy.MOLECULE)
        rmsd_sq.sum().backward()

        # Gradients should exist and be finite (no NaN or Inf)
        assert coords2.grad is not None, "Gradients were not computed"
        assert torch.isfinite(coords2.grad).all(), "Gradients contain NaN or Inf"

    def test_rmsd_gradient_stability_identical_structures(self, any_polymer_torch):
        """Test gradient stability with exactly identical structures.

        The degenerate case where structures are identical. The RMSD is 0,
        but gradients should still be computable and finite.
        """
        import torch
        import ciffy

        p1 = any_polymer_torch
        coords2 = p1.coordinates.clone().detach().requires_grad_(True)
        p2 = p1.with_coordinates(coords2)

        rmsd_sq = ciffy.rmsd(p1, p2, ciffy.MOLECULE)

        # RMSD should be essentially zero, but float32 accumulation errors
        # grow with structure size. Tolerance scales as sqrt(N) * eps * scale.
        n_atoms = p1.size()
        # For 86k atoms, this gives ~0.003 tolerance (float32 eps ~1e-7, coords ~100A)
        tolerance = max(1e-6, (n_atoms ** 0.5) * 1e-7 * 100)
        assert rmsd_sq.item() < tolerance, f"RMSD {rmsd_sq.item()} >= {tolerance} for {n_atoms} atoms"

        rmsd_sq.sum().backward()

        # Gradients should be finite (may be zero, but not NaN/Inf)
        assert coords2.grad is not None, "Gradients were not computed"
        assert torch.isfinite(coords2.grad).all(), "Gradients contain NaN or Inf"

    def test_rmsd_gradient_magnitude_bounded(self, any_polymer_torch):
        """Test that gradient magnitudes are reasonable (not exploding)."""
        import torch
        import ciffy

        p1 = any_polymer_torch
        coords2 = p1.coordinates.clone().detach().requires_grad_(True)
        # Moderate perturbation
        p2 = p1.with_coordinates(coords2 + torch.randn_like(coords2) * 0.5)

        rmsd_sq = ciffy.rmsd(p1, p2, ciffy.MOLECULE)
        rmsd_sq.sum().backward()

        # Gradient magnitude should be bounded (not exploding)
        grad_norm = coords2.grad.norm()
        assert grad_norm < 1e6, f"Gradient norm too large: {grad_norm}"
        assert torch.isfinite(grad_norm), "Gradient norm is not finite"

    def test_rmsd_gradient_stability_single_chain(self, any_polymer_torch):
        """Test gradient stability on single-chain polymer."""
        import torch
        import ciffy

        # Select single chain
        p1 = any_polymer_torch.by_index(0)
        coords2 = p1.coordinates.clone().detach().requires_grad_(True)
        p2 = p1.with_coordinates(coords2 + torch.randn_like(coords2) * 0.1)

        rmsd_sq = ciffy.rmsd(p1, p2, ciffy.MOLECULE)
        rmsd_sq.sum().backward()

        assert coords2.grad is not None
        assert torch.isfinite(coords2.grad).all(), "Gradients contain NaN or Inf"


class TestScatterOperations:
    """Test scatter operations directly."""

    @pytest.mark.parametrize("device", GPU_DEVICES)
    def test_scatter_sum(self, device):
        """Test scatter_sum on GPU devices."""
        skip_if_no_device(device)
        import torch
        from ciffy.backend import scatter_sum

        features = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], device=device)
        index = torch.tensor([0, 0, 1], device=device)

        result = scatter_sum(features, index, dim_size=2)

        expected = torch.tensor([[4., 6.], [5., 6.]], device=device)
        # MPS requires CPU comparison due to precision differences
        if device == "mps":
            assert torch.allclose(result.cpu(), expected.cpu())
        else:
            assert torch.allclose(result, expected)

    @pytest.mark.parametrize("device", GPU_DEVICES)
    def test_scatter_mean(self, device):
        """Test scatter_mean on GPU devices."""
        skip_if_no_device(device)
        import torch
        from ciffy.backend import scatter_mean

        features = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], device=device)
        index = torch.tensor([0, 0, 1], device=device)

        result = scatter_mean(features, index, dim_size=2)

        expected = torch.tensor([[2., 3.], [5., 6.]], device=device)
        # MPS requires CPU comparison due to precision differences
        if device == "mps":
            assert torch.allclose(result.cpu(), expected.cpu())
        else:
            assert torch.allclose(result, expected)
