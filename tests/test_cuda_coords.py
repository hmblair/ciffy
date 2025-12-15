"""Tests for CUDA coordinate conversion operations.

These tests verify that:
1. CUDA implementations match CPU implementations numerically
2. Gradients are correct (via gradcheck and comparison)
3. Round-trip conversions work correctly through the Polymer interface

Tests are organized into:
- TestAutogradFunctions: Direct tests of autograd functions (CPU vs CUDA comparison)
- TestPolymerRoundTrip: Tests through Polymer interface with proper cache invalidation
- TestGradients: Gradient correctness tests
- TestEdgeCases: Edge cases and stress tests
"""

import pytest
import numpy as np

from tests.utils import get_test_cif


# =============================================================================
# Device detection and skip markers
# =============================================================================

def cuda_available():
    """Check if PyTorch CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def cuda_extension_available():
    """Check if ciffy CUDA extension is built and available."""
    try:
        from ciffy.backend.cuda_ops import HAS_CUDA_EXTENSION
        return HAS_CUDA_EXTENSION
    except ImportError:
        return False


requires_cuda = pytest.mark.skipif(
    not cuda_available(),
    reason="CUDA not available"
)

requires_cuda_extension = pytest.mark.skipif(
    not cuda_extension_available(),
    reason="CUDA extension not built"
)


# =============================================================================
# Test: Autograd Functions (CPU vs CUDA comparison)
# =============================================================================

class TestAutogradFunctions:
    """Test autograd functions produce identical results on CPU and CUDA.

    These tests call the autograd functions directly (not through Polymer),
    comparing CUDA results against CPU results.
    """

    @requires_cuda
    @requires_cuda_extension
    def test_cartesian_to_internal_simple(self):
        """Test cartesian_to_internal matches CPU for simple chain."""
        import torch
        from ciffy.backend.autograd import cartesian_to_internal

        coords_np = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.0, 1.5, 0.0],
            [3.5, 1.5, 1.0]
        ], dtype=np.float32)

        indices_np = np.array([
            [0, -1, -1, -1],
            [1,  0, -1, -1],
            [2,  1,  0, -1],
            [3,  2,  1,  0],
        ], dtype=np.int64)

        coords_cpu = torch.tensor(coords_np, dtype=torch.float32)
        indices = torch.tensor(indices_np, dtype=torch.int64)

        # CPU computation
        d_cpu, a_cpu, dh_cpu = cartesian_to_internal(coords_cpu, indices)

        # CUDA computation
        d_cuda, a_cuda, dh_cuda = cartesian_to_internal(
            coords_cpu.cuda(), indices.cuda()
        )

        assert torch.allclose(d_cpu, d_cuda.cpu(), atol=1e-5)
        assert torch.allclose(a_cpu, a_cuda.cpu(), atol=1e-5)
        assert torch.allclose(dh_cpu, dh_cuda.cpu(), atol=1e-5)

    @requires_cuda
    @requires_cuda_extension
    def test_cartesian_to_internal_polymer(self):
        """Test cartesian_to_internal matches CPU for real polymer."""
        import torch
        from ciffy import from_sequence
        from ciffy.backend.autograd import cartesian_to_internal

        polymer = from_sequence("acgu", backend="torch")
        coords = polymer.coordinates.float()

        # Build indices from Z-matrix
        cm = polymer._coord_manager
        zmat = cm.zmatrix
        indices = torch.stack([
            torch.from_numpy(zmat.atom_indices.astype(np.int64)),
            torch.from_numpy(zmat.distance_refs.astype(np.int64)),
            torch.from_numpy(zmat.angle_refs.astype(np.int64)),
            torch.from_numpy(zmat.dihedral_refs.astype(np.int64)),
        ], dim=1)

        # CPU computation
        d_cpu, a_cpu, dh_cpu = cartesian_to_internal(coords, indices)

        # CUDA computation
        d_cuda, a_cuda, dh_cuda = cartesian_to_internal(
            coords.cuda(), indices.cuda()
        )

        assert torch.allclose(d_cpu, d_cuda.cpu(), atol=1e-4)
        assert torch.allclose(a_cpu, a_cuda.cpu(), atol=1e-4)
        assert torch.allclose(dh_cpu, dh_cuda.cpu(), atol=1e-4)

    @requires_cuda
    @requires_cuda_extension
    def test_nerf_reconstruct_simple(self):
        """Test nerf_reconstruct matches CPU for simple chain."""
        import torch
        from ciffy.backend.autograd import nerf_reconstruct

        n_atoms = 4
        indices_np = np.array([
            [0, -1, -1, -1],
            [1,  0, -1, -1],
            [2,  1,  0, -1],
            [3,  2,  1,  0],
        ], dtype=np.int64)

        distances = torch.tensor([0.0, 1.5, 1.5, 1.5], dtype=torch.float32)
        angles = torch.tensor([0.0, 0.0, 1.91, 1.91], dtype=torch.float32)
        dihedrals = torch.tensor([0.0, 0.0, 0.0, 1.57], dtype=torch.float32)
        indices = torch.tensor(indices_np, dtype=torch.int64)

        # CPU reconstruction
        coords_cpu = nerf_reconstruct(indices, distances, angles, dihedrals, n_atoms)

        # CUDA reconstruction
        coords_cuda = nerf_reconstruct(
            indices.cuda(), distances.cuda(), angles.cuda(), dihedrals.cuda(), n_atoms
        )

        assert torch.allclose(coords_cpu, coords_cuda.cpu(), atol=1e-5)

    @requires_cuda
    @requires_cuda_extension
    def test_autograd_roundtrip(self):
        """Test full round-trip through autograd functions matches CPU.

        Note: This tests the raw autograd functions, not Polymer caching.
        The functions always compute (no caching).
        """
        import torch
        from ciffy import from_sequence
        from ciffy.backend.autograd import cartesian_to_internal, nerf_reconstruct

        polymer = from_sequence("acgu", backend="torch")
        coords_orig = polymer.coordinates.float()

        cm = polymer._coord_manager
        zmat = cm.zmatrix
        indices = torch.stack([
            torch.from_numpy(zmat.atom_indices.astype(np.int64)),
            torch.from_numpy(zmat.distance_refs.astype(np.int64)),
            torch.from_numpy(zmat.angle_refs.astype(np.int64)),
            torch.from_numpy(zmat.dihedral_refs.astype(np.int64)),
        ], dim=1)

        n_atoms = len(coords_orig)

        # CPU round-trip
        d, a, dh = cartesian_to_internal(coords_orig, indices)
        coords_cpu = nerf_reconstruct(indices, d, a, dh, n_atoms)

        # CUDA round-trip
        d, a, dh = cartesian_to_internal(coords_orig.cuda(), indices.cuda())
        coords_cuda = nerf_reconstruct(indices.cuda(), d, a, dh, n_atoms)

        assert torch.allclose(coords_cpu, coords_cuda.cpu(), atol=1e-4)


# =============================================================================
# Test: Polymer Round-Trip (with proper cache invalidation)
# =============================================================================

class TestPolymerRoundTrip:
    """Test round-trip conversions through the Polymer interface.

    These tests properly invalidate the coordinate cache by setting
    dihedrals, which triggers reconstruction on the next .coordinates access.
    """

    @requires_cuda
    @requires_cuda_extension
    def test_roundtrip_on_cuda(self):
        """Test round-trip on CUDA through Polymer interface."""
        import torch
        from ciffy import from_sequence
        from ciffy.operations.alignment import kabsch_align

        polymer = from_sequence("acgu", backend="torch").to("cuda")
        orig_coords = polymer.coordinates.clone()

        # Setting dihedrals invalidates Cartesian cache
        dihedrals = polymer.dihedrals.clone()
        polymer.dihedrals = dihedrals  # This marks coordinates as dirty

        # Accessing coordinates triggers reconstruction
        new_coords = polymer.coordinates
        assert new_coords.device.type == "cuda"

        # Compare using Kabsch alignment
        aligned, _, _ = kabsch_align(new_coords.cpu(), orig_coords.cpu())
        rmsd = torch.sqrt(((aligned - orig_coords.cpu()) ** 2).sum(dim=1).mean())

        assert rmsd < 1e-4, f"CUDA round-trip RMSD {rmsd} exceeds threshold"

    @requires_cuda
    @requires_cuda_extension
    def test_roundtrip_matches_cpu(self):
        """Test CUDA round-trip produces same result as CPU."""
        import torch
        from ciffy import from_sequence
        from ciffy.operations.alignment import kabsch_align

        # CPU round-trip
        polymer_cpu = from_sequence("acgu", backend="torch")
        orig_coords_cpu = polymer_cpu.coordinates.clone()
        dihedrals_cpu = polymer_cpu.dihedrals.clone()
        polymer_cpu.dihedrals = dihedrals_cpu
        coords_cpu = polymer_cpu.coordinates

        # CUDA round-trip
        polymer_cuda = from_sequence("acgu", backend="torch").to("cuda")
        dihedrals_cuda = polymer_cuda.dihedrals.clone()
        polymer_cuda.dihedrals = dihedrals_cuda
        coords_cuda = polymer_cuda.coordinates.cpu()

        # Both should produce similar results
        aligned_cpu, _, _ = kabsch_align(coords_cpu, orig_coords_cpu)
        aligned_cuda, _, _ = kabsch_align(coords_cuda, orig_coords_cpu)

        rmsd_cpu = torch.sqrt(((aligned_cpu - orig_coords_cpu) ** 2).sum(dim=1).mean())
        rmsd_cuda = torch.sqrt(((aligned_cuda - orig_coords_cpu) ** 2).sum(dim=1).mean())

        assert rmsd_cpu < 1e-4
        assert rmsd_cuda < 1e-4
        # CUDA and CPU results should be very close
        assert abs(rmsd_cpu - rmsd_cuda) < 1e-4

    @requires_cuda
    @requires_cuda_extension
    def test_internal_coords_on_cuda(self):
        """Test accessing internal coordinates on CUDA device."""
        import torch
        from ciffy import from_sequence

        polymer = from_sequence("acgu", backend="torch").to("cuda")

        distances = polymer.distances
        angles = polymer.angles
        dihedrals = polymer.dihedrals

        assert distances.device.type == "cuda"
        assert angles.device.type == "cuda"
        assert dihedrals.device.type == "cuda"

        # Values should be reasonable
        assert torch.all(distances >= 0)
        assert torch.all(angles >= 0) and torch.all(angles <= np.pi + 1e-5)
        assert torch.all(dihedrals >= -np.pi - 1e-5) and torch.all(dihedrals <= np.pi + 1e-5)

    @requires_cuda
    @requires_cuda_extension
    def test_pdb_roundtrip_on_cuda(self):
        """Test round-trip on real PDB structure."""
        import torch
        from ciffy import load
        from ciffy.operations.alignment import kabsch_align

        polymer = load(get_test_cif("1ZEW")).poly().torch().to("cuda")
        orig_coords = polymer.coordinates.clone()

        # Invalidate cache
        dihedrals = polymer.dihedrals.clone()
        polymer.dihedrals = dihedrals

        # Trigger reconstruction
        new_coords = polymer.coordinates

        aligned, _, _ = kabsch_align(new_coords.cpu(), orig_coords.cpu())
        rmsd = torch.sqrt(((aligned - orig_coords.cpu()) ** 2).sum(dim=1).mean())

        assert rmsd < 1e-3, f"PDB round-trip RMSD {rmsd} exceeds threshold"


# =============================================================================
# Test: Gradients
# =============================================================================

class TestGradients:
    """Test gradient correctness for CUDA operations."""

    @requires_cuda
    @requires_cuda_extension
    def test_cartesian_to_internal_gradcheck(self):
        """Test cartesian_to_internal gradients via gradcheck."""
        import torch
        from ciffy.backend.autograd import cartesian_to_internal

        coords_np = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.0, 1.5, 0.0],
            [3.5, 1.5, 1.0]
        ], dtype=np.float32)

        indices_np = np.array([
            [0, -1, -1, -1],
            [1,  0, -1, -1],
            [2,  1,  0, -1],
            [3,  2,  1,  0],
        ], dtype=np.int64)

        indices = torch.tensor(indices_np, dtype=torch.int64, device="cuda")

        def wrapper(coords):
            coords32 = coords.float()
            d, a, dh = cartesian_to_internal(coords32, indices)
            return d.double(), a.double(), dh.double()

        coords_check = torch.tensor(
            coords_np, requires_grad=True, dtype=torch.float64, device="cuda"
        )
        assert torch.autograd.gradcheck(
            wrapper, coords_check, eps=1e-4, atol=1e-3, rtol=1e-2
        )

    @requires_cuda
    @requires_cuda_extension
    def test_nerf_reconstruct_gradcheck(self):
        """Test nerf_reconstruct gradients via gradcheck."""
        import torch
        from ciffy.backend.autograd import nerf_reconstruct

        n_atoms = 4
        indices_np = np.array([
            [0, -1, -1, -1],
            [1,  0, -1, -1],
            [2,  1,  0, -1],
            [3,  2,  1,  0],
        ], dtype=np.int64)

        distances_np = np.array([0.0, 1.5, 1.5, 1.5], dtype=np.float32)
        angles_np = np.array([0.0, 0.0, 1.91, 1.91], dtype=np.float32)
        dihedrals_np = np.array([0.0, 0.0, 0.0, 1.57], dtype=np.float32)

        indices = torch.tensor(indices_np, dtype=torch.int64, device="cuda")

        def wrapper(distances, angles, dihedrals):
            d32 = distances.float()
            a32 = angles.float()
            dh32 = dihedrals.float()
            coords = nerf_reconstruct(indices, d32, a32, dh32, n_atoms)
            return coords.double()

        distances_check = torch.tensor(distances_np, requires_grad=True, dtype=torch.float64, device="cuda")
        angles_check = torch.tensor(angles_np, requires_grad=True, dtype=torch.float64, device="cuda")
        dihedrals_check = torch.tensor(dihedrals_np, requires_grad=True, dtype=torch.float64, device="cuda")

        assert torch.autograd.gradcheck(
            wrapper, (distances_check, angles_check, dihedrals_check),
            eps=1e-4, atol=1e-3, rtol=1e-2
        )

    @requires_cuda
    @requires_cuda_extension
    def test_cartesian_to_internal_grad_matches_cpu(self):
        """Test CUDA gradients match CPU gradients for cartesian_to_internal."""
        import torch
        from ciffy.backend.autograd import cartesian_to_internal

        coords_np = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.3, 0.2],
            [2.0, 1.5, 0.1],
            [3.5, 1.5, 1.0],
            [4.0, 2.5, 1.5],
        ], dtype=np.float32)

        indices_np = np.array([
            [0, -1, -1, -1],
            [1,  0, -1, -1],
            [2,  1,  0, -1],
            [3,  2,  1,  0],
            [4,  3,  2,  1],
        ], dtype=np.int64)

        indices = torch.tensor(indices_np, dtype=torch.int64)

        # CPU gradient
        coords_cpu = torch.tensor(coords_np, requires_grad=True)
        d, a, dh = cartesian_to_internal(coords_cpu, indices)
        loss = d.sum() + a.sum() + dh.sum()
        loss.backward()
        grad_cpu = coords_cpu.grad.clone()

        # CUDA gradient
        coords_cuda = torch.tensor(coords_np, device="cuda", requires_grad=True)
        d, a, dh = cartesian_to_internal(coords_cuda, indices.cuda())
        loss = d.sum() + a.sum() + dh.sum()
        loss.backward()
        grad_cuda = coords_cuda.grad.cpu()

        assert torch.allclose(grad_cpu, grad_cuda, atol=1e-4)

    @requires_cuda
    @requires_cuda_extension
    def test_nerf_reconstruct_grad_matches_cpu(self):
        """Test CUDA gradients match CPU gradients for nerf_reconstruct."""
        import torch
        from ciffy.backend.autograd import nerf_reconstruct

        n_atoms = 4
        indices_np = np.array([
            [0, -1, -1, -1],
            [1,  0, -1, -1],
            [2,  1,  0, -1],
            [3,  2,  1,  0],
        ], dtype=np.int64)

        distances_np = np.array([0.0, 1.5, 1.5, 1.5], dtype=np.float32)
        angles_np = np.array([0.0, 0.0, 1.91, 1.91], dtype=np.float32)
        dihedrals_np = np.array([0.0, 0.0, 0.0, 1.57], dtype=np.float32)

        # CPU gradient
        indices = torch.tensor(indices_np)
        distances_cpu = torch.tensor(distances_np, requires_grad=True)
        angles_cpu = torch.tensor(angles_np, requires_grad=True)
        dihedrals_cpu = torch.tensor(dihedrals_np, requires_grad=True)

        coords = nerf_reconstruct(indices, distances_cpu, angles_cpu, dihedrals_cpu, n_atoms)
        coords.sum().backward()

        grad_d_cpu = distances_cpu.grad.clone()
        grad_a_cpu = angles_cpu.grad.clone()
        grad_dh_cpu = dihedrals_cpu.grad.clone()

        # CUDA gradient
        distances_cuda = torch.tensor(distances_np, device="cuda", requires_grad=True)
        angles_cuda = torch.tensor(angles_np, device="cuda", requires_grad=True)
        dihedrals_cuda = torch.tensor(dihedrals_np, device="cuda", requires_grad=True)

        coords = nerf_reconstruct(
            indices.cuda(), distances_cuda, angles_cuda, dihedrals_cuda, n_atoms
        )
        coords.sum().backward()

        assert torch.allclose(grad_d_cpu, distances_cuda.grad.cpu(), atol=1e-4)
        assert torch.allclose(grad_a_cpu, angles_cuda.grad.cpu(), atol=1e-4)
        assert torch.allclose(grad_dh_cpu, dihedrals_cuda.grad.cpu(), atol=1e-4)

    @requires_cuda
    @requires_cuda_extension
    def test_differentiability_through_polymer(self):
        """Test gradient flow through Polymer reconstruction on CUDA."""
        import torch
        from ciffy import from_sequence

        polymer = from_sequence("acgu", backend="torch").to("cuda")

        # Enable gradients on dihedrals
        dihedrals = polymer.dihedrals.clone()
        dihedrals.requires_grad_(True)
        polymer.dihedrals = dihedrals

        # Access coordinates (triggers reconstruction)
        coords = polymer.coordinates

        # Compute loss and backpropagate
        loss = coords.pow(2).mean()
        loss.backward()

        # Gradients should exist and be on CUDA
        assert dihedrals.grad is not None
        assert dihedrals.grad.device.type == "cuda"
        assert not torch.all(dihedrals.grad == 0)


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests for CUDA operations."""

    @requires_cuda
    @requires_cuda_extension
    def test_single_atom(self):
        """Test handling of single-atom structure."""
        import torch
        from ciffy.backend.autograd import cartesian_to_internal

        coords = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda")
        indices = torch.tensor([[0, -1, -1, -1]], dtype=torch.int64, device="cuda")

        d, a, dh = cartesian_to_internal(coords, indices)

        assert len(d) == 1
        assert len(a) == 1
        assert len(dh) == 1

    @requires_cuda
    @requires_cuda_extension
    def test_two_atoms(self):
        """Test handling of two-atom structure."""
        import torch
        from ciffy.backend.autograd import cartesian_to_internal

        coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0]
        ], dtype=torch.float32, device="cuda")
        indices = torch.tensor([
            [0, -1, -1, -1],
            [1,  0, -1, -1]
        ], dtype=torch.int64, device="cuda")

        d, a, dh = cartesian_to_internal(coords, indices)

        assert len(d) == 2
        assert d[1].item() == pytest.approx(1.5, abs=1e-5)

    @requires_cuda
    @requires_cuda_extension
    def test_collinear_atoms(self):
        """Test handling of collinear atoms (edge case for angles)."""
        import torch
        from ciffy.backend.autograd import cartesian_to_internal

        coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0]
        ], dtype=torch.float32, device="cuda")
        indices = torch.tensor([
            [0, -1, -1, -1],
            [1,  0, -1, -1],
            [2,  1,  0, -1]
        ], dtype=torch.int64, device="cuda")

        d, a, dh = cartesian_to_internal(coords, indices)

        # Angle should be pi (180 degrees)
        # Note: Collinear atoms are numerically unstable; allow looser tolerance
        assert a[2].item() == pytest.approx(np.pi, abs=1e-2)

    @requires_cuda
    @requires_cuda_extension
    def test_large_structure(self):
        """Test CUDA handles large structures."""
        import torch
        from ciffy.backend.autograd import cartesian_to_internal, nerf_reconstruct

        n_atoms = 1000
        coords_np = np.random.randn(n_atoms, 3).astype(np.float32) * 10

        # Create linear Z-matrix
        indices_np = np.zeros((n_atoms, 4), dtype=np.int64)
        indices_np[:, 0] = np.arange(n_atoms)
        indices_np[1:, 1] = np.arange(n_atoms - 1)
        indices_np[2:, 2] = np.arange(n_atoms - 2)
        indices_np[3:, 3] = np.arange(n_atoms - 3)
        indices_np[0, 1:] = -1
        indices_np[1, 2:] = -1
        indices_np[2, 3] = -1

        coords_cuda = torch.tensor(coords_np, device="cuda")
        indices_cuda = torch.tensor(indices_np, device="cuda")

        d, a, dh = cartesian_to_internal(coords_cuda, indices_cuda)
        assert len(d) == n_atoms

        coords_recon = nerf_reconstruct(indices_cuda, d, a, dh, n_atoms)
        assert coords_recon.shape == (n_atoms, 3)

    @requires_cuda
    @requires_cuda_extension
    def test_cuda_cpu_transfer(self):
        """Test moving between CPU and CUDA preserves internal coordinates."""
        import torch
        from ciffy import from_sequence

        polymer_cpu = from_sequence("acgu", backend="torch")
        dihedrals_cpu = polymer_cpu.dihedrals.clone()

        polymer_cuda = polymer_cpu.to("cuda")
        dihedrals_cuda = polymer_cuda.dihedrals

        assert torch.allclose(dihedrals_cpu, dihedrals_cuda.cpu(), atol=1e-5)

        # Move back to CPU
        polymer_back = polymer_cuda.to("cpu")
        dihedrals_back = polymer_back.dihedrals

        assert torch.allclose(dihedrals_cpu, dihedrals_back, atol=1e-5)
