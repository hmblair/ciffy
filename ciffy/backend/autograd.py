"""
PyTorch autograd functions for internal coordinate conversions.

Provides custom autograd.Function implementations that use C backward passes
for efficient gradient computation through the internal coordinate pipeline.

Functions
---------
cartesian_to_internal
    Convert Cartesian coordinates to internal coordinates (distances, angles,
    dihedrals) with full autograd support. Gradients are computed analytically
    in C for efficiency.

nerf_reconstruct
    Reconstruct Cartesian coordinates from internal coordinates using the NERF
    algorithm. Supports autograd for gradients w.r.t. distances, angles, and
    dihedrals.

Gradient Computation
--------------------
The backward passes are implemented by composing primitive operations:

- **Cross product**: ∂L/∂a = b × grad, ∂L/∂b = grad × a
- **Normalize**: ∂L/∂v = (grad - v̂(v̂·grad)) / |v|
- **Dot product**: ∂L/∂a = grad·b, ∂L/∂b = grad·a
- **atan2**: ∂L/∂y = grad·x/(x²+y²), ∂L/∂x = -grad·y/(x²+y²)

This composition approach ensures numerical correctness by matching the exact
forward computation graph.

Example
-------
>>> import torch
>>> from ciffy.backend.autograd import cartesian_to_internal
>>>
>>> # Create coordinates with gradient tracking
>>> coords = torch.randn(10, 3, requires_grad=True)
>>> indices = torch.tensor([[i, i-1, i-2, i-3] for i in range(3, 10)])
>>>
>>> # Convert to internal coordinates
>>> distances, angles, dihedrals = cartesian_to_internal(coords, indices)
>>>
>>> # Compute loss and backpropagate
>>> loss = dihedrals.sum()
>>> loss.backward()
>>>
>>> # Gradients are now available
>>> print(coords.grad.shape)  # (10, 3)

Notes
-----
- Requires PyTorch and the ciffy C extension to be installed.
- All operations use float32 precision internally.
- The NERF backward pass has approximate gradients for the first 2-3 atoms
  in each chain due to the underdetermined frame construction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "cartesian_to_internal",
    "nerf_reconstruct",
    "CartesianToInternalFunction",
    "NerfReconstructFunction",
    "HAS_TORCH",
    "HAS_C_EXTENSION",
    "HAS_CUDA_EXTENSION",
]

try:
    import torch
    from torch.autograd import Function
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Function = object  # Dummy for type hints

# Import C extension functions
try:
    from .._c import (
        _cartesian_to_internal,
        _cartesian_to_internal_backward,
        _nerf_reconstruct,
        _nerf_reconstruct_backward,
    )
    HAS_C_EXTENSION = True
except ImportError:
    HAS_C_EXTENSION = False

# Import CUDA extension functions
try:
    from .cuda_ops import (
        HAS_CUDA_EXTENSION,
        is_cuda_available,
        cuda_cartesian_to_internal,
        cuda_cartesian_to_internal_backward,
        cuda_nerf_reconstruct,
        cuda_nerf_reconstruct_backward,
    )
except ImportError:
    HAS_CUDA_EXTENSION = False
    is_cuda_available = lambda x: False


class CartesianToInternalFunction(Function):
    """
    Autograd function for Cartesian to internal coordinate conversion.

    Forward: coords -> (distances, angles, dihedrals)
    Backward: grad_internal -> grad_coords
    """

    @staticmethod
    def forward(
        ctx: Any,
        coords: "torch.Tensor",
        indices: "torch.Tensor",
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        Convert Cartesian coordinates to internal coordinates.

        Args:
            ctx: Autograd context for saving tensors.
            coords: (N, 3) float32 tensor of Cartesian coordinates.
            indices: (M, 4) int64 tensor of Z-matrix indices.

        Returns:
            Tuple of (distances, angles, dihedrals), each (M,) float32.
        """
        # Check if we can use CUDA path
        use_cuda = is_cuda_available(coords)
        ctx.use_cuda = use_cuda

        if use_cuda:
            # GPU path: stay on device
            distances, angles, dihedrals = cuda_cartesian_to_internal(coords, indices)
        else:
            # CPU path: convert to numpy for C extension
            coords_np = coords.detach().cpu().numpy().astype(np.float32)
            indices_np = indices.detach().cpu().numpy().astype(np.int64)

            # Call C extension
            distances_np, angles_np, dihedrals_np = _cartesian_to_internal(
                coords_np, indices_np
            )

            # Convert back to tensors
            device = coords.device
            distances = torch.from_numpy(distances_np).to(device)
            angles = torch.from_numpy(angles_np).to(device)
            dihedrals = torch.from_numpy(dihedrals_np).to(device)

        # Save for backward
        ctx.save_for_backward(coords, indices, distances, angles)

        return distances, angles, dihedrals

    @staticmethod
    def backward(
        ctx: Any,
        grad_distances: "torch.Tensor",
        grad_angles: "torch.Tensor",
        grad_dihedrals: "torch.Tensor",
    ) -> tuple["torch.Tensor", None]:
        """
        Backward pass for Cartesian to internal conversion.

        Args:
            ctx: Autograd context with saved tensors.
            grad_distances: (M,) upstream gradients for distances.
            grad_angles: (M,) upstream gradients for angles.
            grad_dihedrals: (M,) upstream gradients for dihedrals.

        Returns:
            Tuple of (grad_coords, None) - None for indices (not differentiable).
        """
        coords, indices, distances, angles = ctx.saved_tensors

        if ctx.use_cuda:
            # GPU path: stay on device
            # Ensure gradients are contiguous (autograd may provide non-contiguous tensors)
            grad_coords = cuda_cartesian_to_internal_backward(
                coords, indices, distances, angles,
                grad_distances.contiguous(),
                grad_angles.contiguous(),
                grad_dihedrals.contiguous()
            )
        else:
            # CPU path: convert to numpy
            coords_np = coords.detach().cpu().numpy().astype(np.float32)
            indices_np = indices.detach().cpu().numpy().astype(np.int64)
            distances_np = distances.detach().cpu().numpy().astype(np.float32)
            angles_np = angles.detach().cpu().numpy().astype(np.float32)
            grad_distances_np = grad_distances.detach().cpu().numpy().astype(np.float32)
            grad_angles_np = grad_angles.detach().cpu().numpy().astype(np.float32)
            grad_dihedrals_np = grad_dihedrals.detach().cpu().numpy().astype(np.float32)

            # Call C backward
            grad_coords_np = _cartesian_to_internal_backward(
                coords_np, indices_np, distances_np, angles_np,
                grad_distances_np, grad_angles_np, grad_dihedrals_np
            )

            # Convert back to tensor
            grad_coords = torch.from_numpy(grad_coords_np).to(coords.device)

        return grad_coords, None


class NerfReconstructFunction(Function):
    """
    Autograd function for NERF reconstruction.

    Forward: (distances, angles, dihedrals) -> coords
    Backward: grad_coords -> (grad_distances, grad_angles, grad_dihedrals)
    """

    @staticmethod
    def forward(
        ctx: Any,
        indices: "torch.Tensor",
        distances: "torch.Tensor",
        angles: "torch.Tensor",
        dihedrals: "torch.Tensor",
        n_atoms: int,
    ) -> "torch.Tensor":
        """
        Reconstruct Cartesian coordinates from internal coordinates.

        Args:
            ctx: Autograd context for saving tensors.
            indices: (M, 4) int64 tensor of Z-matrix indices.
            distances: (M,) float32 tensor of bond lengths.
            angles: (M,) float32 tensor of bond angles.
            dihedrals: (M,) float32 tensor of dihedral angles.
            n_atoms: Total number of atoms.

        Returns:
            coords: (N, 3) float32 tensor of Cartesian coordinates.
        """
        # Check if we can use CUDA path
        use_cuda = is_cuda_available(distances)
        ctx.use_cuda = use_cuda

        if use_cuda:
            # GPU path: stay on device
            # Allocate output tensor on same device
            coords = torch.zeros(n_atoms, 3, dtype=torch.float32, device=distances.device)
            cuda_nerf_reconstruct(coords, indices, distances, angles, dihedrals)
        else:
            # CPU path: convert to numpy
            indices_np = indices.detach().cpu().numpy().astype(np.int64)
            distances_np = distances.detach().cpu().numpy().astype(np.float32)
            angles_np = angles.detach().cpu().numpy().astype(np.float32)
            dihedrals_np = dihedrals.detach().cpu().numpy().astype(np.float32)

            # Call C extension
            coords_np = _nerf_reconstruct(
                indices_np, distances_np, angles_np, dihedrals_np, n_atoms
            )

            # Convert back to tensor
            device = distances.device
            coords = torch.from_numpy(coords_np).to(device)

        # Save for backward
        ctx.save_for_backward(coords, indices, distances, angles, dihedrals)
        ctx.n_atoms = n_atoms

        return coords

    @staticmethod
    def backward(
        ctx: Any,
        grad_coords: "torch.Tensor",
    ) -> tuple[None, "torch.Tensor", "torch.Tensor", "torch.Tensor", None]:
        """
        Backward pass for NERF reconstruction.

        Args:
            ctx: Autograd context with saved tensors.
            grad_coords: (N, 3) upstream gradients for coordinates.

        Returns:
            Tuple of (None, grad_distances, grad_angles, grad_dihedrals, None).
            None for indices and n_atoms (not differentiable).
        """
        coords, indices, distances, angles, dihedrals = ctx.saved_tensors

        if ctx.use_cuda:
            # GPU path: stay on device
            # Ensure gradients are contiguous (autograd may provide non-contiguous tensors)
            _, grad_distances, grad_angles, grad_dihedrals = cuda_nerf_reconstruct_backward(
                coords, indices, distances, angles, dihedrals, grad_coords.contiguous()
            )
        else:
            # CPU path: convert to numpy
            coords_np = coords.detach().cpu().numpy().astype(np.float32)
            indices_np = indices.detach().cpu().numpy().astype(np.int64)
            distances_np = distances.detach().cpu().numpy().astype(np.float32)
            angles_np = angles.detach().cpu().numpy().astype(np.float32)
            dihedrals_np = dihedrals.detach().cpu().numpy().astype(np.float32)
            grad_coords_np = grad_coords.detach().cpu().numpy().astype(np.float32).copy()

            # Call C backward
            grad_distances_np, grad_angles_np, grad_dihedrals_np = _nerf_reconstruct_backward(
                coords_np, indices_np, distances_np, angles_np, dihedrals_np,
                grad_coords_np
            )

            # Convert back to tensors
            device = distances.device
            grad_distances = torch.from_numpy(grad_distances_np).to(device)
            grad_angles = torch.from_numpy(grad_angles_np).to(device)
            grad_dihedrals = torch.from_numpy(grad_dihedrals_np).to(device)

        return None, grad_distances, grad_angles, grad_dihedrals, None


def cartesian_to_internal(
    coords: "torch.Tensor",
    indices: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    Convert Cartesian coordinates to internal coordinates with autograd support.

    Args:
        coords: (N, 3) float32 tensor of Cartesian coordinates.
        indices: (M, 4) int64 tensor of Z-matrix indices.

    Returns:
        Tuple of (distances, angles, dihedrals), each (M,) float32.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for this function")
    if not HAS_C_EXTENSION:
        raise ImportError("C extension is required for this function")

    return CartesianToInternalFunction.apply(coords, indices)


def nerf_reconstruct(
    indices: "torch.Tensor",
    distances: "torch.Tensor",
    angles: "torch.Tensor",
    dihedrals: "torch.Tensor",
    n_atoms: int,
) -> "torch.Tensor":
    """
    Reconstruct Cartesian coordinates from internal coordinates with autograd support.

    Args:
        indices: (M, 4) int64 tensor of Z-matrix indices.
        distances: (M,) float32 tensor of bond lengths.
        angles: (M,) float32 tensor of bond angles.
        dihedrals: (M,) float32 tensor of dihedral angles.
        n_atoms: Total number of atoms.

    Returns:
        coords: (N, 3) float32 tensor of Cartesian coordinates.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for this function")
    if not HAS_C_EXTENSION:
        raise ImportError("C extension is required for this function")

    return NerfReconstructFunction.apply(indices, distances, angles, dihedrals, n_atoms)
