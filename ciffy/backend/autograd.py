"""
PyTorch autograd functions for internal coordinate conversions.

.. warning::
    This module is an **internal implementation detail**. Do not import directly.
    Use ``ciffy.backend.dispatch`` for coordinate conversion operations, or
    the higher-level ``ciffy.internal`` and ``Polymer`` APIs.

Provides custom autograd.Function implementations that use C backward passes
for efficient gradient computation through the internal coordinate pipeline.

Classes
-------
CartesianToInternalFunction
    Autograd function for Cartesian to internal coordinate conversion.

NerfReconstructFunction
    Autograd function for NERF reconstruction from internal coordinates.

Gradient Computation
--------------------
The backward passes are implemented by composing primitive operations:

- **Cross product**: ∂L/∂a = b × grad, ∂L/∂b = grad × a
- **Normalize**: ∂L/∂v = (grad - v̂(v̂·grad)) / |v|
- **Dot product**: ∂L/∂a = grad·b, ∂L/∂b = grad·a
- **atan2**: ∂L/∂y = grad·x/(x²+y²), ∂L/∂x = -grad·y/(x²+y²)

This composition approach ensures numerical correctness by matching the exact
forward computation graph.

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
        _nerf_reconstruct_leveled,
        _nerf_reconstruct_backward_leveled,
    )
    HAS_C_EXTENSION = True
except ImportError:
    HAS_C_EXTENSION = False

# Import CUDA extension functions
try:
    from .cuda_ops import (
        HAS_CUDA_EXTENSION,
        HAS_LEVELED_NERF,
        HAS_ANCHORED_NERF,
        is_cuda_available,
        cuda_cartesian_to_internal,
        cuda_cartesian_to_internal_backward,
        cuda_nerf_reconstruct,
        cuda_nerf_reconstruct_backward,
        cuda_nerf_reconstruct_leveled,
        cuda_nerf_reconstruct_backward_leveled,
        cuda_nerf_reconstruct_leveled_anchored,
        cuda_nerf_reconstruct_backward_leveled_anchored,
    )
except ImportError:
    HAS_CUDA_EXTENSION = False
    HAS_LEVELED_NERF = False
    HAS_ANCHORED_NERF = False
    is_cuda_available = lambda x: False

# Import anchored C extension functions
try:
    from .._c import (
        _nerf_reconstruct_leveled_anchored,
        _nerf_reconstruct_backward_leveled_anchored,
    )
    HAS_C_ANCHORED = True
except ImportError:
    HAS_C_ANCHORED = False


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

    When level_offsets is provided and leveled CUDA is available, uses
    level-parallel reconstruction for significantly better performance.
    """

    @staticmethod
    def forward(
        ctx: Any,
        indices: "torch.Tensor",
        distances: "torch.Tensor",
        angles: "torch.Tensor",
        dihedrals: "torch.Tensor",
        n_atoms: int,
        level_offsets: "torch.Tensor | None" = None,
        anchor_coords: "torch.Tensor | None" = None,
        component_ids: "torch.Tensor | None" = None,
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
            level_offsets: Optional (n_levels+1,) int32 tensor for level-parallel CUDA.
            anchor_coords: Optional (n_components, 3, 3) float32 tensor of anchor positions.
            component_ids: Optional (M,) int32 tensor mapping entries to components.

        Returns:
            coords: (N, 3) float32 tensor of Cartesian coordinates.
        """
        # Check if we can use CUDA path
        use_cuda = is_cuda_available(distances)
        use_anchored = (
            anchor_coords is not None and
            component_ids is not None and
            level_offsets is not None
        )
        use_leveled = use_cuda and HAS_LEVELED_NERF and level_offsets is not None
        ctx.use_cuda = use_cuda
        ctx.use_anchored = use_anchored

        # Convert level_offsets to tensor if needed
        level_offsets_tensor = None
        if level_offsets is not None:
            if not isinstance(level_offsets, torch.Tensor):
                level_offsets_tensor = torch.from_numpy(np.asarray(level_offsets))
            else:
                level_offsets_tensor = level_offsets
            level_offsets_tensor = level_offsets_tensor.to(
                device=distances.device, dtype=torch.int32
            ).contiguous()

        # Convert anchor_coords and component_ids if provided
        anchor_tensor = None
        comp_ids_tensor = None
        if use_anchored:
            if not isinstance(anchor_coords, torch.Tensor):
                anchor_tensor = torch.from_numpy(np.asarray(anchor_coords))
            else:
                anchor_tensor = anchor_coords
            anchor_tensor = anchor_tensor.to(
                device=distances.device, dtype=torch.float32
            ).contiguous()

            if not isinstance(component_ids, torch.Tensor):
                comp_ids_tensor = torch.from_numpy(np.asarray(component_ids))
            else:
                comp_ids_tensor = component_ids
            comp_ids_tensor = comp_ids_tensor.to(
                device=distances.device, dtype=torch.int32
            ).contiguous()

        if use_anchored and use_cuda and HAS_ANCHORED_NERF:
            # GPU path with anchored level-parallel reconstruction
            coords = torch.zeros(n_atoms, 3, dtype=torch.float32, device=distances.device)
            cuda_nerf_reconstruct_leveled_anchored(
                coords, indices, distances, angles, dihedrals,
                level_offsets_tensor, anchor_tensor, comp_ids_tensor
            )
        elif use_anchored and HAS_C_ANCHORED:
            # CPU path with anchored reconstruction
            indices_np = indices.detach().cpu().numpy().astype(np.int64)
            distances_np = distances.detach().cpu().numpy().astype(np.float32)
            angles_np = angles.detach().cpu().numpy().astype(np.float32)
            dihedrals_np = dihedrals.detach().cpu().numpy().astype(np.float32)
            level_off_np = level_offsets_tensor.detach().cpu().numpy().astype(np.int32)
            anchor_np = anchor_tensor.detach().cpu().numpy().astype(np.float32)
            comp_ids_np = comp_ids_tensor.detach().cpu().numpy().astype(np.int32)

            coords_np = _nerf_reconstruct_leveled_anchored(
                indices_np, distances_np, angles_np, dihedrals_np, n_atoms,
                level_off_np, anchor_np, comp_ids_np
            )
            coords = torch.from_numpy(coords_np).to(distances.device)
        elif use_leveled:
            # GPU path with level-parallel reconstruction (non-anchored)
            coords = torch.zeros(n_atoms, 3, dtype=torch.float32, device=distances.device)
            cuda_nerf_reconstruct_leveled(
                coords, indices, distances, angles, dihedrals, level_offsets_tensor
            )
        elif use_cuda:
            # GPU path: sequential (fallback)
            coords = torch.zeros(n_atoms, 3, dtype=torch.float32, device=distances.device)
            cuda_nerf_reconstruct(coords, indices, distances, angles, dihedrals)
        else:
            # CPU path: always use sequential version (leveled has massive OpenMP overhead)
            indices_np = indices.detach().cpu().numpy().astype(np.int64)
            distances_np = distances.detach().cpu().numpy().astype(np.float32)
            angles_np = angles.detach().cpu().numpy().astype(np.float32)
            dihedrals_np = dihedrals.detach().cpu().numpy().astype(np.float32)

            coords_np = _nerf_reconstruct(
                indices_np, distances_np, angles_np, dihedrals_np, n_atoms
            )

            # Convert back to tensor
            device = distances.device
            coords = torch.from_numpy(coords_np).to(device)

        # Save for backward
        ctx.save_for_backward(coords, indices, distances, angles, dihedrals)
        ctx.n_atoms = n_atoms
        ctx.use_leveled = use_leveled
        # Save extra context (not tensors we need gradients for)
        # Detach these to avoid keeping grad history - they're frozen references
        ctx.level_offsets = level_offsets_tensor.detach() if level_offsets_tensor is not None else None
        ctx.anchor_coords = anchor_tensor.detach() if anchor_tensor is not None else None
        ctx.component_ids = comp_ids_tensor.detach() if comp_ids_tensor is not None else None

        return coords

    @staticmethod
    def backward(
        ctx: Any,
        grad_coords: "torch.Tensor",
    ) -> tuple[None, "torch.Tensor", "torch.Tensor", "torch.Tensor", None, None, None, None]:
        """
        Backward pass for NERF reconstruction.

        Args:
            ctx: Autograd context with saved tensors.
            grad_coords: (N, 3) upstream gradients for coordinates.

        Returns:
            Tuple of (None, grad_distances, grad_angles, grad_dihedrals, None, None, None, None).
            None for indices, n_atoms, level_offsets, anchor_coords, and component_ids
            (not differentiable).
        """
        coords, indices, distances, angles, dihedrals = ctx.saved_tensors

        if ctx.use_anchored and ctx.use_cuda and HAS_ANCHORED_NERF:
            # GPU path with anchored level-parallel backward
            _, grad_distances, grad_angles, grad_dihedrals = cuda_nerf_reconstruct_backward_leveled_anchored(
                coords, indices, distances, angles, dihedrals,
                grad_coords.contiguous(), ctx.level_offsets,
                ctx.anchor_coords, ctx.component_ids
            )
        elif ctx.use_anchored and HAS_C_ANCHORED:
            # CPU path with anchored backward
            coords_np = coords.detach().cpu().numpy().astype(np.float32)
            indices_np = indices.detach().cpu().numpy().astype(np.int64)
            distances_np = distances.detach().cpu().numpy().astype(np.float32)
            angles_np = angles.detach().cpu().numpy().astype(np.float32)
            dihedrals_np = dihedrals.detach().cpu().numpy().astype(np.float32)
            grad_coords_np = grad_coords.detach().cpu().numpy().astype(np.float32).copy()
            level_off_np = ctx.level_offsets.cpu().numpy().astype(np.int32)
            anchor_np = ctx.anchor_coords.cpu().numpy().astype(np.float32)
            comp_ids_np = ctx.component_ids.cpu().numpy().astype(np.int32)

            grad_distances_np, grad_angles_np, grad_dihedrals_np = _nerf_reconstruct_backward_leveled_anchored(
                coords_np, indices_np, distances_np, angles_np, dihedrals_np,
                grad_coords_np, level_off_np, anchor_np, comp_ids_np
            )

            device = distances.device
            grad_distances = torch.from_numpy(grad_distances_np).to(device)
            grad_angles = torch.from_numpy(grad_angles_np).to(device)
            grad_dihedrals = torch.from_numpy(grad_dihedrals_np).to(device)
        elif ctx.use_cuda and ctx.use_leveled and ctx.level_offsets is not None:
            # GPU path with level-parallel backward (fastest)
            # Ensure gradients are contiguous (autograd may provide non-contiguous tensors)
            _, grad_distances, grad_angles, grad_dihedrals = cuda_nerf_reconstruct_backward_leveled(
                coords, indices, distances, angles, dihedrals,
                grad_coords.contiguous(), ctx.level_offsets
            )
        elif ctx.use_cuda:
            # GPU path: sequential (fallback)
            # Ensure gradients are contiguous (autograd may provide non-contiguous tensors)
            _, grad_distances, grad_angles, grad_dihedrals = cuda_nerf_reconstruct_backward(
                coords, indices, distances, angles, dihedrals, grad_coords.contiguous()
            )
        else:
            # CPU path: always use sequential version (leveled has massive OpenMP overhead)
            coords_np = coords.detach().cpu().numpy().astype(np.float32)
            indices_np = indices.detach().cpu().numpy().astype(np.int64)
            distances_np = distances.detach().cpu().numpy().astype(np.float32)
            angles_np = angles.detach().cpu().numpy().astype(np.float32)
            dihedrals_np = dihedrals.detach().cpu().numpy().astype(np.float32)
            grad_coords_np = grad_coords.detach().cpu().numpy().astype(np.float32).copy()

            grad_distances_np, grad_angles_np, grad_dihedrals_np = _nerf_reconstruct_backward(
                coords_np, indices_np, distances_np, angles_np, dihedrals_np,
                grad_coords_np
            )

            # Convert back to tensors
            device = distances.device
            grad_distances = torch.from_numpy(grad_distances_np).to(device)
            grad_angles = torch.from_numpy(grad_angles_np).to(device)
            grad_dihedrals = torch.from_numpy(grad_dihedrals_np).to(device)

        return None, grad_distances, grad_angles, grad_dihedrals, None, None, None, None


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
    level_offsets: "torch.Tensor | None" = None,
    anchor_coords: "torch.Tensor | None" = None,
    component_ids: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """
    Reconstruct Cartesian coordinates from internal coordinates with autograd support.

    Args:
        indices: (M, 4) int64 tensor of Z-matrix indices.
        distances: (M,) float32 tensor of bond lengths.
        angles: (M,) float32 tensor of bond angles.
        dihedrals: (M,) float32 tensor of dihedral angles.
        n_atoms: Total number of atoms.
        level_offsets: Optional (n_levels+1,) int32 tensor for level-parallel CUDA.
            When provided and leveled CUDA is available, enables parallel NERF
            reconstruction by processing atoms at the same BFS level simultaneously.
        anchor_coords: Optional (n_components, 3, 3) float32 tensor of anchor positions.
            When provided (with component_ids), atoms are placed directly in the
            reference frame defined by these anchors, eliminating Kabsch rotation.
        component_ids: Optional (M,) int32 tensor mapping entries to components.
            Required when anchor_coords is provided.

    Returns:
        coords: (N, 3) float32 tensor of Cartesian coordinates.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for this function")
    if not HAS_C_EXTENSION:
        raise ImportError("C extension is required for this function")

    return NerfReconstructFunction.apply(
        indices, distances, angles, dihedrals, n_atoms,
        level_offsets, anchor_coords, component_ids
    )
