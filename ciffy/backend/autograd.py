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
    "HAS_CUDA_EXTENSION",
]

try:
    import torch
    from torch.autograd import Function
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Function = object  # Dummy for type hints

# C extension functions (required)
from .._c import (
    _cartesian_to_internal,
    _cartesian_to_internal_backward,
    _nerf_reconstruct_leveled_anchored,
    _nerf_reconstruct_backward_leveled_anchored,
)

# Import CUDA extension functions
try:
    from .cuda_ops import (
        HAS_CUDA_EXTENSION,
        HAS_ANCHORED_NERF,
        is_cuda_available,
        cuda_cartesian_to_internal,
        cuda_cartesian_to_internal_backward,
        cuda_nerf_reconstruct_leveled_anchored,
        cuda_nerf_reconstruct_backward_leveled_anchored,
    )
except ImportError:
    HAS_CUDA_EXTENSION = False
    HAS_ANCHORED_NERF = False
    is_cuda_available = lambda x: False


class CartesianToInternalFunction(Function):
    """
    Autograd function for Cartesian to internal coordinate conversion.

    Forward: coords -> internal (M, 3) [distance, angle, dihedral]
    Backward: grad_internal -> grad_coords
    """

    @staticmethod
    def forward(
        ctx: Any,
        coords: "torch.Tensor",
        indices: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Convert Cartesian coordinates to internal coordinates.

        Args:
            ctx: Autograd context for saving tensors.
            coords: (N, 3) float32 tensor of Cartesian coordinates.
            indices: (M, 4) int64 tensor of Z-matrix indices.

        Returns:
            internal: (M, 3) float32 tensor where each row is [distance, angle, dihedral].
        """
        # Check if we can use CUDA path
        use_cuda = is_cuda_available(coords)
        ctx.use_cuda = use_cuda

        if use_cuda:
            # GPU path: stay on device
            internal = cuda_cartesian_to_internal(coords, indices)
        else:
            # CPU path: convert to numpy for C extension
            coords_np = coords.detach().cpu().numpy().astype(np.float32)
            indices_np = indices.detach().cpu().numpy().astype(np.int64)

            # Call C extension - returns (M, 3) array
            internal_np = _cartesian_to_internal(coords_np, indices_np)

            # Convert back to tensor
            device = coords.device
            internal = torch.from_numpy(internal_np).to(device)

        # Save for backward
        ctx.save_for_backward(coords, indices, internal)

        return internal

    @staticmethod
    def backward(
        ctx: Any,
        grad_internal: "torch.Tensor",
    ) -> tuple["torch.Tensor", None]:
        """
        Backward pass for Cartesian to internal conversion.

        Args:
            ctx: Autograd context with saved tensors.
            grad_internal: (M, 3) upstream gradients for internal coordinates.

        Returns:
            Tuple of (grad_coords, None) - None for indices (not differentiable).
        """
        coords, indices, internal = ctx.saved_tensors

        if ctx.use_cuda:
            # GPU path: stay on device
            # Ensure gradients are contiguous (autograd may provide non-contiguous tensors)
            grad_coords = cuda_cartesian_to_internal_backward(
                coords, indices, internal, grad_internal.contiguous()
            )
        else:
            # CPU path: convert to numpy
            coords_np = coords.detach().cpu().numpy().astype(np.float32)
            indices_np = indices.detach().cpu().numpy().astype(np.int64)
            internal_np = internal.detach().cpu().numpy().astype(np.float32)
            grad_internal_np = grad_internal.detach().cpu().numpy().astype(np.float32)

            # Call C backward
            grad_coords_np = _cartesian_to_internal_backward(
                coords_np, indices_np, internal_np, grad_internal_np
            )

            # Convert back to tensor
            grad_coords = torch.from_numpy(grad_coords_np).to(coords.device)

        return grad_coords, None


class NerfReconstructFunction(Function):
    """
    Autograd function for anchored NERF reconstruction.

    Forward: internal (M, 3) -> coords (N, 3)
    Backward: grad_coords -> grad_internal

    Requires component_offsets, anchor_coords, and component_ids for anchored
    reconstruction which places atoms directly in the reference frame.
    """

    @staticmethod
    def forward(
        ctx: Any,
        indices: "torch.Tensor",
        internal: "torch.Tensor",
        component_offsets: "torch.Tensor | None" = None,
        anchor_coords: "torch.Tensor | None" = None,
        component_ids: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """
        Reconstruct Cartesian coordinates from internal coordinates.

        Args:
            ctx: Autograd context for saving tensors.
            indices: (M, 4) int64 tensor of Z-matrix indices.
                The number of atoms is inferred from the first dimension.
            internal: (M, 3) float32 tensor of internal coordinates.
                Each row: [distance, angle, dihedral].
            component_offsets: (n_components+1,) int32 tensor for component-parallel reconstruction.
            anchor_coords: (n_components, 3, 3) float32 tensor of anchor positions.
            component_ids: (M,) int32 tensor mapping entries to components.

        Returns:
            coords: (N, 3) float32 tensor of Cartesian coordinates.
        """
        if component_offsets is None or anchor_coords is None or component_ids is None:
            raise ValueError(
                "nerf_reconstruct requires component_offsets, anchor_coords, and component_ids. "
                "Use CoordinateManager for automatic setup of these parameters."
            )

        n_atoms = len(indices)

        # Check if we can use CUDA path
        use_cuda = is_cuda_available(internal)
        ctx.use_cuda = use_cuda

        # Convert component_offsets to tensor if needed
        if not isinstance(component_offsets, torch.Tensor):
            comp_off_tensor = torch.from_numpy(np.asarray(component_offsets))
        else:
            comp_off_tensor = component_offsets
        comp_off_tensor = comp_off_tensor.to(
            device=internal.device, dtype=torch.int32
        ).contiguous()

        # Convert anchor_coords to tensor if needed
        if not isinstance(anchor_coords, torch.Tensor):
            anchor_tensor = torch.from_numpy(np.asarray(anchor_coords))
        else:
            anchor_tensor = anchor_coords
        anchor_tensor = anchor_tensor.to(
            device=internal.device, dtype=torch.float32
        ).contiguous()

        # Convert component_ids to tensor if needed
        if not isinstance(component_ids, torch.Tensor):
            comp_ids_tensor = torch.from_numpy(np.asarray(component_ids))
        else:
            comp_ids_tensor = component_ids
        comp_ids_tensor = comp_ids_tensor.to(
            device=internal.device, dtype=torch.int32
        ).contiguous()

        if use_cuda and HAS_ANCHORED_NERF:
            # GPU path with anchored component-parallel reconstruction
            coords = torch.zeros(n_atoms, 3, dtype=torch.float32, device=internal.device)
            cuda_nerf_reconstruct_leveled_anchored(
                coords, indices, internal,
                comp_off_tensor, anchor_tensor, comp_ids_tensor
            )
        else:
            # CPU path with anchored reconstruction
            indices_np = indices.detach().cpu().numpy().astype(np.int64)
            internal_np = internal.detach().cpu().numpy().astype(np.float32)
            comp_off_np = comp_off_tensor.detach().cpu().numpy().astype(np.int32)
            anchor_np = anchor_tensor.detach().cpu().numpy().astype(np.float32)
            comp_ids_np = comp_ids_tensor.detach().cpu().numpy().astype(np.int32)

            coords_np = _nerf_reconstruct_leveled_anchored(
                indices_np, internal_np, n_atoms,
                comp_off_np, anchor_np, comp_ids_np
            )
            coords = torch.from_numpy(coords_np).to(internal.device)

        # Save for backward
        ctx.save_for_backward(coords, indices, internal)
        # Save extra context (not tensors we need gradients for)
        # Detach these to avoid keeping grad history - they're frozen references
        ctx.component_offsets = comp_off_tensor.detach()
        ctx.anchor_coords = anchor_tensor.detach()
        ctx.component_ids = comp_ids_tensor.detach()

        return coords

    @staticmethod
    def backward(
        ctx: Any,
        grad_coords: "torch.Tensor",
    ) -> tuple[None, "torch.Tensor", None, None, None]:
        """
        Backward pass for anchored NERF reconstruction.

        Args:
            ctx: Autograd context with saved tensors.
            grad_coords: (N, 3) upstream gradients for coordinates.

        Returns:
            Tuple of (None, grad_internal, None, None, None).
            None for indices, component_offsets, anchor_coords, and component_ids
            (not differentiable).
        """
        coords, indices, internal = ctx.saved_tensors

        if ctx.use_cuda and HAS_ANCHORED_NERF:
            # GPU path with anchored component-parallel backward
            grad_internal = cuda_nerf_reconstruct_backward_leveled_anchored(
                coords, indices, internal,
                grad_coords.contiguous(), ctx.component_offsets,
                ctx.anchor_coords, ctx.component_ids
            )
        else:
            # CPU path with anchored backward
            coords_np = coords.detach().cpu().numpy().astype(np.float32)
            indices_np = indices.detach().cpu().numpy().astype(np.int64)
            internal_np = internal.detach().cpu().numpy().astype(np.float32)
            grad_coords_np = grad_coords.detach().cpu().numpy().astype(np.float32).copy()
            comp_off_np = ctx.component_offsets.cpu().numpy().astype(np.int32)
            anchor_np = ctx.anchor_coords.cpu().numpy().astype(np.float32)
            comp_ids_np = ctx.component_ids.cpu().numpy().astype(np.int32)

            grad_internal_np = _nerf_reconstruct_backward_leveled_anchored(
                coords_np, indices_np, internal_np,
                grad_coords_np, comp_off_np, anchor_np, comp_ids_np
            )

            device = internal.device
            grad_internal = torch.from_numpy(grad_internal_np).to(device)

        return None, grad_internal, None, None, None


def cartesian_to_internal(
    coords: "torch.Tensor",
    indices: "torch.Tensor",
) -> "torch.Tensor":
    """
    Convert Cartesian coordinates to internal coordinates with autograd support.

    Args:
        coords: (N, 3) float32 tensor of Cartesian coordinates.
        indices: (M, 4) int64 tensor of Z-matrix indices.

    Returns:
        internal: (M, 3) float32 tensor where each row is [distance, angle, dihedral].
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for this function")

    return CartesianToInternalFunction.apply(coords, indices)


def nerf_reconstruct(
    indices: "torch.Tensor",
    internal: "torch.Tensor",
    component_offsets: "torch.Tensor | None" = None,
    anchor_coords: "torch.Tensor | None" = None,
    component_ids: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """
    Reconstruct Cartesian coordinates from internal coordinates with autograd support.

    Args:
        indices: (M, 4) int64 tensor of Z-matrix indices.
            The number of atoms is inferred from the first dimension.
        internal: (M, 3) float32 tensor of internal coordinates.
            Each row: [distance, angle, dihedral].
        component_offsets: Optional (n_components+1,) int32 tensor for component-parallel NERF.
            When provided, enables parallel reconstruction by processing each
            connected component independently.
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

    return NerfReconstructFunction.apply(
        indices, internal,
        component_offsets, anchor_coords, component_ids
    )
