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

import warnings
from typing import Any

import numpy as np

__all__ = [
    "cartesian_to_internal",
    "nerf_reconstruct",
    "CartesianToInternalFunction",
    "NerfReconstructFunction",
    "TORCH_AVAILABLE",
    "CUDA_EXTENSION_AVAILABLE",
]

try:
    import torch
    from torch.autograd import Function
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
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
        CUDA_EXTENSION_AVAILABLE,
        ANCHORED_NERF_AVAILABLE,
        is_cuda_available,
        cuda_cartesian_to_internal,
        cuda_cartesian_to_internal_backward,
        cuda_nerf_reconstruct_leveled_anchored,
        cuda_nerf_reconstruct_backward_leveled_anchored,
    )
except ImportError:
    CUDA_EXTENSION_AVAILABLE = False
    ANCHORED_NERF_AVAILABLE = False
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

        # Save original device for backward pass
        device = coords.device
        ctx.output_device = device

        if use_cuda:
            # GPU path: stay on device
            internal = cuda_cartesian_to_internal(coords, indices)
        else:
            # CPU path: use C extension via buffer protocol
            if not coords.is_cpu:
                warnings.warn(
                    f"Tensor on {device} falling back to CPU for C extension. "
                    "Consider using CUDA tensors with the CUDA extension for best performance.",
                    stacklevel=3
                )
                coords = coords.cpu()
                indices = indices.cpu()

            # Ensure contiguous float32/int64 layout for buffer protocol
            coords_f32 = coords.detach().to(torch.float32).contiguous()
            indices_i64 = indices.detach().to(torch.int64).contiguous()

            # Call C extension (accepts buffer protocol objects) - returns numpy array
            internal_np = _cartesian_to_internal(coords_f32, indices_i64)

            # Convert back to tensor on original device
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
            # CPU path: use C extension via buffer protocol
            # Use saved device from forward pass
            device = ctx.output_device

            # Transfer ALL tensors to CPU for C extension
            # (coords/indices may already be CPU from forward, but internal/grad may not be)
            coords = coords.cpu() if not coords.is_cpu else coords
            indices = indices.cpu() if not indices.is_cpu else indices
            internal = internal.cpu() if not internal.is_cpu else internal
            grad_internal = grad_internal.cpu() if not grad_internal.is_cpu else grad_internal

            # Ensure contiguous layout for buffer protocol
            coords_f32 = coords.detach().to(torch.float32).contiguous()
            indices_i64 = indices.detach().to(torch.int64).contiguous()
            internal_f32 = internal.detach().to(torch.float32).contiguous()
            grad_internal_f32 = grad_internal.detach().to(torch.float32).contiguous()

            # Call C backward (accepts buffer protocol objects)
            grad_coords_np = _cartesian_to_internal_backward(
                coords_f32, indices_i64, internal_f32, grad_internal_f32
            )

            # Convert back to tensor on original device
            grad_coords = torch.from_numpy(grad_coords_np).to(device)

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

        # Save original device for backward pass
        device = internal.device

        if use_cuda and ANCHORED_NERF_AVAILABLE:
            # GPU path with anchored component-parallel reconstruction
            coords = torch.zeros(n_atoms, 3, dtype=torch.float32, device=device)
            cuda_nerf_reconstruct_leveled_anchored(
                coords, indices, internal,
                comp_off_tensor, anchor_tensor, comp_ids_tensor
            )
        else:
            # CPU path: use C extension via buffer protocol
            if not internal.is_cpu:
                warnings.warn(
                    f"Tensor on {device} falling back to CPU for C extension. "
                    "Consider using CUDA tensors with the CUDA extension for best performance.",
                    stacklevel=3
                )
                indices = indices.cpu()
                internal = internal.cpu()
                comp_off_tensor = comp_off_tensor.cpu()
                anchor_tensor = anchor_tensor.cpu()
                comp_ids_tensor = comp_ids_tensor.cpu()

            # Ensure contiguous layout for buffer protocol
            indices_i64 = indices.detach().to(torch.int64).contiguous()
            internal_f32 = internal.detach().to(torch.float32).contiguous()
            comp_off_i32 = comp_off_tensor.detach().to(torch.int32).contiguous()
            anchor_f32 = anchor_tensor.detach().to(torch.float32).contiguous()
            comp_ids_i32 = comp_ids_tensor.detach().to(torch.int32).contiguous()

            # Call C extension (accepts buffer protocol objects)
            coords_np = _nerf_reconstruct_leveled_anchored(
                indices_i64, internal_f32, n_atoms,
                comp_off_i32, anchor_f32, comp_ids_i32
            )
            coords = torch.from_numpy(coords_np).to(device)

        # Save for backward
        ctx.save_for_backward(coords, indices, internal)
        # Save extra context (not tensors we need gradients for)
        # Detach these to avoid keeping grad history - they're frozen references
        ctx.component_offsets = comp_off_tensor.detach()
        ctx.anchor_coords = anchor_tensor.detach()
        ctx.component_ids = comp_ids_tensor.detach()
        ctx.output_device = device  # Save original device for backward

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

        if ctx.use_cuda and ANCHORED_NERF_AVAILABLE:
            # GPU path with anchored component-parallel backward
            # Returns (grad_coords_accum, grad_internal) - we only need grad_internal
            _, grad_internal = cuda_nerf_reconstruct_backward_leveled_anchored(
                coords, indices, internal,
                grad_coords.contiguous(), ctx.component_offsets,
                ctx.anchor_coords, ctx.component_ids
            )
        else:
            # CPU path: use C extension via buffer protocol
            # Use saved device from forward pass
            device = ctx.output_device

            # Transfer ALL tensors to CPU for C extension
            # (some may already be CPU from forward, but others may not be)
            coords = coords.cpu() if not coords.is_cpu else coords
            indices = indices.cpu() if not indices.is_cpu else indices
            internal = internal.cpu() if not internal.is_cpu else internal
            grad_coords = grad_coords.cpu() if not grad_coords.is_cpu else grad_coords
            component_offsets = ctx.component_offsets.cpu() if not ctx.component_offsets.is_cpu else ctx.component_offsets
            anchor_coords = ctx.anchor_coords.cpu() if not ctx.anchor_coords.is_cpu else ctx.anchor_coords
            component_ids = ctx.component_ids.cpu() if not ctx.component_ids.is_cpu else ctx.component_ids

            # Ensure contiguous layout for buffer protocol
            coords_f32 = coords.detach().to(torch.float32).contiguous()
            indices_i64 = indices.detach().to(torch.int64).contiguous()
            internal_f32 = internal.detach().to(torch.float32).contiguous()
            grad_coords_f32 = grad_coords.detach().to(torch.float32).contiguous()
            comp_off_i32 = component_offsets.to(torch.int32).contiguous()
            anchor_f32 = anchor_coords.to(torch.float32).contiguous()
            comp_ids_i32 = component_ids.to(torch.int32).contiguous()

            # Call C backward (accepts buffer protocol objects)
            grad_internal_np = _nerf_reconstruct_backward_leveled_anchored(
                coords_f32, indices_i64, internal_f32,
                grad_coords_f32, comp_off_i32, anchor_f32, comp_ids_i32
            )

            # Transfer result back to original device
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
    if not TORCH_AVAILABLE:
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
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for this function")

    return NerfReconstructFunction.apply(
        indices, internal,
        component_offsets, anchor_coords, component_ids
    )
