"""
CUDA operations dispatch layer for internal coordinate conversions.

This module provides access to the CUDA-accelerated coordinate conversion
functions when available. It handles importing the CUDA extension and
provides fallback mechanisms.

Usage
-----
>>> from ciffy.backend.cuda_ops import CUDA_EXTENSION_AVAILABLE, is_cuda_available
>>> if is_cuda_available(tensor):
...     result = cuda_cartesian_to_internal(coords, indices)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = [
    "CUDA_EXTENSION_AVAILABLE",
    "ANCHORED_NERF_AVAILABLE",
    "is_cuda_available",
    "cuda_cartesian_to_internal",
    "cuda_cartesian_to_internal_backward",
    "cuda_nerf_reconstruct_leveled_anchored",
    "cuda_nerf_reconstruct_backward_leveled_anchored",
]


# Try importing CUDA extension (cartesian_to_internal ops)
try:
    from .._cuda import (
        cartesian_to_internal as _cuda_cartesian_to_internal,
        cartesian_to_internal_backward as _cuda_cartesian_to_internal_backward,
    )
    CUDA_EXTENSION_AVAILABLE = True
except ImportError:
    CUDA_EXTENSION_AVAILABLE = False
    _cuda_cartesian_to_internal = None
    _cuda_cartesian_to_internal_backward = None

# Try importing anchored NERF (places atoms in reference frame from anchors)
try:
    from .._cuda import (
        nerf_reconstruct_leveled_anchored as _cuda_nerf_reconstruct_leveled_anchored,
        nerf_reconstruct_backward_leveled_anchored as _cuda_nerf_reconstruct_backward_leveled_anchored,
    )
    ANCHORED_NERF_AVAILABLE = True
except (ImportError, AttributeError):
    ANCHORED_NERF_AVAILABLE = False
    _cuda_nerf_reconstruct_leveled_anchored = None
    _cuda_nerf_reconstruct_backward_leveled_anchored = None


def is_cuda_available(tensor: "torch.Tensor") -> bool:
    """
    Check if CUDA extension is available and tensor is on CUDA device.

    Args:
        tensor: A PyTorch tensor.

    Returns:
        True if CUDA extension is available and tensor is on CUDA.
    """
    return CUDA_EXTENSION_AVAILABLE and tensor.is_cuda


def cuda_cartesian_to_internal(
    coords: "torch.Tensor",
    indices: "torch.Tensor",
) -> "torch.Tensor":
    """
    GPU: Convert Cartesian to internal coordinates.

    Args:
        coords: (N, 3) float32 CUDA tensor.
        indices: (M, 4) int64 CUDA tensor.

    Returns:
        internal: (M, 3) float32 CUDA tensor with [dist, angle, dihedral] per row.

    Raises:
        RuntimeError: If CUDA extension is not available.
        ValueError: If tensors are not on CUDA device.
    """
    if not CUDA_EXTENSION_AVAILABLE:
        raise RuntimeError("CUDA extension not available")
    if not coords.is_cuda:
        raise ValueError("coords must be a CUDA tensor")
    if not indices.is_cuda:
        raise ValueError("indices must be a CUDA tensor")

    return _cuda_cartesian_to_internal(coords, indices)


def cuda_cartesian_to_internal_backward(
    coords: "torch.Tensor",
    indices: "torch.Tensor",
    internal: "torch.Tensor",
    grad_internal: "torch.Tensor",
) -> "torch.Tensor":
    """
    GPU: Backward pass for Cartesian to internal conversion.

    Args:
        coords: (N, 3) float32 CUDA tensor.
        indices: (M, 4) int64 CUDA tensor.
        internal: (M, 3) float32 CUDA tensor (from forward pass).
        grad_internal: (M, 3) float32 CUDA tensor of upstream gradients.

    Returns:
        grad_coords: (N, 3) float32 CUDA tensor.
    """
    if not CUDA_EXTENSION_AVAILABLE:
        raise RuntimeError("CUDA extension not available")

    return _cuda_cartesian_to_internal_backward(
        coords, indices, internal, grad_internal
    )


def cuda_nerf_reconstruct_leveled_anchored(
    coords: "torch.Tensor",
    indices: "torch.Tensor",
    internal: "torch.Tensor",
    level_offsets: "torch.Tensor",
    anchor_coords: "torch.Tensor",
    component_ids: "torch.Tensor",
) -> "torch.Tensor":
    """
    GPU: Level-parallel anchored NERF reconstruction.

    Places atoms directly in the reference frame defined by anchor coordinates,
    eliminating the need for post-reconstruction Kabsch rotation.

    Args:
        coords: (N, 3) float32 CUDA tensor (will be modified in-place).
        indices: (M, 4) int64 CUDA tensor.
        internal: (M, 3) float32 CUDA tensor.
        level_offsets: (n_levels+1,) int32 CUDA tensor of CSR-style offsets.
        anchor_coords: (n_components, 3, 3) float32 CUDA tensor of anchor positions.
        component_ids: (M,) int32 CUDA tensor mapping entries to components.

    Returns:
        coords tensor (modified in-place).

    Raises:
        RuntimeError: If anchored NERF CUDA kernel is not available.
    """
    if not ANCHORED_NERF_AVAILABLE:
        raise RuntimeError(
            "Anchored NERF CUDA kernel not available. "
            "Rebuild with CUDA support."
        )

    return _cuda_nerf_reconstruct_leveled_anchored(
        coords, indices, internal,
        level_offsets, anchor_coords, component_ids
    )


def cuda_nerf_reconstruct_backward_leveled_anchored(
    coords: "torch.Tensor",
    indices: "torch.Tensor",
    internal: "torch.Tensor",
    grad_coords: "torch.Tensor",
    level_offsets: "torch.Tensor",
    anchor_coords: "torch.Tensor",
    component_ids: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """
    GPU: Backward pass for level-parallel anchored NERF reconstruction.

    Note: Gradients do not flow through anchor_coords (they are frozen references).

    Args:
        coords: (N, 3) float32 CUDA tensor.
        indices: (M, 4) int64 CUDA tensor.
        internal: (M, 3) float32 CUDA tensor.
        grad_coords: (N, 3) float32 CUDA tensor of upstream gradients.
        level_offsets: (n_levels+1,) int32 CUDA tensor of CSR-style offsets.
        anchor_coords: (n_components, 3, 3) float32 CUDA tensor of anchor positions.
        component_ids: (M,) int32 CUDA tensor mapping entries to components.

    Returns:
        Tuple of (grad_coords_accum, grad_internal).

    Raises:
        RuntimeError: If anchored NERF CUDA kernel is not available.
    """
    if not ANCHORED_NERF_AVAILABLE:
        raise RuntimeError(
            "Anchored NERF CUDA kernel not available. "
            "Rebuild with CUDA support."
        )

    return _cuda_nerf_reconstruct_backward_leveled_anchored(
        coords, indices, internal,
        grad_coords, level_offsets, anchor_coords, component_ids
    )
