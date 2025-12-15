"""
CUDA operations dispatch layer for internal coordinate conversions.

This module provides access to the CUDA-accelerated coordinate conversion
functions when available. It handles importing the CUDA extension and
provides fallback mechanisms.

Usage
-----
>>> from ciffy.backend.cuda_ops import HAS_CUDA_EXTENSION, is_cuda_available
>>> if is_cuda_available(tensor):
...     result = cuda_cartesian_to_internal(coords, indices)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = [
    "HAS_CUDA_EXTENSION",
    "is_cuda_available",
    "cuda_cartesian_to_internal",
    "cuda_cartesian_to_internal_backward",
    "cuda_nerf_reconstruct",
    "cuda_nerf_reconstruct_backward",
]


# Try importing CUDA extension
try:
    from .._cuda import (
        cartesian_to_internal as _cuda_cartesian_to_internal,
        cartesian_to_internal_backward as _cuda_cartesian_to_internal_backward,
        nerf_reconstruct as _cuda_nerf_reconstruct,
        nerf_reconstruct_backward as _cuda_nerf_reconstruct_backward,
    )
    HAS_CUDA_EXTENSION = True
except ImportError:
    HAS_CUDA_EXTENSION = False
    _cuda_cartesian_to_internal = None
    _cuda_cartesian_to_internal_backward = None
    _cuda_nerf_reconstruct = None
    _cuda_nerf_reconstruct_backward = None


def is_cuda_available(tensor: "torch.Tensor") -> bool:
    """
    Check if CUDA extension is available and tensor is on CUDA device.

    Args:
        tensor: A PyTorch tensor.

    Returns:
        True if CUDA extension is available and tensor is on CUDA.
    """
    return HAS_CUDA_EXTENSION and tensor.is_cuda


def cuda_cartesian_to_internal(
    coords: "torch.Tensor",
    indices: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    GPU: Convert Cartesian to internal coordinates.

    Args:
        coords: (N, 3) float32 CUDA tensor.
        indices: (M, 4) int64 CUDA tensor.

    Returns:
        Tuple of (distances, angles, dihedrals), each (M,) float32 CUDA tensor.

    Raises:
        RuntimeError: If CUDA extension is not available.
        ValueError: If tensors are not on CUDA device.
    """
    if not HAS_CUDA_EXTENSION:
        raise RuntimeError("CUDA extension not available")
    if not coords.is_cuda:
        raise ValueError("coords must be a CUDA tensor")
    if not indices.is_cuda:
        raise ValueError("indices must be a CUDA tensor")

    return _cuda_cartesian_to_internal(coords, indices)


def cuda_cartesian_to_internal_backward(
    coords: "torch.Tensor",
    indices: "torch.Tensor",
    distances: "torch.Tensor",
    angles: "torch.Tensor",
    grad_distances: "torch.Tensor",
    grad_angles: "torch.Tensor",
    grad_dihedrals: "torch.Tensor",
) -> "torch.Tensor":
    """
    GPU: Backward pass for Cartesian to internal conversion.

    Args:
        coords: (N, 3) float32 CUDA tensor.
        indices: (M, 4) int64 CUDA tensor.
        distances: (M,) float32 CUDA tensor (from forward pass).
        angles: (M,) float32 CUDA tensor (from forward pass).
        grad_distances: (M,) float32 CUDA tensor of upstream gradients.
        grad_angles: (M,) float32 CUDA tensor of upstream gradients.
        grad_dihedrals: (M,) float32 CUDA tensor of upstream gradients.

    Returns:
        grad_coords: (N, 3) float32 CUDA tensor.
    """
    if not HAS_CUDA_EXTENSION:
        raise RuntimeError("CUDA extension not available")

    return _cuda_cartesian_to_internal_backward(
        coords, indices, distances, angles,
        grad_distances, grad_angles, grad_dihedrals
    )


def cuda_nerf_reconstruct(
    coords: "torch.Tensor",
    indices: "torch.Tensor",
    distances: "torch.Tensor",
    angles: "torch.Tensor",
    dihedrals: "torch.Tensor",
) -> "torch.Tensor":
    """
    GPU: NERF reconstruction.

    Args:
        coords: (N, 3) float32 CUDA tensor (will be modified in-place).
        indices: (M, 4) int64 CUDA tensor.
        distances: (M,) float32 CUDA tensor.
        angles: (M,) float32 CUDA tensor.
        dihedrals: (M,) float32 CUDA tensor.

    Returns:
        coords tensor (modified in-place).
    """
    if not HAS_CUDA_EXTENSION:
        raise RuntimeError("CUDA extension not available")

    return _cuda_nerf_reconstruct(coords, indices, distances, angles, dihedrals)


def cuda_nerf_reconstruct_backward(
    coords: "torch.Tensor",
    indices: "torch.Tensor",
    distances: "torch.Tensor",
    angles: "torch.Tensor",
    dihedrals: "torch.Tensor",
    grad_coords: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    GPU: Backward pass for NERF reconstruction.

    Args:
        coords: (N, 3) float32 CUDA tensor.
        indices: (M, 4) int64 CUDA tensor.
        distances: (M,) float32 CUDA tensor.
        angles: (M,) float32 CUDA tensor.
        dihedrals: (M,) float32 CUDA tensor.
        grad_coords: (N, 3) float32 CUDA tensor of upstream gradients.

    Returns:
        Tuple of (grad_coords_accum, grad_distances, grad_angles, grad_dihedrals).
    """
    if not HAS_CUDA_EXTENSION:
        raise RuntimeError("CUDA extension not available")

    return _cuda_nerf_reconstruct_backward(
        coords, indices, distances, angles, dihedrals, grad_coords
    )
