"""
Unified array operations with automatic backend dispatch.

Functions in this module automatically detect the backend from input arrays
and dispatch to the appropriate implementation using a dispatch table pattern.
"""

from __future__ import annotations
from typing import Tuple, Optional, TYPE_CHECKING

from . import get_backend, Backend, Array

if TYPE_CHECKING:
    import torch


# =============================================================================
# Dispatch Table
# =============================================================================

def _get_ops(arr: Array):
    """Get the appropriate ops module for the array's backend."""
    if get_backend(arr) == Backend.TORCH:
        from . import torch_ops
        return torch_ops
    from . import numpy_ops
    return numpy_ops


# =============================================================================
# Scatter Operations
# =============================================================================

def scatter_sum(src: Array, index: Array, dim_size: int) -> Array:
    """
    Sum values into an output array at specified indices.

    Args:
        src: Source array of shape (N, ...).
        index: Index array of shape (N,) with values in [0, dim_size).
        dim_size: Size of output first dimension.

    Returns:
        Array of shape (dim_size, ...) with summed values.
    """
    return _get_ops(src).scatter_sum(src, index, dim_size)


def scatter_mean(src: Array, index: Array, dim_size: int) -> Array:
    """
    Average values into an output array at specified indices.

    Args:
        src: Source array of shape (N, ...).
        index: Index array of shape (N,) with values in [0, dim_size).
        dim_size: Size of output first dimension.

    Returns:
        Array of shape (dim_size, ...) with averaged values.
    """
    return _get_ops(src).scatter_mean(src, index, dim_size)


def scatter_max(src: Array, index: Array, dim_size: int) -> Tuple[Array, Optional[Array]]:
    """
    Maximum values at specified indices.

    Args:
        src: Source array of shape (N, ...).
        index: Index array of shape (N,) with values in [0, dim_size).
        dim_size: Size of output first dimension.

    Returns:
        Tuple of (max_values, argmax_indices). argmax_indices may be None.
    """
    return _get_ops(src).scatter_max(src, index, dim_size)


def scatter_min(src: Array, index: Array, dim_size: int) -> Tuple[Array, Optional[Array]]:
    """
    Minimum values at specified indices.

    Args:
        src: Source array of shape (N, ...).
        index: Index array of shape (N,) with values in [0, dim_size).
        dim_size: Size of output first dimension.

    Returns:
        Tuple of (min_values, argmin_indices). argmin_indices may be None.
    """
    return _get_ops(src).scatter_min(src, index, dim_size)


# =============================================================================
# Array Operations
# =============================================================================

def repeat_interleave(arr: Array, repeats: Array) -> Array:
    """
    Repeat elements of an array along the first axis.

    Args:
        arr: Array to repeat elements from.
        repeats: Number of times to repeat each element.

    Returns:
        Array with repeated elements.
    """
    return _get_ops(arr).repeat_interleave(arr, repeats)


def cdist(x1: Array, x2: Array) -> Array:
    """
    Compute pairwise Euclidean distances.

    Args:
        x1: Array of shape (M, D).
        x2: Array of shape (N, D).

    Returns:
        Distance matrix of shape (M, N).
    """
    return _get_ops(x1).cdist(x1, x2)


def cat(arrays: list, axis: int = 0) -> Array:
    """
    Concatenate arrays along an axis.

    Args:
        arrays: List of arrays to concatenate.
        axis: Axis along which to concatenate.

    Returns:
        Concatenated array.
    """
    if len(arrays) == 0:
        raise ValueError("Cannot concatenate empty list")

    ops = _get_ops(arrays[0])
    # Handle axis/dim naming difference
    if get_backend(arrays[0]) == Backend.TORCH:
        return ops.cat(arrays, dim=axis)
    return ops.cat(arrays, axis=axis)


def multiply(a: Array, b: Array) -> Array:
    """
    Element-wise multiplication.

    Args:
        a: First array.
        b: Second array.

    Returns:
        Element-wise product.
    """
    return _get_ops(a).multiply(a, b)
