"""
NumPy implementations of array operations.

Provides NumPy equivalents of PyTorch operations for backend-agnostic code.
"""

from __future__ import annotations

import numpy as np


def scatter_sum(
    src: np.ndarray,
    index: np.ndarray,
    dim_size: int,
) -> np.ndarray:
    """
    Sum values into an output array at specified indices.

    Args:
        src: Source array of shape (N, ...).
        index: Index array of shape (N,) with values in [0, dim_size).
        dim_size: Size of output first dimension.

    Returns:
        Array of shape (dim_size, ...) with summed values.
    """
    result = np.zeros((dim_size, *src.shape[1:]), dtype=src.dtype)
    np.add.at(result, index, src)
    return result


def scatter_mean(
    src: np.ndarray,
    index: np.ndarray,
    dim_size: int,
) -> np.ndarray:
    """
    Average values into an output array at specified indices.

    Args:
        src: Source array of shape (N, ...).
        index: Index array of shape (N,) with values in [0, dim_size).
        dim_size: Size of output first dimension.

    Returns:
        Array of shape (dim_size, ...) with averaged values.
    """
    sums = scatter_sum(src, index, dim_size)
    counts = np.zeros(dim_size, dtype=np.int64)
    np.add.at(counts, index, 1)
    # Reshape counts for broadcasting
    shape = (-1,) + (1,) * (sums.ndim - 1)
    # Proper division - empty bins get 0 (consistent with torch behavior)
    with np.errstate(invalid='ignore', divide='ignore'):
        result = sums / counts.reshape(shape)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result


def scatter_max(
    src: np.ndarray,
    index: np.ndarray,
    dim_size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Maximum values at specified indices.

    Args:
        src: Source array of shape (N, ...).
        index: Index array of shape (N,) with values in [0, dim_size).
        dim_size: Size of output first dimension.

    Returns:
        Tuple of (max_values, argmax_indices). argmax_indices may be None.
    """
    # Track which bins have values (avoids sentinel collision with real data)
    has_value = np.zeros(dim_size, dtype=bool)
    np.logical_or.at(has_value, index, True)

    # Use appropriate min value based on dtype
    if np.issubdtype(src.dtype, np.floating):
        fill_value = -np.inf
    else:
        fill_value = np.iinfo(src.dtype).min

    result = np.full((dim_size, *src.shape[1:]), fill_value, dtype=src.dtype)
    np.maximum.at(result, index, src)

    # Zero out unfilled bins using vectorized boolean indexing
    result[~has_value] = 0

    return result, None


def scatter_min(
    src: np.ndarray,
    index: np.ndarray,
    dim_size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Minimum values at specified indices.

    Args:
        src: Source array of shape (N, ...).
        index: Index array of shape (N,) with values in [0, dim_size).
        dim_size: Size of output first dimension.

    Returns:
        Tuple of (min_values, argmin_indices). argmin_indices may be None.
    """
    # Track which bins have values (avoids sentinel collision with real data)
    has_value = np.zeros(dim_size, dtype=bool)
    np.logical_or.at(has_value, index, True)

    # Use appropriate max value based on dtype
    if np.issubdtype(src.dtype, np.floating):
        fill_value = np.inf
    else:
        fill_value = np.iinfo(src.dtype).max

    result = np.full((dim_size, *src.shape[1:]), fill_value, dtype=src.dtype)
    np.minimum.at(result, index, src)

    # Zero out unfilled bins using vectorized boolean indexing
    result[~has_value] = 0

    return result, None


def repeat_interleave(arr: np.ndarray, repeats: np.ndarray) -> np.ndarray:
    """
    Repeat elements of an array along the first axis.

    Args:
        arr: Array to repeat elements from.
        repeats: Number of times to repeat each element.

    Returns:
        Array with repeated elements.
    """
    return np.repeat(arr, repeats, axis=0)


def cdist(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances.

    Args:
        x1: Array of shape (M, D).
        x2: Array of shape (N, D).

    Returns:
        Distance matrix of shape (M, N).
    """
    # Efficient computation using broadcasting
    diff = x1[:, None, :] - x2[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def zeros(shape: tuple[int, ...], dtype: type = np.float32) -> np.ndarray:
    """Create a zeros array."""
    return np.zeros(shape, dtype=dtype)


def ones(shape: tuple[int, ...], dtype: type = np.float32) -> np.ndarray:
    """Create a ones array."""
    return np.ones(shape, dtype=dtype)


def arange(n: int, dtype: type = np.int64) -> np.ndarray:
    """Create a range array."""
    return np.arange(n, dtype=dtype)


def cat(arrays: list, axis: int = 0) -> np.ndarray:
    """Concatenate arrays."""
    return np.concatenate(arrays, axis=axis)


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise multiplication."""
    return np.multiply(a, b)


def sign(arr: np.ndarray) -> np.ndarray:
    """Element-wise sign (-1, 0, or 1)."""
    return np.sign(arr)
