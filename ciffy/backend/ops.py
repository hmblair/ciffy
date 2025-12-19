"""
Unified array operations with automatic backend dispatch.

Functions in this module automatically detect the backend from input arrays
and dispatch to the appropriate implementation. Simple operations use inline
if/else, while complex operations use a dispatch table to numpy_ops/torch_ops.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from .core import get_backend, is_torch, Backend, Array

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


def scatter_max(src: Array, index: Array, dim_size: int) -> tuple[Array, Array | None]:
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


def scatter_min(src: Array, index: Array, dim_size: int) -> tuple[Array, Array | None]:
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


def sign(arr: Array) -> Array:
    """
    Element-wise sign (-1, 0, or 1).

    Args:
        arr: Input array.

    Returns:
        Array with sign of each element.
    """
    return _get_ops(arr).sign(arr)


def arange(n: int, like: Array) -> Array:
    """
    Create an integer range [0, n) with same backend/device as reference array.

    Args:
        n: Length of the range.
        like: Reference array to match backend and device.

    Returns:
        Integer array [0, 1, ..., n-1] on same backend/device as `like`.
    """
    ops = _get_ops(like)
    if get_backend(like) == Backend.TORCH:
        return ops.arange(n, device=like.device)
    return ops.arange(n)


# =============================================================================
# Backend Conversion
# =============================================================================

def to_backend(arr: np.ndarray, like: Array) -> Array:
    """
    Convert a numpy array to match the backend of 'like'.

    Args:
        arr: NumPy array to convert.
        like: Template array whose backend to match.

    Returns:
        Array in the same backend as 'like'. If 'like' is torch,
        returns a tensor on the same device.
    """
    if is_torch(like):
        import torch
        result = torch.from_numpy(np.ascontiguousarray(arr))
        if hasattr(like, 'device'):
            result = result.to(like.device)
        return result
    return arr


def convert_backend(arr: Array, like: Array) -> Array:
    """
    Convert arr to match the backend of 'like'.

    More general than to_backend - works with both numpy and torch inputs.

    Args:
        arr: Array to convert.
        like: Template array for backend detection.

    Returns:
        Array in the same backend as 'like'.
    """
    if is_torch(like):
        if not is_torch(arr):
            import torch
            return torch.from_numpy(np.asarray(arr)).to(like.device)
        return arr
    else:
        if is_torch(arr):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)


# =============================================================================
# Linear Algebra
# =============================================================================

def eigh(arr: Array) -> tuple[Array, Array]:
    """
    Eigenvalue decomposition of a symmetric/Hermitian matrix.

    Args:
        arr: Symmetric matrix.

    Returns:
        Tuple of (eigenvalues, eigenvectors) in original backend.
    """
    if is_torch(arr):
        import torch
        return torch.linalg.eigh(arr)
    return np.linalg.eigh(arr)


def det(arr: Array) -> Array:
    """
    Matrix determinant.

    Args:
        arr: Square matrix.

    Returns:
        Determinant value in original backend.
    """
    if is_torch(arr):
        import torch
        return torch.linalg.det(arr)
    return np.linalg.det(arr)


def svd(arr: Array) -> tuple[Array, Array, Array]:
    """
    Full singular value decomposition.

    Args:
        arr: Input matrix.

    Returns:
        Tuple of (U, S, Vh) matrices in original backend.
    """
    if is_torch(arr):
        import torch
        return torch.linalg.svd(arr)
    return np.linalg.svd(arr)


def svdvals(arr: Array) -> Array:
    """
    Singular values of a matrix.

    Args:
        arr: Input matrix.

    Returns:
        Singular values in original backend.
    """
    if is_torch(arr):
        import torch
        return torch.linalg.svdvals(arr)
    return np.linalg.svd(arr, compute_uv=False)


# =============================================================================
# Array Creation (backend-aware)
# =============================================================================

def zeros(size: int, *, like: Array, dtype: str = 'int64') -> Array:
    """
    Create a zeros array matching the backend of 'like'.

    Args:
        size: Length of array.
        like: Template array for backend detection.
        dtype: Data type ('int64', 'float32', 'bool').

    Returns:
        Zeros array in the same backend as 'like'.
    """
    if is_torch(like):
        import torch
        torch_dtype = {'int64': torch.long, 'float32': torch.float32, 'bool': torch.bool}[dtype]
        return torch.zeros(size, dtype=torch_dtype, device=getattr(like, 'device', None))
    np_dtype = {'int64': np.int64, 'float32': np.float32, 'bool': bool}[dtype]
    return np.zeros(size, dtype=np_dtype)


def ones(size: int, *, like: Array, dtype: str = 'int64') -> Array:
    """
    Create a ones array matching the backend of 'like'.

    Args:
        size: Length of array.
        like: Template array for backend detection.
        dtype: Data type ('int64', 'float32').

    Returns:
        Ones array in the same backend as 'like'.
    """
    if is_torch(like):
        import torch
        torch_dtype = {'int64': torch.long, 'float32': torch.float32}[dtype]
        return torch.ones(size, dtype=torch_dtype, device=getattr(like, 'device', None))
    np_dtype = {'int64': np.int64, 'float32': np.float32}[dtype]
    return np.ones(size, dtype=np_dtype)


def array(data: list, *, like: Array, dtype: str = 'int64') -> Array:
    """
    Create an array from data matching the backend of 'like'.

    Args:
        data: List of values.
        like: Template array for backend detection.
        dtype: Data type ('int64', 'float32').

    Returns:
        Array in the same backend as 'like'.
    """
    if is_torch(like):
        import torch
        torch_dtype = {'int64': torch.long, 'float32': torch.float32}[dtype]
        return torch.tensor(data, dtype=torch_dtype, device=getattr(like, 'device', None))
    np_dtype = {'int64': np.int64, 'float32': np.float32}[dtype]
    return np.array(data, dtype=np_dtype)


# =============================================================================
# Utility Operations
# =============================================================================

def nonzero_1d(arr: Array) -> Array:
    """
    Get indices of non-zero elements in a 1D array.

    Args:
        arr: 1D array.

    Returns:
        Indices of non-zero elements in original backend.
    """
    if is_torch(arr):
        return arr.nonzero().squeeze(-1)
    return arr.nonzero()[0]


def to_int64(arr: Array) -> Array:
    """
    Convert array to int64 dtype.

    Args:
        arr: Input array.

    Returns:
        Array with int64 dtype in original backend.
    """
    if is_torch(arr):
        return arr.long()
    return arr.astype(np.int64)


def isin(arr: Array, values: list | tuple) -> Array:
    """
    Check if elements of arr are in values.

    Args:
        arr: Input array to check.
        values: List/tuple of values to check against.

    Returns:
        Boolean array of same shape as arr, True where element is in values.
    """
    if is_torch(arr):
        import torch
        # Convert values to tensor on same device
        test_tensor = torch.tensor(values, device=arr.device, dtype=arr.dtype)
        return torch.isin(arr, test_tensor)
    return np.isin(arr, values)


# =============================================================================
# Math Operations
# =============================================================================

def sqrt(arr: Array) -> Array:
    """
    Element-wise square root.

    Args:
        arr: Input array.

    Returns:
        Square root in original backend.
    """
    if is_torch(arr):
        import torch
        return torch.sqrt(arr)
    return np.sqrt(arr)


def clamp(arr: Array, min_val: float | None = None, max_val: float | None = None) -> Array:
    """
    Clamp array values to a range.

    Args:
        arr: Input array.
        min_val: Minimum value (None for no lower bound).
        max_val: Maximum value (None for no upper bound).

    Returns:
        Clamped array in original backend.
    """
    if is_torch(arr):
        import torch
        return torch.clamp(arr, min=min_val, max=max_val)
    result = arr
    if min_val is not None:
        result = np.maximum(result, min_val)
    if max_val is not None:
        result = np.minimum(result, max_val)
    return result


def argsort(arr: Array) -> Array:
    """
    Return indices that would sort the array.

    Args:
        arr: 1D input array.

    Returns:
        Array of indices that would sort the array.
    """
    if is_torch(arr):
        import torch
        return torch.argsort(arr)
    return np.argsort(arr)


def diff(arr: Array) -> Array:
    """
    Compute differences between consecutive elements.

    Args:
        arr: 1D input array.

    Returns:
        Array of differences (length n-1 for input of length n).
    """
    if is_torch(arr):
        import torch
        return torch.diff(arr)
    return np.diff(arr)


def split_at_indices(arr: Array, split_indices: Array) -> list:
    """
    Split array at specified indices.

    Args:
        arr: Array to split.
        split_indices: 1D array of indices where splits occur.

    Returns:
        List of array chunks.
    """
    if is_torch(arr):
        import torch
        # Convert split_indices to list for torch.tensor_split
        if len(split_indices) == 0:
            return [arr]
        return list(torch.tensor_split(arr, split_indices.cpu().tolist()))
    # NumPy path
    if len(split_indices) == 0:
        return [arr]
    return list(np.split(arr, split_indices))


def topk(arr: Array, k: int, dim: int = -1, largest: bool = True) -> tuple[Array, Array]:
    """
    Find k largest or smallest elements along a dimension.

    Args:
        arr: Input array.
        k: Number of elements to return.
        dim: Dimension along which to find topk.
        largest: If True, find k largest; if False, find k smallest.

    Returns:
        Tuple of (values, indices) in original backend.
    """
    if is_torch(arr):
        import torch
        return torch.topk(arr, k, dim=dim, largest=largest)
    else:
        # NumPy implementation
        if largest:
            # Get indices of k largest
            indices = np.argpartition(arr, -k, axis=dim)
            indices = np.take(indices, range(-k, 0), axis=dim)
        else:
            # Get indices of k smallest
            indices = np.argpartition(arr, k, axis=dim)
            indices = np.take(indices, range(k), axis=dim)

        # Get values and sort by value
        values = np.take_along_axis(arr, indices, axis=dim)
        sort_idx = np.argsort(values, axis=dim)
        if largest:
            sort_idx = np.flip(sort_idx, axis=dim)
        indices = np.take_along_axis(indices, sort_idx, axis=dim)
        values = np.take_along_axis(values, sort_idx, axis=dim)

        return values, indices
