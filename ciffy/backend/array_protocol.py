"""
Backend-agnostic operations using numpy's array protocol.

PyTorch tensors implement `__array__()`, allowing numpy functions to
operate on them directly. This module provides utilities for:
1. Converting results back to the original backend
2. Operations that work transparently via the array protocol
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Union, Tuple

import numpy as np

from . import is_torch, Array

if TYPE_CHECKING:
    import torch


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


def as_numpy(arr: Array) -> np.ndarray:
    """
    Convert array to numpy using the __array__ protocol.

    This works for any object implementing __array__(), including
    PyTorch tensors (which will be detached and moved to CPU).

    Args:
        arr: Array-like object.

    Returns:
        NumPy array.
    """
    if is_torch(arr):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


# =============================================================================
# Linear Algebra Operations (via array protocol)
# =============================================================================

def eigh(arr: Array) -> Tuple[Array, Array]:
    """
    Eigenvalue decomposition of a symmetric/Hermitian matrix.

    Uses numpy's implementation via __array__ protocol, then converts
    results back to the original backend.

    Args:
        arr: Symmetric matrix.

    Returns:
        Tuple of (eigenvalues, eigenvectors) in original backend.
    """
    if is_torch(arr):
        # Use torch's native implementation for better device support
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
