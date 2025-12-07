"""
Backend abstraction for NumPy and PyTorch array operations.

This module provides a unified interface for array operations that can work
with either NumPy arrays or PyTorch tensors. The backend is automatically
detected from the array type.
"""

from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    import torch

# Type alias for arrays that can be either NumPy or PyTorch
Array = Union[np.ndarray, "torch.Tensor"]


class Backend(Enum):
    """Array backend type."""
    NUMPY = "numpy"
    TORCH = "torch"


def get_backend(arr: Array) -> Backend:
    """
    Detect the backend type from an array.

    Args:
        arr: A NumPy array or PyTorch tensor.

    Returns:
        Backend.TORCH if arr is a PyTorch tensor, Backend.NUMPY otherwise.
    """
    # PyTorch tensors have a 'numpy' method, NumPy arrays don't have 'dim'
    if hasattr(arr, 'dim') and callable(getattr(arr, 'dim')):
        return Backend.TORCH
    return Backend.NUMPY


def is_torch(arr: Array) -> bool:
    """Check if array is a PyTorch tensor."""
    return get_backend(arr) == Backend.TORCH


def is_numpy(arr: Array) -> bool:
    """Check if array is a NumPy array."""
    return get_backend(arr) == Backend.NUMPY


def to_numpy(arr: Array) -> np.ndarray:
    """
    Convert an array to NumPy.

    Args:
        arr: A NumPy array or PyTorch tensor.

    Returns:
        NumPy array. If already NumPy, returns as-is.
    """
    if is_torch(arr):
        return arr.detach().cpu().numpy()
    return arr


def to_torch(arr: Array) -> "torch.Tensor":
    """
    Convert an array to PyTorch.

    Args:
        arr: A NumPy array or PyTorch tensor.

    Returns:
        PyTorch tensor. If already PyTorch, returns as-is.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    import torch
    if is_numpy(arr):
        return torch.from_numpy(arr)
    return arr


def size(arr: Array, dim: int = 0) -> int:
    """
    Get the size of an array along a dimension.

    Works with both NumPy (.shape) and PyTorch (.size()).

    Args:
        arr: Array to get size of.
        dim: Dimension to get size along.

    Returns:
        Size of the array along the specified dimension.
    """
    if is_torch(arr):
        return arr.size(dim)
    return arr.shape[dim]
