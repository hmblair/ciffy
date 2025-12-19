"""
Core backend utilities for NumPy and PyTorch array operations.

This module provides the fundamental type definitions and utilities for
backend-agnostic array operations.
"""

from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    import torch

# Try to import torch for isinstance checks (more reliable than duck-typing)
try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None
    _TORCH_AVAILABLE = False

# Type alias for arrays that can be either NumPy or PyTorch
# Note: Using Union instead of | because this is a runtime type alias, not just an annotation
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
    # Use isinstance when torch is available (reliable)
    if _TORCH_AVAILABLE and isinstance(arr, _torch.Tensor):
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


def get_device(arr: Array) -> str | None:
    """
    Get the device of an array.

    Args:
        arr: A NumPy array or PyTorch tensor.

    Returns:
        Device string (e.g., 'cpu', 'cuda:0') for PyTorch tensors,
        None for NumPy arrays.
    """
    if is_torch(arr):
        return str(arr.device)
    return None


def check_compatible(target: Array, source: Array, name: str = "array") -> None:
    """
    Check that source array is compatible with target (same backend and device).

    Args:
        target: The reference array (e.g., existing coordinates).
        source: The array being assigned.
        name: Name of the array for error messages.

    Raises:
        TypeError: If backends don't match.
        ValueError: If devices don't match (for PyTorch tensors).
    """
    target_backend = get_backend(target)
    source_backend = get_backend(source)

    if target_backend != source_backend:
        raise TypeError(
            f"Cannot assign {source_backend.value} {name} to "
            f"{target_backend.value} Polymer. "
            f"Convert using .numpy() or .torch() first."
        )

    if target_backend == Backend.TORCH:
        target_device = str(target.device)
        source_device = str(source.device)
        if target_device != source_device:
            raise ValueError(
                f"Cannot assign {name} on device '{source_device}' to "
                f"Polymer on device '{target_device}'. "
                f"Move tensor using .to('{target_device}') first."
            )


def has_nan(arr: Array) -> bool:
    """Check if array contains any NaN values."""
    if is_torch(arr):
        import torch
        return torch.isnan(arr).any().item()
    return np.any(np.isnan(arr))


def has_inf(arr: Array) -> bool:
    """Check if array contains any Inf values."""
    if is_torch(arr):
        import torch
        return torch.isinf(arr).any().item()
    return np.any(np.isinf(arr))


def any_abs_greater_than(arr: Array, threshold: float) -> bool:
    """Check if any absolute value in array exceeds threshold."""
    if is_torch(arr):
        import torch
        return (torch.abs(arr) > threshold).any().item()
    return np.any(np.abs(arr) > threshold)
