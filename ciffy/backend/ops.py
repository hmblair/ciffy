"""
Unified array operations with automatic backend dispatch.

Functions in this module automatically detect the backend from input arrays
and dispatch to the appropriate implementation.
"""

from __future__ import annotations
from typing import Tuple, Optional, Union, TYPE_CHECKING

import numpy as np

from . import get_backend, Backend, Array

if TYPE_CHECKING:
    import torch


def scatter_sum(
    src: Array,
    index: Array,
    dim_size: int,
) -> Array:
    """
    Sum values into an output array at specified indices.

    Args:
        src: Source array of shape (N, ...).
        index: Index array of shape (N,) with values in [0, dim_size).
        dim_size: Size of output first dimension.

    Returns:
        Array of shape (dim_size, ...) with summed values.
    """
    if get_backend(src) == Backend.TORCH:
        from . import torch_ops
        return torch_ops.scatter_sum(src, index, dim_size)
    from . import numpy_ops
    return numpy_ops.scatter_sum(src, index, dim_size)


def scatter_mean(
    src: Array,
    index: Array,
    dim_size: int,
) -> Array:
    """
    Average values into an output array at specified indices.

    Args:
        src: Source array of shape (N, ...).
        index: Index array of shape (N,) with values in [0, dim_size).
        dim_size: Size of output first dimension.

    Returns:
        Array of shape (dim_size, ...) with averaged values.
    """
    if get_backend(src) == Backend.TORCH:
        from . import torch_ops
        return torch_ops.scatter_mean(src, index, dim_size)
    from . import numpy_ops
    return numpy_ops.scatter_mean(src, index, dim_size)


def scatter_max(
    src: Array,
    index: Array,
    dim_size: int,
) -> Tuple[Array, Optional[Array]]:
    """
    Maximum values at specified indices.

    Args:
        src: Source array of shape (N, ...).
        index: Index array of shape (N,) with values in [0, dim_size).
        dim_size: Size of output first dimension.

    Returns:
        Tuple of (max_values, argmax_indices). argmax_indices may be None.
    """
    if get_backend(src) == Backend.TORCH:
        from . import torch_ops
        return torch_ops.scatter_max(src, index, dim_size)
    from . import numpy_ops
    return numpy_ops.scatter_max(src, index, dim_size)


def scatter_min(
    src: Array,
    index: Array,
    dim_size: int,
) -> Tuple[Array, Optional[Array]]:
    """
    Minimum values at specified indices.

    Args:
        src: Source array of shape (N, ...).
        index: Index array of shape (N,) with values in [0, dim_size).
        dim_size: Size of output first dimension.

    Returns:
        Tuple of (min_values, argmin_indices). argmin_indices may be None.
    """
    if get_backend(src) == Backend.TORCH:
        from . import torch_ops
        return torch_ops.scatter_min(src, index, dim_size)
    from . import numpy_ops
    return numpy_ops.scatter_min(src, index, dim_size)


def repeat_interleave(arr: Array, repeats: Array) -> Array:
    """
    Repeat elements of an array along the first axis.

    Args:
        arr: Array to repeat elements from.
        repeats: Number of times to repeat each element.

    Returns:
        Array with repeated elements.
    """
    if get_backend(arr) == Backend.TORCH:
        from . import torch_ops
        return torch_ops.repeat_interleave(arr, repeats)
    from . import numpy_ops
    return numpy_ops.repeat_interleave(arr, repeats)


def cdist(x1: Array, x2: Array) -> Array:
    """
    Compute pairwise Euclidean distances.

    Args:
        x1: Array of shape (M, D).
        x2: Array of shape (N, D).

    Returns:
        Distance matrix of shape (M, N).
    """
    if get_backend(x1) == Backend.TORCH:
        from . import torch_ops
        return torch_ops.cdist(x1, x2)
    from . import numpy_ops
    return numpy_ops.cdist(x1, x2)


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

    if get_backend(arrays[0]) == Backend.TORCH:
        from . import torch_ops
        return torch_ops.cat(arrays, dim=axis)
    from . import numpy_ops
    return numpy_ops.cat(arrays, axis=axis)


def multiply(a: Array, b: Array) -> Array:
    """
    Element-wise multiplication.

    Args:
        a: First array.
        b: Second array.

    Returns:
        Element-wise product.
    """
    if get_backend(a) == Backend.TORCH:
        from . import torch_ops
        return torch_ops.multiply(a, b)
    from . import numpy_ops
    return numpy_ops.multiply(a, b)
