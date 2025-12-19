"""
Reduction operations for aggregating values across structure levels.

Provides scatter operations to reduce per-atom features to per-residue,
per-chain, or per-molecule features.
"""

from __future__ import annotations
from enum import Enum
from typing import Union, TYPE_CHECKING

import numpy as np

from ..backend import ops as backend
from ..backend import Array, is_torch

if TYPE_CHECKING:
    import torch


class Reduction(Enum):
    """
    Types of reduction operations for aggregating values.

    - NONE: Return values unchanged
    - COLLATE: Group values into a list per index
    - MEAN: Compute mean per index
    - SUM: Compute sum per index
    - MIN: Compute minimum per index (returns values and indices)
    - MAX: Compute maximum per index (returns values and indices)
    """

    NONE = 0
    COLLATE = 1
    MEAN = 2
    SUM = 3
    MIN = 4
    MAX = 5


def scatter_collate(
    features: Array,
    indices: Array,
    dim: int,
    dim_size: int,
) -> list:
    """
    Group features by their indices into a list of arrays.

    Uses O(n log n) sort + split instead of O(n * dim_size) boolean masks.

    Args:
        features: Values to group.
        indices: Index for each value.
        dim: Dimension to reduce (unused, for API compatibility).
        dim_size: Number of unique indices.

    Returns:
        List where each element contains all values for that index.
    """
    if len(indices) == 0:
        return [features[:0] for _ in range(dim_size)]

    # Sort by index: O(n log n)
    sort_order = backend.argsort(indices)
    sorted_features = features[sort_order]
    sorted_indices = indices[sort_order]

    # Find split points where index changes: O(n)
    diffs = backend.diff(sorted_indices)
    split_points = backend.nonzero_1d(diffs) + 1

    # Split into groups: O(n)
    chunks = backend.split_at_indices(sorted_features, split_points)

    # Handle missing indices (indices with no elements)
    # Get actual indices present in data
    if len(chunks) == 0:
        return [features[:0] for _ in range(dim_size)]

    # Build result with empty arrays for missing indices
    unique_indices = sorted_indices[
        backend.cat([backend.array([0], like=sorted_indices), split_points])
    ]

    result = []
    chunk_idx = 0
    for i in range(dim_size):
        if chunk_idx < len(unique_indices) and int(unique_indices[chunk_idx]) == i:
            result.append(chunks[chunk_idx])
            chunk_idx += 1
        else:
            # Empty array for missing index
            result.append(features[:0])

    return result


def _scatter_sum(features: Array, indices: Array, dim: int, dim_size: int) -> Array:
    """Scatter sum wrapper for REDUCTIONS dict."""
    return backend.scatter_sum(features, indices, dim_size)


def _scatter_mean(features: Array, indices: Array, dim: int, dim_size: int) -> Array:
    """Scatter mean wrapper for REDUCTIONS dict."""
    return backend.scatter_mean(features, indices, dim_size)


def _scatter_min(features: Array, indices: Array, dim: int, dim_size: int):
    """Scatter min wrapper for REDUCTIONS dict."""
    return backend.scatter_min(features, indices, dim_size)


def _scatter_max(features: Array, indices: Array, dim: int, dim_size: int):
    """Scatter max wrapper for REDUCTIONS dict."""
    return backend.scatter_max(features, indices, dim_size)


REDUCTIONS = {
    Reduction.NONE: lambda features, indices, dim, dim_size: features,
    Reduction.COLLATE: scatter_collate,
    Reduction.MEAN: _scatter_mean,
    Reduction.SUM: _scatter_sum,
    Reduction.MIN: _scatter_min,
    Reduction.MAX: _scatter_max,
}


# Type alias for reduction results
ReductionResult = Union[
    Array,                              # MEAN, SUM
    tuple[Array, Union[Array, None]],   # MIN, MAX (values, indices)
    list[Array],                        # COLLATE
]


def create_reduction_index(count: int, sizes: Array, device=None) -> Array:
    """
    Create an index array for scatter reduction.

    Args:
        count: Number of unique groups.
        sizes: Number of elements in each group.
        device: Target device for the index (torch only). If None, uses sizes.device.

    Returns:
        Array where element i contains the group index for that element.

    Example:
        >>> create_reduction_index(3, np.array([2, 1, 3]))
        array([0, 0, 1, 2, 2, 2])
    """
    if is_torch(sizes):
        import torch
        target_device = device if device is not None else sizes.device
        return torch.arange(count, device=target_device).repeat_interleave(
            sizes.to(target_device)
        )
    return np.repeat(np.arange(count), sizes)
