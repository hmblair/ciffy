"""
Reduction operations for aggregating values across structure levels.

Provides scatter operations to reduce per-atom features to per-residue,
per-chain, or per-molecule features.
"""

from __future__ import annotations
from enum import Enum
from typing import Union
import torch
from torch_scatter import (
    scatter_sum as t_scatter_sum,
    scatter_mean as t_scatter_mean,
    scatter_max as t_scatter_max,
    scatter_min as t_scatter_min,
)


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
    features: torch.Tensor,
    indices: torch.Tensor,
    dim: int,
    dim_size: int,
) -> list[torch.Tensor]:
    """
    Group features by their indices into a list of tensors.

    Args:
        features: Values to group.
        indices: Index for each value.
        dim: Dimension to reduce (unused, for API compatibility).
        dim_size: Number of unique indices (unused, for API compatibility).

    Returns:
        List where each element contains all values for that index.
    """
    return [
        features[indices == ix]
        for ix in range(indices.max() + 1)
    ]


REDUCTIONS = {
    Reduction.NONE: lambda features, indices, dim, dim_size: features,
    Reduction.COLLATE: scatter_collate,
    Reduction.MEAN: t_scatter_mean,
    Reduction.SUM: t_scatter_sum,
    Reduction.MIN: t_scatter_min,
    Reduction.MAX: t_scatter_max,
}


# Type alias for reduction results
ReductionResult = Union[
    torch.Tensor,
    tuple[torch.Tensor, torch.LongTensor],
    list[torch.Tensor],
]


def create_reduction_index(count: int, sizes: torch.Tensor) -> torch.Tensor:
    """
    Create an index tensor for scatter reduction.

    Args:
        count: Number of unique groups.
        sizes: Number of elements in each group.

    Returns:
        Tensor where element i contains the group index for that element.

    Example:
        >>> create_reduction_index(3, torch.tensor([2, 1, 3]))
        tensor([0, 0, 1, 2, 2, 2])
    """
    return torch.arange(count).repeat_interleave(sizes)
