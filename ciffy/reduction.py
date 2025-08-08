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
    NONE = 0
    COLLATE = 1
    MEAN = 2
    SUM = 3
    MIN = 4
    MAX = 5


def t_scatter_collate(
    features: torch.Tensor,
    indices: torch.Tensor,
    dim: int,
    dim_size: int,
) -> list[torch.Tensor]:
    """
    Return a list of tensors, containing all values corresponding to each value
    of the index.
    """

    return [
        features[indices == ix]
        for ix in range(indices.max() + 1)
    ]


REDUCTIONS = {
    Reduction.NONE: lambda features, indices, dim, dim_size: features,
    Reduction.COLLATE: t_scatter_collate,
    Reduction.MEAN: t_scatter_mean,
    Reduction.SUM: t_scatter_sum,
    Reduction.MIN: t_scatter_min,
    Reduction.MAX: t_scatter_max,
}


_Reduction = Union[
    torch.Tensor,
    tuple[torch.Tensor, torch.LongTensor],
    list[torch.Tensor],
]
