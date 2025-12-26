"""Output projection layers for geometric networks.

These layers project hidden features to structured outputs like matrices,
useful for generating covariance matrices, transformation matrices, etc.

Classes:
    MatrixOutput: Full-rank matrix projection
    LowRankMatrixOutput: Low-rank factorized matrix projection
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_weights(module: nn.Module) -> None:
    """Initialize weights using Xavier uniform for linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class MatrixOutput(nn.Module):
    """Output layer that generates a full-rank matrix from hidden features.

    Projects hidden features to a flattened matrix and reshapes to the target
    dimensions. Uses fused operations for torch.compile optimization.

    Args:
        hidden_dim: Dimension of input hidden features.
        out_rows: Number of rows in output matrix.
        out_cols: Number of columns in output matrix.
        bias: Whether to include bias in the projection.

    Example:
        >>> output = MatrixOutput(hidden_dim=64, out_rows=8, out_cols=16)
        >>> h = torch.randn(100, 64)  # hidden features
        >>> matrix = output(h)  # shape: (100, 8, 16)
    """

    def __init__(
        self,
        hidden_dim: int,
        out_rows: int,
        out_cols: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.out_rows = out_rows
        self.out_cols = out_cols
        self.out_dim_flat = out_rows * out_cols

        self.layer = nn.Linear(hidden_dim, self.out_dim_flat, bias=bias)
        self.apply(_init_weights)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Project hidden features to matrix.

        Args:
            h: Hidden features of shape (..., hidden_dim).

        Returns:
            Matrix of shape (..., out_rows, out_cols).
        """
        *b, _ = h.size()
        out = F.linear(h, self.layer.weight, self.layer.bias)
        return out.view(*b, self.out_rows, self.out_cols)


class LowRankMatrixOutput(nn.Module):
    """Output layer that generates a low-rank factorized matrix from hidden features.

    Projects hidden features to two factor matrices A and B, then computes
    A @ B to produce the output matrix. This significantly reduces parameters
    while maintaining the output shape.

    Parameter count comparison for output matrix of shape (M, N):
    - Full-rank: hidden_dim * M * N
    - Low-rank:  hidden_dim * (M * rank + rank * N) = hidden_dim * rank * (M + N)

    For M=N=64 and rank=8: full-rank = 4096*h, low-rank = 1024*h (4x reduction)

    Uses fused operations for torch.compile optimization.

    Args:
        hidden_dim: Dimension of input hidden features.
        out_rows: Number of rows in output matrix (M).
        out_cols: Number of columns in output matrix (N).
        rank: Rank of the factorization. Must be positive.
        bias: Whether to include bias in the projections.

    Raises:
        ValueError: If rank is less than 1.

    Example:
        >>> output = LowRankMatrixOutput(hidden_dim=64, out_rows=8, out_cols=16, rank=4)
        >>> h = torch.randn(100, 64)  # hidden features
        >>> matrix = output(h)  # shape: (100, 8, 16)
    """

    def __init__(
        self,
        hidden_dim: int,
        out_rows: int,
        out_cols: int,
        rank: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if rank < 1:
            raise ValueError(f"rank must be positive, got {rank}")

        self.out_rows = out_rows
        self.out_cols = out_cols
        self.rank = rank

        # Left factor: (M, rank), Right factor: (rank, N)
        self.layer_left = nn.Linear(hidden_dim, out_rows * rank, bias=bias)
        self.layer_right = nn.Linear(hidden_dim, rank * out_cols, bias=bias)
        self.apply(_init_weights)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Project hidden features to low-rank factorized matrix.

        Args:
            h: Hidden features of shape (..., hidden_dim).

        Returns:
            Matrix of shape (..., out_rows, out_cols).
        """
        *b, _ = h.size()
        left = F.linear(h, self.layer_left.weight, self.layer_left.bias)
        right = F.linear(h, self.layer_right.weight, self.layer_right.bias)
        left = left.view(*b, self.out_rows, self.rank)
        right = right.view(*b, self.rank, self.out_cols)
        return left @ right


__all__ = [
    "MatrixOutput",
    "LowRankMatrixOutput",
]
