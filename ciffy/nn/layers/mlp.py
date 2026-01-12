"""
Multi-layer perceptron with modern architecture choices.

This module provides a unified MLP implementation following best practices
from LLaMA, PaLM, etc:

- **RMSNorm**: Pre-normalization for residual blocks
- **SiLU**: Smooth activation (also known as Swish)
- **Optional residual**: For stacking blocks

Example:
    >>> from ciffy.nn.layers import MLP
    >>>
    >>> # Simple projection
    >>> proj = MLP(256, 64)
    >>>
    >>> # Residual block for stacking
    >>> block = MLP(256, 256, residual=True)
    >>> x = block(block(block(x)))  # Stack 3 blocks
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import RMSNorm


class MLP(nn.Module):
    """
    Multi-layer perceptron with modern architecture choices.

    Uses RMSNorm (pre-norm) and SiLU activation following best practices
    from LLaMA, PaLM, etc. Supports optional residual connections for
    stacking blocks.

    Args:
        in_dim: Input dimension.
        out_dim: Output dimension.
        hidden_dims: Hidden layer dimensions. If None, uses single hidden
            layer with 2 * max(in_dim, out_dim) neurons.
        residual: Whether to add residual connection (requires in_dim == out_dim).
        dropout: Dropout probability.
        zero_init: Whether to zero-initialize the final layer weights.
            Useful for residual blocks to start as identity.

    Example:
        >>> # Simple projection
        >>> proj = MLP(256, 64)
        >>>
        >>> # Deep MLP with custom hidden dims
        >>> deep = MLP(64, 10, hidden_dims=[128, 64])
        >>>
        >>> # Residual block for stacking
        >>> block = MLP(256, 256, residual=True)
        >>> x = block(block(block(x)))  # Stack 3 blocks
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Optional[list[int]] = None,
        residual: bool = False,
        dropout: float = 0.0,
        zero_init: bool = False,
    ):
        super().__init__()

        if residual and in_dim != out_dim:
            raise ValueError(
                f"Residual connection requires in_dim == out_dim, "
                f"got in_dim={in_dim}, out_dim={out_dim}"
            )

        self.residual = residual

        # Default: single hidden layer with 2x dimension
        if hidden_dims is None:
            hidden_dims = [2 * max(in_dim, out_dim)]

        # Build layers
        dims = [in_dim] + hidden_dims + [out_dim]
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1])
            for i in range(len(dims) - 1)
        ])

        # Pre-norm for residual blocks
        self.norm = RMSNorm(in_dim) if residual else None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Optional zero initialization for residual blocks
        if zero_init:
            final_layer = self.layers[-1]
            assert isinstance(final_layer, nn.Linear)
            nn.init.zeros_(final_layer.weight)
            if final_layer.bias is not None:
                nn.init.zeros_(final_layer.bias)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass through MLP."""
        residual = x if self.residual else None

        # Pre-norm for residual blocks
        if self.norm is not None:
            x = self.norm(x)

        # Hidden layers with activation and dropout
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.silu(x)
            if self.dropout is not None:
                x = self.dropout(x)

        # Final layer
        x = self.layers[-1](x)
        if self.dropout is not None:
            x = self.dropout(x)

        # Residual connection
        if residual is not None:
            x = residual + x

        return x
