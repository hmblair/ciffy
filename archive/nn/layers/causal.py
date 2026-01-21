"""
Causal (autoregressive) transformer components.

This module provides a convenience function for creating causal transformers.
The main implementation is in transformer.py with is_causal=True.

Example:
    >>> from ciffy.nn.layers import CausalTransformer
    >>>
    >>> model = CausalTransformer(d_model=256, num_layers=6, num_heads=8)
    >>> x = torch.randn(2, 100, 256)  # (batch, seq, dim)
    >>> out = model(x)  # Each position only sees previous positions

    # Equivalent to:
    >>> from ciffy.nn.layers import Transformer
    >>> model = Transformer(d_model=256, num_layers=6, num_heads=8, is_causal=True)
"""

from __future__ import annotations

from typing import Optional

import torch

from .transformer import Transformer


def create_causal_mask(seq_len: int, device: "torch.device") -> "torch.Tensor":
    """
    Create a causal attention mask.

    Args:
        seq_len: Sequence length.
        device: Device to create mask on.

    Returns:
        Boolean mask of shape (seq_len, seq_len) where True = masked (cannot attend).
        Upper triangular (excluding diagonal) is True.
    """
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


def CausalTransformer(
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: Optional[int] = None,
    dropout: float = 0.0,
    max_seq_len: int = 2048,
    qk_norm: bool = False,
    layer_scale_init: Optional[float] = None,
) -> Transformer:
    """
    Create a causal (autoregressive) transformer.

    This is a convenience function that creates a Transformer with is_causal=True.
    Each position can only attend to previous positions, enabling left-to-right
    generation.

    Args:
        d_model: Model dimension.
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads.
        d_ff: Feedforward hidden dimension (default: auto-computed for SwiGLU).
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length for RoPE.
        qk_norm: Whether to apply QK-Norm for training stability.
        layer_scale_init: Initial value for layer scale parameters.

    Returns:
        Transformer instance with causal masking enabled.

    Example:
        >>> model = CausalTransformer(d_model=256, num_layers=6, num_heads=8)
        >>> x = torch.randn(2, 100, 256)  # (batch, seq, dim)
        >>> out = model(x)  # (batch, seq, dim) - each position sees only past
    """
    return Transformer(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        max_seq_len=max_seq_len,
        use_rope=True,
        qk_norm=qk_norm,
        layer_scale_init=layer_scale_init,
        is_causal=True,
    )
