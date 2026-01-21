"""Time embeddings for flow matching and continuous normalizing flows.

This module provides embeddings for time-conditioned models used in
flow matching and continuous normalizing flows.

Example:
    >>> from ciffy.nn.flow import SinusoidalTimeEmbedding
    >>>
    >>> embed = SinusoidalTimeEmbedding(dim=128)
    >>> t = torch.rand(32)  # Time values in [0, 1]
    >>> t_emb = embed(t)    # (32, 128)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for flow matching.

    Embeds scalar time values into a higher-dimensional space using
    sinusoidal functions at different frequencies. This is the standard
    positional encoding adapted for continuous time values.

    Args:
        dim: Output embedding dimension.

    Example:
        >>> embed = SinusoidalTimeEmbedding(dim=128)
        >>> t = torch.tensor([0.0, 0.5, 1.0])
        >>> t_emb = embed(t)  # (3, 128)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embed time values.

        Args:
            t: Time values of shape (batch,) or scalar, typically in [0, 1].

        Returns:
            Embeddings of shape (batch, dim).
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


__all__ = ["SinusoidalTimeEmbedding"]
