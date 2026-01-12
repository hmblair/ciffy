"""
Causal (autoregressive) transformer components.

This module provides decoder-only transformer architecture for autoregressive
generation, following GPT/LLaMA design patterns:

- **Causal masking**: Each position can only attend to previous positions
- **Pre-LN**: LayerNorm before attention/FFN for stable training
- **RMSNorm**: Simpler, faster normalization
- **RoPE**: Rotary Position Embeddings for relative position encoding
- **SwiGLU**: Gated activation for improved performance

Example:
    >>> from ciffy.nn.layers import CausalTransformer
    >>>
    >>> model = CausalTransformer(d_model=256, num_layers=6, num_heads=8)
    >>> x = torch.randn(2, 100, 256)  # (batch, seq, dim)
    >>> out = model(x)  # Each position only sees previous positions
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import RMSNorm, RotaryPositionEmbedding, SwiGLU

logger = logging.getLogger(__name__)


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
    # True means "masked" (cannot attend)
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


class CausalMultiHeadAttention(nn.Module):
    """
    Multi-head attention with causal masking and Rotary Position Embeddings.

    Each position can only attend to itself and previous positions (autoregressive).
    Uses Flash Attention when available for O(n) memory.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)

    def forward(
        self,
        x: "torch.Tensor",
        padding_mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """
        Apply causal multi-head attention.

        Args:
            x: Input tensor of shape (batch, seq, d_model)
            padding_mask: Optional padding mask (batch, seq) where True = padded

        Returns:
            Output tensor of shape (batch, seq, d_model)
        """
        B, L, D = x.shape

        if D != self.d_model:
            raise ValueError(
                f"CausalMultiHeadAttention: d_model mismatch. "
                f"Expected {self.d_model}, got {D}"
            )

        # Project to Q, K, V
        qkv = self.qkv_proj(x).view(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply rotary position embeddings
        q, k = self.rope(q, k, seq_len=L)

        # Use Flash Attention with causal mask if available
        if hasattr(F, "scaled_dot_product_attention"):
            if padding_mask is None:
                # No padding - use built-in causal mask (most efficient)
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=True,
                )
            else:
                # Combine causal mask with padding mask
                # Create causal mask: (L, L)
                causal = create_causal_mask(L, x.device)
                # Expand padding mask for keys: (B, 1, 1, L)
                pad_mask = padding_mask.unsqueeze(1).unsqueeze(2)
                # Combined mask: (B, 1, L, L)
                combined = causal.unsqueeze(0).unsqueeze(0) | pad_mask
                attn_mask = combined.float().masked_fill(combined, float("-inf"))

                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
        else:
            # Manual implementation with causal mask
            scale = self.head_dim ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale

            # Apply causal mask
            causal_mask = create_causal_mask(L, x.device)
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            # Apply padding mask if provided
            if padding_mask is not None:
                attn = attn.masked_fill(
                    padding_mask.unsqueeze(1).unsqueeze(2),
                    float("-inf")
                )

            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)


class CausalTransformerBlock(nn.Module):
    """
    Pre-LN Causal Transformer block.

    Architecture:
        x = x + CausalAttention(RMSNorm(x))
        x = x + SwiGLU(RMSNorm(x))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attn = CausalMultiHeadAttention(d_model, num_heads, dropout, max_seq_len)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: "torch.Tensor",
        padding_mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        x = x + self.dropout(self.attn(self.norm1(x), padding_mask=padding_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class CausalTransformer(nn.Module):
    """
    Decoder-only (GPT-style) Transformer for autoregressive generation.

    Each position can only attend to previous positions, enabling left-to-right
    generation. Uses modern architecture choices from LLaMA/GPT-NeoX.

    Args:
        d_model: Model dimension.
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads.
        d_ff: Feedforward hidden dimension (default: auto-computed for SwiGLU).
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length for RoPE.

    Example:
        >>> model = CausalTransformer(d_model=256, num_layers=6, num_heads=8)
        >>> x = torch.randn(2, 100, 256)  # (batch, seq, dim)
        >>> out = model(x)  # (batch, seq, dim) - each position sees only past
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.layers = nn.ModuleList([
            CausalTransformerBlock(d_model, num_heads, d_ff, dropout, max_seq_len)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)

    def forward(
        self,
        x: "torch.Tensor",
        padding_mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """
        Process input through causal transformer layers.

        Args:
            x: Input tensor (batch, seq, d_model)
            padding_mask: Padding mask (batch, seq) where True = padded

        Returns:
            Output tensor (batch, seq, d_model)
        """
        if x.dim() != 3:
            raise ValueError(
                f"CausalTransformer: input must be 3D (batch, seq, d_model), "
                f"got {x.dim()}D with shape {tuple(x.shape)}"
            )

        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)

        return self.final_norm(x)
