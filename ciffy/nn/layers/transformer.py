"""
Modern Transformer implementation with best practices.

This module provides a reusable transformer architecture following
modern design choices from LLaMA, GPT-NeoX, and PaLM:

- **Pre-LN**: LayerNorm before attention/FFN for stable training
- **RMSNorm**: Simpler, faster normalization
- **RoPE**: Rotary Position Embeddings for better length generalization
- **SwiGLU**: Gated activation for improved performance

Example:
    >>> from ciffy.nn import Transformer
    >>>
    >>> model = Transformer(d_model=256, num_layers=4, num_heads=8)
    >>> x = torch.randn(2, 100, 256)  # (batch, seq, dim)
    >>> out = model(x)
"""

from __future__ import annotations

import logging
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None
    F = None

logger = logging.getLogger(__name__)


def _check_tensor(
    tensor: "torch.Tensor",
    name: str,
    expected_dims: Optional[int] = None,
    check_nan: bool = True,
    check_inf: bool = True,
) -> None:
    """
    Validate tensor for common issues that cause CUDA errors.

    Args:
        tensor: Tensor to validate.
        name: Name for error messages.
        expected_dims: Expected number of dimensions, or None to skip.
        check_nan: Whether to check for NaN values.
        check_inf: Whether to check for Inf values.

    Raises:
        ValueError: If tensor has wrong dimensions.
        RuntimeError: If tensor contains NaN or Inf values.
    """
    if expected_dims is not None and tensor.dim() != expected_dims:
        raise ValueError(
            f"{name}: expected {expected_dims}D tensor, got {tensor.dim()}D "
            f"with shape {tuple(tensor.shape)}"
        )

    if check_nan and torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        raise RuntimeError(
            f"{name}: tensor contains {nan_count} NaN values. "
            f"Shape: {tuple(tensor.shape)}, dtype: {tensor.dtype}"
        )

    if check_inf and torch.isinf(tensor).any():
        inf_count = torch.isinf(tensor).sum().item()
        raise RuntimeError(
            f"{name}: tensor contains {inf_count} Inf values. "
            f"Shape: {tuple(tensor.shape)}, dtype: {tensor.dtype}"
        )


class RMSNorm(nn.Module if TORCH_AVAILABLE else object):
    """
    Root Mean Square Layer Normalization.

    Simpler and faster than LayerNorm. Normalizes by RMS without centering.

    Reference: Zhang & Sennrich (2019) https://arxiv.org/abs/1910.07467
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Apply RMS normalization.

        Args:
            x: Input tensor of any shape, normalized along last dimension.

        Returns:
            Normalized tensor with same shape as input.

        Raises:
            RuntimeError: If input contains NaN values.
        """
        # Check for NaN (would propagate through normalization)
        if torch.isnan(x).any():
            nan_count = torch.isnan(x).sum().item()
            raise RuntimeError(
                f"RMSNorm: input contains {nan_count} NaN values. "
                f"Shape: {tuple(x.shape)}"
            )

        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryPositionEmbedding(nn.Module if TORCH_AVAILABLE else object):
    """
    Rotary Position Embeddings (RoPE).

    Encodes position by rotating query/key vectors, enabling the model to
    learn relative positions and generalize to longer sequences.

    Reference: Su et al. (2021) https://arxiv.org/abs/2104.09864
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(f"RoPE dimension must be even, got {dim}")

        self.dim = dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._update_cache(max_seq_len)

    def _update_cache(self, seq_len: int) -> None:
        positions = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, q: "torch.Tensor", k: "torch.Tensor", seq_len: Optional[int] = None
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """
        Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor of shape (batch, heads, seq, head_dim)
            k: Key tensor of shape (batch, heads, seq, head_dim)
            seq_len: Sequence length (defaults to q.shape[2])

        Returns:
            Rotated (q, k) tensors.

        Raises:
            ValueError: If seq_len is invalid or tensors have wrong shape.
        """
        # Validate tensor shapes FIRST (before accessing shape indices)
        if q.dim() != 4:
            raise ValueError(
                f"RoPE: query must be 4D (batch, heads, seq, head_dim), "
                f"got {q.dim()}D with shape {tuple(q.shape)}"
            )
        if k.dim() != 4:
            raise ValueError(
                f"RoPE: key must be 4D (batch, heads, seq, head_dim), "
                f"got {k.dim()}D with shape {tuple(k.shape)}"
            )

        # Now safe to access shape indices
        if seq_len is None:
            seq_len = q.shape[2]

        # Validate sequence length
        if seq_len <= 0:
            raise ValueError(
                f"RoPE: seq_len must be positive, got {seq_len}"
            )

        # Validate head dimension matches RoPE dimension
        if q.shape[-1] != self.dim:
            raise ValueError(
                f"RoPE: head_dim mismatch. RoPE initialized with dim={self.dim}, "
                f"but query has head_dim={q.shape[-1]}"
            )

        if seq_len > self.cos_cached.shape[0]:
            self._update_cache(seq_len)

        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)

    def _rotate(self, x: "torch.Tensor", cos: "torch.Tensor", sin: "torch.Tensor") -> "torch.Tensor":
        """Apply rotary embedding to input tensor."""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin


class SwiGLU(nn.Module if TORCH_AVAILABLE else object):
    """
    SwiGLU feedforward network.

    Gated Linear Unit with Swish activation. Generally outperforms GELU/ReLU FFN.

    Reference: Shazeer (2020) https://arxiv.org/abs/2002.05202
    """

    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.0):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        super().__init__()

        if d_ff is None:
            d_ff = int(4 * d_model * 2 / 3)
            d_ff = ((d_ff + 63) // 64) * 64  # Round to multiple of 64

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class MultiHeadAttention(nn.Module if TORCH_AVAILABLE else object):
    """
    Multi-head attention with Rotary Position Embeddings.

    Uses efficient scaled dot-product attention (Flash Attention) when available.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
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

    def forward(self, x: "torch.Tensor", mask: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        """
        Apply multi-head attention.

        Args:
            x: Input tensor of shape (batch, seq, d_model)
            mask: Padding mask of shape (batch, seq) where True = masked

        Returns:
            Output tensor of shape (batch, seq, d_model)

        Raises:
            ValueError: If input shapes are invalid.
            RuntimeError: If tensors contain NaN/Inf values.
        """
        # Validate input shape
        if x.dim() != 3:
            raise ValueError(
                f"MultiHeadAttention: input must be 3D (batch, seq, d_model), "
                f"got {x.dim()}D with shape {tuple(x.shape)}"
            )

        B, L, D = x.shape

        if D != self.d_model:
            raise ValueError(
                f"MultiHeadAttention: input d_model mismatch. "
                f"Expected {self.d_model}, got {D}. Input shape: {tuple(x.shape)}"
            )

        if L <= 0:
            raise ValueError(
                f"MultiHeadAttention: sequence length must be positive, got {L}"
            )

        # Validate mask if provided
        if mask is not None:
            if mask.dim() != 2:
                raise ValueError(
                    f"MultiHeadAttention: mask must be 2D (batch, seq), "
                    f"got {mask.dim()}D with shape {tuple(mask.shape)}"
                )
            if mask.shape[0] != B or mask.shape[1] != L:
                raise ValueError(
                    f"MultiHeadAttention: mask shape {tuple(mask.shape)} "
                    f"doesn't match input batch/seq ({B}, {L})"
                )

        # Check for NaN in input (common cause of CUDA errors)
        if torch.isnan(x).any():
            nan_count = torch.isnan(x).sum().item()
            raise RuntimeError(
                f"MultiHeadAttention: input contains {nan_count} NaN values. "
                f"This often indicates numerical instability in earlier layers."
            )

        qkv = self.qkv_proj(x).view(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q, k = self.rope(q, k, seq_len=L)

        if hasattr(F, "scaled_dot_product_attention"):
            attn_mask = None
            if mask is not None:
                attn_mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, L, -1)
                attn_mask = attn_mask.float().masked_fill(attn_mask, float("-inf"))

            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        else:
            scale = self.head_dim ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            if mask is not None:
                attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))
            attn = F.softmax(attn, dim=-1)

            # Check for NaN after softmax (can happen with all-masked sequences)
            if torch.isnan(attn).any():
                raise RuntimeError(
                    f"MultiHeadAttention: NaN in attention weights after softmax. "
                    f"This typically occurs when all positions are masked in a sequence."
                )

            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)


class TransformerBlock(nn.Module if TORCH_AVAILABLE else object):
    """
    Pre-LN Transformer block with RMSNorm, RoPE, and SwiGLU.

    Architecture:
        x = x + Attention(RMSNorm(x))
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
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        super().__init__()

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout, max_seq_len)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: "torch.Tensor", mask: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        x = x + self.dropout(self.attn(self.norm1(x), mask=mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class Transformer(nn.Module if TORCH_AVAILABLE else object):
    """
    Modern Transformer encoder.

    Uses Pre-LN architecture with RMSNorm, RoPE, and SwiGLU - following
    best practices from LLaMA, GPT-NeoX, and PaLM.

    Args:
        d_model: Model dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        d_ff: Feedforward hidden dimension (default: auto-computed for SwiGLU)
        dropout: Dropout probability
        max_seq_len: Maximum sequence length for RoPE

    Example:
        >>> model = Transformer(d_model=256, num_layers=4, num_heads=8)
        >>> x = torch.randn(2, 100, 256)
        >>> out = model(x)  # (2, 100, 256)
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
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout, max_seq_len)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)

    def forward(self, x: "torch.Tensor", mask: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        """
        Process input through transformer layers.

        Args:
            x: Input tensor (batch, seq, d_model)
            mask: Padding mask (batch, seq) where True = masked/ignored

        Returns:
            Output tensor (batch, seq, d_model)

        Raises:
            ValueError: If input shapes are invalid.
            RuntimeError: If tensors contain NaN/Inf values.
        """
        # Validate input tensor
        if x.dim() != 3:
            raise ValueError(
                f"Transformer: input must be 3D (batch, seq, d_model), "
                f"got {x.dim()}D with shape {tuple(x.shape)}"
            )

        B, L, D = x.shape

        if D != self.d_model:
            raise ValueError(
                f"Transformer: input d_model mismatch. "
                f"Expected {self.d_model}, got {D}. "
                f"Input shape: {tuple(x.shape)}"
            )

        if L <= 0:
            raise ValueError(
                f"Transformer: sequence length must be positive, got {L}. "
                f"This may indicate empty input data."
            )

        if B <= 0:
            raise ValueError(
                f"Transformer: batch size must be positive, got {B}."
            )

        # Validate mask if provided
        if mask is not None:
            if mask.dim() != 2:
                raise ValueError(
                    f"Transformer: mask must be 2D (batch, seq), "
                    f"got {mask.dim()}D with shape {tuple(mask.shape)}"
                )
            if mask.shape[0] != B:
                raise ValueError(
                    f"Transformer: mask batch size {mask.shape[0]} "
                    f"doesn't match input batch size {B}"
                )
            if mask.shape[1] != L:
                raise ValueError(
                    f"Transformer: mask sequence length {mask.shape[1]} "
                    f"doesn't match input sequence length {L}"
                )

            # Check if all positions are masked (would cause NaN in attention)
            if mask.all():
                raise ValueError(
                    f"Transformer: all positions are masked. "
                    f"At least one position must be unmasked per sequence."
                )

            # Check per-sequence: warn if any sequence is fully masked
            fully_masked = mask.all(dim=1)
            if fully_masked.any():
                n_fully_masked = fully_masked.sum().item()
                raise ValueError(
                    f"Transformer: {n_fully_masked} sequence(s) have all positions masked. "
                    f"This will cause NaN in attention weights."
                )

        # Check for NaN/Inf in input
        if torch.isnan(x).any():
            nan_count = torch.isnan(x).sum().item()
            raise RuntimeError(
                f"Transformer: input contains {nan_count} NaN values. "
                f"Check your data preprocessing and embedding layers."
            )

        if torch.isinf(x).any():
            inf_count = torch.isinf(x).sum().item()
            raise RuntimeError(
                f"Transformer: input contains {inf_count} Inf values. "
                f"This may indicate exploding gradients or invalid operations."
            )

        # Process through layers
        for i, layer in enumerate(self.layers):
            x = layer(x, mask=mask)

            # Check for NaN propagation after each layer (debug mode only)
            if logger.isEnabledFor(logging.DEBUG) and torch.isnan(x).any():
                nan_count = torch.isnan(x).sum().item()
                logger.debug(
                    f"Transformer layer {i}: output contains {nan_count} NaN values"
                )

        return self.final_norm(x)
