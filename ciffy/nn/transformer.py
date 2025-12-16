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
        if seq_len is None:
            seq_len = q.shape[2]

        if seq_len > self.cos_cached.shape[0]:
            self._update_cache(seq_len)

        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        return self._apply(q, cos, sin), self._apply(k, cos, sin)

    def _apply(self, x: "torch.Tensor", cos: "torch.Tensor", sin: "torch.Tensor") -> "torch.Tensor":
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
        B, L, _ = x.shape

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
        Args:
            x: Input tensor (batch, seq, d_model)
            mask: Padding mask (batch, seq) where True = masked/ignored

        Returns:
            Output tensor (batch, seq, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.final_norm(x)
