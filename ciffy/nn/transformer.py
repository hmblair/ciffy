"""
Modern Transformer implementation with best practices.

This module provides a reusable transformer architecture following
modern design choices:

- **Pre-LN**: LayerNorm before attention/FFN for stable training
- **RMSNorm**: Simpler, faster normalization (optional)
- **RoPE**: Rotary Position Embeddings for better length generalization
- **SwiGLU**: Gated activation for improved performance

Based on architectures from LLaMA, GPT-NeoX, and PaLM.

Example:
    >>> from ciffy.nn.transformer import Transformer
    >>>
    >>> # Create transformer encoder
    >>> model = Transformer(
    ...     d_model=256,
    ...     num_layers=4,
    ...     num_heads=8,
    ...     use_rope=True,
    ...     use_swiglu=True,
    ... )
    >>>
    >>> # Forward pass
    >>> x = torch.randn(2, 100, 256)  # (batch, seq, dim)
    >>> out = model(x)
"""

from __future__ import annotations

import math
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


# =============================================================================
# RMSNorm
# =============================================================================


class RMSNorm(nn.Module if TORCH_AVAILABLE else object):
    """
    Root Mean Square Layer Normalization.

    Simpler and slightly faster than LayerNorm, used in LLaMA and other
    modern architectures. Normalizes by RMS without centering.

    RMSNorm(x) = x / RMS(x) * gamma
    where RMS(x) = sqrt(mean(x^2) + eps)

    Args:
        dim: Feature dimension to normalize
        eps: Small constant for numerical stability

    Reference:
        Zhang & Sennrich (2019) "Root Mean Square Layer Normalization"
        https://arxiv.org/abs/1910.07467
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for RMSNorm")
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return x / rms * self.weight


# =============================================================================
# Rotary Position Embeddings (RoPE)
# =============================================================================


class RotaryPositionEmbedding(nn.Module if TORCH_AVAILABLE else object):
    """
    Rotary Position Embeddings (RoPE).

    Encodes position information by rotating query and key vectors in
    2D subspaces. This allows the model to learn relative positions
    naturally and generalizes better to longer sequences than absolute
    position embeddings.

    The rotation is applied as:
        q_rotated = q * cos(theta) + rotate_half(q) * sin(theta)

    where theta depends on position and dimension.

    Args:
        dim: Head dimension (must be even)
        max_seq_len: Maximum sequence length for precomputation
        base: Base for the frequency computation (default 10000)

    Reference:
        Su et al. (2021) "RoFormer: Enhanced Transformer with Rotary Position Embedding"
        https://arxiv.org/abs/2104.09864
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for RotaryPositionEmbedding")
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(f"RoPE dimension must be even, got {dim}")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequency bands
        # inv_freq shape: (dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin for positions up to max_seq_len
        self._update_cos_sin_cache(max_seq_len)

    def _update_cos_sin_cache(self, seq_len: int) -> None:
        """Precompute cos/sin values for given sequence length."""
        # positions: (seq_len,)
        positions = torch.arange(seq_len, device=self.inv_freq.device)

        # freqs: (seq_len, dim // 2)
        freqs = torch.outer(positions, self.inv_freq)

        # Duplicate for pairs: (seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)

        # Cache cos and sin
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        q: "torch.Tensor",
        k: "torch.Tensor",
        seq_len: Optional[int] = None,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """
        Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor of shape (batch, heads, seq, head_dim)
            k: Key tensor of shape (batch, heads, seq, head_dim)
            seq_len: Sequence length (inferred from q if not provided)

        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes as input
        """
        if seq_len is None:
            seq_len = q.shape[2]

        # Extend cache if needed
        if seq_len > self.cos_cached.shape[0]:
            self._update_cos_sin_cache(seq_len)

        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq, dim)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq, dim)

        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)

        return q_rotated, k_rotated

    def _apply_rotary(
        self,
        x: "torch.Tensor",
        cos: "torch.Tensor",
        sin: "torch.Tensor",
    ) -> "torch.Tensor":
        """Apply rotary embedding to a single tensor."""
        # x shape: (batch, heads, seq, head_dim)
        # Split into pairs and rotate
        x_rotated = self._rotate_half(x)
        return x * cos + x_rotated * sin

    @staticmethod
    def _rotate_half(x: "torch.Tensor") -> "torch.Tensor":
        """Rotate half the hidden dims."""
        # Split into first and second half
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        # Rotate: [x1, x2] -> [-x2, x1]
        return torch.cat([-x2, x1], dim=-1)


# =============================================================================
# SwiGLU Feedforward
# =============================================================================


class SwiGLU(nn.Module if TORCH_AVAILABLE else object):
    """
    SwiGLU feedforward network.

    Gated Linear Unit with Swish activation, used in PaLM, LLaMA, etc.
    Generally outperforms standard FFN with GELU or ReLU.

    SwiGLU(x) = (Linear1(x) * Swish(Linear2(x))) @ Linear3

    The hidden dimension is typically (2/3) * 4 * d_model to match
    parameter count of standard FFN with 4 * d_model hidden.

    Args:
        d_model: Model dimension
        d_ff: Feedforward hidden dimension (default: 4 * d_model * 2/3)
        dropout: Dropout probability
        bias: Whether to use bias in linear layers

    Reference:
        Shazeer (2020) "GLU Variants Improve Transformer"
        https://arxiv.org/abs/2002.05202
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for SwiGLU")
        super().__init__()

        # Default hidden dim: 4 * d_model * (2/3) to match param count
        if d_ff is None:
            d_ff = int(4 * d_model * 2 / 3)
            # Round to multiple of 64 for efficiency
            d_ff = ((d_ff + 63) // 64) * 64

        self.w1 = nn.Linear(d_model, d_ff, bias=bias)  # Gate projection
        self.w2 = nn.Linear(d_model, d_ff, bias=bias)  # Up projection
        self.w3 = nn.Linear(d_ff, d_model, bias=bias)  # Down projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # SwiGLU: Swish(xW1) * (xW2) then project down
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class StandardFFN(nn.Module if TORCH_AVAILABLE else object):
    """
    Standard feedforward network with GELU activation.

    FFN(x) = GELU(xW1 + b1)W2 + b2

    Args:
        d_model: Model dimension
        d_ff: Feedforward hidden dimension (default: 4 * d_model)
        dropout: Dropout probability
        bias: Whether to use bias in linear layers
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for StandardFFN")
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.dropout(self.w2(F.gelu(self.w1(x))))


# =============================================================================
# Multi-Head Attention with RoPE
# =============================================================================


class MultiHeadAttention(nn.Module if TORCH_AVAILABLE else object):
    """
    Multi-head attention with optional Rotary Position Embeddings.

    Supports both standard attention and RoPE-based attention.
    Uses efficient scaled dot-product attention when available.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Attention dropout probability
        use_rope: Whether to use Rotary Position Embeddings
        max_seq_len: Maximum sequence length for RoPE
        bias: Whether to use bias in projections
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rope: bool = True,
        max_seq_len: int = 2048,
        bias: bool = False,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MultiHeadAttention")
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections (packed QKV for efficiency)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

        # Optional RoPE
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)
        else:
            self.rope = None

    def forward(
        self,
        x: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """
        Forward pass for self-attention.

        Args:
            x: Input tensor of shape (batch, seq, d_model)
            mask: Optional attention mask (True = masked/ignored)

        Returns:
            Output tensor of shape (batch, seq, d_model)
        """
        B, L, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE if enabled
        if self.use_rope:
            q, k = self.rope(q, k, seq_len=L)

        # Scaled dot-product attention
        # Try to use Flash Attention if available (PyTorch 2.0+)
        if hasattr(F, "scaled_dot_product_attention"):
            # Convert mask to attention mask format
            attn_mask = None
            if mask is not None:
                # mask shape: (B, L) where True = ignore
                # Need shape: (B, 1, 1, L) or (B, 1, L, L)
                attn_mask = mask.unsqueeze(1).unsqueeze(2)
                attn_mask = attn_mask.expand(-1, -1, L, -1)
                # Convert to float mask where -inf = ignore
                attn_mask = attn_mask.float().masked_fill(attn_mask, float("-inf"))

            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        else:
            # Manual attention computation
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            if mask is not None:
                # mask shape: (B, L) -> (B, 1, 1, L)
                mask = mask.unsqueeze(1).unsqueeze(2)
                attn = attn.masked_fill(mask, float("-inf"))

            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)


# =============================================================================
# Transformer Block (Pre-LN)
# =============================================================================


class TransformerBlock(nn.Module if TORCH_AVAILABLE else object):
    """
    Pre-LN Transformer block.

    Uses Pre-LayerNorm architecture for more stable training:
        x = x + Attention(Norm(x))
        x = x + FFN(Norm(x))

    This differs from the original Post-LN Transformer which applies
    normalization after the residual connection.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feedforward hidden dimension
        dropout: Dropout probability
        use_rope: Whether to use Rotary Position Embeddings
        use_swiglu: Whether to use SwiGLU (vs standard GELU FFN)
        use_rmsnorm: Whether to use RMSNorm (vs LayerNorm)
        max_seq_len: Maximum sequence length for RoPE
        bias: Whether to use bias in linear layers
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        use_rope: bool = True,
        use_swiglu: bool = True,
        use_rmsnorm: bool = True,
        max_seq_len: int = 2048,
        bias: bool = False,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TransformerBlock")
        super().__init__()

        # Normalization layers
        norm_class = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.norm1 = norm_class(d_model)
        self.norm2 = norm_class(d_model)

        # Attention
        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
            bias=bias,
        )

        # Feedforward
        if use_swiglu:
            self.ffn = SwiGLU(d_model, d_ff, dropout=dropout, bias=bias)
        else:
            self.ffn = StandardFFN(d_model, d_ff, dropout=dropout, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """
        Forward pass with Pre-LN residual connections.

        Args:
            x: Input tensor of shape (batch, seq, d_model)
            mask: Optional attention mask (True = masked/ignored)

        Returns:
            Output tensor of shape (batch, seq, d_model)
        """
        # Pre-LN attention block
        x = x + self.dropout(self.attn(self.norm1(x), mask=mask))

        # Pre-LN feedforward block
        x = x + self.dropout(self.ffn(self.norm2(x)))

        return x


# =============================================================================
# Full Transformer
# =============================================================================


class Transformer(nn.Module if TORCH_AVAILABLE else object):
    """
    Modern Transformer encoder with configurable architecture.

    Supports various architecture choices:
    - Pre-LN vs Post-LN (Pre-LN default)
    - RoPE vs absolute position embeddings
    - SwiGLU vs standard GELU FFN
    - RMSNorm vs LayerNorm

    This is a bidirectional encoder suitable for tasks like
    classification, encoding sequences, etc.

    Args:
        d_model: Model dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        d_ff: Feedforward hidden dimension (default: auto based on use_swiglu)
        dropout: Dropout probability
        use_rope: Whether to use Rotary Position Embeddings
        use_swiglu: Whether to use SwiGLU activation
        use_rmsnorm: Whether to use RMSNorm
        max_seq_len: Maximum sequence length
        bias: Whether to use bias in linear layers

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
        use_rope: bool = True,
        use_swiglu: bool = True,
        use_rmsnorm: bool = True,
        max_seq_len: int = 2048,
        bias: bool = False,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Transformer")
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_rope = use_rope

        # Stack of transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                use_rope=use_rope,
                use_swiglu=use_swiglu,
                use_rmsnorm=use_rmsnorm,
                max_seq_len=max_seq_len,
                bias=bias,
            )
            for _ in range(num_layers)
        ])

        # Final normalization (important for Pre-LN)
        norm_class = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.final_norm = norm_class(d_model)

    def forward(
        self,
        x: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """
        Forward pass through all transformer layers.

        Args:
            x: Input tensor of shape (batch, seq, d_model)
            mask: Optional padding mask of shape (batch, seq)
                  where True = masked/ignored positions

        Returns:
            Output tensor of shape (batch, seq, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask=mask)

        return self.final_norm(x)
