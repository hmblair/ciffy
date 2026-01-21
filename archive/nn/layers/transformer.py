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

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Simpler and faster than LayerNorm. Normalizes by RMS without centering.
    Uses fused rsqrt for better GPU performance.

    Reference: Zhang & Sennrich (2019) https://arxiv.org/abs/1910.07467
    """

    def __init__(self, dim: int, eps: float = 1e-6):
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
        """
        # Fused: rsqrt is faster than sqrt + division
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class AdaLN(nn.Module):
    """Adaptive Layer Normalization (AdaLN) for conditioning.

    Modulates normalized features based on a conditioning vector.
    Used in DiT, SDXL, and other modern diffusion transformers.

    Supports both:
    - Global conditioning: cond shape (batch, cond_dim) - same modulation for all positions
    - Per-position conditioning: cond shape (batch, seq_len, cond_dim) - different per position

    Formula:
        y = norm(x) * (1 + scale) + shift
        where (scale, shift) = linear(cond)

    Reference: Peebles & Xie (2023) "Scalable Diffusion Models with Transformers"
    """

    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6):
        """Initialize AdaLN.

        Args:
            dim: Hidden dimension to normalize.
            cond_dim: Dimension of conditioning vector.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.norm = RMSNorm(dim, eps)

        # Project conditioning to scale and shift
        self.proj = nn.Linear(cond_dim, 2 * dim)

        # Initialize to identity transform (scale=0, shift=0)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self, x: "torch.Tensor", cond: "torch.Tensor"
    ) -> "torch.Tensor":
        """Apply adaptive normalization.

        Args:
            x: Input tensor of shape (batch, seq_len, dim).
            cond: Conditioning tensor of shape (batch, cond_dim) for global conditioning,
                or (batch, seq_len, cond_dim) for per-position conditioning.

        Returns:
            Modulated tensor of shape (batch, seq_len, dim).
        """
        # Get scale and shift from conditioning
        scale_shift = self.proj(cond)  # (batch, 2*dim) or (batch, seq_len, 2*dim)
        scale, shift = scale_shift.chunk(2, dim=-1)

        # Apply normalization
        x_norm = self.norm(x)

        # Handle global vs per-position conditioning
        if cond.dim() == 2:
            # Global: (batch, cond_dim) -> unsqueeze for broadcasting
            return x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        else:
            # Per-position: (batch, seq_len, cond_dim) -> already correct shape
            return x_norm * (1 + scale) + shift


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).

    Encodes position by rotating query/key vectors, enabling the model to
    learn relative positions and generalize to longer sequences.

    Reference: Su et al. (2021) https://arxiv.org/abs/2104.09864
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
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

    def apply_by_positions(
        self,
        x: "torch.Tensor",
        positions: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Apply rotary embedding using explicit position indices.

        Useful for edge-wise attention where Q and K have different positions.

        Args:
            x: Tensor of shape (..., head_dim) to rotate.
            positions: Position indices of shape broadcastable to x[..., 0].
                Each position determines the rotation angle.

        Returns:
            Rotated tensor of same shape as x.

        Example:
            >>> rope = RotaryPositionEmbedding(dim=64)
            >>> # Edge-wise attention: Q at target positions, K at source positions
            >>> q = torch.randn(N, nheads, k, 64)  # queries
            >>> k = torch.randn(N, nheads, k, 64)  # keys
            >>> q_pos = torch.arange(N).view(N, 1, 1)  # target positions
            >>> k_pos = neighbor_idx.unsqueeze(1)      # source positions (N, 1, k)
            >>> q_rot = rope.apply_by_positions(q, q_pos)
            >>> k_rot = rope.apply_by_positions(k, k_pos)
        """
        max_pos = int(positions.max().item()) + 1
        if max_pos > self.cos_cached.shape[0]:
            self._update_cache(max_pos)

        # Gather cos/sin for each position
        cos = self.cos_cached[positions]  # (..., head_dim)
        sin = self.sin_cached[positions]  # (..., head_dim)

        return self._rotate(x, cos, sin)


class SwiGLU(nn.Module):
    """
    SwiGLU feedforward network.

    Gated Linear Unit with Swish activation. Generally outperforms GELU/ReLU FFN.
    Uses fused w1/w2 projection for better GPU utilization.

    Reference: Shazeer (2020) https://arxiv.org/abs/2002.05202
    """

    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.0):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model
            d_ff = ((d_ff + 63) // 64) * 64  # Round to multiple of 64

        self.d_ff = d_ff
        # Fused projection: single matmul instead of two separate ones
        self.w12 = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # Single projection + chunk is faster than two separate projections
        x12 = self.w12(x)
        x1, x2 = x12.split(self.d_ff, dim=-1)
        return self.dropout(self.w3(F.silu(x1) * x2))


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with optional Rotary Position Embeddings.

    Uses efficient scaled dot-product attention (Flash Attention) when available.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length for RoPE.
        use_rope: Whether to use Rotary Position Embeddings. Set to False
            when using custom attention biases (e.g., distance-based).
        qk_norm: Whether to apply RMSNorm to queries and keys before the dot
            product. Improves training stability by preventing attention logit
            explosion. From ViT-22B (Dehghani et al., 2023).
        is_causal: Whether to apply causal (autoregressive) masking. Each position
            can only attend to itself and previous positions.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        use_rope: bool = True,
        qk_norm: bool = False,
        is_causal: bool = False,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_rope = use_rope
        self.qk_norm = qk_norm
        self.is_causal = is_causal

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        if use_rope:
            self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)
        else:
            self.rope = None

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        x: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
        attn_bias: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """
        Apply multi-head attention.

        Args:
            x: Input tensor of shape (batch, seq, d_model)
            mask: Padding mask of shape (batch, seq) where True = masked
            attn_bias: Optional attention bias of shape (batch, heads, seq, seq)
                or (batch, seq, seq) to be broadcast across heads. Added to
                attention scores before softmax. Use -inf for hard masking.

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

        qkv = self.qkv_proj(x).view(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE if enabled
        if self.rope is not None:
            q, k = self.rope(q, k, seq_len=L)

        # Apply QK-Norm if enabled (after RoPE, before dot product)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Build combined attention mask from padding mask and custom bias
        combined_mask = None

        if mask is not None:
            # Padding mask: (batch, seq) -> (batch, 1, 1, seq)
            combined_mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, L, -1)
            combined_mask = combined_mask.float().masked_fill(combined_mask, float("-inf"))

        if attn_bias is not None:
            # attn_bias: (batch, heads, seq, seq) or (batch, seq, seq)
            if attn_bias.dim() == 3:
                attn_bias = attn_bias.unsqueeze(1)  # Add heads dim
            if combined_mask is not None:
                combined_mask = combined_mask + attn_bias
            else:
                combined_mask = attn_bias

        # Add causal mask if needed
        if self.is_causal and combined_mask is not None:
            # Create causal mask and combine with existing mask
            causal = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
            causal_mask = causal.float().masked_fill(causal, float("-inf"))
            combined_mask = combined_mask + causal_mask

        if hasattr(F, "scaled_dot_product_attention"):
            # Use efficient is_causal=True when no other masks are present
            if self.is_causal and combined_mask is None:
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=True,
                )
            else:
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=combined_mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
        else:
            scale = self.head_dim ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale

            # Apply causal mask for manual implementation
            if self.is_causal and combined_mask is None:
                causal = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
                attn = attn.masked_fill(causal, float("-inf"))
            elif combined_mask is not None:
                attn = attn + combined_mask

            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer block with RMSNorm, optional RoPE, and SwiGLU.

    Architecture:
        x = x + layer_scale * Attention(RMSNorm(x))
        x = x + layer_scale * SwiGLU(RMSNorm(x))

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        d_ff: Feedforward hidden dimension (default: auto-computed for SwiGLU).
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length for RoPE.
        use_rope: Whether to use Rotary Position Embeddings.
        qk_norm: Whether to apply QK-Norm for training stability.
        layer_scale_init: Initial value for layer scale parameters. If None,
            layer scale is disabled. Typical values: 1e-4 to 1e-6. From CaiT/DeiT-III.
        is_causal: Whether to apply causal (autoregressive) masking.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        use_rope: bool = True,
        qk_norm: bool = False,
        layer_scale_init: Optional[float] = None,
        is_causal: bool = False,
    ):
        super().__init__()

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout, max_seq_len, use_rope, qk_norm, is_causal)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

        if layer_scale_init is not None:
            self.gamma_1 = nn.Parameter(torch.full((d_model,), layer_scale_init))
            self.gamma_2 = nn.Parameter(torch.full((d_model,), layer_scale_init))
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def forward(
        self,
        x: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
        attn_bias: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        attn_out = self.dropout(self.attn(self.norm1(x), mask=mask, attn_bias=attn_bias))
        ffn_out = self.dropout(self.ffn(self.norm2(x)))

        if self.gamma_1 is not None:
            attn_out = self.gamma_1 * attn_out
            ffn_out = self.gamma_2 * ffn_out

        x = x + attn_out
        x = x + ffn_out
        return x


class AdaLNTransformerBlock(nn.Module):
    """Transformer block with Adaptive Layer Normalization (AdaLN).

    Uses AdaLN for conditioning on timestep (or other signals) instead of
    fixed normalization. This is the key innovation from DiT for diffusion.

    Architecture:
        x = x + layer_scale * Attention(AdaLN(x, cond))
        x = x + layer_scale * SwiGLU(AdaLN(x, cond))

    Args:
        d_model: Model dimension.
        cond_dim: Conditioning vector dimension.
        num_heads: Number of attention heads.
        d_ff: Feedforward hidden dimension (default: auto-computed for SwiGLU).
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length for RoPE.
        qk_norm: Whether to apply QK-Norm for training stability.
        layer_scale_init: Initial value for layer scale parameters. If None,
            layer scale is disabled. Typical values: 1e-4 to 1e-6.
    """

    def __init__(
        self,
        d_model: int,
        cond_dim: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        qk_norm: bool = False,
        layer_scale_init: Optional[float] = None,
    ):
        super().__init__()

        self.adaln1 = AdaLN(d_model, cond_dim)
        self.adaln2 = AdaLN(d_model, cond_dim)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout, max_seq_len, use_rope=True, qk_norm=qk_norm)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

        if layer_scale_init is not None:
            self.gamma_1 = nn.Parameter(torch.full((d_model,), layer_scale_init))
            self.gamma_2 = nn.Parameter(torch.full((d_model,), layer_scale_init))
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def forward(
        self,
        x: "torch.Tensor",
        cond: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """Forward pass with conditioning.

        Args:
            x: Input tensor (batch, seq, d_model).
            cond: Conditioning tensor (batch, cond_dim).
            mask: Optional padding mask (batch, seq).

        Returns:
            Output tensor (batch, seq, d_model).
        """
        attn_out = self.dropout(self.attn(self.adaln1(x, cond), mask=mask))
        ffn_out = self.dropout(self.ffn(self.adaln2(x, cond)))

        if self.gamma_1 is not None:
            attn_out = self.gamma_1 * attn_out
            ffn_out = self.gamma_2 * ffn_out

        x = x + attn_out
        x = x + ffn_out
        return x


class AdaLNTransformer(nn.Module):
    """Transformer with Adaptive Layer Normalization conditioning.

    Like the standard Transformer but each layer receives a conditioning
    vector that modulates the normalization. Used in DiT and modern
    diffusion transformers.

    Args:
        d_model: Model dimension.
        cond_dim: Conditioning vector dimension.
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads.
        d_ff: Feedforward hidden dimension (default: auto-computed for SwiGLU).
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length for RoPE.
        qk_norm: Whether to apply QK-Norm for training stability.
        layer_scale_init: Initial value for layer scale parameters. If None,
            layer scale is disabled. Typical values: 1e-4 to 1e-6.

    Example:
        >>> model = AdaLNTransformer(d_model=256, cond_dim=256, num_layers=4, num_heads=8)
        >>> x = torch.randn(2, 100, 256)
        >>> cond = torch.randn(2, 256)  # e.g., timestep embedding
        >>> out = model(x, cond)  # (2, 100, 256)
    """

    def __init__(
        self,
        d_model: int,
        cond_dim: int,
        num_layers: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        qk_norm: bool = False,
        layer_scale_init: Optional[float] = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.cond_dim = cond_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.layers = nn.ModuleList([
            AdaLNTransformerBlock(
                d_model, cond_dim, num_heads, d_ff, dropout, max_seq_len, qk_norm, layer_scale_init
            )
            for _ in range(num_layers)
        ])
        # Final norm also uses AdaLN for consistency
        self.final_adaln = AdaLN(d_model, cond_dim)

    def forward(
        self,
        x: "torch.Tensor",
        cond: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """Process input through transformer layers with conditioning.

        Args:
            x: Input tensor (batch, seq, d_model).
            cond: Conditioning tensor (batch, cond_dim).
            mask: Padding mask (batch, seq) where True = masked/ignored.

        Returns:
            Output tensor (batch, seq, d_model).
        """
        for layer in self.layers:
            x = layer(x, cond, mask=mask)
        return self.final_adaln(x, cond)


class Transformer(nn.Module):
    """
    Modern Transformer encoder.

    Uses Pre-LN architecture with RMSNorm, optional RoPE, and SwiGLU - following
    best practices from LLaMA, GPT-NeoX, and PaLM.

    Args:
        d_model: Model dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        d_ff: Feedforward hidden dimension (default: auto-computed for SwiGLU)
        dropout: Dropout probability
        max_seq_len: Maximum sequence length for RoPE
        use_rope: Whether to use Rotary Position Embeddings. Set to False
            when using custom attention biases (e.g., distance-based).
        qk_norm: Whether to apply QK-Norm for training stability. Recommended
            for large models or when experiencing NaN issues.
        layer_scale_init: Initial value for layer scale parameters. If None,
            layer scale is disabled. Typical values: 1e-4 to 1e-6. From CaiT/DeiT-III.
        is_causal: Whether to apply causal (autoregressive) masking. Each position
            can only attend to itself and previous positions.

    Example:
        >>> model = Transformer(d_model=256, num_layers=4, num_heads=8)
        >>> x = torch.randn(2, 100, 256)
        >>> out = model(x)  # (2, 100, 256)
        >>>
        >>> # Without RoPE, with custom attention bias
        >>> model = Transformer(d_model=256, num_layers=4, num_heads=8, use_rope=False)
        >>> attn_bias = torch.randn(2, 8, 100, 100)  # distance-based bias
        >>> out = model(x, attn_bias=attn_bias)
        >>>
        >>> # Causal (autoregressive) transformer
        >>> model = Transformer(d_model=256, num_layers=4, num_heads=8, is_causal=True)
        >>> out = model(x)  # Each position only sees previous positions
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        use_rope: bool = True,
        qk_norm: bool = False,
        layer_scale_init: Optional[float] = None,
        is_causal: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, dropout, max_seq_len, use_rope, qk_norm, layer_scale_init, is_causal
            )
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)

    def forward(
        self,
        x: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
        attn_bias: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """
        Process input through transformer layers.

        Args:
            x: Input tensor (batch, seq, d_model)
            mask: Padding mask (batch, seq) where True = masked/ignored
            attn_bias: Optional attention bias of shape (batch, heads, seq, seq)
                or (batch, seq, seq). Added to attention scores before softmax.

        Returns:
            Output tensor (batch, seq, d_model)

        Raises:
            ValueError: If input shapes are invalid.
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

        for layer in self.layers:
            x = layer(x, mask=mask, attn_bias=attn_bias)

        return self.final_norm(x)
