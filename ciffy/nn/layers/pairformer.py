"""
Pairformer: AlphaFold3-style transformer for pair representations.

This module implements the Pairformer architecture from AlphaFold3,
which processes pair representations using triangular operations:

- **Triangular Multiplicative Update**: Aggregates information along shared indices
- **Triangular Attention**: Row/column-wise attention on pair matrices
- **Pair Transition**: Standard feedforward on pair representation

Example:
    >>> from ciffy.nn.layers import Pairformer
    >>>
    >>> model = Pairformer(d_pair=128, num_layers=4, num_heads=8)
    >>> pair = torch.randn(2, 100, 100, 128)  # (batch, seq, seq, d_pair)
    >>> pair_out = model(pair)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None
    F = None

from .transformer import RMSNorm, SwiGLU

logger = logging.getLogger(__name__)


def _check_pair_tensor(
    tensor: "torch.Tensor",
    name: str,
    d_pair: int,
) -> None:
    """
    Validate pair tensor for common issues.

    Args:
        tensor: Tensor to validate.
        name: Name for error messages.
        d_pair: Expected pair dimension.

    Raises:
        ValueError: If tensor shape is invalid.
        RuntimeError: If tensor contains NaN values.
    """
    if tensor.dim() != 4:
        raise ValueError(
            f"{name}: expected 4D tensor (batch, seq, seq, d_pair), "
            f"got {tensor.dim()}D with shape {tuple(tensor.shape)}"
        )

    B, L1, L2, D = tensor.shape
    if L1 != L2:
        raise ValueError(
            f"{name}: pair tensor must be square in sequence dimensions, "
            f"got shape {tuple(tensor.shape)}"
        )

    if D != d_pair:
        raise ValueError(
            f"{name}: d_pair mismatch. Expected {d_pair}, got {D}. "
            f"Input shape: {tuple(tensor.shape)}"
        )

    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        raise RuntimeError(
            f"{name}: tensor contains {nan_count} NaN values. "
            f"Shape: {tuple(tensor.shape)}, dtype: {tensor.dtype}"
        )


class TriangularMultiplicativeUpdate(nn.Module if TORCH_AVAILABLE else object):
    """
    Triangular multiplicative update for pair representations.

    Updates pair[i,j] by aggregating information along shared indices:
    - Outgoing: sum over k of (pair[i,k] * pair[j,k])
    - Incoming: sum over k of (pair[k,i] * pair[k,j])

    Args:
        d_pair: Dimension of pair representation.
        d_hidden: Hidden dimension for gating projections (default: d_pair).
        direction: "outgoing" or "incoming".
        dropout: Dropout probability.

    Example:
        >>> tmu = TriangularMultiplicativeUpdate(d_pair=128, direction="outgoing")
        >>> pair = torch.randn(2, 100, 100, 128)
        >>> pair_updated = tmu(pair)
    """

    def __init__(
        self,
        d_pair: int,
        d_hidden: Optional[int] = None,
        direction: str = "outgoing",
        dropout: float = 0.0,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        super().__init__()

        if direction not in ("outgoing", "incoming"):
            raise ValueError(f"direction must be 'outgoing' or 'incoming', got {direction}")

        self.d_pair = d_pair
        self.d_hidden = d_hidden if d_hidden is not None else d_pair
        self.direction = direction

        # Layer norm
        self.layer_norm_in = RMSNorm(d_pair)

        # Projections for left and right gates
        self.proj_a = nn.Linear(d_pair, self.d_hidden, bias=False)
        self.proj_b = nn.Linear(d_pair, self.d_hidden, bias=False)
        self.gate_a = nn.Linear(d_pair, self.d_hidden, bias=False)
        self.gate_b = nn.Linear(d_pair, self.d_hidden, bias=False)

        # Output projection
        self.layer_norm_out = RMSNorm(self.d_hidden)
        self.proj_out = nn.Linear(self.d_hidden, d_pair, bias=False)
        self.gate_out = nn.Linear(d_pair, d_pair, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        pair: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """
        Apply triangular multiplicative update.

        Args:
            pair: Pair representation (batch, seq, seq, d_pair).
            mask: Optional mask (batch, seq) where True = masked position.

        Returns:
            Updated pair representation (batch, seq, seq, d_pair).
        """
        _check_pair_tensor(pair, "TriangularMultiplicativeUpdate", self.d_pair)

        # Pre-norm
        pair_normed = self.layer_norm_in(pair)

        # Project and gate
        a = self.proj_a(pair_normed) * torch.sigmoid(self.gate_a(pair_normed))
        b = self.proj_b(pair_normed) * torch.sigmoid(self.gate_b(pair_normed))

        # Apply mask if provided
        if mask is not None:
            # mask: (B, L) -> expand to (B, L, L, d_hidden)
            mask_2d = mask.unsqueeze(2) | mask.unsqueeze(1)  # (B, L, L)
            mask_expanded = mask_2d.unsqueeze(-1)  # (B, L, L, 1)
            a = a.masked_fill(mask_expanded, 0.0)
            b = b.masked_fill(mask_expanded, 0.0)

        # Triangular multiplication
        if self.direction == "outgoing":
            # pair[i,j] = sum_k a[i,k] * b[j,k]
            out = torch.einsum("bikd,bjkd->bijd", a, b)
        else:  # incoming
            # pair[i,j] = sum_k a[k,i] * b[k,j]
            out = torch.einsum("bkid,bkjd->bijd", a, b)

        # Output projection with gating
        out = self.layer_norm_out(out)
        out = self.proj_out(out) * torch.sigmoid(self.gate_out(pair_normed))

        return self.dropout(out)


class TriangularAttention(nn.Module if TORCH_AVAILABLE else object):
    """
    Triangular self-attention for pair representations.

    Attention operates along one axis of the pair matrix:
    - Starting (row-wise): position (i,j) attends to positions (i,k) for all k
    - Ending (col-wise): position (i,j) attends to positions (k,j) for all k

    Args:
        d_pair: Dimension of pair representation.
        num_heads: Number of attention heads.
        direction: "starting" (row-wise) or "ending" (col-wise).
        dropout: Dropout probability.

    Example:
        >>> tri_attn = TriangularAttention(d_pair=128, num_heads=8, direction="starting")
        >>> pair = torch.randn(2, 100, 100, 128)
        >>> pair_updated = tri_attn(pair)
    """

    def __init__(
        self,
        d_pair: int,
        num_heads: int,
        direction: str = "starting",
        dropout: float = 0.0,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        super().__init__()

        if direction not in ("starting", "ending"):
            raise ValueError(f"direction must be 'starting' or 'ending', got {direction}")

        if d_pair % num_heads != 0:
            raise ValueError(f"d_pair ({d_pair}) must be divisible by num_heads ({num_heads})")

        self.d_pair = d_pair
        self.num_heads = num_heads
        self.head_dim = d_pair // num_heads
        self.direction = direction

        # Pre-norm
        self.layer_norm = RMSNorm(d_pair)

        # QKV projections
        self.proj_q = nn.Linear(d_pair, d_pair, bias=False)
        self.proj_k = nn.Linear(d_pair, d_pair, bias=False)
        self.proj_v = nn.Linear(d_pair, d_pair, bias=False)

        # Pair bias (use pair representation to bias attention)
        self.bias_proj = nn.Linear(d_pair, num_heads, bias=False)

        # Output projection with gating
        self.proj_out = nn.Linear(d_pair, d_pair, bias=False)
        self.gate_out = nn.Linear(d_pair, d_pair, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        pair: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """
        Apply triangular attention.

        Args:
            pair: Pair representation (batch, seq, seq, d_pair).
            mask: Optional mask (batch, seq) where True = masked position.

        Returns:
            Updated pair representation (batch, seq, seq, d_pair).
        """
        _check_pair_tensor(pair, "TriangularAttention", self.d_pair)

        B, L, _, D = pair.shape

        # Pre-norm
        pair_normed = self.layer_norm(pair)

        # QKV projections
        q = self.proj_q(pair_normed)
        k = self.proj_k(pair_normed)
        v = self.proj_v(pair_normed)

        # Pair bias: (B, L, L, num_heads)
        bias = self.bias_proj(pair_normed)

        if self.direction == "starting":
            # Row-wise attention: (i,j) attends to (i,k) for all k
            # Reshape: treat each row as a batch
            # q, k, v: (B, L, L, D) -> (B*L, L, num_heads, head_dim)
            q = q.reshape(B * L, L, self.num_heads, self.head_dim)
            k = k.reshape(B * L, L, self.num_heads, self.head_dim)
            v = v.reshape(B * L, L, self.num_heads, self.head_dim)

            # bias: (B, L, L, H) -> (B*L, H, L) for broadcasting
            # For row i, bias[i,j,k] should bias attention from position j to position k
            # We want bias[i,:,:] for each row
            bias = bias.permute(0, 1, 3, 2).reshape(B * L, self.num_heads, L)
        else:  # ending
            # Column-wise attention: (i,j) attends to (k,j) for all k
            # Transpose to make columns the "rows" for attention
            q = q.transpose(1, 2).reshape(B * L, L, self.num_heads, self.head_dim)
            k = k.transpose(1, 2).reshape(B * L, L, self.num_heads, self.head_dim)
            v = v.transpose(1, 2).reshape(B * L, L, self.num_heads, self.head_dim)

            # bias: (B, L, L, H) -> transpose -> (B, L, L, H) -> (B*L, H, L)
            bias = bias.permute(0, 2, 3, 1).reshape(B * L, self.num_heads, L)

        # Transpose for attention: (B*L, num_heads, L, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention with bias
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Add pair bias: (B*L, H, L, L)
        attn = attn + bias.unsqueeze(-1)

        # Apply mask if provided
        if mask is not None:
            # Expand mask for attention: (B, L) -> (B*L, 1, 1, L)
            if self.direction == "starting":
                # For row-wise, mask positions in each row
                mask_expanded = mask.unsqueeze(1).expand(-1, L, -1).reshape(B * L, 1, 1, L)
            else:
                # For column-wise, mask positions in each column
                mask_expanded = mask.unsqueeze(2).expand(-1, -1, L).reshape(B * L, 1, 1, L)
            attn = attn.masked_fill(mask_expanded, float("-inf"))

        attn = F.softmax(attn, dim=-1)

        # Handle NaN from all-masked rows (replace with zeros)
        attn = torch.nan_to_num(attn, nan=0.0)

        attn = self.dropout(attn)

        # Apply attention
        out = torch.matmul(attn, v)  # (B*L, H, L, head_dim)

        # Reshape back
        out = out.transpose(1, 2).reshape(B * L, L, D)

        if self.direction == "starting":
            out = out.reshape(B, L, L, D)
        else:
            out = out.reshape(B, L, L, D).transpose(1, 2)

        # Output projection with gating
        out = self.proj_out(out) * torch.sigmoid(self.gate_out(pair_normed))

        return self.dropout(out)


class PairTransition(nn.Module if TORCH_AVAILABLE else object):
    """
    Feedforward transition layer for pair representations.

    Standard SwiGLU feedforward with pre-normalization.

    Args:
        d_pair: Dimension of pair representation.
        d_ff: Feedforward hidden dimension (default: 4 * d_pair).
        dropout: Dropout probability.

    Example:
        >>> transition = PairTransition(d_pair=128)
        >>> pair = torch.randn(2, 100, 100, 128)
        >>> pair_updated = transition(pair)
    """

    def __init__(
        self,
        d_pair: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        super().__init__()

        self.d_pair = d_pair
        self.layer_norm = RMSNorm(d_pair)
        self.ffn = SwiGLU(d_pair, d_ff, dropout)

    def forward(self, pair: "torch.Tensor") -> "torch.Tensor":
        """
        Apply pair transition.

        Args:
            pair: Pair representation (batch, seq, seq, d_pair).

        Returns:
            Updated pair representation (batch, seq, seq, d_pair).
        """
        _check_pair_tensor(pair, "PairTransition", self.d_pair)
        return self.ffn(self.layer_norm(pair))


class OuterProductMean(nn.Module if TORCH_AVAILABLE else object):
    """
    Compute outer product of single representations to update pair.

    Projects single representation to lower dimension, computes outer product,
    and projects back to pair dimension.

    Args:
        d_single: Dimension of single representation.
        d_pair: Dimension of pair representation.
        d_hidden: Hidden dimension for projection (default: 32).

    Example:
        >>> opm = OuterProductMean(d_single=256, d_pair=128)
        >>> single = torch.randn(2, 100, 256)
        >>> pair_update = opm(single)
    """

    def __init__(
        self,
        d_single: int,
        d_pair: int,
        d_hidden: int = 32,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        super().__init__()

        self.d_single = d_single
        self.d_pair = d_pair
        self.d_hidden = d_hidden

        self.layer_norm = RMSNorm(d_single)
        self.proj_a = nn.Linear(d_single, d_hidden, bias=False)
        self.proj_b = nn.Linear(d_single, d_hidden, bias=False)
        self.proj_out = nn.Linear(d_hidden * d_hidden, d_pair, bias=False)

    def forward(
        self,
        single: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """
        Compute outer product mean.

        Args:
            single: Single representation (batch, seq, d_single).
            mask: Optional mask (batch, seq) where True = masked position.

        Returns:
            Pair update (batch, seq, seq, d_pair).
        """
        if single.dim() != 3:
            raise ValueError(
                f"OuterProductMean: expected 3D tensor (batch, seq, d_single), "
                f"got {single.dim()}D with shape {tuple(single.shape)}"
            )

        B, L, D = single.shape
        if D != self.d_single:
            raise ValueError(
                f"OuterProductMean: d_single mismatch. Expected {self.d_single}, got {D}."
            )

        # Pre-norm
        single_normed = self.layer_norm(single)

        # Project
        a = self.proj_a(single_normed)  # (B, L, d_hidden)
        b = self.proj_b(single_normed)  # (B, L, d_hidden)

        # Apply mask
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # (B, L, 1)
            a = a.masked_fill(mask_expanded, 0.0)
            b = b.masked_fill(mask_expanded, 0.0)

        # Outer product: (B, L, d_hidden) x (B, L, d_hidden) -> (B, L, L, d_hidden, d_hidden)
        outer = torch.einsum("bid,bjc->bijdc", a, b)

        # Flatten and project
        outer = outer.reshape(B, L, L, self.d_hidden * self.d_hidden)
        pair_update = self.proj_out(outer)

        return pair_update


class PairToSingleAttention(nn.Module if TORCH_AVAILABLE else object):
    """
    Aggregate pair representation to update single representation.

    Uses attention-weighted mean of pair features along one axis.

    Args:
        d_pair: Dimension of pair representation.
        d_single: Dimension of single representation.
        num_heads: Number of attention heads.

    Example:
        >>> p2s = PairToSingleAttention(d_pair=128, d_single=256, num_heads=8)
        >>> pair = torch.randn(2, 100, 100, 128)
        >>> single = torch.randn(2, 100, 256)
        >>> single_update = p2s(pair, single)
    """

    def __init__(
        self,
        d_pair: int,
        d_single: int,
        num_heads: int = 8,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        super().__init__()

        if d_single % num_heads != 0:
            raise ValueError(f"d_single ({d_single}) must be divisible by num_heads ({num_heads})")

        self.d_pair = d_pair
        self.d_single = d_single
        self.num_heads = num_heads
        self.head_dim = d_single // num_heads

        self.layer_norm_pair = RMSNorm(d_pair)
        self.layer_norm_single = RMSNorm(d_single)

        # Query from single, K/V from pair
        self.proj_q = nn.Linear(d_single, d_single, bias=False)
        self.proj_k = nn.Linear(d_pair, d_single, bias=False)
        self.proj_v = nn.Linear(d_pair, d_single, bias=False)

        self.proj_out = nn.Linear(d_single, d_single, bias=False)

    def forward(
        self,
        pair: "torch.Tensor",
        single: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """
        Aggregate pair to single.

        Args:
            pair: Pair representation (batch, seq, seq, d_pair).
            single: Single representation (batch, seq, d_single).
            mask: Optional mask (batch, seq) where True = masked position.

        Returns:
            Single update (batch, seq, d_single).
        """
        _check_pair_tensor(pair, "PairToSingleAttention", self.d_pair)

        B, L, _, _ = pair.shape

        # Pre-norm
        pair_normed = self.layer_norm_pair(pair)
        single_normed = self.layer_norm_single(single)

        # Query from single: (B, L, d_single) -> (B, L, H, head_dim)
        q = self.proj_q(single_normed).view(B, L, self.num_heads, self.head_dim)

        # K/V from pair rows: (B, L, L, d_pair) -> (B, L, L, d_single) -> (B, L, L, H, head_dim)
        k = self.proj_k(pair_normed).view(B, L, L, self.num_heads, self.head_dim)
        v = self.proj_v(pair_normed).view(B, L, L, self.num_heads, self.head_dim)

        # Attention: q[i] attends to k[i,:] (row i of pair matrix)
        # q: (B, L, H, D) -> (B, H, L, D)
        # k: (B, L, L, H, D) -> (B, H, L, L, D) (for each position i, keys from row i)
        q = q.permute(0, 2, 1, 3)  # (B, H, L, D)
        k = k.permute(0, 3, 1, 2, 4)  # (B, H, L, L, D)
        v = v.permute(0, 3, 1, 2, 4)  # (B, H, L, L, D)

        # For position i: score[j] = q[i] @ k[i,j]
        # scores: (B, H, L, L)
        scale = self.head_dim ** -0.5
        scores = torch.einsum("bhid,bhijd->bhij", q, k) * scale

        # Apply mask
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            scores = scores.masked_fill(mask_expanded, float("-inf"))

        attn = F.softmax(scores, dim=-1)

        # Apply attention: out[i] = sum_j attn[i,j] * v[i,j]
        out = torch.einsum("bhij,bhijd->bhid", attn, v)  # (B, H, L, D)

        # Reshape
        out = out.permute(0, 2, 1, 3).reshape(B, L, self.d_single)

        return self.proj_out(out)


class PairformerBlock(nn.Module if TORCH_AVAILABLE else object):
    """
    Single Pairformer block with triangular operations.

    Architecture (Pre-LN):
        pair = pair + TriMulUpdate_outgoing(pair)
        pair = pair + TriMulUpdate_incoming(pair)
        pair = pair + TriAttn_starting(pair)
        pair = pair + TriAttn_ending(pair)
        pair = pair + Transition(pair)

    Args:
        d_pair: Dimension of pair representation.
        d_hidden_mul: Hidden dimension for multiplicative updates (default: d_pair).
        num_heads: Number of attention heads.
        d_ff: Feedforward hidden dimension.
        dropout: Dropout probability.

    Example:
        >>> block = PairformerBlock(d_pair=128, num_heads=8)
        >>> pair = torch.randn(2, 100, 100, 128)
        >>> pair = block(pair)
    """

    def __init__(
        self,
        d_pair: int,
        d_hidden_mul: Optional[int] = None,
        num_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        super().__init__()

        self.d_pair = d_pair

        # Triangular multiplicative updates
        self.tri_mul_out = TriangularMultiplicativeUpdate(
            d_pair, d_hidden_mul, direction="outgoing", dropout=dropout
        )
        self.tri_mul_in = TriangularMultiplicativeUpdate(
            d_pair, d_hidden_mul, direction="incoming", dropout=dropout
        )

        # Triangular attention
        self.tri_attn_start = TriangularAttention(
            d_pair, num_heads, direction="starting", dropout=dropout
        )
        self.tri_attn_end = TriangularAttention(
            d_pair, num_heads, direction="ending", dropout=dropout
        )

        # Pair transition
        self.transition = PairTransition(d_pair, d_ff, dropout)

    def forward(
        self,
        pair: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """
        Apply Pairformer block.

        Args:
            pair: Pair representation (batch, seq, seq, d_pair).
            mask: Optional mask (batch, seq) where True = masked position.

        Returns:
            Updated pair representation (batch, seq, seq, d_pair).
        """
        pair = pair + self.tri_mul_out(pair, mask)
        pair = pair + self.tri_mul_in(pair, mask)
        pair = pair + self.tri_attn_start(pair, mask)
        pair = pair + self.tri_attn_end(pair, mask)
        pair = pair + self.transition(pair)
        return pair


class Pairformer(nn.Module if TORCH_AVAILABLE else object):
    """
    Full Pairformer transformer for pair representations.

    Stacks multiple PairformerBlocks with optional single representation track.

    Args:
        d_pair: Dimension of pair representation.
        num_layers: Number of Pairformer blocks.
        d_hidden_mul: Hidden dimension for multiplicative updates (default: d_pair).
        num_heads: Number of attention heads.
        d_ff: Feedforward hidden dimension.
        dropout: Dropout probability.
        d_single: Dimension of single representation (None = no single track).

    Example:
        >>> # Pair-only
        >>> model = Pairformer(d_pair=128, num_layers=4, num_heads=8)
        >>> pair = torch.randn(2, 100, 100, 128)
        >>> pair_out = model(pair)

        >>> # With single track
        >>> model = Pairformer(d_pair=128, num_layers=4, num_heads=8, d_single=256)
        >>> single = torch.randn(2, 100, 256)
        >>> pair_out, single_out = model(pair, single=single)
    """

    def __init__(
        self,
        d_pair: int,
        num_layers: int,
        d_hidden_mul: Optional[int] = None,
        num_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        d_single: Optional[int] = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        super().__init__()

        self.d_pair = d_pair
        self.num_layers = num_layers
        self.d_single = d_single

        # Pairformer blocks
        self.blocks = nn.ModuleList([
            PairformerBlock(d_pair, d_hidden_mul, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Optional single track components
        if d_single is not None:
            self.outer_product_mean = nn.ModuleList([
                OuterProductMean(d_single, d_pair)
                for _ in range(num_layers)
            ])
            self.pair_to_single = nn.ModuleList([
                PairToSingleAttention(d_pair, d_single, num_heads)
                for _ in range(num_layers)
            ])
            # Single track transformer layers (simple attention + FFN)
            self.single_attn = nn.ModuleList([
                nn.MultiheadAttention(d_single, num_heads, dropout=dropout, batch_first=True)
                for _ in range(num_layers)
            ])
            self.single_ffn = nn.ModuleList([
                SwiGLU(d_single, dropout=dropout)
                for _ in range(num_layers)
            ])
            self.single_norm1 = nn.ModuleList([RMSNorm(d_single) for _ in range(num_layers)])
            self.single_norm2 = nn.ModuleList([RMSNorm(d_single) for _ in range(num_layers)])

        # Final norms
        self.final_norm_pair = RMSNorm(d_pair)
        if d_single is not None:
            self.final_norm_single = RMSNorm(d_single)

    def forward(
        self,
        pair: "torch.Tensor",
        single: Optional["torch.Tensor"] = None,
        mask: Optional["torch.Tensor"] = None,
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", "torch.Tensor"]]:
        """
        Process pair (and optionally single) representation through Pairformer.

        Args:
            pair: Pair representation (batch, seq, seq, d_pair).
            single: Optional single representation (batch, seq, d_single).
            mask: Optional padding mask (batch, seq) where True = masked position.

        Returns:
            If single is None: Updated pair (batch, seq, seq, d_pair).
            If single provided: Tuple of (pair, single).
        """
        _check_pair_tensor(pair, "Pairformer", self.d_pair)

        if self.d_single is not None and single is None:
            raise ValueError(
                "Pairformer initialized with d_single but single tensor not provided"
            )

        if self.d_single is None and single is not None:
            raise ValueError(
                "Pairformer initialized without d_single but single tensor provided"
            )

        for i, block in enumerate(self.blocks):
            # Single -> Pair update
            if single is not None:
                pair = pair + self.outer_product_mean[i](single, mask)

            # Pair block
            pair = block(pair, mask)

            # Pair -> Single update
            if single is not None:
                single = single + self.pair_to_single[i](pair, single, mask)

                # Single self-attention
                single_normed = self.single_norm1[i](single)
                attn_mask = None
                if mask is not None:
                    # Convert to attention mask format
                    attn_mask = mask.unsqueeze(1).expand(-1, single.shape[1], -1)
                attn_out, _ = self.single_attn[i](
                    single_normed, single_normed, single_normed,
                    key_padding_mask=mask,
                    attn_mask=attn_mask,
                )
                single = single + attn_out

                # Single FFN
                single = single + self.single_ffn[i](self.single_norm2[i](single))

        # Final normalization
        pair = self.final_norm_pair(pair)

        if single is not None:
            single = self.final_norm_single(single)
            return pair, single

        return pair
