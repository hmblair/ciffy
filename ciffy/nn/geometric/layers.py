"""Equivariant neural network layers for geometric deep learning.

This module provides SE(3)-equivariant layers for building geometric
deep learning models. These layers maintain equivariance to rotations
and translations when processing 3D molecular or point cloud data.

Uses k-NN based sparse attention with O(N*k) complexity.

Classes:
    RadialWeight: Neural network for computing tensor product weights
    EquivariantLinear: Linear layer preserving spherical tensor structure
    EquivariantGating: Norm-based gating for spherical tensors
    EquivariantTransition: MLP transition layer for spherical tensors
    EquivariantConvolution: SE(3)-equivariant convolution
    EquivariantLayerNorm: Equivariant layer normalization
    Attention: k-NN based multi-head attention
    EquivariantAttention: SE(3)-equivariant sparse attention
    EquivariantTransformerBlock: Transformer block with sparse attention
    EquivariantTransformer: Full equivariant transformer
"""
from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .representations import Repr, ProductRepr
from .equivariant import (
    RepNorm,
    EquivariantBasis,
    EquivariantBases,
    _EquivariantBasisLegacy,
    RadialBasisFunctions,
    SequencePositionEncoding,
)
from .output import MatrixOutput, LowRankMatrixOutput
from ..layers.transformer import RMSNorm


def _compute_num_basis(repr: ProductRepr, rank: int | None) -> int:
    """Compute the number of basis elements for a ProductRepr and rank.

    Args:
        repr: ProductRepr specifying input and output representations.
        rank: Maximum filter degree relative to minimum coupling.
            0 = only l=|l1-l2| terms, None = all terms up to l1+l2.

    Returns:
        Number of basis elements.
    """
    return repr.num_basis(rank)


def _init_weights(module: nn.Module) -> None:
    """Initialize weights using Xavier uniform for linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def build_knn_graph(
    coordinates: torch.Tensor,
    k: int,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Build k-nearest neighbor graph from 3D coordinates.

    Uses chunked computation for large N to reduce peak memory from O(N²) to
    O(chunk_size * N). For small N, uses direct computation.

    Args:
        coordinates: Node coordinates of shape (N, 3).
        k: Number of nearest neighbors per node.
        chunk_size: Maximum chunk size for distance computation. Larger values
            are faster but use more memory. Default 2048 uses ~32MB per chunk.

    Returns:
        Neighbor indices of shape (N, k), where neighbor_idx[i] contains
        the k nearest neighbor indices for node i (excluding self).

    Example:
        >>> coords = torch.randn(100, 3)
        >>> neighbor_idx = build_knn_graph(coords, k=16)
        >>> neighbor_idx.shape
        torch.Size([100, 16])
    """
    N = coordinates.size(0)

    # Handle edge case where N <= k
    if N <= k + 1:
        # Use all other nodes as neighbors
        all_idx = torch.arange(N, device=coordinates.device)
        neighbor_idx = all_idx.unsqueeze(0).expand(N, -1)
        # Exclude self: create mask and gather non-self indices
        mask = neighbor_idx != torch.arange(N, device=coordinates.device).unsqueeze(1)
        # Pad if necessary
        result = torch.zeros(N, k, dtype=torch.long, device=coordinates.device)
        for i in range(N):
            others = neighbor_idx[i, mask[i]]
            result[i, :len(others)] = others
            if len(others) < k:
                # Repeat last neighbor to fill
                result[i, len(others):] = others[-1] if len(others) > 0 else 0
        return result

    # For small N, use direct computation (faster, memory is acceptable)
    if N <= chunk_size:
        dists = torch.cdist(coordinates, coordinates)  # (N, N)
        _, indices = dists.topk(k + 1, largest=False, dim=-1)  # (N, k+1)
        return indices[:, 1:]  # Exclude self

    # For large N, use chunked computation to reduce peak memory
    # Peak memory: O(chunk_size * N) instead of O(N²)
    neighbor_idx = torch.empty(N, k, dtype=torch.long, device=coordinates.device)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        # Compute distances from this chunk to all nodes
        chunk_dists = torch.cdist(coordinates[start:end], coordinates)  # (chunk, N)
        # Get k+1 nearest neighbors for this chunk
        _, chunk_indices = chunk_dists.topk(k + 1, largest=False, dim=-1)
        # Exclude self (distance 0 is always to self at same index)
        neighbor_idx[start:end] = chunk_indices[:, 1:]

    # Debug assertion: verify all indices are valid
    assert neighbor_idx.max() < N, f"k-NN returned invalid index: max={neighbor_idx.max()}, N={N}"
    assert neighbor_idx.min() >= 0, f"k-NN returned negative index: min={neighbor_idx.min()}"

    return neighbor_idx


class RadialWeight(nn.Module):
    """Compute tensor product weights from invariant edge features.

    A neural network that maps edge features to weights for tensor product
    contractions. Used in equivariant message passing networks.

    Supports two output formats:
    - Legacy format (for factored basis): output shape (num_basis, out_mult * in_mult)
    - New format (for explicit basis): output shape (num_basis, out_mult, in_mult)

    Also supports low-rank decomposition for parameter efficiency.

    Args:
        edge_dim: Dimension of input edge features.
        hidden_dim: Hidden layer dimension.
        num_basis: Number of equivariant basis elements.
        in_mult: Input multiplicity.
        out_mult: Output multiplicity.
        dropout: Dropout probability.
        rank: Rank for low-rank decomposition. If None, uses full-rank.

    Example:
        >>> weight_net = RadialWeight(16, 32, num_basis=4, in_mult=8, out_mult=8)
        >>> edge_features = torch.randn(100, 16)
        >>> weights = weight_net(edge_features)  # (100, 4, 8, 8)
    """

    def __init__(
        self: RadialWeight,
        edge_dim: int,
        hidden_dim: int,
        num_basis: int,
        in_mult: int,
        out_mult: int,
        dropout: float = 0,
        rank: int | None = None,
    ) -> None:
        super().__init__()

        self.num_basis = num_basis
        self.in_mult = in_mult
        self.out_mult = out_mult
        self.rank = rank

        # Output dimensions
        self.out_rows = num_basis * out_mult
        self.out_cols = in_mult

        # Hidden layer with fused ops
        self.layer1 = nn.Linear(edge_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Matrix output layer (full-rank or low-rank)
        if rank is None:
            self.matrix_output = MatrixOutput(hidden_dim, self.out_rows, self.out_cols)
        else:
            self.matrix_output = LowRankMatrixOutput(
                hidden_dim, self.out_rows, self.out_cols, rank
            )

        self.apply(_init_weights)

    def forward(self: RadialWeight, x: torch.Tensor) -> torch.Tensor:
        """Compute weights from edge features.

        Uses fused linear+GELU operations for better performance with torch.compile.

        Args:
            x: Edge features of shape (..., edge_dim).

        Returns:
            Weights of shape (..., num_basis, out_mult, in_mult).
        """
        *batch_dims, _ = x.shape

        # Fused linear + GELU (approximate='tanh' is faster and torch.compile friendly)
        h = F.linear(x, self.layer1.weight, self.layer1.bias)
        h = F.gelu(h, approximate='tanh')
        h = self.dropout(h)

        # Get output and reshape to (*, num_basis, out_mult, in_mult)
        out = self.matrix_output(h)  # (*, num_basis * out_mult, in_mult)
        return out.view(*batch_dims, self.num_basis, self.out_mult, self.in_mult)


class EquivariantLinear(nn.Module):
    """Linear layer that preserves spherical tensor structure.

    Applies separate linear transformations to each irrep degree,
    preserving SO(3) equivariance. Only degree-0 (scalar) components
    can have bias terms.

    Args:
        repr: Input representation.
        out_repr: Output representation (must have same lvals as repr).
        dropout: Dropout probability.
        activation: Activation function (applied only to scalars).
        bias: Whether to include bias for scalar components.

    Raises:
        ValueError: If repr and out_repr have different lvals.

    Example:
        >>> repr_in = Repr(lvals=[0, 1, 2], mult=8)
        >>> repr_out = Repr(lvals=[0, 1, 2], mult=16)
        >>> layer = EquivariantLinear(repr_in, repr_out)
        >>> x = torch.randn(32, 8, 9)  # batch, mult, dim
        >>> y = layer(x)  # shape: (32, 16, 9)
    """

    def __init__(
        self: EquivariantLinear,
        repr: Repr,
        out_repr: Repr,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if repr.lvals != out_repr.lvals:
            raise ValueError(
                "EquivariantLinear cannot modify the degrees of a representation. "
                f"Got input lvals={repr.lvals}, output lvals={out_repr.lvals}"
            )

        self.repr = repr

        # Weight matrix for linear transformation
        self.weight = nn.Parameter(
            torch.empty(repr.nreps() * out_repr.mult, repr.mult)
        )
        nn.init.xavier_uniform_(self.weight)

        # Indices for gathering correct degrees
        indices = torch.tensor(repr.indices(), dtype=torch.long)
        self.register_buffer('indices', indices)

        self.expanddims = (1, out_repr.mult, repr.dim())
        self.outdims = (repr.nreps(), out_repr.mult, repr.dim())

        # Bias only for scalar (degree-0) components
        nscalar, scalar_locs = repr.find_scalar()
        self.scalar_locs = scalar_locs

        if nscalar > 0 and bias:
            self.bias = nn.Parameter(
                torch.zeros(out_repr.mult, nscalar),
                requires_grad=True,
            )
        else:
            self.bias = None

        self.dropout = nn.Dropout(dropout)
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self: EquivariantLinear, f: torch.Tensor) -> torch.Tensor:
        """Apply equivariant linear transformation.

        Args:
            f: Spherical tensor of shape (..., mult, dim).

        Returns:
            Transformed tensor of shape (..., out_mult, dim).
        """
        GATHER_DIM = -3

        *b, _, _ = f.shape

        # Apply linear transformation
        out = (self.weight @ f).view(*b, *self.outdims)
        out = self.dropout(out)

        # Gather components for each degree
        ix = self.indices.expand(*b, *self.expanddims)
        out = out.gather(dim=GATHER_DIM, index=ix).squeeze(GATHER_DIM)

        # Add bias and activation to scalar components
        if self.bias is not None:
            out = out.clone()
            out[..., self.scalar_locs] = self.activation(
                out[..., self.scalar_locs] + self.bias
            )

        return out


class EquivariantGating(nn.Module):
    """Norm-based gating for spherical tensors.

    Computes norms of each irrep component, processes them through
    a linear layer and sigmoid, then uses the result to gate the
    original tensor. This provides a learnable, equivariant nonlinearity.

    Args:
        repr: The representation of input tensors.
        dropout: Dropout probability.

    Example:
        >>> repr = Repr(lvals=[0, 1, 2], mult=8)
        >>> gate = EquivariantGating(repr)
        >>> x = torch.randn(32, 8, 9)
        >>> y = gate(x)  # shape: (32, 8, 9)
    """

    def __init__(
        self: EquivariantGating,
        repr: Repr,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.repr = repr
        self.norm = RepNorm(repr)

        # Linear layer for processing norms
        self.linear = nn.Linear(
            repr.nreps() * repr.mult,
            repr.nreps() * repr.mult,
        )

        # Indices mapping norms back to full dimension
        self.register_buffer('ix', torch.tensor(repr.indices(), dtype=torch.long))

        self.outdims = (repr.mult, repr.nreps())
        self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

        self.apply(_init_weights)

    def forward(self: EquivariantGating, st: torch.Tensor) -> torch.Tensor:
        """Apply gating to spherical tensor.

        Args:
            st: Spherical tensor of shape (..., mult, dim).

        Returns:
            Gated tensor of shape (..., mult, dim).
        """
        # Compute norms
        norms = self.norm(st)

        *b, _, _ = norms.size()
        norms = norms.flatten(-2, -1)

        # Process through linear layer
        norms = self.linear(norms).view(*b, *self.outdims)

        # Apply activation and dropout
        norms = self.activation(norms)
        norms = self.dropout(norms)

        # Gate the input
        return st * norms[..., self.ix]


class EquivariantTransition(nn.Module):
    """Transition layer for equivariant transformers.

    A feed-forward network that expands to a hidden representation
    and then projects back, with gating nonlinearity.

    Args:
        repr: Input/output representation.
        hidden_repr: Hidden representation (typically 4x multiplicity).

    Example:
        >>> repr = Repr(lvals=[0, 1], mult=8)
        >>> hidden = Repr(lvals=[0, 1], mult=32)
        >>> transition = EquivariantTransition(repr, hidden)
    """

    def __init__(
        self: EquivariantTransition,
        repr: Repr,
        hidden_repr: Repr,
    ) -> None:
        super().__init__()

        self.proj1 = EquivariantLinear(repr, hidden_repr, activation=None)
        self.gating = EquivariantGating(hidden_repr)
        self.proj2 = EquivariantLinear(hidden_repr, repr, activation=None)

    def forward(self: EquivariantTransition, x: torch.Tensor) -> torch.Tensor:
        """Apply transition layer.

        Args:
            x: Input tensor of shape (..., mult, dim).

        Returns:
            Output tensor of shape (..., mult, dim).
        """
        x = self.proj1(x)
        x = self.gating(x)
        return self.proj2(x)


class EquivariantConvolution(nn.Module):
    """SE(3)-equivariant convolution using Clebsch-Gordan basis.

    Performs equivariant message passing using explicit basis matrices
    computed from Clebsch-Gordan coefficients. Bases are precomputed
    externally (via EquivariantBasis or EquivariantBases) and passed to
    forward() for efficiency when the same bases are reused.

    The `rank` parameter controls the expressivity vs efficiency
    trade-off in the basis:

    - rank=0: Only l=|l1-l2| terms (rank-1, fast, equivalent to legacy)
    - rank=k: Terms up to l=|l1-l2|+k
    - rank=None: All terms up to l=l1+l2 (full SO(3) expressivity)

    Args:
        repr: ProductRepr specifying input/output representations.
        edge_dim: Dimension of invariant edge features.
        hidden_dim: Hidden dimension for radial weight network.
        num_basis: Number of basis elements (from EquivariantBasis.num_basis).
        dropout: Dropout probability.
        radial_weight_rank: Rank for low-rank radial weight decomposition.
            If None, uses full-rank weights. Lower rank reduces parameters.

    Example:
        >>> repr = ProductRepr(Repr([0, 1]), Repr([0, 1]))
        >>> basis_module = EquivariantBasis(repr, rank=0)
        >>> conv = EquivariantConvolution(repr, edge_dim=16, hidden_dim=32,
        ...                               num_basis=basis_module.num_basis)
        >>> basis = basis_module(displacements)  # Precompute
        >>> output = conv(basis, edge_feats, f, src_idx)
    """

    def __init__(
        self: EquivariantConvolution,
        repr: ProductRepr,
        edge_dim: int,
        hidden_dim: int,
        num_basis: int,
        dropout: float = 0.0,
        radial_weight_rank: int | None = None,
    ) -> None:
        super().__init__()

        self.repr = repr

        # Dimensions
        self.in_mult = repr.rep1.mult
        self.out_mult = repr.rep2.mult
        self.in_dim = repr.rep1.dim()
        self.out_dim = repr.rep2.dim()
        self.num_basis = num_basis

        # Create radial weight network
        self.rwlin = RadialWeight(
            edge_dim,
            hidden_dim,
            num_basis=num_basis,
            in_mult=self.in_mult,
            out_mult=self.out_mult,
            dropout=dropout,
            rank=radial_weight_rank,
        )

    def forward(
        self: EquivariantConvolution,
        basis: torch.Tensor,
        edge_feats: torch.Tensor,
        f: torch.Tensor,
        src_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Apply equivariant convolution.

        Args:
            basis: Precomputed basis matrices of shape (E, num_basis, in_dim, out_dim),
                where in_dim and out_dim are the input and output representation
                dimensions (rep1.dim() and rep2.dim() respectively).
            edge_feats: Invariant edge features of shape (E, edge_dim).
            f: Node features of shape (N, in_mult, in_dim).
            src_idx: Source node indices for each edge, shape (E,).

        Returns:
            Convolved features of shape (E, out_mult, out_dim).
        """
        # Debug assertion: verify source indices are valid
        N = f.size(0)
        assert src_idx.max() < N, f"EquivariantConvolution: src_idx max ({src_idx.max()}) >= N ({N})"
        assert src_idx.min() >= 0, f"EquivariantConvolution: src_idx min ({src_idx.min()}) < 0"

        # Compute radial weights: (E, num_basis, out_mult, in_mult)
        weights = self.rwlin(edge_feats)

        # Gather source features: (E, in_mult, in_dim)
        f_src = f[src_idx]

        # Step 1: Contract basis with input features
        # f_src: (E, in_mult, in_dim), basis: (E, num_basis, in_dim, out_dim)
        # -> contracted: (E, num_basis, in_mult, out_dim)
        contracted = f_src.unsqueeze(1) @ basis

        # Step 2: Apply weights and sum over basis
        # weights: (E, num_basis, out_mult, in_mult) @ contracted: (E, num_basis, in_mult, out_dim)
        # -> (E, num_basis, out_mult, out_dim) -> sum over basis -> (E, out_mult, out_dim)
        output = (weights @ contracted).sum(dim=1)

        return output


class EquivariantLayerNorm(nn.Module):
    """Equivariant layer normalization.

    Normalizes spherical tensors by their irrep norms while
    preserving equivariance. Applies standard LayerNorm to
    the norms across the multiplicity dimension.

    Args:
        repr: The representation of input tensors.
        epsilon: Small constant for numerical stability.

    Example:
        >>> repr = Repr(lvals=[0, 1, 2], mult=8)
        >>> ln = EquivariantLayerNorm(repr)
        >>> x = torch.randn(32, 8, 9)
        >>> y = ln(x)
    """

    EPSILON = 1e-8

    def __init__(
        self: EquivariantLayerNorm,
        repr: Repr,
        epsilon: float = EPSILON,
    ) -> None:
        super().__init__()

        self.mult = repr.mult
        self.norm = RepNorm(repr)
        # LayerNorm(1) is degenerate (always outputs zeros), so skip it for mult=1
        self.lnorm = nn.LayerNorm(repr.mult) if repr.mult > 1 else None
        self.epsilon = epsilon

        self.register_buffer('ix', torch.tensor(repr.indices(), dtype=torch.long))

    def forward(self: EquivariantLayerNorm, f: torch.Tensor) -> torch.Tensor:
        """Apply equivariant layer normalization.

        Args:
            f: Spherical tensor of shape (..., mult, dim).

        Returns:
            Normalized tensor of shape (..., mult, dim).
        """
        # Compute norms: shape (..., mult, nreps)
        norms = self.norm(f)

        # For mult=1, just normalize by the norm (no cross-channel normalization)
        if self.lnorm is None:
            norms_r = 1.0 / (norms + self.epsilon)
            return f * norms_r[..., self.ix]

        # LayerNorm expects mult as the last dimension
        # norms has shape (..., mult, nreps), need to transpose for LayerNorm
        norms_t = norms.transpose(-2, -1)  # (..., nreps, mult)
        lnorms_t = self.lnorm(norms_t)  # normalize over mult (last dim)
        lnorms = lnorms_t.transpose(-2, -1)  # (..., mult, nreps)

        # Renormalize features
        norms_r = lnorms / (norms + self.epsilon)
        return f * norms_r[..., self.ix]


class Attention(nn.Module):
    """k-NN based multi-head attention with regular neighbor structure.

    Each node attends only to its k nearest neighbors, enabling
    efficient local attention with O(N*k) complexity instead of O(N^2).

    Args:
        hidden_size: Total hidden dimension.
        nheads: Number of attention heads.
        dropout: Dropout probability for attention weights.

    Example:
        >>> attn = Attention(64, nheads=4)
        >>> keys = torch.randn(100, 64)
        >>> queries = torch.randn(100, 64)
        >>> values = torch.randn(100, 64)
        >>> neighbor_idx = torch.randint(0, 100, (100, 16))
        >>> output = attn(keys, queries, values, neighbor_idx)  # (100, 64)
    """

    def __init__(
        self: Attention,
        hidden_size: int,
        nheads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_size % nheads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by nheads ({nheads})"
            )

        self.hidden_size = hidden_size
        self.nheads = nheads
        self.head_dim = hidden_size // nheads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(
        self: Attention,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        neighbor_idx: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute k-NN attention.

        Args:
            keys: Key vectors of shape (N, hidden_size).
            queries: Query vectors of shape (N, hidden_size).
            values: Value vectors of shape (N, hidden_size).
            neighbor_idx: Neighbor indices of shape (N, k).
            mask: Optional attention mask of shape (N, k).
                  True values are masked out.

        Returns:
            Attended values of shape (N, hidden_size).
        """
        N, k = neighbor_idx.shape

        # Reshape for multi-head: (N, nheads, head_dim)
        queries = queries.view(N, self.nheads, self.head_dim)
        keys = keys.view(N, self.nheads, self.head_dim)
        values = values.view(N, self.nheads, self.head_dim)

        # Gather neighbor keys and values: (N, k, nheads, head_dim)
        neighbor_keys = keys[neighbor_idx]
        neighbor_values = values[neighbor_idx]

        # Compute attention scores: (N, nheads, k)
        # queries: (N, nheads, head_dim) -> (N, nheads, 1, head_dim)
        # neighbor_keys: (N, k, nheads, head_dim) -> (N, nheads, k, head_dim)
        queries = queries.unsqueeze(2)  # (N, nheads, 1, head_dim)
        neighbor_keys = neighbor_keys.transpose(1, 2).contiguous()  # (N, nheads, k, head_dim)
        neighbor_values = neighbor_values.transpose(1, 2).contiguous()  # (N, nheads, k, head_dim)

        scores = (queries @ neighbor_keys.transpose(-2, -1)).squeeze(2) * self.scale
        # scores: (N, nheads, k)

        # Apply mask if provided
        if mask is not None:
            # mask: (N, k) -> (N, 1, k)
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))

        # Softmax over k neighbors
        weights = F.softmax(scores, dim=-1)  # (N, nheads, k)
        weights = self.dropout(weights)

        # Weighted sum of neighbor values: (N, nheads, head_dim)
        # weights: (N, nheads, k) -> (N, nheads, 1, k)
        # neighbor_values: (N, nheads, k, head_dim)
        output = (weights.unsqueeze(2) @ neighbor_values).squeeze(2)

        # Reshape back: (N, hidden_size)
        return output.view(N, self.hidden_size)


class EquivariantAttention(nn.Module):
    """SE(3)-equivariant k-NN attention.

    Q, K, V are all computed as edge features via single convolution:
        1. Q, K, V = conv(neighbor_features, edge)  # All edge-level
        2. scores = Q_ij · K_ij                     # Per-edge dot product
        3. output = softmax(scores) @ V             # Weighted sum of values

    This design supports different input/output lvals since the convolution
    can transform between representations via tensor product with edge
    spherical harmonics.

    Args:
        repr: ProductRepr specifying input (rep1) and output (rep2) representations.
        edge_dim: Dimension of invariant edge features.
        edge_hidden_dim: Hidden dimension for radial networks.
        nheads: Number of attention heads.
        dropout: Dropout for convolutions.
        attn_dropout: Dropout for attention weights.
        scale_type: Attention scaling - "sqrt_head_dim", "sqrt_dim", "learned", or "none".
        radial_weight_rank: Rank for low-rank radial weight decomposition.
            If None, uses full-rank weights. Lower rank reduces parameters.
        qk_norm: If True, apply RMSNorm to queries and keys before computing
            attention scores. Improves training stability for deep networks.
        use_rope: If True, apply Rotary Position Embeddings to queries and keys.
            Requires seq_pos to be passed to forward(). Replaces the need for
            separate SequencePositionEncoding on edge features.
        rank: Maximum filter degree for equivariant basis.
            0 = rank-1 (fast, default), None = full SO(3) expressivity.
    """

    def __init__(
        self: EquivariantAttention,
        repr: ProductRepr,
        edge_dim: int,
        edge_hidden_dim: int,
        nheads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        scale_type: str = "sqrt_head_dim",
        radial_weight_rank: int | None = None,
        qk_norm: bool = False,
        use_rope: bool = False,
        rank: int | None = 0,
    ) -> None:
        super().__init__()

        if scale_type not in ("sqrt_head_dim", "sqrt_dim", "learned", "none"):
            raise ValueError(f"scale_type must be 'sqrt_head_dim', 'sqrt_dim', 'learned', or 'none'")

        self.repr = repr
        self.nheads = nheads
        self.scale_type = scale_type
        self.qk_norm = qk_norm
        self.use_rope = use_rope
        self.rank = rank

        # Input/output dimensions
        in_mult = repr.rep1.mult
        in_dim = repr.rep1.dim()
        out_mult = repr.rep2.mult
        out_dim = repr.rep2.dim()

        self.in_mult = in_mult
        self.in_dim = in_dim
        self.out_mult = out_mult
        self.out_dim = out_dim

        # Hidden size for attention (output space)
        self.hidden_size = out_mult * out_dim
        self.attn_hidden_size = out_mult * out_dim

        if self.attn_hidden_size % nheads != 0:
            raise ValueError(
                f"attention hidden_size ({self.attn_hidden_size}) must be "
                f"divisible by nheads ({nheads})"
            )

        self.head_dim = self.attn_hidden_size // nheads

        # Initialize scale factor based on scale_type
        self._init_scale()

        # QK normalization (improves training stability for deep networks)
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        # Rotary Position Embeddings
        if use_rope:
            from ..layers.transformer import RotaryPositionEmbedding
            self.rope = RotaryPositionEmbedding(self.head_dim)
        else:
            self.rope = None

        # Q, K, V all computed via single edge-level convolution
        rep1_copy = deepcopy(repr.rep1)
        rep2_copy = deepcopy(repr.rep2)
        rep2_copy.mult = 3 * out_mult  # Q, K, V concatenated
        repr_qkv = ProductRepr(rep1_copy, rep2_copy)

        num_basis = _compute_num_basis(repr_qkv, rank)
        self.conv_qkv = EquivariantConvolution(
            repr_qkv, edge_dim, edge_hidden_dim,
            num_basis=num_basis,
            dropout=dropout, radial_weight_rank=radial_weight_rank
        )

        # Output projection (operates on rep2)
        self.out_proj = EquivariantLinear(repr.rep2, repr.rep2, activation=None, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def _init_scale(self) -> None:
        """Initialize attention scale factor."""
        if self.scale_type == "sqrt_head_dim":
            self.scale = self.head_dim ** -0.5
        elif self.scale_type == "sqrt_dim":
            self.scale = self.attn_hidden_size ** -0.5
        elif self.scale_type == "learned":
            self.scale = nn.Parameter(torch.tensor(self.head_dim ** -0.5))
        else:  # none
            self.scale = 1.0

    def forward(
        self: EquivariantAttention,
        basis: torch.Tensor,
        edge_feats: torch.Tensor,
        f: torch.Tensor,
        neighbor_idx: torch.Tensor,
        mask: torch.Tensor | None = None,
        seq_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply equivariant k-NN attention.

        Args:
            basis: Precomputed equivariant basis of shape (N, k, num_basis, dim1, dim2).
            edge_feats: Edge features of shape (N, k, edge_dim).
            f: Node features of shape (N, in_mult, in_dim).
            neighbor_idx: Neighbor indices of shape (N, k).
            mask: Optional attention mask of shape (N, k).
            seq_pos: Optional sequence positions of shape (N,). Required if
                use_rope=True. Used to compute relative positions for RoPE.

        Returns:
            Updated node features of shape (N, out_mult, out_dim).
        """
        if self.rope is not None and seq_pos is None:
            raise ValueError("seq_pos must be provided when use_rope=True")

        N, k = neighbor_idx.shape

        # Handle empty neighbor case
        if k == 0:
            return f.new_zeros(N, self.out_mult, self.out_dim)

        # Flatten for convolution
        src_idx = neighbor_idx.flatten()
        basis_flat = basis.view(N * k, *basis.shape[2:])  # (N*k, num_basis, dim1, dim2)
        edge_feats_flat = edge_feats.view(N * k, -1)

        # Compute Q, K, V as edge features via single convolution
        qkv = self.conv_qkv(basis_flat, edge_feats_flat, f, src_idx)
        qkv = qkv.view(N, k, 3 * self.out_mult, self.out_dim)

        # Split into Q, K, V
        queries, keys, values = qkv.chunk(3, dim=2)

        # Reshape for multi-head attention: (N, k, nheads, head_dim)
        q_heads = queries.flatten(-2, -1).view(N, k, self.nheads, self.head_dim)
        k_heads = keys.flatten(-2, -1).view(N, k, self.nheads, self.head_dim)
        v_heads = values.flatten(-2, -1).view(N, k, self.nheads, self.head_dim)

        # Transpose to (N, nheads, k, head_dim) and make contiguous for efficient ops
        q_heads = q_heads.transpose(1, 2).contiguous()
        k_heads = k_heads.transpose(1, 2).contiguous()
        v_heads = v_heads.transpose(1, 2).contiguous()

        # Apply QK normalization if enabled
        if self.q_norm is not None:
            q_heads = self.q_norm(q_heads)
            k_heads = self.k_norm(k_heads)

        # Apply RoPE if enabled
        if self.rope is not None:
            # Q position: target node index (N,) -> (N, 1, k) broadcast to all neighbors
            q_pos = seq_pos.view(N, 1, 1).expand(-1, -1, k)
            # K position: source node index from neighbor_idx (N, k) -> (N, 1, k)
            k_pos = seq_pos[neighbor_idx].unsqueeze(1)
            q_heads = self.rope.apply_by_positions(q_heads, q_pos)
            k_heads = self.rope.apply_by_positions(k_heads, k_pos)

        # Edge attention scores: element-wise Q · K, sum over head_dim
        scores = (q_heads * k_heads).sum(dim=-1) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weight edge values and aggregate
        weighted = attn_weights.unsqueeze(-1) * v_heads
        output = weighted.sum(dim=2)

        # Reshape back to (N, out_mult, out_dim)
        output = output.view(N, self.out_mult, self.out_dim)

        return self.out_proj(output)


class EquivariantTransformerBlock(nn.Module):
    """Transformer block with k-NN attention.

    Combines equivariant attention with optional transition layer,
    layer normalization, and residual connections.

    Args:
        repr: ProductRepr for the block.
        edge_dim: Edge feature dimension.
        edge_hidden_dim: Hidden dimension for edge processing.
        nheads: Number of attention heads.
        dropout: Dropout probability.
        attn_dropout: Attention dropout probability.
        transition: Whether to include transition layer.
        residual_scale: Scale factor for residual connections. Use < 1.0
            (e.g., 0.1-0.5) for deep networks to improve gradient flow.
        scale_type: Attention scaling strategy.
        skip_type: Skip connection type - "scaled", "gated", or "none".
        radial_weight_rank: Rank for low-rank radial weight decomposition.
            If None, uses full-rank weights.
        qk_norm: If True, apply RMSNorm to queries and keys before attention.
        use_rope: If True, apply Rotary Position Embeddings to queries and keys.
        rank: Maximum filter degree for equivariant basis.
            0 = rank-1 (fast, default), None = full SO(3) expressivity.
    """

    def __init__(
        self: EquivariantTransformerBlock,
        repr: ProductRepr,
        edge_dim: int,
        edge_hidden_dim: int,
        nheads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        transition: bool = False,
        residual_scale: float = 1.0,
        scale_type: str = "sqrt_head_dim",
        skip_type: str = "scaled",
        radial_weight_rank: int | None = None,
        qk_norm: bool = False,
        use_rope: bool = False,
        rank: int | None = 0,
    ) -> None:
        super().__init__()

        if skip_type not in ("scaled", "gated", "none"):
            raise ValueError(f"skip_type must be 'scaled', 'gated', or 'none', got {skip_type}")

        self.prepr = repr
        self.residual_scale = residual_scale
        self.skip_type = skip_type

        self.attn = EquivariantAttention(
            repr, edge_dim, edge_hidden_dim, nheads, dropout, attn_dropout,
            scale_type=scale_type,
            radial_weight_rank=radial_weight_rank, qk_norm=qk_norm,
            use_rope=use_rope, rank=rank,
        )

        self.ln1 = EquivariantLayerNorm(repr.rep1)

        # Check if skip connection is possible (requires matching dimensions)
        deg_match = repr.rep1.lvals == repr.rep2.lvals
        mult_match = repr.rep1.mult == repr.rep2.mult
        self.can_skip = deg_match and mult_match

        # Learnable gate for gated skip connections
        if skip_type == "gated" and self.can_skip:
            self.gate = nn.Parameter(torch.zeros(1))
        else:
            self.gate = None

        # Optional transition layer
        if transition:
            self.ln2 = EquivariantLayerNorm(repr.rep2)
            hidden_repr = deepcopy(repr.rep2)
            hidden_repr.mult = repr.rep2.mult * 4
            self.transition = EquivariantTransition(repr.rep2, hidden_repr)
            # Separate gate for transition if gated
            if skip_type == "gated" and self.can_skip:
                self.gate_transition = nn.Parameter(torch.zeros(1))
            else:
                self.gate_transition = None
        else:
            self.ln2 = None
            self.transition = None
            self.gate_transition = None

    def _apply_skip(
        self, residual: torch.Tensor, output: torch.Tensor, gate: nn.Parameter | None
    ) -> torch.Tensor:
        """Apply skip connection based on skip_type."""
        if not self.can_skip or self.skip_type == "none":
            return output
        elif self.skip_type == "scaled":
            return residual + self.residual_scale * output
        elif self.skip_type == "gated":
            g = torch.sigmoid(gate)
            return residual * (1 - g) + output * g
        return output

    def forward(
        self: EquivariantTransformerBlock,
        basis: torch.Tensor,
        features: torch.Tensor,
        edge_feats: torch.Tensor,
        neighbor_idx: torch.Tensor,
        mask: torch.Tensor | None = None,
        seq_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply transformer block.

        Args:
            basis: Precomputed equivariant basis of shape (N, k, num_basis, dim1, dim2).
            features: Node features of shape (N, mult, dim).
            edge_feats: Edge features of shape (N, k, edge_dim).
            neighbor_idx: Neighbor indices of shape (N, k).
            mask: Optional attention mask of shape (N, k).
            seq_pos: Optional sequence positions of shape (N,). Required if use_rope=True.

        Returns:
            Updated node features.
        """
        # Pre-LN transformer variant
        residual = features
        features = self.ln1(features)
        features = self.attn(basis, edge_feats, features, neighbor_idx, mask, seq_pos)
        features = self._apply_skip(residual, features, self.gate)

        if self.transition is not None:
            residual = features
            features = self.ln2(features)
            features = self.transition(features)
            features = self._apply_skip(residual, features, self.gate_transition)

        return features


class EquivariantTransformer(nn.Module):
    """SE(3)-equivariant transformer for geometric point clouds.

    A full transformer architecture that processes 3D point clouds
    while maintaining SE(3) equivariance. Uses k-NN attention with
    O(N*k) complexity.

    Supports different input/output lvals since the convolution can
    transform between representations via tensor product with edge
    spherical harmonics.

    Args:
        in_repr: Input representation.
        out_repr: Output representation.
        hidden_repr: Hidden representation for intermediate layers.
        hidden_layers: Number of hidden transformer blocks.
        edge_dim: Edge feature dimension (RBF dimension).
        edge_hidden_dim: Hidden dimension for edge processing.
        k_neighbors: Number of nearest neighbors for attention.
        nheads: Number of attention heads.
        dropout: Dropout probability.
        attn_dropout: Attention dropout probability.
        transition: Whether to include transition layers.
        residual_scale: Scale factor for residual connections. Use < 1.0
            (e.g., 0.1-0.5) for deep networks to improve gradient flow.
        scale_type: Attention scaling - "sqrt_head_dim", "sqrt_dim", "learned", "none".
        skip_type: Skip connection type - "scaled", "gated", or "none".
        rbf_type: Radial basis function type - "gaussian", "bessel", or "polynomial".
        rbf_r_min: Minimum radius for RBF initialization.
        rbf_r_max: Maximum radius for RBF initialization/cutoff.
        radial_weight_rank: Rank for low-rank radial weight decomposition.
            If None, uses full-rank weights. Lower rank significantly reduces parameters.
        seq_pos_dim: Dimension for sequence position encoding. If 0 or None,
            no sequence position encoding is used. When enabled, encodes the
            relative sequence distance between nodes (useful for polymers where
            sequential vs non-sequential neighbors have different meanings).
        seq_pos_max_distance: Maximum sequence distance to encode. Default 128.
        seq_pos_learnable: If True, use learnable position embeddings instead of sinusoidal.
        qk_norm: If True, apply RMSNorm to queries and keys before attention.
            Improves training stability for deep networks.
        use_rope: If True, apply Rotary Position Embeddings to Q/K instead of edge-based
            position encoding. Requires seq_pos to be passed to forward(). Can be used
            together with seq_pos_dim (RoPE on Q/K, edge encoding on edge features).
        rank: Maximum filter degree for equivariant basis.
            0 = rank-1 (fast, default), None = full SO(3) expressivity.

    Example:
        >>> in_repr = Repr(lvals=[0], mult=32)  # Scalar input
        >>> out_repr = Repr(lvals=[0], mult=2)  # Scalar output
        >>> hidden_repr = Repr(lvals=[0, 1], mult=16)  # Vectors in hidden
        >>> model = EquivariantTransformer(
        ...     in_repr, out_repr, hidden_repr,
        ...     hidden_layers=4,
        ...     edge_dim=16,
        ...     edge_hidden_dim=32,
        ...     k_neighbors=16,
        ...     seq_pos_dim=8,  # Add sequence position encoding
        ... )
        >>> seq_pos = torch.arange(N)  # residue indices
        >>> output = model(coordinates, node_features, seq_pos=seq_pos)
    """

    def __init__(
        self: EquivariantTransformer,
        in_repr: Repr,
        out_repr: Repr,
        hidden_repr: Repr,
        hidden_layers: int,
        edge_dim: int,
        edge_hidden_dim: int,
        k_neighbors: int,
        nheads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        transition: bool = False,
        residual_scale: float = 1.0,
        scale_type: str = "sqrt_head_dim",
        skip_type: str = "scaled",
        rbf_type: str = "gaussian",
        rbf_r_min: float = 0.0,
        rbf_r_max: float = 10.0,
        radial_weight_rank: int | None = None,
        seq_pos_dim: int | None = None,
        seq_pos_max_distance: int = 128,
        seq_pos_learnable: bool = False,
        qk_norm: bool = False,
        use_rope: bool = False,
        rank: int | None = 0,
    ) -> None:
        super().__init__()

        # Sequence position encoding (optional)
        self.seq_pos_dim = seq_pos_dim or 0
        self.qk_norm = qk_norm
        self.use_rope = use_rope
        if self.seq_pos_dim > 0:
            self.seq_pos_enc = SequencePositionEncoding(
                dim=self.seq_pos_dim,
                max_seq_distance=seq_pos_max_distance,
                learnable=seq_pos_learnable,
            )
        else:
            self.seq_pos_enc = None

        # Total edge dimension = RBF + sequence position
        total_edge_dim = edge_dim + self.seq_pos_dim

        self.edge_dim = edge_dim
        self.total_edge_dim = total_edge_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.nheads = nheads
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.use_transition = transition
        self.residual_scale = residual_scale
        self.k_neighbors = k_neighbors
        self.scale_type = scale_type
        self.skip_type = skip_type
        self.radial_weight_rank = radial_weight_rank
        self.rank = rank

        self.in_repr = in_repr
        self.out_repr = out_repr
        self.hidden_repr = hidden_repr

        # Final layer norm and output projection
        out_repr_tmp = deepcopy(out_repr)
        out_repr_tmp.mult = hidden_repr.mult
        self.final_ln = EquivariantLayerNorm(out_repr_tmp)
        self.proj = EquivariantLinear(out_repr_tmp, out_repr, activation=None, bias=True)

        # Build sequence of representations
        reprs = [in_repr] + [hidden_repr] * hidden_layers + [out_repr_tmp]

        # Create layers and track product representations
        # With fused K/V, we only need one ProductRepr per layer
        layers = []
        preprs = []
        for i in range(len(reprs) - 1):
            repr1, repr2 = reprs[i], reprs[i + 1]
            prepr = ProductRepr(deepcopy(repr1), deepcopy(repr2))
            preprs.append(prepr)
            layers.append(self._construct_layer(prepr))

        self.layers = nn.ModuleList(layers)
        self.num_layers = len(layers)

        # Compute equivariant bases for all layers (with deduplication)
        self.bases = EquivariantBases(*preprs, rank=rank)

        # Radial basis functions for edge features
        self.rbf = RadialBasisFunctions(
            edge_dim,
            r_min=rbf_r_min,
            r_max=rbf_r_max,
            rbf_type=rbf_type,
        )

    def _construct_layer(
        self: EquivariantTransformer,
        prepr: ProductRepr,
    ) -> EquivariantTransformerBlock:
        """Construct a single transformer block."""
        return EquivariantTransformerBlock(
            prepr,
            self.total_edge_dim,  # Use total_edge_dim (RBF + seq pos)
            self.edge_hidden_dim,
            self.nheads,
            self.dropout,
            self.attn_dropout,
            self.use_transition,
            self.residual_scale,
            scale_type=self.scale_type,
            skip_type=self.skip_type,
            radial_weight_rank=self.radial_weight_rank,
            qk_norm=self.qk_norm,
            use_rope=self.use_rope,
            rank=self.rank,
        )

    def forward(
        self: EquivariantTransformer,
        coordinates: torch.Tensor,
        node_features: torch.Tensor,
        edge_features: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        seq_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply equivariant transformer.

        Args:
            coordinates: Node coordinates of shape (N, 3).
            node_features: Node features of shape (N, mult, dim).
            edge_features: Optional edge features of shape (N, k, edge_dim).
                If None, uses distances.
            mask: Optional attention mask of shape (N, k).
            seq_pos: Optional sequence positions of shape (N,). Required if
                seq_pos_dim > 0 was set during initialization. For polymers,
                this is typically the residue index (0, 1, 2, ...).

        Returns:
            Output features of shape (N, out_mult, out_dim).

        Raises:
            ValueError: If node_features shape doesn't match in_repr.
            ValueError: If seq_pos_dim > 0 but seq_pos is not provided.
        """
        if not self.in_repr.verify(node_features):
            raise ValueError(
                f"Node features shape {node_features.size()} does not match "
                f"input representation (mult={self.in_repr.mult}, dim={self.in_repr.dim()})"
            )

        if self.seq_pos_enc is not None and seq_pos is None:
            raise ValueError(
                "seq_pos must be provided when seq_pos_dim > 0. "
                "Pass sequence positions (e.g., residue indices) or set seq_pos_dim=0."
            )

        if self.use_rope and seq_pos is None:
            raise ValueError(
                "seq_pos must be provided when use_rope=True. "
                "Pass sequence positions (e.g., residue indices) or set use_rope=False."
            )

        N = coordinates.size(0)

        # Handle empty input
        if N == 0:
            return node_features.new_zeros(0, self.out_repr.mult, self.out_repr.dim())

        # Build k-NN graph from coordinates
        neighbor_idx = build_knn_graph(coordinates, self.k_neighbors)
        k = neighbor_idx.size(1)

        # Compute displacements for (N, k) neighbor pairs
        neighbor_coords = coordinates[neighbor_idx]  # (N, k, 3)
        displacements = coordinates.unsqueeze(1) - neighbor_coords  # (N, k, 3)

        # Compute edge features from distances using RBFs if not provided
        if edge_features is None:
            distances = displacements.norm(dim=-1)  # (N, k)
            edge_features = self.rbf(distances)  # (N, k, edge_dim)

        # Add sequence position encoding if configured
        if self.seq_pos_enc is not None:
            seq_pos_features = self.seq_pos_enc(seq_pos, neighbor_idx)  # (N, k, seq_pos_dim)
            edge_features = torch.cat([edge_features, seq_pos_features], dim=-1)

        # Compute all bases at once (one per layer, with deduplication)
        all_bases_flat = self.bases(displacements.view(N * k, 3))

        # Pass through layers with precomputed bases
        for i, layer in enumerate(self.layers):
            # Get basis for this layer: (N*k, num_basis, dim1, dim2)
            basis_flat = all_bases_flat[i]
            # Reshape to (N, k, num_basis, dim1, dim2)
            basis = basis_flat.view(N, k, *basis_flat.shape[1:])

            node_features = layer(
                basis, node_features, edge_features, neighbor_idx, mask,
                seq_pos=seq_pos if self.use_rope else None
            )

        # Final layer norm and output projection
        node_features = self.final_ln(node_features)
        return self.proj(node_features)

    def compile(
        self: EquivariantTransformer,
        mode: str = "default",
        fullgraph: bool = False,
    ) -> EquivariantTransformer:
        """Compile the model using torch.compile for faster execution.

        Applies torch.compile to optimize the forward pass with kernel fusion
        and graph optimizations. Best speedups are seen on GPU with repeated
        inference or training iterations.

        Args:
            mode: Compilation mode. Options:
                - "default": Balanced compilation (default, good memory usage)
                - "reduce-overhead": Uses CUDA graphs (faster but high memory usage)
                - "max-autotune": Maximum optimization (slower compile, faster runtime)
            fullgraph: If True, requires the entire forward to compile as one graph.
                Set to False (default) to allow graph breaks for compatibility.

        Returns:
            Self for method chaining.

        Example:
            >>> model = EquivariantTransformer(...).compile()
            >>> output = model(coords, features)  # Compiled execution
        """
        # Compile the forward method in-place
        self.forward = torch.compile(
            self.forward,
            mode=mode,
            fullgraph=fullgraph,
        )
        return self
