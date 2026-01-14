"""Equivariant neural network primitives.

This module provides neural network layers and operations that are equivariant to
3D rotations (SO(3)). These are the building blocks for constructing
rotation-equivariant networks for 3D data like molecules and point clouds.

Classes:
    RepNorm: Compute norms of spherical tensor components
    SphericalHarmonic: Compute spherical harmonics features
    RadialBasisFunctions: Learnable radial basis function expansion
    SequencePositionEncoding: Sinusoidal encoding for relative sequence positions
    EquivariantBasis: Compute equivariant basis matrices for a product representation
    EquivariantBases: Compute equivariant bases for multiple product representations
"""
from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn as nn

from .representations import Repr, ProductRepr

# Dimension constants for spherical tensor indexing
FEATURE_DIM = -2  # Multiplicity dimension
REPR_DIM = -1     # Representation dimension

# Lazy import for sphericart
_sphericart: Any = None
_sphericart_available: bool | None = None


def _get_sphericart() -> Any:
    """Lazily import sphericart and cache the result."""
    global _sphericart, _sphericart_available

    if _sphericart_available is None:
        # Check if sphericart is disabled via environment variable
        if os.environ.get("CIFFY_DISABLE_SPHERICART", "").lower() in ("1", "true", "yes"):
            _sphericart_available = False
        else:
            try:
                import sphericart.torch as sc
                _sphericart = sc
                _sphericart_available = True
            except ImportError:
                _sphericart_available = False

    return _sphericart


def _spherical_harmonics_pytorch(
    x: torch.Tensor,
    lmax: int,
    normalized: bool = True,
) -> torch.Tensor:
    """Pure PyTorch implementation of real spherical harmonics for l=0,1,2.

    Uses the same conventions as sphericart for consistency.

    Args:
        x: Coordinates of shape (..., 3) in sphericart convention (z, x, y order).
        lmax: Maximum degree (only 0, 1, 2 supported).
        normalized: If True, normalize by 1/sqrt(4*pi) factor.

    Returns:
        Spherical harmonics of shape (..., (lmax+1)^2).
    """
    if lmax > 2:
        raise ValueError(
            f"PyTorch fallback only supports lmax <= 2, got {lmax}. "
            "Install sphericart for higher degrees: pip install sphericart"
        )

    # Extract coordinates (sphericart convention: input is z, x, y)
    z = x[..., 0]
    xc = x[..., 1]
    y = x[..., 2]

    # Compute r^2 and r with numerical stability
    r2 = xc * xc + y * y + z * z
    r = torch.sqrt(r2.clamp(min=1e-12))

    # Normalization factors for real spherical harmonics
    # Standard convention: Y_l^m normalized so integral over sphere = 1
    if normalized:
        # Normalized SH (sphericart default)
        norm_l0 = 0.28209479177387814  # 1/sqrt(4*pi)
        norm_l1 = 0.4886025119029199   # sqrt(3/(4*pi))
        norm_l2_0 = 0.31539156525252005  # sqrt(5/(16*pi))
        norm_l2_1 = 1.0925484305920792  # sqrt(15/(4*pi))
        norm_l2_2 = 0.5462742152960396  # sqrt(15/(16*pi))
    else:
        norm_l0 = 1.0
        norm_l1 = 1.0
        norm_l2_0 = 1.0
        norm_l2_1 = 1.0
        norm_l2_2 = 1.0

    results = []

    # l=0: Y_0^0 = norm_l0 (constant)
    results.append(torch.full_like(z, norm_l0))

    if lmax >= 1:
        # l=1: Y_1^{-1}, Y_1^0, Y_1^1
        # Standard ordering: m = -1, 0, 1
        inv_r = 1.0 / r
        results.append(norm_l1 * y * inv_r)   # Y_1^{-1}
        results.append(norm_l1 * z * inv_r)   # Y_1^0
        results.append(norm_l1 * xc * inv_r)  # Y_1^1

    if lmax >= 2:
        # l=2: Y_2^{-2}, Y_2^{-1}, Y_2^0, Y_2^1, Y_2^2
        inv_r2 = 1.0 / r2.clamp(min=1e-12)
        results.append(norm_l2_1 * xc * y * inv_r2)           # Y_2^{-2}
        results.append(norm_l2_1 * y * z * inv_r2)            # Y_2^{-1}
        results.append(norm_l2_0 * (3 * z * z - r2) * inv_r2) # Y_2^0
        results.append(norm_l2_1 * xc * z * inv_r2)           # Y_2^1
        results.append(norm_l2_2 * (xc * xc - y * y) * inv_r2) # Y_2^2

    return torch.stack(results, dim=-1)


class RepNorm(nn.Module):
    """Compute norms of spherical tensor components.

    For a spherical tensor with multiple irrep components, computes
    the norm of each component separately. This produces rotation-invariant
    features that can be used for gating or as input to invariant networks.

    Args:
        repr: The representation specifying the tensor structure.

    Example:
        >>> repr = Repr(lvals=[0, 1, 2])  # scalar + vector + rank-2
        >>> norm = RepNorm(repr)
        >>> st = torch.randn(32, 9)  # batch of spherical tensors
        >>> norms = norm(st)  # shape: (32, 3) - one norm per irrep
    """

    def __init__(self: RepNorm, repr: Repr) -> None:
        super().__init__()
        self.num_reps = repr.nreps()

        # Create indices mapping each dimension to its irrep index
        # e.g., for lvals=[0,1,2]: indices = [0, 1,1,1, 2,2,2,2,2]
        cdims = repr.cumdims()
        indices = []
        for i in range(self.num_reps):
            size = cdims[i + 1] - cdims[i]
            indices.extend([i] * size)
        self.register_buffer('indices', torch.tensor(indices, dtype=torch.long))

    def forward(self: RepNorm, st: torch.Tensor) -> torch.Tensor:
        """Compute the norm of each irrep component.

        Uses vectorized scatter_add for GPU efficiency, avoiding Python loops.

        Args:
            st: Spherical tensor of shape (..., dim).

        Returns:
            Norms of shape (..., nreps).
        """
        # Compute squared values
        sq = st * st

        # Allocate output and use scatter_add to sum within each irrep
        result = torch.zeros(
            *sq.shape[:-1], self.num_reps,
            device=sq.device, dtype=sq.dtype
        )
        ix = self.indices.expand(sq.shape)
        result.scatter_add_(-1, ix, sq)

        return result.sqrt()


class SphericalHarmonic(nn.Module):
    """Compute spherical harmonic features for 3D coordinates.

    Spherical harmonics form a complete orthonormal basis for functions
    on the sphere. They are the natural features for SO(3)-equivariant
    networks, as they transform predictably under rotations.

    Uses the sphericart library when available, with a pure PyTorch
    fallback for lmax <= 2.

    Args:
        lmax: Maximum degree of spherical harmonics to compute.
            Total features = (lmax + 1)^2.
        normalized: If True, compute normalized spherical harmonics.

    Raises:
        ImportError: If sphericart is not installed and lmax > 2.

    Example:
        >>> sh = SphericalHarmonic(lmax=2)
        >>> coords = torch.randn(100, 3)
        >>> features = sh(coords)  # shape: (100, 9)
    """

    def __init__(
        self: SphericalHarmonic,
        lmax: int,
        normalized: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(lmax, int) or lmax < 0:
            raise ValueError(f"lmax must be a non-negative integer, got {lmax}")

        self.lmax = lmax
        self.normalized = normalized

        sc = _get_sphericart()
        if sc is not None:
            # Use sphericart for efficient computation
            self.sh = sc.SphericalHarmonics(lmax, normalized)
            self._use_sphericart = True
        elif lmax <= 2:
            # Use PyTorch fallback for low-degree harmonics
            self.sh = None
            self._use_sphericart = False
        else:
            raise ImportError(
                f"sphericart is required for lmax > 2, got lmax={lmax}. "
                "Install with: pip install sphericart"
            )

        # Index permutation for sphericart coordinate convention
        self.register_buffer('ix', torch.tensor([2, 0, 1], dtype=torch.int64))

    def forward(self: SphericalHarmonic, x: torch.Tensor) -> torch.Tensor:
        """Compute spherical harmonic features for points.

        Compatible with torch.autocast (AMP). Sphericart only supports float32/64,
        so autocast is temporarily disabled during computation.

        Args:
            x: Coordinates of shape (..., N, 3).

        Returns:
            Spherical harmonic features of shape (..., N, (lmax+1)^2).
            NaN values (from zero vectors) are replaced with zeros.
        """
        *b, n, _ = x.shape

        # Handle empty input
        if n == 0:
            out_dim = (self.lmax + 1) ** 2
            return x.new_zeros(*b, 0, out_dim)

        x_flat = x.view(-1, 3)

        # Permute coordinates for sphericart convention
        x_permuted = x_flat[:, self.ix]

        # Handle dtype (sphericart only supports float32/64)
        dtype = x_permuted.dtype
        if dtype not in [torch.float32, torch.float64]:
            x_permuted = x_permuted.to(torch.float32)

        if self._use_sphericart:
            # Use sphericart with autocast disabled
            with torch.amp.autocast(x_permuted.device.type, enabled=False):
                sh = self.sh.compute(x_permuted)
        else:
            # Use PyTorch fallback
            sh = _spherical_harmonics_pytorch(x_permuted, self.lmax, self.normalized)

        # Restore original dtype and handle NaN
        sh = sh.to(dtype)
        sh = torch.nan_to_num(sh, nan=0.0)

        return sh.view(*b, n, -1)

    def pairwise(self: SphericalHarmonic, x: torch.Tensor) -> torch.Tensor:
        """Compute spherical harmonics for pairwise relative positions.

        Args:
            x: Point cloud of shape (N, 3).

        Returns:
            Pairwise features of shape (N, N, (lmax+1)^2).
        """
        # Compute pairwise relative positions
        pairwise = x[:, None] - x[None, :]

        # Compute spherical harmonics
        sh = self(pairwise)

        return torch.nan_to_num(sh, nan=0.0)


class RadialBasisFunctions(nn.Module):
    """Learnable radial basis function expansion.

    Expands scalar distance values into a set of basis functions.
    Useful for encoding distances in equivariant networks where
    edge features should be rotation-invariant.

    Supports multiple RBF types:
        - "gaussian": Standard Gaussian RBFs with learnable centers and widths
        - "bessel": Spherical Bessel functions (smooth at boundaries)
        - "polynomial": Polynomial envelope functions

    Args:
        num_functions: Number of basis functions.
        r_min: Minimum distance for center initialization.
        r_max: Maximum distance for center initialization (cutoff for bessel/polynomial).
        rbf_type: Type of radial basis function ("gaussian", "bessel", "polynomial").

    Attributes:
        mu: Learnable centers (gaussian only).
        sigma: Learnable widths (gaussian only).

    Example:
        >>> rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0)
        >>> distances = torch.rand(100) * 10  # distances in [0, 10]
        >>> features = rbf(distances)  # shape: (100, 16)
    """

    def __init__(
        self: RadialBasisFunctions,
        num_functions: int,
        r_min: float = 0.0,
        r_max: float = 10.0,
        rbf_type: str = "gaussian",
    ) -> None:
        super().__init__()

        if not isinstance(num_functions, int) or num_functions < 1:
            raise ValueError(f"num_functions must be a positive integer, got {num_functions}")
        if r_max <= r_min:
            raise ValueError(f"r_max ({r_max}) must be greater than r_min ({r_min})")

        valid_types = {"gaussian", "bessel", "polynomial"}
        if rbf_type not in valid_types:
            raise ValueError(f"rbf_type must be one of {valid_types}, got '{rbf_type}'")

        self.rbf_type = rbf_type
        self.num_functions = num_functions
        self.r_min = r_min
        self.r_max = r_max

        if rbf_type == "gaussian":
            # Initialize centers evenly spaced in [r_min, r_max]
            self.mu = nn.Parameter(
                torch.linspace(r_min, r_max, num_functions),
                requires_grad=True,
            )

            # Initialize widths based on spacing between centers
            spacing = (r_max - r_min) / max(num_functions - 1, 1)
            self.sigma = nn.Parameter(
                torch.full((num_functions,), spacing),
                requires_grad=True,
            )

        elif rbf_type == "bessel":
            # Spherical Bessel basis: j_0(n*pi*r/r_max)
            # Frequencies for each basis function
            self.register_buffer(
                "bessel_freqs",
                torch.arange(1, num_functions + 1, dtype=torch.float32) * math.pi / r_max,
            )

        elif rbf_type == "polynomial":
            # Polynomial envelope with smooth cutoff
            # Each basis function is: (1 - r/r_max)^(p+n) where n is the basis index
            self.register_buffer(
                "poly_powers",
                torch.arange(2, num_functions + 2, dtype=torch.float32),
            )

    def forward(self: RadialBasisFunctions, x: torch.Tensor) -> torch.Tensor:
        """Evaluate radial basis functions at input values.

        Args:
            x: Input distances of shape (...).

        Returns:
            Basis function values of shape (..., num_functions).
        """
        if self.rbf_type == "gaussian":
            return self._forward_gaussian(x)
        elif self.rbf_type == "bessel":
            return self._forward_bessel(x)
        elif self.rbf_type == "polynomial":
            return self._forward_polynomial(x)
        else:
            raise ValueError(f"Unknown rbf_type: {self.rbf_type}")

    def _forward_gaussian(self: RadialBasisFunctions, x: torch.Tensor) -> torch.Tensor:
        """Gaussian RBF: exp(-((x - mu) / sigma)^2)."""
        diff = (x[..., None] - self.mu) / self.sigma.abs().clamp(min=1e-6)
        return torch.exp(-diff ** 2)

    def _forward_bessel(self: RadialBasisFunctions, x: torch.Tensor) -> torch.Tensor:
        """Spherical Bessel RBF with smooth cutoff.

        Uses sinc functions (spherical Bessel j_0) multiplied by
        a smooth envelope that goes to zero at r_max.
        """
        # Smooth cutoff envelope: (1 - (r/r_max)^2)^2 for r < r_max, 0 otherwise
        r_scaled = (x / self.r_max).clamp(max=1.0)
        envelope = (1 - r_scaled ** 2) ** 2

        # Spherical Bessel j_0(k*r) = sin(k*r) / (k*r)
        # We use sinc which handles r=0 correctly
        kr = x[..., None] * self.bessel_freqs
        # sinc(x) = sin(pi*x) / (pi*x), so we need to adjust
        bessel = torch.sinc(kr / math.pi)

        return envelope[..., None] * bessel

    def _forward_polynomial(self: RadialBasisFunctions, x: torch.Tensor) -> torch.Tensor:
        """Polynomial envelope RBF.

        Uses polynomial functions with smooth cutoff at r_max.
        Each basis: (1 - r/r_max)^p for different powers p.
        """
        # Normalize to [0, 1] and clip
        r_scaled = (x / self.r_max).clamp(min=0.0, max=1.0)

        # (1 - r/r_max)^p for each power p
        one_minus_r = 1.0 - r_scaled
        return one_minus_r[..., None] ** self.poly_powers


class SequencePositionEncoding(nn.Module):
    """Sinusoidal encoding for relative sequence positions.

    Encodes the relative distance in sequence space (|i - j|) between
    nodes using sinusoidal functions, similar to the original transformer
    positional encoding but applied to relative positions.

    This allows the model to learn patterns based on sequence distance,
    which is important for polymers where:
    - Sequential neighbors (|i-j|=1) are covalently bonded
    - Distant in sequence but close in space = tertiary contacts

    Args:
        dim: Output dimension for position encoding.
        max_seq_distance: Maximum sequence distance to encode. Distances
            beyond this are clipped. Default 128 covers most local patterns.
        learnable: If True, use learnable embeddings instead of sinusoidal.

    Example:
        >>> enc = SequencePositionEncoding(dim=16)
        >>> seq_pos = torch.arange(50)  # residue indices
        >>> neighbor_idx = torch.randint(0, 50, (50, 16))
        >>> pos_features = enc(seq_pos, neighbor_idx)  # (50, 16, 16)
    """

    def __init__(
        self: SequencePositionEncoding,
        dim: int,
        max_seq_distance: int = 128,
        learnable: bool = False,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.max_seq_distance = max_seq_distance
        self.learnable = learnable

        if learnable:
            # Learnable embeddings for each distance bucket
            # Use 2*max + 1 to handle signed distances if needed
            self.embedding = nn.Embedding(2 * max_seq_distance + 1, dim)
        else:
            # Pre-compute sinusoidal encoding
            # Frequencies: 10000^(-2i/d) for i in [0, d/2)
            position = torch.arange(max_seq_distance + 1).float().unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
            )
            pe = torch.zeros(max_seq_distance + 1, dim)
            pe[:, 0::2] = torch.sin(position * div_term)
            if dim > 1:
                pe[:, 1::2] = torch.cos(position * div_term[:dim // 2])
            self.register_buffer("pe", pe)

    def forward(
        self: SequencePositionEncoding,
        seq_pos: torch.Tensor,
        neighbor_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute sequence position encoding for edges.

        Args:
            seq_pos: Sequence positions for each node, shape (N,).
                For residue-level: residue index (0, 1, 2, ...).
                For atom-level: can be residue index or atom index.
            neighbor_idx: Neighbor indices of shape (N, k).

        Returns:
            Position encoding of shape (N, k, dim).
        """
        N, k = neighbor_idx.shape

        # Debug assertion: verify neighbor indices are valid for seq_pos
        assert neighbor_idx.max() < len(seq_pos), (
            f"SequencePositionEncoding: neighbor_idx max ({neighbor_idx.max()}) >= seq_pos length ({len(seq_pos)})"
        )

        # Get sequence positions for self and neighbors
        self_pos = seq_pos.unsqueeze(1)  # (N, 1)
        neighbor_pos = seq_pos[neighbor_idx]  # (N, k)

        # Compute absolute sequence distance
        seq_dist = (self_pos - neighbor_pos).abs()  # (N, k)

        # Clip to max distance
        seq_dist = seq_dist.clamp(max=self.max_seq_distance)

        if self.learnable:
            # Shift to handle embedding index (center at max_seq_distance)
            signed_dist = self_pos - neighbor_pos
            signed_dist = signed_dist.clamp(-self.max_seq_distance, self.max_seq_distance)
            idx = (signed_dist + self.max_seq_distance).long()
            return self.embedding(idx)
        else:
            # Look up pre-computed sinusoidal encoding
            return self.pe[seq_dist.long()]


class EquivariantBasis(nn.Module):
    """Compute SO(3)-equivariant basis matrices using Clebsch-Gordan coefficients.

    Given 3D displacement vectors, computes explicit basis matrices suitable for
    equivariant convolutions. The basis matrices satisfy the equivariance property:
        B(R @ x) = D_out(R) @ B(x) @ D_in(R)^T

    The basis is parameterized by `rank` which controls the trade-off between
    expressivity and efficiency:
        - rank=0: Only l=|l1-l2| terms (rank-1 approximation, fastest)
        - rank=k: Terms up to l=|l1-l2|+k
        - rank=None: All terms up to l=l1+l2 (full SO(3) expressivity)

    The implementation uses a vectorized single-matmul approach for efficiency:
    precomputed CG coefficients are packed into a tensor that can be contracted
    with spherical harmonics in one operation.

    Args:
        repr: ProductRepr specifying the input and output representations.
        rank: Maximum filter degree relative to the minimum coupling.
            0 = rank-1 (fast, current default), None = full expressivity.

    Example:
        >>> repr = ProductRepr(Repr([0, 1]), Repr([0, 1]))
        >>> basis = EquivariantBasis(repr, rank=0)  # Rank-1 (fast)
        >>> displacements = torch.randn(100, 3)
        >>> B = basis(displacements)  # (100, num_basis, dim1, dim2)
        >>>
        >>> basis_full = EquivariantBasis(repr, rank=None)  # Full SO(3)
        >>> B_full = basis_full(displacements)  # More basis elements
    """

    def __init__(
        self: EquivariantBasis,
        repr: ProductRepr,
        rank: int | None = 0,
    ) -> None:
        super().__init__()

        self.repr = repr
        self.rank = rank
        self.dim1 = repr.rep1.dim()
        self.dim2 = repr.rep2.dim()

        # Maximum l needed for spherical harmonics
        self._lmax = repr.lmax()

        # Spherical harmonic calculator
        self.sh = SphericalHarmonic(self._lmax)

        # Get the coupling tensor from ProductRepr
        coupling_coeff, num_basis = repr.cg_tensor(rank)
        self.num_basis = num_basis

        # Register as buffer: (sh_dim, num_basis * dim1 * dim2)
        self.register_buffer('coupling_coeff', coupling_coeff)

    def forward(self: EquivariantBasis, x: torch.Tensor) -> torch.Tensor:
        """Compute equivariant basis matrices.

        Args:
            x: Displacement vectors of shape (N, 3).

        Returns:
            Basis matrices of shape (N, num_basis, dim1, dim2).
        """
        N = x.size(0)

        # Compute spherical harmonics: (N, sh_dim)
        sh = self.sh(x)
        sh = torch.nan_to_num(sh, nan=0.0)

        # Single matmul: (N, sh_dim) @ (sh_dim, num_basis * dim1 * dim2)
        # -> (N, num_basis * dim1 * dim2)
        out = sh @ self.coupling_coeff

        # Reshape to (N, num_basis, dim1, dim2)
        return out.view(N, self.num_basis, self.dim1, self.dim2)


class _EquivariantBasisLegacy(nn.Module):
    """Legacy equivariant basis returning factored coefficients.

    This is the original implementation that returns a tuple of coefficient
    tensors for low-rank tensor product contractions. Kept for backward
    compatibility with existing code.

    Args:
        repr: ProductRepr specifying the input and output representations.
    """

    def __init__(self: _EquivariantBasisLegacy, repr: ProductRepr) -> None:
        super().__init__()

        self.outdims1 = (repr.rep1.dim(), repr.rep1.nreps())
        self.outdims2 = (repr.rep2.nreps(), repr.rep2.dim())

        # Spherical harmonic calculator
        self.sh = SphericalHarmonic(repr.lmax())

        # Pre-compute slice indices for efficient forward pass
        cdims1 = repr.rep1.cumdims()
        self.slices1 = [
            (cdims1[j], cdims1[j + 1], l ** 2, (l + 1) ** 2, j)
            for j, l in enumerate(repr.rep1.lvals)
        ]

        cdims2 = repr.rep2.cumdims()
        self.slices2 = [
            (cdims2[j], cdims2[j + 1], l ** 2, (l + 1) ** 2, j)
            for j, l in enumerate(repr.rep2.lvals)
        ]

    def forward(
        self: _EquivariantBasisLegacy,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute equivariant basis coefficients.

        Args:
            x: Displacement vectors of shape (N, 3).

        Returns:
            Tuple of (coeff1, coeff2) where:
            - coeff1 has shape (N, rep1.dim(), rep1.nreps())
            - coeff2 has shape (N, rep2.nreps(), rep2.dim())
        """
        # Get spherical harmonic features
        sh = self.sh(x)
        sh = torch.nan_to_num(sh, nan=0.0)

        # Build coefficient matrices using pre-computed slices
        coeff1 = torch.zeros(x.size(0), *self.outdims1, device=x.device, dtype=x.dtype)
        coeff2 = torch.zeros(x.size(0), *self.outdims2, device=x.device, dtype=x.dtype)

        # Normalize SH values to unit RMS for each l-degree
        sh_normalized = sh * math.sqrt(4 * math.pi)

        for c_low, c_high, sh_low, sh_high, j in self.slices1:
            coeff1[..., c_low:c_high, j] = sh_normalized[..., sh_low:sh_high]

        for c_low, c_high, sh_low, sh_high, j in self.slices2:
            coeff2[..., j, c_low:c_high] = sh_normalized[..., sh_low:sh_high]

        return coeff1, coeff2


class EquivariantBases(nn.Module):
    """Compute equivariant bases for multiple product representations.

    Efficiently computes basis matrices for multiple ProductRepr objects,
    avoiding redundant computation for identical representations.

    Args:
        *reprs: Variable number of ProductRepr objects.
        rank: Maximum filter degree for all bases.
            0 = rank-1 (fast), None = full expressivity.

    Example:
        >>> repr1 = ProductRepr(Repr([0, 1]), Repr([0, 1]))
        >>> repr2 = ProductRepr(Repr([0, 1, 2]), Repr([0, 1, 2]))
        >>> bases = EquivariantBases(repr1, repr2, rank=0)
        >>> displacements = torch.randn(100, 3)
        >>> basis_list = bases(displacements)  # tuple of basis tensors
    """

    def __init__(
        self: EquivariantBases,
        *reprs: ProductRepr,
        rank: int | None = 0,
    ) -> None:
        super().__init__()

        self.rank = rank

        # Deduplicate representations to avoid redundant computation
        self.unique_reprs: list[ProductRepr] = []
        self.comps = nn.ModuleList()
        self.repr_ix: list[int] = []

        repr_count = -1
        for repr in reprs:
            if repr not in self.unique_reprs:
                self.unique_reprs.append(repr)
                self.comps.append(EquivariantBasis(repr, rank=rank))
                repr_count += 1
            self.repr_ix.append(repr_count)

    def forward(
        self: EquivariantBases,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Compute basis matrices for all representations.

        Args:
            x: Displacement vectors of shape (N, 3).

        Returns:
            Tuple of basis tensors, one for each input ProductRepr.
            Each tensor has shape (N, num_basis, dim1, dim2).
        """
        # Compute unique bases
        ms = [comp(x) for comp in self.comps]
        # Expand to required outputs without recomputation
        return tuple(ms[ix] for ix in self.repr_ix)
