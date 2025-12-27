"""
Structural alignment using the Kabsch algorithm.

Provides functions for computing optimal rotations and RMSD between
polymer structures or coordinate arrays.
"""

from __future__ import annotations
from functools import singledispatch
from typing import TYPE_CHECKING, Tuple, overload

import numpy as np

from ..backend import is_torch, Array, svd, svdvals, det, multiply

if TYPE_CHECKING:
    import torch
    from ..polymer import Polymer
    from ..biochemistry import Scale


# =============================================================================
# Core Kabsch alignment functions (work on raw coordinate arrays)
# =============================================================================

def kabsch_rotation(coords1: Array, coords2: Array) -> Array:
    """
    Compute the optimal rotation matrix to align coords1 onto coords2.

    Uses the Kabsch algorithm (SVD of the cross-covariance matrix) to find
    the rotation that minimizes RMSD. Coordinates should be pre-centered.

    Args:
        coords1: First coordinate set, shape (N, 3). Should be centered.
        coords2: Second coordinate set, shape (N, 3). Should be centered.

    Returns:
        Rotation matrix R of shape (3, 3). Apply as: coords1_aligned = coords1 @ R.T
    """
    # Cross-covariance matrix H = X^T @ Y
    H = coords1.T @ coords2

    # SVD: H = U @ S @ Vt
    U, S, Vt = svd(H)

    # Optimal rotation: R = V @ U^T
    R = Vt.T @ U.T

    # Handle reflection case (det(R) = -1)
    if det(R) < 0:
        if is_torch(Vt):
            Vt = Vt.clone()
        else:
            Vt = Vt.copy()
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R


def kabsch_align(
    coords1: Array,
    coords2: Array,
    center: bool = True,
) -> Tuple[Array, Array, Array]:
    """
    Align coords1 onto coords2 using the Kabsch algorithm.

    Args:
        coords1: Coordinates to transform, shape (N, 3).
        coords2: Target coordinates, shape (N, 3).
        center: Whether to center coordinates before alignment.

    Returns:
        Tuple of (aligned_coords1, rotation_matrix, translation).
        - aligned_coords1: Transformed coords1, shape (N, 3)
        - rotation_matrix: Optimal rotation R, shape (3, 3)
        - translation: Centroid of coords2, shape (3,)
    """
    if is_torch(coords1):
        mean_fn = lambda x: x.mean(dim=0)
    else:
        mean_fn = lambda x: x.mean(axis=0)

    if center:
        centroid1 = mean_fn(coords1)
        centroid2 = mean_fn(coords2)
        coords1_centered = coords1 - centroid1
        coords2_centered = coords2 - centroid2
    else:
        centroid2 = mean_fn(coords2) * 0  # Zero translation
        coords1_centered = coords1
        coords2_centered = coords2

    # Compute optimal rotation
    R = kabsch_rotation(coords1_centered, coords2_centered)

    # Apply rotation and translate to target centroid
    aligned = coords1_centered @ R.T + centroid2

    return aligned, R, centroid2


def _rmsd_single(coords1: Array, coords2: Array) -> float:
    """
    Compute Kabsch-aligned RMSD between two coordinate sets.

    This is the core RMSD computation used by both Polymer and array versions.

    Args:
        coords1: First coordinates, shape (N, 3).
        coords2: Second coordinates, shape (N, 3).

    Returns:
        RMSD value (float).
    """
    aligned, _, _ = kabsch_align(coords1, coords2, center=True)
    diff = aligned - coords2

    if is_torch(diff):
        import torch
        msd = (diff ** 2).mean()
        return torch.sqrt(torch.clamp(msd, min=0.0)).item()
    else:
        msd = (diff ** 2).mean()
        return float(np.sqrt(max(msd, 0.0)))


# =============================================================================
# Unified RMSD interface using singledispatch
# =============================================================================

# Type stubs for static type checking
@overload
def rmsd(a: Array, b: Array, scale: None = None) -> Array | float: ...

@overload
def rmsd(a: "Polymer", b: "Polymer", scale: "Scale | None" = None) -> Array: ...


@singledispatch
def rmsd(a, b, scale=None):
    """
    Compute Kabsch-aligned RMSD between structures or coordinate arrays.

    This function dispatches based on the input type:
    - Polymer objects: Uses optimized hierarchical computation
    - numpy/torch arrays: Direct coordinate-based computation

    Args:
        a: First structure (Polymer) or coordinates (array).
        b: Second structure (Polymer) or coordinates (array).
        scale: For Polymer inputs, the scale at which to compute RMSD
            (default: MOLECULE). Ignored for array inputs.

    Returns:
        For Polymer: Array of RMSD values, one per scale unit.
        For arrays (N, 3): Single RMSD value (float).
        For arrays (B, N, 3): Array of B RMSD values.

    Examples:
        >>> import ciffy
        >>> # Polymer RMSD
        >>> p1 = ciffy.load("struct1.cif")
        >>> p2 = ciffy.load("struct2.cif")
        >>> rmsd_val = ciffy.rmsd(p1, p2)
        >>>
        >>> # Array RMSD (single pair)
        >>> coords1 = np.random.randn(100, 3)
        >>> coords2 = np.random.randn(100, 3)
        >>> rmsd_val = ciffy.rmsd(coords1, coords2)
        >>>
        >>> # Batched array RMSD
        >>> batch1 = np.random.randn(32, 100, 3)
        >>> batch2 = np.random.randn(32, 100, 3)
        >>> rmsd_vals = ciffy.rmsd(batch1, batch2)  # shape (32,)
    """
    raise TypeError(
        f"rmsd() not supported for type {type(a).__name__}. "
        f"Expected Polymer or numpy/torch array."
    )


@rmsd.register(np.ndarray)
def _rmsd_ndarray(a: np.ndarray, b: np.ndarray, scale=None) -> np.ndarray | float:
    """RMSD for numpy arrays."""
    if not isinstance(b, np.ndarray):
        raise TypeError(f"Both inputs must be numpy arrays, got {type(b).__name__}")

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    if a.ndim == 2:
        # Single pair: (N, 3)
        if a.shape[1] != 3:
            raise ValueError(f"Expected shape (N, 3), got {a.shape}")
        return _rmsd_single(a, b)

    elif a.ndim == 3:
        # Batch: (B, N, 3)
        if a.shape[2] != 3:
            raise ValueError(f"Expected shape (B, N, 3), got {a.shape}")
        return np.array([_rmsd_single(c1, c2) for c1, c2 in zip(a, b)])

    else:
        raise ValueError(f"Expected 2D (N, 3) or 3D (B, N, 3) array, got {a.ndim}D")


# Register torch.Tensor if available
try:
    import torch

    @rmsd.register(torch.Tensor)
    def _rmsd_tensor(a: torch.Tensor, b: torch.Tensor, scale=None) -> torch.Tensor | float:
        """RMSD for torch tensors."""
        if not isinstance(b, torch.Tensor):
            raise TypeError(f"Both inputs must be torch tensors, got {type(b).__name__}")

        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

        if a.ndim == 2:
            # Single pair: (N, 3)
            if a.shape[1] != 3:
                raise ValueError(f"Expected shape (N, 3), got {a.shape}")
            return _rmsd_single(a, b)

        elif a.ndim == 3:
            # Batch: (B, N, 3)
            if a.shape[2] != 3:
                raise ValueError(f"Expected shape (B, N, 3), got {a.shape}")
            return torch.tensor([_rmsd_single(c1, c2) for c1, c2 in zip(a, b)])

        else:
            raise ValueError(f"Expected 2D (N, 3) or 3D (B, N, 3) tensor, got {a.ndim}D")

except ImportError:
    pass


# =============================================================================
# Polymer-specific functions
# =============================================================================

def coordinate_covariance(
    polymer1: "Polymer",
    polymer2: "Polymer",
    scale: "Scale",
) -> Array:
    """
    Compute coordinate covariance matrices between two polymers.

    The covariance is computed by taking the outer product of coordinates
    and reducing at the specified scale.

    Args:
        polymer1: First polymer structure.
        polymer2: Second polymer structure (must have same atom count).
        scale: Scale at which to compute covariance (e.g., MOLECULE).

    Returns:
        Array of covariance matrices, one per scale unit.
    """
    outer_prod = multiply(
        polymer1.coordinates[:, None, :],
        polymer2.coordinates[:, :, None],
    )
    return polymer1.reduce(outer_prod, scale)


def _rmsd_polymer(
    polymer1: "Polymer",
    polymer2: "Polymer",
    scale: "Scale | None" = None,
) -> Array:
    """
    Compute Kabsch distance (aligned RMSD) between polymer structures.

    Uses singular value decomposition to find the optimal rotation
    that minimizes the distance between the two structures. The
    polymers must have the same number of atoms and atom ordering.

    Args:
        polymer1: First polymer structure.
        polymer2: Second polymer structure.
        scale: Scale at which to compute distance. Default is MOLECULE.

    Returns:
        Array of RMSD values (Angstroms), one per scale unit.

    Note:
        Single-atom molecules (ions, water) produce degenerate covariance
        matrices, leading to numerical instability in the SVD. Use .poly()
        to exclude non-polymer atoms before computing RMSD if your structure
        contains such molecules.
    """
    from ..biochemistry import Scale

    if scale is None:
        scale = Scale.MOLECULE

    # Center both structures
    polymer1_c, _ = polymer1.center(scale)
    polymer2_c, _ = polymer2.center(scale)

    # Compute coordinate covariance
    cov = coordinate_covariance(polymer1_c, polymer2_c, scale)

    # SVD to find optimal rotation
    sigma = svdvals(cov)
    cov_det = det(cov)

    # Handle reflection case
    if is_torch(sigma):
        import torch
        sigma = sigma.clone()
        sigma[cov_det < 0, -1] = -sigma[cov_det < 0, -1]
    else:
        sigma = sigma.copy()
        sigma[cov_det < 0, -1] = -sigma[cov_det < 0, -1]
    sigma = sigma.mean(-1)

    # Get variances of both point clouds
    var1 = polymer1_c.moment(2, scale).mean(-1)
    var2 = polymer2_c.moment(2, scale).mean(-1)

    # Compute Kabsch distance (RMSD)
    msd = var1 + var2 - 2 * sigma
    if is_torch(msd):
        import torch
        return torch.sqrt(torch.clamp(msd, min=0.0))
    else:
        return np.sqrt(np.maximum(msd, 0.0))


# Register Polymer type - must be done after Polymer is importable
# We use a lazy registration pattern to avoid circular imports
_polymer_registered = False
_rmsd_dispatch = rmsd  # Keep reference to singledispatch object


def _ensure_polymer_registered():
    """Lazily register Polymer type with rmsd dispatcher."""
    global _polymer_registered
    if _polymer_registered:
        return

    from ..polymer import Polymer
    _rmsd_dispatch.register(Polymer, _rmsd_polymer)
    _polymer_registered = True


# Type stubs for the public wrapper function
@overload
def rmsd(a: Array, b: Array, scale: None = None) -> Array | float: ...

@overload
def rmsd(a: "Polymer", b: "Polymer", scale: "Scale | None" = None) -> Array: ...


def rmsd(a: "Array | Polymer", b: "Array | Polymer", scale: "Scale | None" = None) -> Array | float:
    """
    Compute Kabsch-aligned RMSD between structures or coordinate arrays.

    This function dispatches based on the input type:
    - Polymer objects: Uses optimized hierarchical computation
    - numpy/torch arrays: Direct coordinate-based computation

    Args:
        a: First structure (Polymer) or coordinates (array).
        b: Second structure (Polymer) or coordinates (array).
        scale: For Polymer inputs, the scale at which to compute RMSD
            (default: MOLECULE). Ignored for array inputs.

    Returns:
        For Polymer: Array of RMSD values, one per scale unit.
        For arrays (N, 3): Single RMSD value (float).
        For arrays (B, N, 3): Array of B RMSD values.

    Examples:
        >>> import ciffy
        >>> # Polymer RMSD
        >>> p1 = ciffy.load("struct1.cif")
        >>> p2 = ciffy.load("struct2.cif")
        >>> rmsd_val = ciffy.rmsd(p1, p2)
        >>>
        >>> # Array RMSD (single pair)
        >>> coords1 = np.random.randn(100, 3)
        >>> coords2 = np.random.randn(100, 3)
        >>> rmsd_val = ciffy.rmsd(coords1, coords2)
        >>>
        >>> # Batched array RMSD
        >>> batch1 = np.random.randn(32, 100, 3)
        >>> batch2 = np.random.randn(32, 100, 3)
        >>> rmsd_vals = ciffy.rmsd(batch1, batch2)  # shape (32,)
    """
    _ensure_polymer_registered()
    return _rmsd_dispatch(a, b, scale)


# Legacy alias for backwards compatibility
kabsch_distance = rmsd


def align(
    polymer1: "Polymer",
    polymer2: "Polymer",
    scale: "Scale | None" = None,
) -> Tuple["Polymer", "Polymer"]:
    """
    Align two polymer structures using the Kabsch algorithm.

    Computes the optimal rotation to superimpose polymer2 onto polymer1,
    returning both polymers with polymer2 transformed.

    Args:
        polymer1: Reference polymer (unchanged).
        polymer2: Mobile polymer (will be aligned to polymer1).
        scale: Scale at which to compute alignment. Default is MOLECULE.
            Use CHAIN to align each chain independently.

    Returns:
        Tuple of (polymer1, aligned_polymer2).
        - polymer1: Unchanged reference structure.
        - aligned_polymer2: polymer2 rotated and translated to minimize
            RMSD with polymer1.

    Examples:
        >>> import ciffy
        >>> p1 = ciffy.load("reference.cif")
        >>> p2 = ciffy.load("mobile.cif")
        >>> ref, aligned = ciffy.align(p1, p2)
        >>> rmsd = ciffy.rmsd(ref, aligned)  # Should be minimal

    Note:
        Both polymers must have the same number of atoms and atom ordering.
        For per-chain alignment, use scale=ciffy.CHAIN.
    """
    from copy import copy
    from ..biochemistry import Scale

    if scale is None:
        scale = Scale.MOLECULE

    if polymer1.size() != polymer2.size():
        raise ValueError(
            f"Polymers must have same size: {polymer1.size()} vs {polymer2.size()}"
        )

    # Get coordinates
    coords1 = polymer1.coordinates
    coords2 = polymer2.coordinates

    # Align polymer2 coordinates onto polymer1
    aligned_coords, _, _ = kabsch_align(coords2, coords1, center=True)

    # Create new polymer with aligned coordinates
    aligned_polymer2 = copy(polymer2)
    aligned_polymer2.coordinates = aligned_coords

    return polymer1, aligned_polymer2
