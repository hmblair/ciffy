"""
Structural alignment using the Kabsch algorithm.

Provides functions for computing optimal rotations and RMSD between
polymer structures.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Tuple

import numpy as np

from ..backend import is_torch, Array, svd, svdvals, det, multiply

if TYPE_CHECKING:
    from ..polymer import Polymer
    from ..types import Scale


# =============================================================================
# Core Kabsch alignment functions
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


def kabsch_distance(
    polymer1: "Polymer",
    polymer2: "Polymer",
    scale: "Scale" = None,
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
    from ..types import Scale

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


def align(
    polymer1: "Polymer",
    polymer2: "Polymer",
    scale: "Scale" = None,
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
    from ..types import Scale

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
