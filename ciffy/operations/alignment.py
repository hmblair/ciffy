"""
Structural alignment using the Kabsch algorithm.

Provides functions for computing optimal rotations and aligning
polymer structures or coordinate arrays.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Tuple


from ..backend import is_torch, Array, svd, det, has_nan, has_inf

if TYPE_CHECKING:
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

    Raises:
        ValueError: If SVD fails due to invalid coordinates (NaN, inf, or
            extreme values causing overflow).
    """
    # Cross-covariance matrix H = X^T @ Y
    H = coords1.T @ coords2

    # SVD: H = U @ S @ Vt
    try:
        U, S, Vt = svd(H)
    except Exception as e:
        # Check for common causes and provide helpful error message
        if has_nan(H) or has_inf(H):
            raise ValueError(
                "RMSD failed: covariance matrix contains NaN or infinity. "
                "Check input coordinates for invalid values."
            ) from e
        raise ValueError(
            "RMSD failed: SVD did not converge. "
            "This may indicate extreme coordinate values causing overflow."
        ) from e

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


def intersect(
    a: "Polymer",
    b: "Polymer",
) -> Tuple["Polymer", "Polymer"]:
    """
    Return polymers containing only atoms present in both structures.

    For each residue position, finds atoms that exist in both polymers
    and returns new polymers containing only those atoms. Useful for
    computing RMSD between structures with missing or extra atoms.

    Args:
        a: First polymer.
        b: Second polymer.

    Returns:
        Tuple of (a_matched, b_matched) containing only shared atoms.

    Raises:
        ValueError: If polymers have different residue counts.

    Examples:
        >>> # RMSD with robustness to missing atoms
        >>> a_matched, b_matched = ciffy.intersect(pred, ref)
        >>> rmsd_val = ciffy.rmsd(a_matched, b_matched)

    Note:
        Atoms are matched by atom type within each residue position.
        The returned polymers have identical atom counts and ordering.
    """
    from ..biochemistry import Scale, Atom
    from ..backend import ops

    n_res_a = a.size(Scale.RESIDUE)
    n_res_b = b.size(Scale.RESIDUE)

    if n_res_a != n_res_b:
        raise ValueError(
            f"Polymers must have same residue count: {n_res_a} vs {n_res_b}"
        )

    # Get residue membership and atom types
    res_a = a.membership(Scale.RESIDUE)
    res_b = b.membership(Scale.RESIDUE)

    # Create compound keys: (residue_idx, atom_type)
    keys_a = res_a * Atom.count() + a.atoms
    keys_b = res_b * Atom.count() + b.atoms

    # Find atoms in a that have matching key in b, and vice versa
    mask_a = ops.isin(keys_a, keys_b)
    mask_b = ops.isin(keys_b, keys_a)

    # Get indices of matching atoms
    indices_a = ops.nonzero_1d(mask_a)
    indices_b = ops.nonzero_1d(mask_b)

    # Sort by key to ensure consistent ordering between polymers
    order_a = ops.argsort(keys_a[indices_a])
    order_b = ops.argsort(keys_b[indices_b])

    sorted_indices_a = indices_a[order_a]
    sorted_indices_b = indices_b[order_b]

    return a[sorted_indices_a], b[sorted_indices_b]
