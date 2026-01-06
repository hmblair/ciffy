"""
Kabsch alignment algorithm for coordinate arrays.

Provides functions for computing optimal rotations and aligning
coordinate arrays using SVD-based alignment.
"""

from __future__ import annotations

from ..backend import is_torch, Array, svd, det, has_nan, has_inf


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
                "Kabsch alignment failed: covariance matrix contains NaN or infinity. "
                "Check input coordinates for invalid values."
            ) from e
        raise ValueError(
            "Kabsch alignment failed: SVD did not converge. "
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
) -> tuple[Array, Array, Array]:
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


__all__ = [
    "kabsch_rotation",
    "kabsch_align",
]
