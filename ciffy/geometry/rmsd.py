"""
Coordinate-level RMSD computation using Kabsch alignment.

Provides functions for computing root mean square deviation between
coordinate arrays, with support for batched inputs.
"""

from __future__ import annotations

import numpy as np

from ..backend import Array, is_torch, svd, det, has_nan, has_inf, sqrt, clamp, clone


def rmsd_coords(coords1: Array, coords2: Array, eps: float = 0.0) -> Array:
    """
    Compute Kabsch-aligned RMSD between coordinate sets.

    Supports both single pairs (N, 3) and batched inputs (B, N, 3).

    Args:
        coords1: First coordinates, shape (N, 3) or (B, N, 3).
        coords2: Second coordinates, shape (N, 3) or (B, N, 3).
        eps: Small value added before sqrt for gradient stability.

    Returns:
        RMSD value(s). Shape () for 2D input, (B,) for 3D input.
    """
    if type(coords1) != type(coords2):
        raise TypeError(f"Both inputs must be same type, got {type(coords1).__name__} and {type(coords2).__name__}")

    if coords1.shape != coords2.shape:
        raise ValueError(f"Shape mismatch: {coords1.shape} vs {coords2.shape}")

    if coords1.ndim == 2:
        if coords1.shape[1] != 3:
            raise ValueError(f"Expected shape (N, 3), got {coords1.shape}")
    elif coords1.ndim == 3:
        if coords1.shape[2] != 3:
            raise ValueError(f"Expected shape (B, N, 3), got {coords1.shape}")
    else:
        raise ValueError(f"Expected 2D (N, 3) or 3D (B, N, 3) array, got {coords1.ndim}D")

    # Handle both 2D and 3D by adding batch dim if needed
    squeeze = coords1.ndim == 2
    if squeeze:
        coords1 = coords1[None, :, :]  # (1, N, 3)
        coords2 = coords2[None, :, :]

    # Center coordinates: mean over atoms (axis=1)
    if is_torch(coords1):
        centroid1 = coords1.mean(dim=1, keepdim=True)
        centroid2 = coords2.mean(dim=1, keepdim=True)
    else:
        centroid1 = coords1.mean(axis=1, keepdims=True)
        centroid2 = coords2.mean(axis=1, keepdims=True)

    c1 = coords1 - centroid1  # (B, N, 3)
    c2 = coords2 - centroid2  # (B, N, 3)

    # Cross-covariance: H = c1^T @ c2, batched as (B, 3, N) @ (B, N, 3) -> (B, 3, 3)
    if is_torch(c1):
        H = c1.transpose(-2, -1) @ c2
    else:
        H = np.swapaxes(c1, -2, -1) @ c2

    # Batched SVD
    try:
        U, S, Vt = svd(H)
    except Exception as e:
        if has_nan(H) or has_inf(H):
            raise ValueError(
                "RMSD failed: covariance matrix contains NaN or infinity. "
                "Check input coordinates for invalid values."
            ) from e
        raise ValueError(
            "RMSD failed: SVD did not converge. "
            "This may indicate extreme coordinate values causing overflow."
        ) from e

    # Optimal rotation: R = V @ U^T -> (B, 3, 3)
    if is_torch(Vt):
        R = Vt.transpose(-2, -1) @ U.transpose(-2, -1)
    else:
        R = np.swapaxes(Vt, -2, -1) @ np.swapaxes(U, -2, -1)

    # Handle reflection case (det(R) < 0)
    d = det(R)
    Vt = clone(Vt, detach=False)  # Allow gradients through SVD
    if is_torch(Vt):
        mask = d < 0
        Vt[mask, -1, :] *= -1
        R = Vt.transpose(-2, -1) @ U.transpose(-2, -1)
    else:
        mask = d < 0
        Vt[mask, -1, :] *= -1
        R = np.swapaxes(Vt, -2, -1) @ np.swapaxes(U, -2, -1)

    # Apply rotation: aligned = c1 @ R^T + centroid2
    if is_torch(R):
        aligned = c1 @ R.transpose(-2, -1) + centroid2
    else:
        aligned = c1 @ np.swapaxes(R, -2, -1) + centroid2

    # Compute RMSD
    diff = aligned - coords2
    if is_torch(diff):
        msd = (diff ** 2).mean(dim=(-2, -1))  # Mean over N and 3
    else:
        msd = (diff ** 2).mean(axis=(-2, -1))

    result = sqrt(clamp(msd, min_val=0.0) + eps)

    # Remove batch dim if input was 2D
    if squeeze:
        if is_torch(result):
            result = result.squeeze(0)
        else:
            result = result.squeeze()

    return result


__all__ = [
    "rmsd_coords",
]
