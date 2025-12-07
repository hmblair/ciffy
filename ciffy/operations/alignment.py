"""
Structural alignment using the Kabsch algorithm.

Provides functions for computing optimal rotations and RMSD between
polymer structures.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from ..backend import is_torch, Array

if TYPE_CHECKING:
    from ..polymer import Polymer
    from ..types import Scale


def _svdvals(arr: Array) -> Array:
    """Compute singular values, backend-agnostic."""
    if is_torch(arr):
        import torch
        return torch.linalg.svdvals(arr)
    # NumPy returns (U, S, Vh), we just want S
    return np.linalg.svd(arr, compute_uv=False)


def _det(arr: Array) -> Array:
    """Compute determinant, backend-agnostic."""
    if is_torch(arr):
        import torch
        return torch.linalg.det(arr)
    return np.linalg.det(arr)


def _multiply(a: Array, b: Array) -> Array:
    """Element-wise multiplication, backend-agnostic."""
    if is_torch(a):
        import torch
        return torch.multiply(a, b)
    return np.multiply(a, b)


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
    outer_prod = _multiply(
        polymer1.coordinates[:, None, :],
        polymer2.coordinates[:, :, None],
    )
    return polymer1.reduce(outer_prod, scale)


def kabsch_distance(
    polymer1: "Polymer",
    polymer2: "Polymer",
    scale: "Scale",
) -> Array:
    """
    Compute Kabsch distance (aligned RMSD) between polymer structures.

    Uses singular value decomposition to find the optimal rotation
    that minimizes the distance between the two structures. The
    polymers must have the same number of atoms and atom ordering.

    Args:
        polymer1: First polymer structure.
        polymer2: Second polymer structure.
        scale: Scale at which to compute distance (e.g., MOLECULE).

    Returns:
        Array of squared distances, one per scale unit.

    Note:
        The returned value is the squared distance. Take sqrt() for RMSD.
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
    sigma = _svdvals(cov)
    det = _det(cov)

    # Handle reflection case
    if is_torch(sigma):
        sigma = sigma.clone()
        sigma[det < 0, -1] = -sigma[det < 0, -1]
    else:
        sigma = sigma.copy()
        sigma[det < 0, -1] = -sigma[det < 0, -1]
    sigma = sigma.mean(-1)

    # Get variances of both point clouds
    var1 = polymer1_c.moment(2, scale).mean(-1)
    var2 = polymer2_c.moment(2, scale).mean(-1)

    # Compute Kabsch distance
    return var1 + var2 - 2 * sigma
