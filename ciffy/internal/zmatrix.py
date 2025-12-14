"""
Internal coordinate computation helpers.

Provides functions for computing bond lengths, bond angles, and dihedral angles
from Cartesian coordinates. These are used by CoordinateManager for converting
between Cartesian and internal coordinate representations.
"""

from __future__ import annotations

import numpy as np

from ..backend import Array, is_torch


def _compute_distance(p1: Array, p2: Array) -> Array:
    """
    Compute Euclidean distance between two points.

    Args:
        p1: First point (3,).
        p2: Second point (3,).

    Returns:
        Scalar distance.
    """
    diff = p1 - p2
    if is_torch(diff):
        import torch
        return torch.sqrt((diff ** 2).sum())
    return np.sqrt((diff ** 2).sum())


def _compute_angle(p1: Array, p2: Array, p3: Array) -> Array:
    """
    Compute bond angle at p2.

    The angle is computed as the angle between vectors (p1-p2) and (p3-p2).

    Args:
        p1: First point (3,).
        p2: Vertex point (3,).
        p3: Third point (3,).

    Returns:
        Angle in radians [0, pi].
    """
    v1 = p1 - p2
    v2 = p3 - p2

    if is_torch(v1):
        import torch
        # Normalize to avoid numerical issues
        v1_norm = torch.norm(v1)
        v2_norm = torch.norm(v2)
        cos_angle = (v1 * v2).sum() / (v1_norm * v2_norm + 1e-8)
        return torch.acos(torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7))
    else:
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm + 1e-8)
        return np.arccos(np.clip(cos_angle, -1.0, 1.0))


def _compute_dihedral(p1: Array, p2: Array, p3: Array, p4: Array) -> Array:
    """
    Compute dihedral (torsion) angle for four points.

    The dihedral angle is the angle between the planes defined by
    (p1, p2, p3) and (p2, p3, p4). Uses atan2 for numerical stability
    and correct quadrant determination.

    Args:
        p1: First point (3,).
        p2: Second point (3,).
        p3: Third point (3,).
        p4: Fourth point (3,).

    Returns:
        Dihedral angle in radians [-pi, pi].
    """
    # Bond vectors
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    if is_torch(b1):
        import torch

        # Normal vectors to planes
        n1 = torch.cross(b1, b2, dim=-1)
        n2 = torch.cross(b2, b3, dim=-1)

        # Normalize
        n1_norm = torch.norm(n1) + 1e-8
        n2_norm = torch.norm(n2) + 1e-8
        n1 = n1 / n1_norm
        n2 = n2 / n2_norm

        # Calculate m1 = n1 x b2_normalized
        b2_norm = torch.norm(b2) + 1e-8
        b2_unit = b2 / b2_norm
        m1 = torch.cross(n1, b2_unit, dim=-1)

        # atan2(y, x) where y = n2 . m1, x = n2 . n1
        x = (n1 * n2).sum()
        y = (m1 * n2).sum()

        return torch.atan2(y, x)
    else:
        # NumPy version
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        n1_norm = np.linalg.norm(n1) + 1e-8
        n2_norm = np.linalg.norm(n2) + 1e-8
        n1 = n1 / n1_norm
        n2 = n2 / n2_norm

        b2_norm = np.linalg.norm(b2) + 1e-8
        b2_unit = b2 / b2_norm
        m1 = np.cross(n1, b2_unit)

        x = np.dot(n1, n2)
        y = np.dot(m1, n2)

        return np.arctan2(y, x)
