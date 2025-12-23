"""
Cremer-Pople puckering coordinates for ring conformations.

This module implements the Cremer-Pople puckering coordinates, which provide
an intrinsic (reference-free) description of ring conformations. For any given
ring geometry, the puckering parameters are uniquely determined regardless of
molecular orientation or coordinate system.

For a k-membered ring, puckering is described by:
- 5-ring: 2 parameters (q₂, φ₂) - amplitude and phase
- 6-ring: 3 parameters (Q, θ, φ) - total amplitude, polar, azimuthal

These coordinates are ideal for machine learning applications because:
1. Same geometry always gives same values (intrinsic)
2. Smooth and continuous (suitable for optimization)
3. Physically meaningful (relate to ring conformations)

Reference:
    Cremer, D.; Pople, J.A. (1975). "A general definition of ring puckering
    coordinates". J. Am. Chem. Soc. 97(6): 1354-1358.
"""

from __future__ import annotations

import numpy as np

from ..backend import Array, is_torch, to_numpy

__all__ = [
    # 5-ring puckering
    "compute_puckering_5ring",
    "apply_puckering_5ring",
    # 6-ring puckering
    "compute_puckering_6ring",
    "apply_puckering_6ring",
    # Utilities
    "flatten_ring_to_plane",
    "compute_mean_plane",
]


# =============================================================================
# Mean Plane Computation
# =============================================================================


def _get_consistent_normal(coords_np: np.ndarray) -> np.ndarray:
    """
    Get a consistently oriented normal for ring coordinates.

    Uses SVD to find the best-fit plane, then orients the normal
    using the right-hand rule (cross product of consecutive edges).
    """
    k = len(coords_np)
    center = coords_np.mean(axis=0)
    centered = coords_np - center

    # SVD to find best-fit plane
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[2]

    # Orient normal using right-hand rule
    cross_sum = np.zeros(3)
    for i in range(k):
        j = (i + 1) % k
        m = (i + 2) % k
        v1 = coords_np[j] - coords_np[i]
        v2 = coords_np[m] - coords_np[j]
        cross_sum += np.cross(v1, v2)

    if np.dot(normal, cross_sum) < 0:
        normal = -normal

    return normal


def compute_mean_plane(coords: Array) -> tuple[Array, Array]:
    """
    Compute the mean plane of a ring.

    Uses SVD to find the best-fit plane through the ring atoms.
    The plane passes through the centroid and has unit normal.

    The normal orientation is determined by the right-hand rule:
    following the ring atoms in order, the normal points according to
    the cross product of consecutive edges.

    Args:
        coords: (k, 3) coordinates of ring atoms in order

    Returns:
        (center, normal): tuple of (3,) arrays where:
        - center: centroid of ring atoms
        - normal: unit normal to the mean plane

    Notes:
        For nearly planar rings, the normal is well-defined.
        For non-planar rings, this gives the least-squares best-fit plane.
    """
    coords_np = to_numpy(coords)
    center = coords_np.mean(axis=0)
    normal = _get_consistent_normal(coords_np)

    if is_torch(coords):
        import torch
        center = torch.from_numpy(center).to(dtype=coords.dtype, device=coords.device)
        normal = torch.from_numpy(normal).to(dtype=coords.dtype, device=coords.device)

    return center, normal


def flatten_ring_to_plane(
    coords: Array,
    preserve_center: bool = True,
) -> Array:
    """
    Project ring atoms onto their mean plane.

    This creates a planar ring that preserves the in-plane structure
    but removes all out-of-plane displacements.

    Args:
        coords: (k, 3) coordinates of ring atoms
        preserve_center: If True, keep centroid at original position

    Returns:
        (k, 3) planar ring coordinates
    """
    coords_np = to_numpy(coords)
    center = coords_np.mean(axis=0)
    centered = coords_np - center

    # Get consistently oriented normal
    normal = _get_consistent_normal(coords_np)

    # Project each atom onto the plane
    # plane_point = point - (point · normal) * normal
    z = np.dot(centered, normal)
    flat = centered - np.outer(z, normal)

    if preserve_center:
        flat = flat + center

    if is_torch(coords):
        import torch
        flat = torch.from_numpy(flat).to(dtype=coords.dtype, device=coords.device)

    return flat


# =============================================================================
# 5-Ring Puckering (Envelope/Twist Conformations)
# =============================================================================


def compute_puckering_5ring(coords: Array) -> tuple[float, float]:
    """
    Compute Cremer-Pople puckering for a 5-membered ring.

    A 5-membered ring has 2 puckering degrees of freedom, described by
    amplitude q₂ and phase φ₂. The phase determines the conformation:
    - φ₂ = 0, π/2, π, ... → envelope (E) conformations
    - φ₂ = π/10, 3π/10, ... → twist (T) conformations

    The out-of-plane displacements follow:
        z_j = q₂ · √(2/5) · cos(φ₂ + 4πj/5)

    Args:
        coords: (5, 3) coordinates of ring atoms in cyclic order

    Returns:
        (q2, phi2): puckering amplitude (Å) and phase (radians in [-π, π])

    Raises:
        ValueError: If coords doesn't have shape (5, 3)

    Example:
        >>> # C2'-endo ribose has φ₂ ≈ π (162°)
        >>> # C3'-endo ribose has φ₂ ≈ 0 (18°)
        >>> q2, phi2 = compute_puckering_5ring(ribose_coords)
    """
    coords_np = to_numpy(coords)

    if coords_np.shape != (5, 3):
        raise ValueError(f"Expected (5, 3) array, got {coords_np.shape}")

    # Get consistently oriented normal
    center = coords_np.mean(axis=0)
    centered = coords_np - center
    normal = _get_consistent_normal(coords_np)

    # Out-of-plane displacements
    z = np.dot(centered, normal)

    # Cremer-Pople extraction formulas for 5-ring (m=2 mode):
    # A = Σ_j z_j cos(4πj/5)
    # B = -Σ_j z_j sin(4πj/5)
    # q₂ = √(2/5) · √(A² + B²)
    # φ₂ = atan2(B, A)
    angles = 4.0 * np.pi * np.arange(5) / 5.0
    A = np.sum(z * np.cos(angles))
    B = -np.sum(z * np.sin(angles))

    q2 = np.sqrt(2.0 / 5.0) * np.sqrt(A**2 + B**2)
    phi2 = np.arctan2(B, A)

    return float(q2), float(phi2)


def apply_puckering_5ring(
    flat_coords: Array,
    q2: float,
    phi2: float,
) -> Array:
    """
    Apply Cremer-Pople puckering to a planar 5-ring.

    Given a planar ring and puckering parameters, this computes the
    puckered coordinates by displacing atoms out of the plane.

    Args:
        flat_coords: (5, 3) planar ring coordinates (or nearly planar)
        q2: puckering amplitude in Angstroms (typically 0.3-0.5 Å for ribose)
        phi2: puckering phase in radians

    Returns:
        (5, 3) puckered ring coordinates

    Raises:
        ValueError: If flat_coords doesn't have shape (5, 3)

    Notes:
        The input doesn't need to be perfectly planar - any existing
        out-of-plane components are replaced with the new puckering.
    """
    coords_np = to_numpy(flat_coords)

    if coords_np.shape != (5, 3):
        raise ValueError(f"Expected (5, 3) array, got {coords_np.shape}")

    # Get consistently oriented normal
    center = coords_np.mean(axis=0)
    centered = coords_np - center
    normal = _get_consistent_normal(coords_np)

    # Flatten by removing current out-of-plane component
    z_current = np.dot(centered, normal)
    flat = centered - np.outer(z_current, normal)

    # Compute new out-of-plane displacements from puckering params
    # z_j = q₂ · √(2/5) · cos(φ₂ + 4πj/5)
    angles = 4.0 * np.pi * np.arange(5) / 5.0
    z_new = q2 * np.sqrt(2.0 / 5.0) * np.cos(phi2 + angles)

    # Apply displacements along normal
    puckered = flat + np.outer(z_new, normal) + center

    if is_torch(flat_coords):
        import torch
        puckered = torch.from_numpy(puckered).to(
            dtype=flat_coords.dtype, device=flat_coords.device
        )

    return puckered


# =============================================================================
# 6-Ring Puckering (Chair/Boat/Twist-Boat Conformations)
# =============================================================================


def compute_puckering_6ring(coords: Array) -> tuple[float, float, float]:
    """
    Compute Cremer-Pople puckering for a 6-membered ring.

    A 6-membered ring has 3 puckering degrees of freedom, described in
    spherical-like coordinates (Q, θ, φ):
    - Q: total puckering amplitude (Å)
    - θ: polar angle (radians), distinguishes chair vs boat/twist
    - φ: azimuthal angle (radians), orientation of boat/twist

    Conformational interpretation:
    - θ = 0: 4C1 chair
    - θ = π: 1C4 chair (inverted)
    - θ = π/2: boat/twist-boat family (φ determines which)
    - φ = 0, π/3, 2π/3, ...: boat conformations
    - φ = π/6, π/2, 5π/6, ...: twist-boat conformations

    The out-of-plane displacements follow:
        z_j = √(1/3) · q₂ · cos(φ + 2πj/3) + √(1/6) · q₃ · (-1)^j

    where Q² = q₂² + q₃², θ = arccos(q₃/Q), and φ is the phase of q₂.

    Args:
        coords: (6, 3) coordinates of ring atoms in cyclic order

    Returns:
        (Q, theta, phi): total amplitude (Å), polar angle, azimuthal angle

    Raises:
        ValueError: If coords doesn't have shape (6, 3)
    """
    coords_np = to_numpy(coords)

    if coords_np.shape != (6, 3):
        raise ValueError(f"Expected (6, 3) array, got {coords_np.shape}")

    # Get consistently oriented normal
    center = coords_np.mean(axis=0)
    centered = coords_np - center
    normal = _get_consistent_normal(coords_np)

    # Out-of-plane displacements
    z = np.dot(centered, normal)

    # Cremer-Pople extraction for 6-ring:
    # m=2 mode (twist-boat): uses 2πj·2/6 = 2πj/3
    # m=3 mode (chair): uses (-1)^j pattern

    # Twist-boat mode (q₂, φ₂)
    angles = 2.0 * np.pi * np.arange(6) / 3.0  # 2π·m·j/N with m=2, N=6
    A2 = np.sum(z * np.cos(angles))
    B2 = -np.sum(z * np.sin(angles))
    q2 = np.sqrt(1.0 / 3.0) * np.sqrt(A2**2 + B2**2)
    phi2 = np.arctan2(B2, A2)

    # Chair mode (q₃)
    alternating = np.array([1, -1, 1, -1, 1, -1], dtype=np.float64)
    q3 = np.sqrt(1.0 / 6.0) * np.sum(z * alternating)

    # Convert to spherical coordinates
    Q = np.sqrt(q2**2 + q3**2)

    if Q < 1e-10:
        # Nearly planar ring
        theta = 0.0
        phi = 0.0
    else:
        # Clamp for numerical stability
        cos_theta = np.clip(q3 / Q, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        phi = phi2

    return float(Q), float(theta), float(phi)


def apply_puckering_6ring(
    flat_coords: Array,
    Q: float,
    theta: float,
    phi: float,
) -> Array:
    """
    Apply Cremer-Pople puckering to a planar 6-ring.

    Given a planar ring and puckering parameters, this computes the
    puckered coordinates by displacing atoms out of the plane.

    Args:
        flat_coords: (6, 3) planar ring coordinates (or nearly planar)
        Q: total puckering amplitude in Angstroms (typically 0.5-0.6 Å)
        theta: polar angle in radians (0 = chair, π/2 = boat)
        phi: azimuthal angle in radians

    Returns:
        (6, 3) puckered ring coordinates

    Raises:
        ValueError: If flat_coords doesn't have shape (6, 3)

    Notes:
        The input doesn't need to be perfectly planar - any existing
        out-of-plane components are replaced with the new puckering.
    """
    coords_np = to_numpy(flat_coords)

    if coords_np.shape != (6, 3):
        raise ValueError(f"Expected (6, 3) array, got {coords_np.shape}")

    # Get consistently oriented normal
    center = coords_np.mean(axis=0)
    centered = coords_np - center
    normal = _get_consistent_normal(coords_np)

    # Flatten by removing current out-of-plane component
    z_current = np.dot(centered, normal)
    flat = centered - np.outer(z_current, normal)

    # Convert spherical to q₂, q₃
    q2 = Q * np.sin(theta)
    q3 = Q * np.cos(theta)

    # Compute new out-of-plane displacements
    # z_j = √(1/3) · q₂ · cos(φ + 2πj/3) + √(1/6) · q₃ · (-1)^j
    angles = 2.0 * np.pi * np.arange(6) / 3.0
    alternating = np.array([1, -1, 1, -1, 1, -1], dtype=np.float64)

    z_new = (
        np.sqrt(1.0 / 3.0) * q2 * np.cos(phi + angles) +
        np.sqrt(1.0 / 6.0) * q3 * alternating
    )

    # Apply displacements along normal
    puckered = flat + np.outer(z_new, normal) + center

    if is_torch(flat_coords):
        import torch
        puckered = torch.from_numpy(puckered).to(
            dtype=flat_coords.dtype, device=flat_coords.device
        )

    return puckered
