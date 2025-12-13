"""
Cartesian to internal coordinate conversion.

Computes bond lengths, bond angles, and dihedral angles from Cartesian
coordinates using a Z-matrix representation.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from ..backend import Array, is_torch

if TYPE_CHECKING:
    from ..polymer import Polymer

from .graph import ZMatrixEntry, build_zmatrix, zmatrix_to_indices
from .internal_polymer import InternalPolymer

# Try to import C extension
try:
    from .._c import _cartesian_to_internal as _c_cartesian_to_internal
    _HAS_C_EXTENSION = True
except ImportError:
    _HAS_C_EXTENSION = False


def cartesian_to_internal(polymer: "Polymer") -> InternalPolymer:
    """
    Convert Cartesian coordinates to internal coordinates.

    Computes bond lengths, bond angles, and dihedral angles from the
    Cartesian coordinate representation. The conversion is differentiable
    when using PyTorch backend.

    Args:
        polymer: Polymer with Cartesian coordinates.

    Returns:
        InternalPolymer with computed internal coordinates.
    """
    coords = polymer.coordinates
    n_atoms = coords.shape[0]

    # Build Z-matrix (defines reference atoms for each coordinate)
    zmatrix = build_zmatrix(polymer)
    n_zmatrix = len(zmatrix)

    # Detect orphan atoms (not in Z-matrix - no bonds)
    zmatrix_atoms = {entry.atom_idx for entry in zmatrix}
    orphan_atoms = [i for i in range(n_atoms) if i not in zmatrix_atoms]

    # Store orphan coordinates
    if orphan_atoms:
        if is_torch(coords):
            import torch
            orphan_coords = torch.stack([coords[i] for i in orphan_atoms])
        else:
            orphan_coords = np.stack([coords[i] for i in orphan_atoms])
    else:
        orphan_coords = None

    # Use C extension if available (works for NumPy and Torch tensors)
    use_c = _HAS_C_EXTENSION

    if use_c:
        # Build indices array and ensure coords are contiguous float32 for C extension
        indices = zmatrix_to_indices(zmatrix)

        if is_torch(coords):
            import torch
            coords_f32 = coords.detach().cpu().to(torch.float32).numpy()
        else:
            coords_f32 = np.ascontiguousarray(coords, dtype=np.float32)

        # Call C extension (returns float32 NumPy arrays)
        distances_np, angles_np, dihedrals_np = _c_cartesian_to_internal(coords_f32, indices)

        if is_torch(coords):
            import torch
            distances = torch.from_numpy(distances_np).to(device=coords.device, dtype=coords.dtype)
            angles = torch.from_numpy(angles_np).to(device=coords.device, dtype=coords.dtype)
            dihedrals = torch.from_numpy(dihedrals_np).to(device=coords.device, dtype=coords.dtype)
        else:
            distances, angles, dihedrals = distances_np, angles_np, dihedrals_np
    else:
        # Python fallback (also used for PyTorch)
        if is_torch(coords):
            import torch
            distances = torch.zeros(n_zmatrix, dtype=coords.dtype, device=coords.device)
            angles = torch.zeros(n_zmatrix, dtype=coords.dtype, device=coords.device)
            dihedrals = torch.zeros(n_zmatrix, dtype=coords.dtype, device=coords.device)
        else:
            distances = np.zeros(n_zmatrix, dtype=np.float32)
            angles = np.zeros(n_zmatrix, dtype=np.float32)
            dihedrals = np.zeros(n_zmatrix, dtype=np.float32)

        # Compute internal coordinates for each atom
        for i, entry in enumerate(zmatrix):
            atom_idx = entry.atom_idx

            if entry.distance_ref >= 0:
                # Compute bond length
                distances[i] = _compute_distance(
                    coords[atom_idx],
                    coords[entry.distance_ref],
                )

            if entry.angle_ref >= 0:
                # Compute bond angle
                angles[i] = _compute_angle(
                    coords[atom_idx],
                    coords[entry.distance_ref],
                    coords[entry.angle_ref],
                )

            if entry.dihedral_ref >= 0:
                # Compute dihedral angle
                dihedrals[i] = _compute_dihedral(
                    coords[atom_idx],
                    coords[entry.distance_ref],
                    coords[entry.angle_ref],
                    coords[entry.dihedral_ref],
                )

    return InternalPolymer(
        distances=distances,
        angles=angles,
        dihedrals=dihedrals,
        zmatrix=zmatrix,
        source_polymer=polymer,
        orphan_atoms=orphan_atoms,
        orphan_coords=orphan_coords,
    )


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
