"""
SE(3) transforms and residue frame computation for molecular modeling.

This module provides functions for:
- SE(3) transform operations (rotation to/from axis-angle, relative transforms)
- Residue frame computation (reference frames at specific atoms)
- Residue positioning (aligning residues for chain building)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..backend import Array, is_torch
from ..backend.ops import to_backend, stack, unsqueeze, sin, acos, clamp

if TYPE_CHECKING:
    from ..biochemistry import Residue
    from ..biochemistry.linking import FrameDefinition
from .primitives import cross, dot, normalize, clone, to_scalar


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class LocalCoordinates:
    """Coordinates in a local frame with SE(3) transform to position globally.

    Used when building polymer chains autoregressively. The coordinates
    are expressed in a local reference frame, and the transform positions
    them relative to the previous residue's linking frame.

    Attributes:
        coordinates: (n_atoms, 3) atom positions in local frame.
        transform: (6,) SE(3) as [axis_angle_x, axis_angle_y, axis_angle_z,
            translation_x, translation_y, translation_z]. The first 3 elements
            are the axis-angle rotation (direction is axis, magnitude is angle
            in radians). The last 3 are the translation vector.

    Example:
        >>> from ciffy.geometry import LocalCoordinates
        >>> local = LocalCoordinates(coords, transform)
        >>> polymer = polymer.append(Residue.A, local)
    """

    coordinates: "Array"
    transform: "Array"


# =============================================================================
# SE(3) Transform Operations
# =============================================================================


def _eye3(like: Array) -> Array:
    """3x3 identity matrix matching backend of reference array."""
    if is_torch(like):
        import torch
        return torch.eye(3, dtype=like.dtype, device=like.device)
    return np.eye(3, dtype=like.dtype)


def _trace(R: Array) -> Array:
    """Matrix trace."""
    if is_torch(R):
        import torch
        return torch.trace(R)
    return np.trace(R)


def _acos_safe(x: Array) -> Array:
    """Arccos with safe clamping."""
    if is_torch(x):
        import torch
        return torch.acos(torch.clamp(x, -1.0, 1.0))
    return np.arccos(np.clip(x, -1.0, 1.0))


def rotation_to_axis_angle(R: Array) -> Array:
    """
    Convert rotation matrix to axis-angle representation.

    Uses the Rodrigues formula inverse.

    Args:
        R: (3, 3) single rotation matrix or (..., 3, 3) batch of matrices.

    Returns:
        (3,) or (..., 3) axis-angle vector(s) where direction is axis
        and magnitude is angle.
    """
    single_input = R.ndim == 2
    if single_input:
        R = R[None]

    batch_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)  # (N, 3, 3)

    # angle = acos((trace(R) - 1) / 2)
    traces = R_flat[..., 0, 0] + R_flat[..., 1, 1] + R_flat[..., 2, 2]
    angles = acos(clamp((traces - 1) / 2, -1.0, 1.0))

    # axis from skew-symmetric part: [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]
    axis = stack([
        R_flat[..., 2, 1] - R_flat[..., 1, 2],
        R_flat[..., 0, 2] - R_flat[..., 2, 0],
        R_flat[..., 1, 0] - R_flat[..., 0, 1],
    ], axis=-1)
    axis = axis / (2 * sin(angles)[..., None] + 1e-8)

    result = axis * angles[..., None]
    result = result.reshape(*batch_shape, 3)

    if single_input:
        result = result[0]

    return result


def rodrigues(axis_angles: Array) -> Array:
    """
    Convert axis-angle vectors to rotation matrices (Rodrigues' formula).

    R = I + sin(θ)K + (1-cos(θ))K²

    where K is the skew-symmetric matrix of the unit axis.

    Backend-agnostic implementation that handles both single and batched inputs.

    Args:
        axis_angles: Either (3,) single axis-angle vector or (n, 3) batch of vectors.
            Direction is the rotation axis, magnitude is the rotation angle.

    Returns:
        (3, 3) rotation matrix for single input, or (n, 3, 3) for batched input.
    """
    from ..backend import (
        norm as backend_norm, sin, cos, eye, zeros_nd, ones_like, where,
        unsqueeze, expand,
    )

    # Handle single vs batched input
    single_input = axis_angles.ndim == 1
    if single_input:
        axis_angles = unsqueeze(axis_angles, 0)  # (1, 3)

    n = len(axis_angles)

    # Compute angle magnitudes
    angles = backend_norm(axis_angles, axis=1, keepdims=True)  # (n, 1)
    safe_angles = where(angles < 1e-8, ones_like(angles), angles)
    axes = axis_angles / safe_angles

    # Build skew-symmetric matrices K
    K = zeros_nd((n, 3, 3), like=axis_angles)
    K[:, 0, 1] = -axes[:, 2]
    K[:, 0, 2] = axes[:, 1]
    K[:, 1, 0] = axes[:, 2]
    K[:, 1, 2] = -axes[:, 0]
    K[:, 2, 0] = -axes[:, 1]
    K[:, 2, 1] = axes[:, 0]

    # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
    I = expand(unsqueeze(eye(3, like=axis_angles), 0), (n, -1, -1))
    sin_a = unsqueeze(sin(angles), -1)  # (n, 1, 1)
    cos_a = unsqueeze(cos(angles), -1)

    result = I + sin_a * K + (1 - cos_a) * (K @ K)

    # Return single matrix if input was single
    if single_input:
        return result[0]
    return result


def axis_angle_to_rotation(axis_angle: Array) -> Array:
    """
    Convert axis-angle to rotation matrix (Rodrigues' formula).

    R = I + sin(t)K + (1-cos(t))K^2

    where K is the skew-symmetric matrix of the unit axis.

    Args:
        axis_angle: (3,) axis-angle vector (direction is axis, magnitude is angle).

    Returns:
        (3, 3) rotation matrix.

    Note:
        This is a convenience wrapper around :func:`rodrigues` for single vectors.
        For batched inputs, use :func:`rodrigues` directly.
    """
    return rodrigues(axis_angle)


def compute_relative_transform(
    origin1: Array,
    R1: Array,
    origin2: Array,
    R2: Array,
) -> Array:
    """
    Compute relative SE(3) transform from frame 1 to frame 2.

    The transform encodes how to get from frame 1 to frame 2:
    - Rotation: R_rel = R1.T @ R2
    - Translation: expressed in frame 1's coordinate system

    Args:
        origin1: (3,) position of frame 1.
        R1: (3, 3) rotation matrix of frame 1.
        origin2: (3,) position of frame 2.
        R2: (3, 3) rotation matrix of frame 2.

    Returns:
        (6,) transform vector [axis-angle (3), translation in frame1 coords (3)].
    """
    R_rel = R1.T @ R2
    axis_angle = rotation_to_axis_angle(R_rel)
    t_world = origin2 - origin1
    t_local = R1.T @ t_world

    if is_torch(origin1):
        import torch
        return torch.cat([axis_angle, t_local])
    return np.concatenate([axis_angle, t_local]).astype(np.float32)


def apply_relative_transform(
    origin: Array,
    R: Array,
    transform: Array,
) -> tuple[Array, Array]:
    """
    Apply relative transform to get a new frame from a source frame.

    This is the inverse of compute_relative_transform.

    Args:
        origin: (3,) position of source frame.
        R: (3, 3) rotation matrix of source frame.
        transform: (6,) vector [axis-angle (3), translation in source coords (3)].

    Returns:
        (origin2, R2): Position and rotation of target frame.
    """
    axis_angle = transform[:3]
    t_local = transform[3:]
    R_rel = axis_angle_to_rotation(axis_angle)
    R2 = R @ R_rel
    t_world = R @ t_local
    origin2 = origin + t_world
    return origin2, R2


# =============================================================================
# Residue Frame Computation
# =============================================================================
# These functions compute reference frames at specific atoms for residue linking.
# All frames return (origin, R) where R is a 3x3 rotation matrix with orthonormal
# columns representing the local x, y, z axes.


def _find_atom_index(atoms: Array, vals: np.ndarray) -> int:
    """Find index of first atom matching any value in vals.

    Raises:
        ValueError: If no matching atom is found.
    """
    vals_backend = to_backend(vals, atoms)
    mask = (atoms[:, None] == vals_backend).any(axis=1 if not is_torch(atoms) else -1)
    if is_torch(atoms):
        indices = mask.nonzero(as_tuple=True)[0]
        if len(indices) == 0:
            raise ValueError(f"No atom found matching values {vals.tolist()}")
        return indices[0]
    if not mask.any():
        raise ValueError(f"No atom found matching values {vals.tolist()}")
    return mask.argmax()


def extract_frame_positions(
    coords: Array,
    atoms: Array,
    frame_def: "FrameDefinition",
) -> Array:
    """
    Extract the 3 atom positions needed for frame computation.

    Uses vectorized 2D comparison to find atoms matching any value in each
    AtomGroup, removing the need to specify a specific residue type.

    Args:
        coords: (N, 3) or (..., N, 3) coordinates.
        atoms: (N,) atom type values.
        frame_def: Frame definition with AtomGroups.

    Returns:
        (3, 3) or (..., 3, 3) positions [origin, axis_ref, plane_ref].
    """
    origin_idx = _find_atom_index(atoms, frame_def.origin.index())
    axis_ref_idx = _find_atom_index(atoms, frame_def.axis_ref.index())
    plane_ref_idx = _find_atom_index(atoms, frame_def.plane_ref.index())

    return stack([
        coords[..., origin_idx, :],
        coords[..., axis_ref_idx, :],
        coords[..., plane_ref_idx, :],
    ], axis=-2)


def frame_from_positions(positions: Array) -> tuple[Array, Array]:
    """
    Compute a coordinate frame from 3 atom positions.

    Convention:
    - Z points FROM positions[0] TOWARD positions[1]
    - X is perpendicular to Z, toward positions[2] (via Gram-Schmidt)
    - Y = Z × X (completes right-handed system)

    Args:
        positions: (3, 3) or (..., 3, 3) - [origin, axis_ref, plane_ref].

    Returns:
        origin: (3,) or (..., 3) frame origin.
        R: (3, 3) or (..., 3, 3) rotation matrix [x, y, z] as columns.
    """
    origin = clone(positions[..., 0, :])
    axis_ref = positions[..., 1, :]
    plane_ref = positions[..., 2, :]

    # Z-axis: origin → axis_ref
    z_axis = normalize(axis_ref - origin)

    # X-axis: perpendicular to Z, toward plane_ref (Gram-Schmidt)
    plane_vec = plane_ref - origin
    proj = dot(plane_vec, z_axis)
    if positions.ndim > 2:
        proj = unsqueeze(proj, -1)
    x_axis = normalize(plane_vec - proj * z_axis)
    y_axis = cross(z_axis, x_axis)
    R = stack([x_axis, y_axis, z_axis], axis=-1)

    return origin, R


def rigid_align(
    coords: Array,
    current_origin: Array,
    current_R: Array,
    target_origin: Array,
    target_R: Array,
) -> Array:
    """
    Align coordinates so current frame matches target frame via rigid transform.

    Computes and applies the rigid transformation (rotation + translation)
    needed to move current_origin/current_R to target_origin/target_R.

    Args:
        coords: (n_atoms, 3) coordinates to transform.
        current_origin: (3,) current frame origin.
        current_R: (3, 3) current frame rotation.
        target_origin: (3,) target frame origin.
        target_R: (3, 3) target frame rotation.

    Returns:
        (n_atoms, 3) transformed coordinates.
    """
    # R_correction @ current_R = target_R
    # => R_correction = target_R @ current_R.T
    if is_torch(coords):
        R_correction = target_R @ current_R.T
        rotated_origin = R_correction @ current_origin
        t_correction = target_origin - rotated_origin
        positioned = (R_correction @ coords.T).T + t_correction
    else:
        R_correction = target_R @ current_R.T
        rotated_origin = R_correction @ current_origin
        t_correction = target_origin - rotated_origin
        positioned = (R_correction @ coords.T).T + t_correction
        positioned = positioned.astype(np.float32)

    return positioned


# =============================================================================
# Residue Type Detection
# =============================================================================


def is_purine(residue: "Residue") -> bool:
    """
    Check if a residue is a purine (has N9 atom).

    Purines (A, G, DA, DG) have an N9 atom connecting the base to the sugar.
    Pyrimidines (C, U, DC, DT) have an N1 atom instead.

    Args:
        residue: Residue type to check.

    Returns:
        True if purine (has N9), False if pyrimidine (has N1).
    """
    return hasattr(residue, 'N9')
