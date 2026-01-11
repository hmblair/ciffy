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
from ..backend.ops import to_backend, stack, unsqueeze, sin, cos, acos, clamp, cat, norm

if TYPE_CHECKING:
    from ..biochemistry import Residue
    from ..biochemistry.linking import FrameDefinition
from ..backend import clone
from .primitives import cross, dot, normalize, to_scalar


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
    # Normalize axis directly (more stable than dividing by 2*sin(angle))
    axis_norm = norm(axis, axis=-1, keepdims=True)
    axis = axis / (axis_norm + 1e-8)

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
    R_rel = rodrigues(axis_angle)
    R2 = R @ R_rel
    t_world = R @ t_local
    origin2 = origin + t_world
    return origin2, R2


def geodesic_so3(
    pred: Array,
    target: Array,
    reduction: str = "mean",
) -> Array:
    """
    Compute geodesic distance on SO(3) between rotations.

    The geodesic distance is the angle of the rotation needed to go from
    one orientation to another: θ = arccos((trace(R_target^T @ R_pred) - 1) / 2).

    This is the proper metric on SO(3), unlike Euclidean distance on axis-angle
    which treats the curved rotation manifold as flat.

    Args:
        pred: (N, 3) predicted axis-angle rotations.
        target: (N, 3) target axis-angle rotations.
        reduction: "mean", "sum", or "none".

    Returns:
        Geodesic distance(s) in radians. If reduction="none", returns (N,) angles.
    """
    R_pred = rodrigues(pred)      # (N, 3, 3)
    R_target = rodrigues(target)  # (N, 3, 3)

    # R_diff = R_target^T @ R_pred
    # For batched: transpose last two dims
    if is_torch(pred):
        R_diff = R_target.transpose(-1, -2) @ R_pred
    else:
        R_diff = np.swapaxes(R_target, -1, -2) @ R_pred

    # trace(R) = 1 + 2*cos(θ), so θ = arccos((trace - 1) / 2)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    cos_angle = (trace - 1) / 2
    cos_angle = clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
    angles = acos(cos_angle)

    if reduction == "mean":
        return angles.mean()
    elif reduction == "sum":
        return angles.sum()
    return angles


def se3_loss(
    pred: Array,
    target: Array,
    rotation_weight: float = 1.0,
    translation_weight: float = 1.0,
) -> Array:
    """
    Compute SE(3) loss with geodesic distance for rotation and MSE for translation.

    This is the proper loss for SE(3) transforms, using:
    - Geodesic distance on SO(3) for the rotation component
    - Mean squared error for the translation component

    The geodesic distance returns angles in radians, so rotation_weight=1.0
    means 1 radian of rotation error equals 1 unit of loss.

    Args:
        pred: (N, 6) predicted transforms [axis_angle(3), translation(3)].
        target: (N, 6) target transforms [axis_angle(3), translation(3)].
        rotation_weight: Weight for rotation loss (default 1.0).
        translation_weight: Weight for translation loss (default 1.0).

    Returns:
        Scalar loss combining rotation geodesic and translation MSE.

    Example:
        >>> pred_transforms = model(polymer)
        >>> loss = se3_loss(pred_transforms, target_transforms)
    """
    pred_rot, pred_trans = pred[..., :3], pred[..., 3:]
    target_rot, target_trans = target[..., :3], target[..., 3:]

    # Geodesic distance on SO(3) - returns mean angle in radians
    rotation_loss = geodesic_so3(pred_rot, target_rot, reduction="mean")

    # MSE for translation
    translation_loss = ((pred_trans - target_trans) ** 2).mean()

    return rotation_weight * rotation_loss + translation_weight * translation_loss


# =============================================================================
# 6D Rotation Representation
# =============================================================================
# The 6D rotation representation (Zhou et al., 2019) is continuous and avoids
# singularities present in axis-angle near θ=0 and θ=π. It represents rotations
# as the first two columns of the rotation matrix, reconstructing the third
# via Gram-Schmidt orthonormalization.


def rotation_matrix_to_6d(R: Array) -> Array:
    """
    Convert rotation matrix to 6D representation (first two columns).

    Args:
        R: (..., 3, 3) rotation matrix or matrices.

    Returns:
        (..., 6) 6D representation [col1, col2] flattened.
    """
    # Extract first two columns and flatten
    col1 = R[..., :, 0]  # (..., 3)
    col2 = R[..., :, 1]  # (..., 3)
    return cat([col1, col2], axis=-1)


def rotation_6d_to_matrix(r6d: Array) -> Array:
    """
    Convert 6D representation to rotation matrix via Gram-Schmidt.

    Args:
        r6d: (..., 6) 6D representation [a1, a2].

    Returns:
        (..., 3, 3) rotation matrix.
    """
    a1 = r6d[..., :3]
    a2 = r6d[..., 3:6]

    # Gram-Schmidt orthonormalization (backend-agnostic)
    b1 = normalize(a1)
    # Compute dot product and expand for broadcasting
    dot_val = dot(b1, a2)
    if r6d.ndim > 1:
        dot_val = unsqueeze(dot_val, -1)
    b2 = normalize(a2 - dot_val * b1)
    b3 = cross(b1, b2)
    return stack([b1, b2, b3], axis=-1)


def axis_angle_to_6d(axis_angle: Array) -> Array:
    """
    Convert axis-angle rotation to 6D representation.

    Args:
        axis_angle: (..., 3) axis-angle rotation vectors.

    Returns:
        (..., 6) 6D rotation representation.
    """
    R = rodrigues(axis_angle)  # (..., 3, 3)
    return rotation_matrix_to_6d(R)


def rotation_6d_to_axis_angle(r6d: Array) -> Array:
    """
    Convert 6D rotation representation to axis-angle.

    Args:
        r6d: (..., 6) 6D rotation representation.

    Returns:
        (..., 3) axis-angle rotation vectors.
    """
    R = rotation_6d_to_matrix(r6d)  # (..., 3, 3)
    return rotation_to_axis_angle(R)


def geodesic_so3_6d(
    pred: Array,
    target: Array,
    reduction: str = "mean",
) -> Array:
    """
    Compute geodesic distance on SO(3) between 6D rotation representations.

    Args:
        pred: (N, 6) predicted 6D rotations.
        target: (N, 6) target 6D rotations.
        reduction: "mean", "sum", or "none".

    Returns:
        Geodesic distance(s) in radians.
    """
    R_pred = rotation_6d_to_matrix(pred)      # (N, 3, 3)
    R_target = rotation_6d_to_matrix(target)  # (N, 3, 3)

    # R_diff = R_target^T @ R_pred (transpose last two dims)
    R_target_T = R_target.swapaxes(-1, -2) if hasattr(R_target, 'swapaxes') else np.swapaxes(R_target, -1, -2)
    R_diff = R_target_T @ R_pred

    # trace(R) = 1 + 2*cos(θ), so θ = arccos((trace - 1) / 2)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    cos_angle = (trace - 1) / 2
    cos_angle = clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
    angles = acos(cos_angle)

    if reduction == "mean":
        return angles.mean()
    elif reduction == "sum":
        return angles.sum()
    return angles


def se3_loss_6d(
    pred: Array,
    target: Array,
    rotation_weight: float = 1.0,
    translation_weight: float = 1.0,
) -> Array:
    """
    Compute SE(3) loss for 9D transforms (6D rotation + 3D translation).

    Args:
        pred: (N, 9) predicted transforms [6D_rotation(6), translation(3)].
        target: (N, 9) target transforms [6D_rotation(6), translation(3)].
        rotation_weight: Weight for rotation loss.
        translation_weight: Weight for translation loss.

    Returns:
        Scalar loss combining rotation geodesic and translation MSE.
    """
    pred_rot, pred_trans = pred[..., :6], pred[..., 6:]
    target_rot, target_trans = target[..., :6], target[..., 6:]

    rotation_loss = geodesic_so3_6d(pred_rot, target_rot, reduction="mean")
    translation_loss = ((pred_trans - target_trans) ** 2).mean()

    return rotation_weight * rotation_loss + translation_weight * translation_loss


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
    R_correction = target_R @ current_R.T
    rotated_origin = R_correction @ current_origin
    t_correction = target_origin - rotated_origin
    return (R_correction @ coords.T).T + t_correction


