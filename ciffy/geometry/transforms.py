"""
SE(3) transforms and residue frame computation for molecular modeling.

This module provides functions for:
- Quaternion operations (rotation representation)
- SE(3) transform operations (relative transforms between frames)
- Residue frame computation (reference frames at specific atoms)
- Residue positioning (aligning residues for chain building)

Quaternions are represented as (w, x, y, z) where w is the scalar part.
The convention is to always use the canonical form with w >= 0 to avoid
the antipodal ambiguity (q and -q represent the same rotation).
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
        transform: (7,) SE(3) as [quaternion (4), translation (3)].
            Quaternion is (w, x, y, z) with w >= 0 (canonical form).

    Example:
        >>> from ciffy.geometry import LocalCoordinates
        >>> local = LocalCoordinates(coords, transform)
        >>> polymer = polymer.append(Residue.A, local)
    """

    coordinates: "Array"
    transform: "Array"


# =============================================================================
# Quaternion Operations
# =============================================================================
# Quaternions are represented as (w, x, y, z) where w is the scalar part.
# We use the canonical form with w >= 0 to avoid the antipodal ambiguity.


def _eye3(like: Array) -> Array:
    """3x3 identity matrix matching backend of reference array."""
    if is_torch(like):
        import torch
        return torch.eye(3, dtype=like.dtype, device=like.device)
    return np.eye(3, dtype=like.dtype)


def rotation_matrix_to_quaternion(R: Array) -> Array:
    """
    Convert rotation matrix to quaternion (w, x, y, z).

    Uses Shepperd's method which is numerically stable for all rotations.
    Returns canonical form with w >= 0.

    Args:
        R: (3, 3) single rotation matrix or (..., 3, 3) batch of matrices.

    Returns:
        (4,) or (..., 4) quaternion(s) in (w, x, y, z) format with w >= 0.
    """
    single_input = R.ndim == 2
    if single_input:
        R = R[None]

    batch_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    n = R_flat.shape[0]

    if is_torch(R):
        import torch
        quat = torch.zeros((n, 4), dtype=R.dtype, device=R.device)
    else:
        quat = np.zeros((n, 4), dtype=R.dtype)

    # Shepperd's method: choose the largest diagonal element to avoid division by small numbers
    trace = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]

    # Case 1: trace > 0 (w is largest)
    mask1 = trace > 0
    if is_torch(R):
        s1 = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4w
    else:
        s1 = np.sqrt(trace[mask1] + 1.0) * 2
    quat[mask1, 0] = 0.25 * s1  # w
    quat[mask1, 1] = (R_flat[mask1, 2, 1] - R_flat[mask1, 1, 2]) / s1  # x
    quat[mask1, 2] = (R_flat[mask1, 0, 2] - R_flat[mask1, 2, 0]) / s1  # y
    quat[mask1, 3] = (R_flat[mask1, 1, 0] - R_flat[mask1, 0, 1]) / s1  # z

    # Case 2: R[0,0] is largest diagonal
    mask2 = ~mask1 & (R_flat[:, 0, 0] > R_flat[:, 1, 1]) & (R_flat[:, 0, 0] > R_flat[:, 2, 2])
    if is_torch(R):
        s2 = torch.sqrt(1.0 + R_flat[mask2, 0, 0] - R_flat[mask2, 1, 1] - R_flat[mask2, 2, 2]) * 2
    else:
        s2 = np.sqrt(1.0 + R_flat[mask2, 0, 0] - R_flat[mask2, 1, 1] - R_flat[mask2, 2, 2]) * 2
    quat[mask2, 0] = (R_flat[mask2, 2, 1] - R_flat[mask2, 1, 2]) / s2  # w
    quat[mask2, 1] = 0.25 * s2  # x
    quat[mask2, 2] = (R_flat[mask2, 0, 1] + R_flat[mask2, 1, 0]) / s2  # y
    quat[mask2, 3] = (R_flat[mask2, 0, 2] + R_flat[mask2, 2, 0]) / s2  # z

    # Case 3: R[1,1] is largest diagonal
    mask3 = ~mask1 & ~mask2 & (R_flat[:, 1, 1] > R_flat[:, 2, 2])
    if is_torch(R):
        s3 = torch.sqrt(1.0 + R_flat[mask3, 1, 1] - R_flat[mask3, 0, 0] - R_flat[mask3, 2, 2]) * 2
    else:
        s3 = np.sqrt(1.0 + R_flat[mask3, 1, 1] - R_flat[mask3, 0, 0] - R_flat[mask3, 2, 2]) * 2
    quat[mask3, 0] = (R_flat[mask3, 0, 2] - R_flat[mask3, 2, 0]) / s3  # w
    quat[mask3, 1] = (R_flat[mask3, 0, 1] + R_flat[mask3, 1, 0]) / s3  # x
    quat[mask3, 2] = 0.25 * s3  # y
    quat[mask3, 3] = (R_flat[mask3, 1, 2] + R_flat[mask3, 2, 1]) / s3  # z

    # Case 4: R[2,2] is largest diagonal
    mask4 = ~mask1 & ~mask2 & ~mask3
    if is_torch(R):
        s4 = torch.sqrt(1.0 + R_flat[mask4, 2, 2] - R_flat[mask4, 0, 0] - R_flat[mask4, 1, 1]) * 2
    else:
        s4 = np.sqrt(1.0 + R_flat[mask4, 2, 2] - R_flat[mask4, 0, 0] - R_flat[mask4, 1, 1]) * 2
    quat[mask4, 0] = (R_flat[mask4, 1, 0] - R_flat[mask4, 0, 1]) / s4  # w
    quat[mask4, 1] = (R_flat[mask4, 0, 2] + R_flat[mask4, 2, 0]) / s4  # x
    quat[mask4, 2] = (R_flat[mask4, 1, 2] + R_flat[mask4, 2, 1]) / s4  # y
    quat[mask4, 3] = 0.25 * s4  # z

    # Normalize and ensure canonical form (w >= 0)
    quat = normalize_quaternion(quat)

    quat = quat.reshape(*batch_shape, 4)
    if single_input:
        quat = quat[0]

    return quat


def quaternion_to_rotation_matrix(q: Array) -> Array:
    """
    Convert quaternion (w, x, y, z) to rotation matrix.

    Args:
        q: (4,) single quaternion or (..., 4) batch of quaternions.

    Returns:
        (3, 3) or (..., 3, 3) rotation matrix.
    """
    single_input = q.ndim == 1
    if single_input:
        q = q[None]

    batch_shape = q.shape[:-1]
    q_flat = q.reshape(-1, 4)

    w, x, y, z = q_flat[:, 0], q_flat[:, 1], q_flat[:, 2], q_flat[:, 3]

    # Rotation matrix from quaternion
    # R = [[1-2(y²+z²), 2(xy-wz), 2(xz+wy)],
    #      [2(xy+wz), 1-2(x²+z²), 2(yz-wx)],
    #      [2(xz-wy), 2(yz+wx), 1-2(x²+y²)]]

    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    if is_torch(q):
        import torch
        R = torch.stack([
            torch.stack([1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
            torch.stack([2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx)], dim=-1),
            torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2)], dim=-1),
        ], dim=-2)
    else:
        R = np.stack([
            np.stack([1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy)], axis=-1),
            np.stack([2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx)], axis=-1),
            np.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2)], axis=-1),
        ], axis=-2)

    R = R.reshape(*batch_shape, 3, 3)
    if single_input:
        R = R[0]

    return R


def normalize_quaternion(q: Array) -> Array:
    """
    Normalize quaternion to unit length and canonical form (w >= 0).

    Args:
        q: (..., 4) quaternion(s).

    Returns:
        (..., 4) normalized quaternion(s) with w >= 0.
    """
    # Normalize to unit length
    q_norm = norm(q, axis=-1, keepdims=True)
    q = q / (q_norm + 1e-8)

    # Canonical form: w >= 0 (flip sign if w < 0)
    if is_torch(q):
        import torch
        sign = torch.sign(q[..., :1])
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    else:
        sign = np.sign(q[..., :1])
        sign = np.where(sign == 0, 1.0, sign)

    return q * sign


def quaternion_multiply(q1: Array, q2: Array) -> Array:
    """
    Multiply two quaternions (Hamilton product).

    Args:
        q1: (..., 4) first quaternion(s).
        q2: (..., 4) second quaternion(s).

    Returns:
        (..., 4) product quaternion(s).
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return stack([w, x, y, z], axis=-1)


def quaternion_conjugate(q: Array) -> Array:
    """
    Compute quaternion conjugate (inverse for unit quaternions).

    Args:
        q: (..., 4) quaternion(s).

    Returns:
        (..., 4) conjugate quaternion(s).
    """
    if is_torch(q):
        import torch
        return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)
    return np.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)


def geodesic_so3_quat(
    pred: Array,
    target: Array,
    reduction: str = "mean",
) -> Array:
    """
    Compute geodesic distance on SO(3) between quaternion rotations.

    The geodesic distance is: θ = 2 * arccos(|q1 · q2|)

    Args:
        pred: (N, 4) predicted quaternions.
        target: (N, 4) target quaternions.
        reduction: "mean", "sum", or "none".

    Returns:
        Geodesic distance(s) in radians.
    """
    # Quaternion dot product (take absolute value due to antipodal equivalence)
    dot_product = (pred * target).sum(axis=-1)
    if is_torch(pred):
        import torch
        dot_product = torch.abs(dot_product)
    else:
        dot_product = np.abs(dot_product)

    # Clamp for numerical stability
    dot_product = clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)

    # Geodesic distance: θ = 2 * arccos(|q1 · q2|)
    angles = 2 * acos(dot_product)

    if reduction == "mean":
        return angles.mean()
    elif reduction == "sum":
        return angles.sum()
    return angles


def se3_loss_quat(
    pred: Array,
    target: Array,
    rotation_weight: float = 1.0,
    translation_weight: float = 1.0,
) -> Array:
    """
    Compute SE(3) loss for 7D transforms (quaternion + translation).

    Args:
        pred: (N, 7) predicted transforms [quaternion(4), translation(3)].
        target: (N, 7) target transforms [quaternion(4), translation(3)].
        rotation_weight: Weight for rotation loss.
        translation_weight: Weight for translation loss.

    Returns:
        Scalar loss combining rotation geodesic and translation MSE.
    """
    pred_rot, pred_trans = pred[..., :4], pred[..., 4:]
    target_rot, target_trans = target[..., :4], target[..., 4:]

    rotation_loss = geodesic_so3_quat(pred_rot, target_rot, reduction="mean")
    translation_loss = ((pred_trans - target_trans) ** 2).mean()

    return rotation_weight * rotation_loss + translation_weight * translation_loss


# =============================================================================
# SE(3) Transform Operations
# =============================================================================


def compute_relative_transform(
    origin1: Array,
    R1: Array,
    origin2: Array,
    R2: Array,
) -> Array:
    """
    Compute relative SE(3) transform from frame 1 to frame 2.

    The transform encodes how to get from frame 1 to frame 2:
    - Rotation: R_rel = R1.T @ R2 (converted to quaternion)
    - Translation: expressed in frame 1's coordinate system

    Args:
        origin1: (3,) position of frame 1.
        R1: (3, 3) rotation matrix of frame 1.
        origin2: (3,) position of frame 2.
        R2: (3, 3) rotation matrix of frame 2.

    Returns:
        (7,) transform vector [quaternion (4), translation in frame1 coords (3)].
    """
    R_rel = R1.T @ R2
    quat = rotation_matrix_to_quaternion(R_rel)
    t_world = origin2 - origin1
    t_local = R1.T @ t_world

    if is_torch(origin1):
        import torch
        return torch.cat([quat, t_local])
    return np.concatenate([quat, t_local]).astype(np.float32)


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
        transform: (7,) vector [quaternion (4), translation in source coords (3)].

    Returns:
        (origin2, R2): Position and rotation of target frame.
    """
    quat = transform[:4]
    t_local = transform[4:]
    R_rel = quaternion_to_rotation_matrix(quat)
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
    R_correction = target_R @ current_R.T
    rotated_origin = R_correction @ current_origin
    t_correction = target_origin - rotated_origin
    return (R_correction @ coords.T).T + t_correction
