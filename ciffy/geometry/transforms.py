"""
SE(3) transforms and residue frame computation for molecular modeling.

This module provides functions for:
- SE(3) transform operations (rotation to/from axis-angle, relative transforms)
- Residue frame computation (reference frames at specific atoms)
- Residue positioning (aligning residues for chain building)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..backend import Array, is_torch

if TYPE_CHECKING:
    from ..biochemistry import Residue
from .primitives import cross, dot, norm, normalize, clone, to_scalar


# =============================================================================
# SE(3) Transform Operations
# =============================================================================


def _stack_columns(x: Array, y: Array, z: Array) -> Array:
    """Stack three vectors as columns of a matrix."""
    if is_torch(x):
        import torch
        return torch.stack([x, y, z], dim=1)
    return np.column_stack([x, y, z])


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

    Uses the Rodrigues formula inverse. Handles edge cases for identity
    and 180-degree rotations.

    Args:
        R: (3, 3) rotation matrix.

    Returns:
        (3,) axis-angle vector where direction is axis and magnitude is angle.
    """
    angle = _acos_safe((_trace(R) - 1) / 2)
    angle_scalar = to_scalar(angle)

    if angle_scalar < 1e-6:
        # Near identity - return zero vector
        if is_torch(R):
            import torch
            return torch.zeros(3, dtype=R.dtype, device=R.device)
        return np.zeros(3, dtype=np.float32)

    if np.pi - angle_scalar < 1e-6:
        # Near 180 degrees - extract axis from R + I
        M = R + _eye3(R)
        if is_torch(R):
            col_norms = M.norm(dim=0)
            k = col_norms.argmax().item()
            axis = M[:, k] / col_norms[k]
        else:
            col_norms = np.linalg.norm(M, axis=0)
            k = np.argmax(col_norms)
            axis = M[:, k] / col_norms[k]
        return axis * angle

    # Standard case
    if is_torch(R):
        import torch
        axis = torch.stack([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ])
        axis = axis / (2 * torch.sin(angle) + 1e-8)
    else:
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ])
        axis = axis / (2 * np.sin(angle) + 1e-8)

    return axis * angle


def axis_angle_to_rotation(axis_angle: Array) -> Array:
    """
    Convert axis-angle to rotation matrix (Rodrigues' formula).

    R = I + sin(t)K + (1-cos(t))K^2

    where K is the skew-symmetric matrix of the unit axis.

    Args:
        axis_angle: (3,) axis-angle vector (direction is axis, magnitude is angle).

    Returns:
        (3, 3) rotation matrix.
    """
    angle = norm(axis_angle)
    angle_scalar = to_scalar(angle)

    if angle_scalar < 1e-8:
        return _eye3(axis_angle)

    axis = axis_angle / angle

    if is_torch(axis_angle):
        import torch
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], dtype=axis_angle.dtype, device=axis_angle.device)
        I = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
        return I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
    else:
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], dtype=np.float32)
        return np.eye(3, dtype=np.float32) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


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


def _arbitrary_perpendicular(z_axis: Array) -> Array:
    """
    Compute an arbitrary unit vector perpendicular to z_axis.

    Used when no perpendicular reference atom is available (e.g., N frame in proteins).

    Args:
        z_axis: (3,) or (..., 3) normalized vector(s).

    Returns:
        Unit vector(s) perpendicular to z_axis, same shape as input.
    """
    if is_torch(z_axis):
        import torch
        # Choose reference that's not parallel to z_axis
        ref = torch.zeros_like(z_axis)
        ref[..., 0] = 1.0
        # If too parallel, use y-axis instead
        parallel = (dot(z_axis, ref).abs() > 0.9) if z_axis.ndim == 1 else None
        if parallel is not None and parallel:
            ref = torch.zeros_like(z_axis)
            ref[..., 1] = 1.0
    else:
        ref = np.zeros_like(z_axis)
        ref[..., 0] = 1.0
        if abs(dot(z_axis, ref)) > 0.9:
            ref = np.zeros_like(z_axis)
            ref[..., 1] = 1.0

    return normalize(cross(ref, z_axis))


def compute_frame_from_indices(
    coords: Array,
    frame_cols: Array | tuple[int, int, int | None],
    z_toward_origin: bool,
) -> tuple[Array, Array]:
    """
    Compute coordinate frame using pre-resolved column indices.

    This is the fast path for frame computation - pure tensor math with no
    Python attribute lookups. Supports both single residues and batches.

    Args:
        coords: (n_atoms, 3) or (batch, n_atoms, 3) coordinates.
        frame_cols: Column indices for frame computation. Can be either:
            - np.ndarray shape (3,): [origin_col, z_ref_col, perp_ref_col]
              with -1 indicating missing perp_ref (preferred API)
            - tuple: (origin_col, z_ref_col, perp_ref_col) with None for
              missing perp_ref (backward compatibility)
        z_toward_origin: If True, Z points from z_ref toward origin.
            If False, Z points from origin toward z_ref.

    Returns:
        origin: (3,) or (batch, 3) frame origin position.
        R: (3, 3) or (batch, 3, 3) rotation matrix with [x, y, z] as columns.

    Example:
        >>> # Array API (preferred)
        >>> frame_cols = np.array([0, 1, 2], dtype=np.int32)  # or -1 for no perp
        >>> origin, R = compute_frame_from_indices(coords, frame_cols, True)
        >>>
        >>> # Tuple API (backward compatible)
        >>> frame_cols = link_def.prev_frame.resolve(residue, atom_to_col)
        >>> origin, R = compute_frame_from_indices(coords, frame_cols, z_toward_origin)
    """
    # Handle both array (new API) and tuple (backward compat)
    if isinstance(frame_cols, np.ndarray):
        origin_col = int(frame_cols[0])
        z_ref_col = int(frame_cols[1])
        perp_ref_col = int(frame_cols[2]) if frame_cols[2] >= 0 else None
    else:
        origin_col, z_ref_col, perp_ref_col = frame_cols

    # Direct indexing - works for (..., n_atoms, 3)
    origin_pos = coords[..., origin_col, :]
    z_ref_pos = coords[..., z_ref_col, :]

    # Compute Z-axis
    if z_toward_origin:
        z_axis = normalize(origin_pos - z_ref_pos)
    else:
        z_axis = normalize(z_ref_pos - origin_pos)

    # Compute X-axis (perpendicular to Z)
    if perp_ref_col is not None:
        perp_pos = coords[..., perp_ref_col, :]
        # Project perp direction onto plane perpendicular to Z
        perp_dir = perp_pos - z_ref_pos
        x_axis = normalize(cross(perp_dir, z_axis))
    else:
        x_axis = _arbitrary_perpendicular(z_axis)

    # Y-axis completes right-handed system
    y_axis = cross(z_axis, x_axis)

    # Stack into rotation matrix
    origin = clone(origin_pos)
    R = _stack_columns(x_axis, y_axis, z_axis)

    return origin, R


def compute_o3p_frame(
    coords: Array,
    atom_to_col: dict[int, int],
    residue: "Residue",
) -> tuple[Array, Array]:
    """
    Compute the O3' frame for a nucleotide residue.

    This frame is used as the outgoing link point for backbone connectivity.

    Frame definition:
    - Origin: O3' atom
    - Z-axis: Along C3'->O3' bond (outward direction)
    - X-axis: Perpendicular, in the C4'-C3'-O3' plane
    - Y-axis: Completes right-handed system

    Args:
        coords: (n_atoms, 3) residue coordinates.
        atom_to_col: Dict mapping atom type value to column index.
        residue: Residue enum (e.g., Residue.A).

    Returns:
        origin: (3,) O3' position.
        R: (3, 3) rotation matrix [x, y, z] as columns.
    """
    c4p = coords[atom_to_col[residue.C4p.value]]
    c3p = coords[atom_to_col[residue.C3p.value]]
    o3p = coords[atom_to_col[residue.O3p.value]]

    origin = clone(o3p)

    z_axis = normalize(o3p - c3p)
    y_temp = c4p - c3p
    x_axis = normalize(cross(y_temp, z_axis))
    y_axis = cross(z_axis, x_axis)

    R = _stack_columns(x_axis, y_axis, z_axis)
    return origin, R


def compute_p_frame(
    coords: Array,
    atom_to_col: dict[int, int],
    residue: "Residue",
) -> tuple[Array, Array]:
    """
    Compute the P frame for a nucleotide residue.

    This frame is used as the incoming link point for backbone connectivity.

    Frame definition:
    - Origin: P atom
    - Z-axis: Along O5'->P bond
    - X-axis: Perpendicular, toward OP1
    - Y-axis: Completes right-handed system

    Args:
        coords: (n_atoms, 3) residue coordinates.
        atom_to_col: Dict mapping atom type value to column index.
        residue: Residue enum (e.g., Residue.A).

    Returns:
        origin: (3,) P position.
        R: (3, 3) rotation matrix [x, y, z] as columns.
    """
    p = coords[atom_to_col[residue.P.value]]
    o5p = coords[atom_to_col[residue.O5p.value]]
    op1 = coords[atom_to_col[residue.OP1.value]]

    origin = clone(p)

    z_axis = normalize(p - o5p)
    y_temp = op1 - p
    x_axis = normalize(cross(y_temp, z_axis))
    y_axis = cross(z_axis, x_axis)

    R = _stack_columns(x_axis, y_axis, z_axis)
    return origin, R


def compute_c_frame(
    coords: Array,
    atom_to_col: dict[int, int],
    residue: "Residue",
) -> tuple[Array, Array]:
    """
    Compute the carbonyl C frame for a protein residue.

    This frame is used as the outgoing link point for peptide bonds.

    Frame definition:
    - Origin: C atom (carbonyl carbon)
    - Z-axis: Along CA->C bond (outward direction)
    - X-axis: Perpendicular, toward O
    - Y-axis: Completes right-handed system

    Args:
        coords: (n_atoms, 3) residue coordinates.
        atom_to_col: Dict mapping atom type value to column index.
        residue: Residue enum (e.g., Residue.ALA).

    Returns:
        origin: (3,) C position.
        R: (3, 3) rotation matrix [x, y, z] as columns.
    """
    ca = coords[atom_to_col[residue.CA.value]]
    c = coords[atom_to_col[residue.C.value]]
    o = coords[atom_to_col[residue.O.value]]

    origin = clone(c)

    z_axis = normalize(c - ca)
    y_temp = o - c
    x_axis = normalize(cross(y_temp, z_axis))
    y_axis = cross(z_axis, x_axis)

    R = _stack_columns(x_axis, y_axis, z_axis)
    return origin, R


def compute_n_frame(
    coords: Array,
    atom_to_col: dict[int, int],
    residue: "Residue",
) -> tuple[Array, Array]:
    """
    Compute the amide N frame for a protein residue.

    This frame is used as the incoming link point for peptide bonds.

    Frame definition:
    - Origin: N atom (amide nitrogen)
    - Z-axis: Along C(prev)->N bond direction
    - X-axis: Perpendicular, toward CA
    - Y-axis: Completes right-handed system

    Note: For the incoming frame, we use CA->N as the Z-axis
    (pointing toward the incoming bond).

    Args:
        coords: (n_atoms, 3) residue coordinates.
        atom_to_col: Dict mapping atom type value to column index.
        residue: Residue enum (e.g., Residue.ALA).

    Returns:
        origin: (3,) N position.
        R: (3, 3) rotation matrix [x, y, z] as columns.
    """
    n = coords[atom_to_col[residue.N.value]]
    ca = coords[atom_to_col[residue.CA.value]]

    origin = clone(n)

    # Z points from N toward CA (incoming direction from previous C)
    z_axis = normalize(ca - n)
    # Use CA for perpendicular direction
    y_temp = ca - n
    # Need a different reference - use the H on N if available, else use CA
    # For simplicity, construct perpendicular from arbitrary vector
    if is_torch(n):
        import torch
        ref = torch.tensor([1.0, 0.0, 0.0], dtype=n.dtype, device=n.device)
        if abs(dot(z_axis, ref)) > 0.9:
            ref = torch.tensor([0.0, 1.0, 0.0], dtype=n.dtype, device=n.device)
    else:
        ref = np.array([1.0, 0.0, 0.0], dtype=n.dtype)
        if abs(dot(z_axis, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=n.dtype)

    x_axis = normalize(cross(ref, z_axis))
    y_axis = cross(z_axis, x_axis)

    R = _stack_columns(x_axis, y_axis, z_axis)
    return origin, R


def compute_prev_frame(
    coords: Array,
    atom_to_col: dict[int, int],
    residue: "Residue",
) -> tuple[Array, Array]:
    """
    Compute the outgoing (previous) frame for a residue.

    Uses LinkingDefinition to determine the frame atoms based on molecule type.
    This is the convenience wrapper - for performance-critical code, use
    compute_frame_from_indices() with pre-resolved column indices.

    Args:
        coords: (n_atoms, 3) residue coordinates.
        atom_to_col: Dict mapping atom type value to column index.
        residue: Residue enum.

    Returns:
        origin: (3,) frame origin position.
        R: (3, 3) rotation matrix.

    Raises:
        ValueError: If no linking definition exists for this molecule type.
    """
    from ..biochemistry.linking import LINKING_BY_TYPE

    link_def = LINKING_BY_TYPE.get(residue.molecule_type)
    if link_def is None:
        raise ValueError(
            f"No linking definition for molecule type {residue.molecule_type}"
        )

    frame_cols = link_def.prev_frame.resolve(residue, atom_to_col)
    return compute_frame_from_indices(coords, frame_cols, link_def.prev_frame.z_toward_origin)


def compute_next_frame(
    coords: Array,
    atom_to_col: dict[int, int],
    residue: "Residue",
) -> tuple[Array, Array]:
    """
    Compute the incoming (next) frame for a residue.

    Uses LinkingDefinition to determine the frame atoms based on molecule type.
    This is the convenience wrapper - for performance-critical code, use
    compute_frame_from_indices() with pre-resolved column indices.

    Args:
        coords: (n_atoms, 3) residue coordinates.
        atom_to_col: Dict mapping atom type value to column index.
        residue: Residue enum.

    Returns:
        origin: (3,) frame origin position.
        R: (3, 3) rotation matrix.

    Raises:
        ValueError: If no linking definition exists for this molecule type.
    """
    from ..biochemistry.linking import LINKING_BY_TYPE

    link_def = LINKING_BY_TYPE.get(residue.molecule_type)
    if link_def is None:
        raise ValueError(
            f"No linking definition for molecule type {residue.molecule_type}"
        )

    frame_cols = link_def.next_frame.resolve(residue, atom_to_col)
    return compute_frame_from_indices(coords, frame_cols, link_def.next_frame.z_toward_origin)


# =============================================================================
# Residue Positioning
# =============================================================================


def position_residue(
    prev_coords: Array,
    next_coords: Array,
    prev_atom_to_col: dict[int, int],
    next_atom_to_col: dict[int, int],
    prev_residue: "Residue",
    next_residue: "Residue",
    transform: Array | None = None,
) -> Array:
    """
    Position a residue relative to the previous residue.

    This function places next_coords so that the incoming link point of the
    next residue aligns with the outgoing link point of the previous residue.

    Works with both NumPy and PyTorch arrays (auto-detected from input).

    Args:
        prev_coords: (n_atoms, 3) positioned coordinates of previous residue.
        next_coords: (n_atoms, 3) coordinates of residue to position.
        prev_atom_to_col: Dict mapping atom type value to column index for prev residue.
        next_atom_to_col: Dict mapping atom type value to column index for next residue.
        prev_residue: Residue enum for previous residue.
        next_residue: Residue enum for next residue.
        transform: Optional (6,) SE(3) transform [axis-angle, translation].
            If None, uses linear extension along the Z-axis with standard bond length.
            If provided, applies the learned transform from flow models.

    Returns:
        (n_atoms, 3) positioned coordinates of the next residue.

    Example (linear extension for templates):
        >>> positioned = position_residue(
        ...     prev_coords, next_coords,
        ...     prev_atom_to_col, next_atom_to_col,
        ...     Residue.A, Residue.C,
        ...     transform=None,  # Linear extension
        ... )

    Example (SE(3) transform for flow models):
        >>> positioned = position_residue(
        ...     prev_coords, next_coords,
        ...     prev_atom_to_col, next_atom_to_col,
        ...     Residue.A, Residue.C,
        ...     transform=learned_transform,  # From flow model
        ... )
    """
    from ..biochemistry.linking import LINKING_BY_TYPE

    # Compute outgoing frame from previous residue
    prev_origin, prev_R = compute_prev_frame(
        prev_coords, prev_atom_to_col, prev_residue
    )

    if transform is None:
        # Linear extension: translate along global Z-axis with appropriate spacing.
        # This keeps all residues with the same orientation extending in a line.
        from ..biochemistry.linking import LINKING_BY_TYPE

        link_def = LINKING_BY_TYPE.get(prev_residue.molecule_type)

        # Calculate spacing: use backbone length (P to O3') + bond length
        # This ensures residues don't overlap while maintaining correct connectivity.
        if link_def is not None:
            prev_link_atom = getattr(prev_residue, link_def.prev_atom)
            next_link_atom = getattr(next_residue, link_def.next_atom)

            prev_link_pos = prev_coords[prev_atom_to_col[prev_link_atom.value]]
            next_link_pos = next_coords[next_atom_to_col[next_link_atom.value]]

            # Get P position of previous residue to calculate backbone span
            prev_p_atom = getattr(prev_residue, link_def.next_atom)  # P atom
            if prev_p_atom.value in prev_atom_to_col:
                prev_p_pos = prev_coords[prev_atom_to_col[prev_p_atom.value]]
                # Backbone span is distance from P to O3' plus bond length
                backbone_span = norm(prev_link_pos - prev_p_pos)
                spacing = backbone_span + link_def.bond_length
            else:
                # First residue may not have P, use default spacing
                spacing = 6.0
        else:
            spacing = 6.0

        if is_torch(prev_coords):
            import torch
            offset = torch.zeros(3, dtype=prev_coords.dtype, device=prev_coords.device)
            offset[2] = spacing
        else:
            offset = np.array([0.0, 0.0, spacing], dtype=prev_coords.dtype)

        prev_centroid = prev_coords.mean(axis=0)
        next_centroid = next_coords.mean(axis=0)
        return next_coords + (prev_centroid + offset - next_centroid)

    # Apply SE(3) transform (for learned/non-default transforms)
    target_origin, target_R = apply_relative_transform(
        prev_origin, prev_R, transform
    )

    # Compute incoming frame from next residue (current position)
    next_origin, next_R = compute_next_frame(
        next_coords, next_atom_to_col, next_residue
    )

    # Compute rigid transformation to align next frame to target frame
    R_correction = target_R @ next_R.T
    t_correction = target_origin - R_correction @ next_origin

    # Apply transformation
    return (R_correction @ next_coords.T).T + t_correction


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


# =============================================================================
# Glycosidic Frame (for nucleotide bases)
# =============================================================================


def compute_glycosidic_frame(
    coords: Array,
    atom_to_col: dict[int, int],
    residue: "Residue",
) -> tuple[Array, Array]:
    """
    Compute the glycosidic frame for a nucleotide residue.

    Frame definition:
    - Origin: C1' atom
    - X-axis: Toward N9 (purines) or N1 (pyrimidines)
    - Z-axis: Normal to the C1'-N9-C4 plane
    - Y-axis: Completes right-handed system

    Args:
        coords: (n_atoms, 3) coordinates (numpy or torch).
        atom_to_col: Dict mapping atom type value to column index.
        residue: Residue type.

    Returns:
        origin: (3,) C1' position.
        R: (3, 3) rotation matrix [x, y, z] as columns.
    """
    c1p_idx = atom_to_col[residue.C1p.value]
    c4_idx = atom_to_col[residue.C4.value]

    # N9 for purines (A, G), N1 for pyrimidines (C, U)
    if is_purine(residue):
        n_idx = atom_to_col[residue.N9.value]
    else:
        n_idx = atom_to_col[residue.N1.value]

    origin = clone(coords[c1p_idx])
    n_pos = coords[n_idx]
    c4_pos = coords[c4_idx]

    x_axis = normalize(n_pos - origin)
    y_temp = c4_pos - origin
    z_axis = normalize(cross(x_axis, y_temp))
    y_axis = cross(z_axis, x_axis)

    # Build rotation matrix with columns [x, y, z]
    if is_torch(coords):
        import torch
        R = torch.stack([x_axis, y_axis, z_axis], dim=1)
    else:
        import numpy as np
        R = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32)
        origin = origin.astype(np.float32)

    return origin, R


# =============================================================================
# Fast Residue Positioning (pre-resolved indices)
# =============================================================================


def position_residue_fast(
    prev_coords: Array,
    next_coords: Array,
    transform: Array,
    prev_frame_cols: Array | tuple[int, int, int | None],
    prev_z_toward_origin: bool,
    next_frame_cols: Array | tuple[int, int, int | None],
    next_z_toward_origin: bool,
) -> Array:
    """
    Position residue 2 relative to residue 1 using pre-resolved frame indices.

    This is the fast path for residue positioning. Uses pre-resolved column
    indices to compute frames with pure tensor math (no Python attribute lookups).
    The frame indices should be computed once at model initialization.

    Works with both NumPy and PyTorch arrays (auto-detected from input).

    Args:
        prev_coords: (n_atoms, 3) coordinates of previous residue.
        next_coords: (n_atoms, 3) coordinates of next residue (in canonical frame).
        transform: (6,) SE(3) transform [axis-angle, translation].
        prev_frame_cols: Column indices for outgoing frame. Can be either:
            - np.ndarray shape (3,): [origin, z_ref, perp_ref] with -1 for missing
            - tuple: (origin, z_ref, perp_ref) with None for missing
        prev_z_toward_origin: Z-axis direction for prev frame.
        next_frame_cols: Column indices for incoming frame (same format as prev).
        next_z_toward_origin: Z-axis direction for next frame.

    Returns:
        (n_atoms, 3) positioned coordinates of next residue.

    Example:
        >>> # Array API (preferred) - using FrameIndices from builder
        >>> frame_indices = _resolve_frame_indices(residue_idx, atom_indices)
        >>> positioned = position_residue_fast(
        ...     prev_coords, next_coords, transform,
        ...     frame_indices.prev_cols, frame_indices.prev_z_toward,
        ...     frame_indices.next_cols, frame_indices.next_z_toward,
        ... )
    """
    # Compute outgoing frame from prev_coords using pre-resolved indices
    prev_origin, prev_R = compute_frame_from_indices(
        prev_coords, prev_frame_cols, prev_z_toward_origin
    )

    # Apply transform to get target incoming frame
    target_origin, target_R = apply_relative_transform(
        prev_origin, prev_R, transform
    )

    # Compute current incoming frame from next_coords using pre-resolved indices
    current_origin, current_R = compute_frame_from_indices(
        next_coords, next_frame_cols, next_z_toward_origin
    )

    # Compute rigid transformation to align current frame to target frame
    R_correction = target_R @ current_R.T
    t_correction = target_origin - R_correction @ current_origin

    # Apply transformation
    return (R_correction @ next_coords.T).T + t_correction
