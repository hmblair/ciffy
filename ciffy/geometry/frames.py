"""
Frame computation utilities for residue alignment and positioning.

This module provides functions for:
- Glycosidic frame computation (for residue alignment)
- Link frame computation (for chain building)
- Residue alignment and transform extraction

All functions are backend-agnostic (work with numpy and torch).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ciffy.backend import Array, is_torch, zeros_nd
from .primitives import normalize, cross, clone
from .transforms import (
    extract_frame_positions,
    frame_from_positions,
    compute_relative_transform,
    apply_relative_transform,
    is_purine,
)

if TYPE_CHECKING:
    from ciffy.biochemistry import Residue


# =============================================================================
# Frame Computation
# =============================================================================


def compute_glycosidic_frame(
    coords: Array,
    atoms: Array,
    residue: "Residue",
) -> tuple[Array, Array]:
    """
    Compute the glycosidic frame for residue alignment.

    Frame definition:
    - Origin: C1' atom
    - X-axis: Toward N9 (purines) or N1 (pyrimidines)
    - Z-axis: Normal to the C1'-N9-C4 plane
    - Y-axis: Completes right-handed system

    Args:
        coords: (n_atoms, 3) or (batch, n_atoms, 3) coordinates.
        atoms: (n_atoms,) atom type values.
        residue: Residue type (e.g., Residue.A).

    Returns:
        origin: (3,) or (batch, 3) C1' position.
        R: (3, 3) or (batch, 3, 3) rotation matrix [x, y, z] as columns.
    """
    from ciffy.biochemistry.linking import (
        PURINE_GLYCOSIDIC_FRAME,
        PYRIMIDINE_GLYCOSIDIC_FRAME,
    )

    # Select frame based on residue type
    frame_def = PURINE_GLYCOSIDIC_FRAME if is_purine(residue) else PYRIMIDINE_GLYCOSIDIC_FRAME

    # Extract positions and compute frame
    positions = extract_frame_positions(coords, atoms, frame_def)

    # Glycosidic frame uses X-primary convention (X toward base)
    # while frame_from_positions uses Z-primary
    # So we compute manually here for the X-primary convention
    origin = clone(positions[..., 0, :])
    n_pos = positions[..., 1, :]
    c4_pos = positions[..., 2, :]

    x_axis = normalize(n_pos - origin)
    y_temp = c4_pos - origin
    z_axis = normalize(cross(x_axis, y_temp))
    y_axis = cross(z_axis, x_axis)

    # Build rotation matrix
    if is_torch(coords):
        import torch
        R = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    else:
        R = np.stack([x_axis, y_axis, z_axis], axis=-1).astype(np.float32)
        origin = origin.astype(np.float32)

    return origin, R


def align_to_frame(
    coords: Array,
    atoms: Array,
    residue: "Residue",
) -> Array:
    """
    Align each residue to a canonical local frame (glycosidic frame).

    Args:
        coords: (n_instances, n_atoms, 3) coordinate array (numpy or torch).
        atoms: (n_atoms,) atom type values.
        residue: Residue type.

    Returns:
        Aligned coordinates with same shape as input.
    """
    from ciffy.backend import zeros_like

    n_instances = coords.shape[0]
    aligned = zeros_like(coords)

    for i in range(n_instances):
        origin, R = compute_glycosidic_frame(coords[i], atoms, residue)
        aligned[i] = (coords[i] - origin) @ R

    return aligned


def align_and_compute_transform(
    coords: Array,
    next_coords: Array | None,
    atoms: Array,
    residue: "Residue",
) -> tuple[Array, Array]:
    """
    Align residue to glycosidic frame and compute link transform.

    This is the canonical preprocessing for flow model training and inference.
    Ensures consistency between training data extraction and encode().

    Backend-agnostic: works with both numpy and torch arrays.

    Args:
        coords: (n_atoms, 3) coordinates of current residue.
        next_coords: (n_atoms, 3) coordinates of next residue, or None.
            If None, returns zero transform.
        atoms: (n_atoms,) atom type values.
        residue: Residue type.

    Returns:
        aligned_coords: (n_atoms, 3) coords in glycosidic frame.
        transform: (6,) SE(3) transform [axis-angle, translation].
    """
    from ciffy.biochemistry.linking import LINKING_BY_TYPE

    # Align to glycosidic frame
    origin, R = compute_glycosidic_frame(coords, atoms, residue)
    aligned_coords = (coords - origin) @ R

    if next_coords is not None:
        # Transform next residue to same frame
        aligned_next = (next_coords - origin) @ R

        # Get linking definition
        link_def = LINKING_BY_TYPE.get(residue.molecule_type)
        if link_def is None:
            raise ValueError(f"No linking definition for molecule type {residue.molecule_type}")

        # Compute link frames
        prev_positions = extract_frame_positions(aligned_coords, atoms, link_def.prev_frame)
        o3p_origin, o3p_R = frame_from_positions(prev_positions)

        next_positions = extract_frame_positions(aligned_next, atoms, link_def.next_frame)
        p_origin, p_R = frame_from_positions(next_positions)

        transform = compute_relative_transform(o3p_origin, o3p_R, p_origin, p_R)
    else:
        transform = zeros_nd((6,), like=coords)

    return aligned_coords, transform


def position_next_residue(
    coords1: Array,
    coords2: Array,
    rel_transform: Array,
    atoms: Array,
    residue: "Residue",
) -> Array:
    """
    Position residue 2 relative to residue 1 using the link transform.

    This is the inverse of transform extraction: given coords1 and a transform,
    position coords2 so that its P frame matches the target derived from
    coords1's O3' frame + transform.

    Args:
        coords1: (n_atoms, 3) coordinates of first residue (numpy or torch).
        coords2: (n_atoms, 3) coordinates of second residue (in canonical frame).
        rel_transform: (6,) SE(3) transform [axis-angle, translation].
        atoms: (n_atoms,) atom type values.
        residue: Residue type.

    Returns:
        (n_atoms, 3) positioned coordinates of second residue.
    """
    from ciffy.biochemistry.linking import LINKING_BY_TYPE

    # Get linking definition
    link_def = LINKING_BY_TYPE.get(residue.molecule_type)
    if link_def is None:
        raise ValueError(f"No linking definition for molecule type {residue.molecule_type}")

    # Compute O3' frame from coords1
    prev_positions = extract_frame_positions(coords1, atoms, link_def.prev_frame)
    o3p_origin, o3p_R = frame_from_positions(prev_positions)

    # Apply transform to get target P frame
    target_p_origin, target_p_R = apply_relative_transform(o3p_origin, o3p_R, rel_transform)

    # Compute current P frame from coords2
    next_positions = extract_frame_positions(coords2, atoms, link_def.next_frame)
    current_p_origin, current_p_R = frame_from_positions(next_positions)

    # Compute rigid transformation to align current P frame to target P frame
    R_correction = target_p_R @ current_p_R.T
    t_correction = target_p_origin - R_correction @ current_p_origin

    # Apply transformation
    coords2_positioned = (R_correction @ coords2.T).T + t_correction

    if not is_torch(coords2_positioned):
        coords2_positioned = coords2_positioned.astype(np.float32)

    return coords2_positioned


__all__ = [
    "compute_glycosidic_frame",
    "align_to_frame",
    "align_and_compute_transform",
    "position_next_residue",
]
