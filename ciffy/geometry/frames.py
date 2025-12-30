"""
Frame computation with precomputed indices for efficient geometry operations.

This module provides:
- FrameIndices: Precomputed column indices for fast frame computation
- Frame computation functions that use pre-resolved indices
- Residue alignment and positioning utilities

All functions are backend-agnostic (work with numpy and torch).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ciffy.backend import Array, is_torch, zeros_nd, stack
from .primitives import normalize, cross, clone
from .transforms import (
    compute_frame_from_indices,
    compute_relative_transform,
    apply_relative_transform,
    is_purine,
)

if TYPE_CHECKING:
    from ciffy.biochemistry import Residue


# =============================================================================
# FrameIndices: Precomputed column indices for frame computation
# =============================================================================


@dataclass
class FrameIndices:
    """
    Precomputed column indices for fast frame computation.

    Stores the column indices needed to compute backbone link frames (O3', P)
    and alignment frames (glycosidic) for a specific residue type and atom
    ordering. Created once at initialization, then used for all frame ops.

    All indices are stored as numpy int32 arrays with -1 for unused slots.
    The format matches compute_frame_from_indices: [origin, z_ref, perp_ref].

    Attributes:
        prev_cols: (3,) Column indices for outgoing frame (O3' for RNA, C for protein).
        prev_z_toward: Z-axis direction for prev frame.
        next_cols: (3,) Column indices for incoming frame (P for RNA, N for protein).
        next_z_toward: Z-axis direction for next frame.
        glycosidic_cols: (3,) Column indices for glycosidic alignment frame.

    Example:
        >>> indices = FrameIndices.from_atoms(atoms, Residue.A)
        >>> origin, R = compute_frame_from_indices(coords, indices.prev_cols, indices.prev_z_toward)
    """

    prev_cols: np.ndarray      # (3,) int32: [origin, z_ref, perp_ref]
    prev_z_toward: bool        # Z-axis direction for prev frame
    next_cols: np.ndarray      # (3,) int32: [origin, z_ref, perp_ref]
    next_z_toward: bool        # Z-axis direction for next frame
    glycosidic_cols: np.ndarray  # (3,) int32: [C1', N9/N1, C4]

    @classmethod
    def from_atoms(cls, atoms: np.ndarray, residue: "Residue") -> "FrameIndices":
        """
        Create FrameIndices from an atoms array and residue type.

        Args:
            atoms: 1D array of atom type indices defining column order.
            residue: Residue type (e.g., Residue.A).

        Returns:
            FrameIndices with precomputed column indices.

        Raises:
            ValueError: If required atoms are missing from the atoms array.
        """
        from ciffy.biochemistry.linking import LINKING_BY_TYPE

        # Build atom -> column mapping (done once here, not at runtime)
        atoms_list = atoms.tolist() if hasattr(atoms, 'tolist') else list(atoms)

        def find_col(atom_value: int, name: str) -> int:
            try:
                return atoms_list.index(atom_value)
            except ValueError:
                raise ValueError(f"Atom {name} (value={atom_value}) not in atoms array")

        # Get linking definition for this molecule type
        link_def = LINKING_BY_TYPE.get(residue.molecule_type)
        if link_def is None:
            raise ValueError(f"No linking definition for {residue.molecule_type}")

        # Resolve prev frame (O3' for RNA, C for protein)
        prev_frame = link_def.prev_frame
        prev_cols = np.array([
            find_col(getattr(residue, prev_frame.origin).value, prev_frame.origin),
            find_col(getattr(residue, prev_frame.z_ref).value, prev_frame.z_ref),
            find_col(getattr(residue, prev_frame.perp_ref).value, prev_frame.perp_ref)
            if prev_frame.perp_ref else -1,
        ], dtype=np.int32)

        # Resolve next frame (P for RNA, N for protein)
        next_frame = link_def.next_frame
        next_cols = np.array([
            find_col(getattr(residue, next_frame.origin).value, next_frame.origin),
            find_col(getattr(residue, next_frame.z_ref).value, next_frame.z_ref),
            find_col(getattr(residue, next_frame.perp_ref).value, next_frame.perp_ref)
            if next_frame.perp_ref else -1,
        ], dtype=np.int32)

        # Resolve glycosidic frame (C1', N9/N1, C4)
        c1p_col = find_col(residue.C1p.value, "C1p")
        c4_col = find_col(residue.C4.value, "C4")
        if is_purine(residue):
            n_col = find_col(residue.N9.value, "N9")
        else:
            n_col = find_col(residue.N1.value, "N1")
        glycosidic_cols = np.array([c1p_col, n_col, c4_col], dtype=np.int32)

        return cls(
            prev_cols=prev_cols,
            prev_z_toward=prev_frame.z_toward_origin,
            next_cols=next_cols,
            next_z_toward=next_frame.z_toward_origin,
            glycosidic_cols=glycosidic_cols,
        )

    def to_torch(self, device: str = "cpu") -> "FrameIndices":
        """Convert indices to torch tensors on specified device."""
        import torch
        return FrameIndices(
            prev_cols=torch.from_numpy(self.prev_cols).to(device),
            prev_z_toward=self.prev_z_toward,
            next_cols=torch.from_numpy(self.next_cols).to(device),
            next_z_toward=self.next_z_toward,
            glycosidic_cols=torch.from_numpy(self.glycosidic_cols).to(device),
        )


# =============================================================================
# Frame Computation (using precomputed indices)
# =============================================================================


def compute_glycosidic_frame_indexed(
    coords: Array,
    indices: FrameIndices | np.ndarray,
) -> tuple[Array, Array]:
    """
    Compute the glycosidic frame for residue alignment using precomputed indices.

    Frame definition:
    - Origin: C1' atom
    - X-axis: Toward N9 (purines) or N1 (pyrimidines)
    - Z-axis: Normal to the C1'-N9-C4 plane
    - Y-axis: Completes right-handed system

    Args:
        coords: (n_atoms, 3) or (batch, n_atoms, 3) coordinates.
        indices: FrameIndices or (3,) array of [C1', N9/N1, C4] column indices.

    Returns:
        origin: (3,) or (batch, 3) C1' position.
        R: (3, 3) or (batch, 3, 3) rotation matrix [x, y, z] as columns.
    """
    if isinstance(indices, FrameIndices):
        cols = indices.glycosidic_cols
    else:
        cols = indices

    # Glycosidic frame has different construction than link frames
    # X-axis is primary (toward base), not Z-axis
    c1p = coords[..., int(cols[0]), :]
    n_pos = coords[..., int(cols[1]), :]
    c4_pos = coords[..., int(cols[2]), :]

    origin = clone(c1p)
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
    indices: FrameIndices | np.ndarray,
) -> Array:
    """
    Align each residue to a canonical local frame (glycosidic frame).

    Args:
        coords: (n_instances, n_atoms, 3) coordinate array (numpy or torch).
        indices: FrameIndices or (3,) array of glycosidic column indices.

    Returns:
        Aligned coordinates with same shape as input.
    """
    from ciffy.backend import zeros_like

    n_instances = coords.shape[0]
    aligned = zeros_like(coords)

    for i in range(n_instances):
        origin, R = compute_glycosidic_frame_indexed(coords[i], indices)
        aligned[i] = (coords[i] - origin) @ R

    return aligned


def align_and_compute_transform(
    coords: Array,
    next_coords: Array | None,
    indices: FrameIndices,
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
        indices: FrameIndices for this residue type.

    Returns:
        aligned_coords: (n_atoms, 3) coords in glycosidic frame.
        transform: (6,) SE(3) transform [axis-angle, translation].
    """
    # Align to glycosidic frame
    origin, R = compute_glycosidic_frame_indexed(coords, indices)
    aligned_coords = (coords - origin) @ R

    if next_coords is not None:
        # Transform next residue to same frame
        aligned_next = (next_coords - origin) @ R

        # Compute link transform
        o3p_origin, o3p_R = compute_frame_from_indices(
            aligned_coords, indices.prev_cols, indices.prev_z_toward
        )
        p_origin, p_R = compute_frame_from_indices(
            aligned_next, indices.next_cols, indices.next_z_toward
        )
        transform = compute_relative_transform(o3p_origin, o3p_R, p_origin, p_R)
    else:
        transform = zeros_nd((6,), like=coords)

    return aligned_coords, transform


def position_next_residue(
    coords1: Array,
    coords2: Array,
    rel_transform: Array,
    indices: FrameIndices,
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
        indices: Precomputed FrameIndices for this residue type.

    Returns:
        (n_atoms, 3) positioned coordinates of second residue.
    """
    # Compute O3' frame from coords1
    o3p_origin, o3p_R = compute_frame_from_indices(
        coords1, indices.prev_cols, indices.prev_z_toward
    )

    # Apply transform to get target P frame
    target_p_origin, target_p_R = apply_relative_transform(o3p_origin, o3p_R, rel_transform)

    # Compute current P frame from coords2
    current_p_origin, current_p_R = compute_frame_from_indices(
        coords2, indices.next_cols, indices.next_z_toward
    )

    # Compute rigid transformation to align current P frame to target P frame
    R_correction = target_p_R @ current_p_R.T
    t_correction = target_p_origin - R_correction @ current_p_origin

    # Apply transformation
    coords2_positioned = (R_correction @ coords2.T).T + t_correction

    if not is_torch(coords2_positioned):
        coords2_positioned = coords2_positioned.astype(np.float32)

    return coords2_positioned
