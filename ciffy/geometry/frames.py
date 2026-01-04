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

from ciffy.backend import Array, zeros_nd
from .transforms import (
    extract_frame_positions,
    frame_from_positions,
    compute_relative_transform,
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

    Frame definition (Z-primary convention):
    - Origin: C1' atom
    - Z-axis: Toward N9 (purines) or N1 (pyrimidines)
    - X-axis: Derived toward C4/C2 via Gram-Schmidt
    - Y-axis: Completes right-handed system (Y = Z × X)

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

    # Extract positions and compute frame using canonical Z-primary convention
    positions = extract_frame_positions(coords, atoms, frame_def)
    return frame_from_positions(positions)


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


__all__ = [
    "compute_glycosidic_frame",
    "align_and_compute_transform",
]
