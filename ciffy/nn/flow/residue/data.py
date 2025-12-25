"""
Data extraction and preprocessing for residue conformations.

This module provides:
- NumPy functions for training-time data extraction
- PyTorch functions for GPU-accelerated inference

Frame computation and SE(3) transforms are implemented in ciffy.geometry.
This module provides wrapper functions that accept atoms list (instead of
atom_to_col dict) for backward compatibility with training code.
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from collections import Counter
from typing import TYPE_CHECKING

import ciffy
from ciffy.backend import to_numpy
from ciffy.types import Scale
from ciffy.operations.reduction import Reduction

# Import shared geometry primitives
from ciffy.geometry import (
    compute_o3p_frame as _compute_o3p_frame_geometry,
    compute_p_frame as _compute_p_frame_geometry,
    compute_relative_transform as _compute_relative_transform_geometry,
    apply_relative_transform as _apply_relative_transform_geometry,
    axis_angle_to_rotation as _axis_angle_to_rotation_geometry,
    rotation_to_axis_angle as _rotation_to_axis_angle_geometry,
    position_residue as _position_residue_geometry,
    compute_frame_from_indices as _compute_frame_from_indices_geometry,
    cross, normalize, clone,
)

if TYPE_CHECKING:
    from ciffy.biochemistry import Residue


# =============================================================================
# Frame Computation Helpers (single source of truth for each frame type)
# =============================================================================


def compute_glycosidic_frame(
    coords: np.ndarray,
    atoms: list[int],
    residue: "Residue",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the glycosidic frame for a residue.

    Frame definition:
    - Origin: C1' atom
    - X-axis: Toward N9 (purines) or N1 (pyrimidines)
    - Z-axis: Normal to the C1'-N9-C4 plane
    - Y-axis: Completes right-handed system

    Args:
        coords: (n_atoms, 3) coordinates.
        atoms: List of atom type indices.
        residue: Residue type.

    Returns:
        origin: (3,) C1' position.
        R: (3, 3) rotation matrix [x, y, z] as columns.
    """
    atom_to_col = {a: i for i, a in enumerate(atoms)}

    c1p_idx = atom_to_col[residue.C1p.value]
    c4_idx = atom_to_col[residue.C4.value]

    # N9 for purines (A, G), N1 for pyrimidines (C, U)
    try:
        n_idx = atom_to_col[residue.N9.value]
    except (KeyError, AttributeError):
        n_idx = atom_to_col[residue.N1.value]

    origin = coords[c1p_idx].copy()
    n_pos = coords[n_idx]
    c4_pos = coords[c4_idx]

    x_axis = n_pos - origin
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_temp = c4_pos - origin
    z_axis = np.cross(x_axis, y_temp)
    z_axis = z_axis / np.linalg.norm(z_axis)

    y_axis = np.cross(z_axis, x_axis)

    R = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32)
    return origin.astype(np.float32), R


def compute_o3p_frame(
    coords: np.ndarray,
    atoms: list[int],
    residue: "Residue",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the O3' frame for a residue (used for backbone linking).

    This is a wrapper around ciffy.geometry.compute_o3p_frame that accepts
    an atoms list instead of atom_to_col dict for backward compatibility.

    Frame definition:
    - Origin: O3' atom
    - Z-axis: Along C3'->O3' bond
    - X-axis: Perpendicular, in the C4'-C3'-O3' plane
    - Y-axis: Completes right-handed system

    Args:
        coords: (n_atoms, 3) coordinates.
        atoms: List of atom type indices.
        residue: Residue type.

    Returns:
        origin: (3,) O3' position.
        R: (3, 3) rotation matrix.
    """
    atom_to_col = {a: i for i, a in enumerate(atoms)}
    return _compute_o3p_frame_geometry(coords, atom_to_col, residue)


def compute_p_frame(
    coords: np.ndarray,
    atoms: list[int],
    residue: "Residue",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the P frame for a residue (used for backbone linking).

    This is a wrapper around ciffy.geometry.compute_p_frame that accepts
    an atoms list instead of atom_to_col dict for backward compatibility.

    Frame definition:
    - Origin: P atom
    - Z-axis: Along O5'->P bond
    - X-axis: Perpendicular, toward OP1
    - Y-axis: Completes right-handed system

    Args:
        coords: (n_atoms, 3) coordinates.
        atoms: List of atom type indices.
        residue: Residue type.

    Returns:
        origin: (3,) P position.
        R: (3, 3) rotation matrix.
    """
    atom_to_col = {a: i for i, a in enumerate(atoms)}
    return _compute_p_frame_geometry(coords, atom_to_col, residue)


# =============================================================================
# SE(3) Transform Helpers
# =============================================================================
# These are thin wrappers around ciffy.geometry functions for backward compatibility.


def _rotation_matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to axis-angle representation."""
    return _rotation_to_axis_angle_geometry(R)


def _axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle to rotation matrix (Rodrigues' formula)."""
    return _axis_angle_to_rotation_geometry(axis_angle)


def compute_relative_transform(
    origin1: np.ndarray,
    R1: np.ndarray,
    origin2: np.ndarray,
    R2: np.ndarray,
) -> np.ndarray:
    """
    Compute relative SE(3) transform from frame 1 to frame 2.

    Wrapper around ciffy.geometry.compute_relative_transform.

    Args:
        origin1, R1: First frame (position and rotation).
        origin2, R2: Second frame (position and rotation).

    Returns:
        6D vector: [axis-angle (3), translation in frame1 coords (3)].
    """
    return _compute_relative_transform_geometry(origin1, R1, origin2, R2)


def apply_relative_transform(
    origin1: np.ndarray,
    R1: np.ndarray,
    rel_transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply relative transform to get frame 2 from frame 1.

    Wrapper around ciffy.geometry.apply_relative_transform.

    Args:
        origin1, R1: Source frame.
        rel_transform: 6D vector [axis-angle (3), translation (3)].

    Returns:
        origin2, R2: Target frame.
    """
    return _apply_relative_transform_geometry(origin1, R1, rel_transform)


# =============================================================================
# PyTorch GPU-Compatible Implementations
# =============================================================================
# The geometry module now supports both NumPy and PyTorch backends via inline dispatch.
# These wrappers are kept for backward compatibility.


def _axis_angle_to_rotation_matrix_torch(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle to rotation matrix (Rodrigues' formula).

    Args:
        axis_angle: (3,) axis-angle vector.

    Returns:
        (3, 3) rotation matrix.
    """
    return _axis_angle_to_rotation_geometry(axis_angle)


def compute_o3p_frame_torch(
    coords: torch.Tensor,
    atom_to_col: dict[int, int],
    residue: "Residue",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the O3' frame for a residue (PyTorch version).

    Uses ciffy.geometry.compute_o3p_frame which has inline backend dispatch.

    Args:
        coords: (n_atoms, 3) coordinates tensor.
        atom_to_col: Dict mapping atom type index to column index.
        residue: Residue type.

    Returns:
        origin: (3,) O3' position.
        R: (3, 3) rotation matrix.
    """
    return _compute_o3p_frame_geometry(coords, atom_to_col, residue)


def compute_p_frame_torch(
    coords: torch.Tensor,
    atom_to_col: dict[int, int],
    residue: "Residue",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the P frame for a residue (PyTorch version).

    Uses ciffy.geometry.compute_p_frame which has inline backend dispatch.

    Args:
        coords: (n_atoms, 3) coordinates tensor.
        atom_to_col: Dict mapping atom type index to column index.
        residue: Residue type.

    Returns:
        origin: (3,) P position.
        R: (3, 3) rotation matrix.
    """
    return _compute_p_frame_geometry(coords, atom_to_col, residue)


def apply_relative_transform_torch(
    origin1: torch.Tensor,
    R1: torch.Tensor,
    rel_transform: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply relative transform to get frame 2 from frame 1 (PyTorch version).

    Uses ciffy.geometry.apply_relative_transform which has inline backend dispatch.

    Args:
        origin1: (3,) source frame origin.
        R1: (3, 3) source frame rotation.
        rel_transform: (6,) vector [axis-angle (3), translation (3)].

    Returns:
        origin2, R2: Target frame origin and rotation.
    """
    return _apply_relative_transform_geometry(origin1, R1, rel_transform)


def position_next_residue_torch(
    coords1: torch.Tensor,
    coords2: torch.Tensor,
    rel_transform: torch.Tensor,
    atom_to_col: dict[int, int],
    residue: "Residue",
) -> torch.Tensor:
    """
    Position residue 2 relative to residue 1 using the link transform (PyTorch version).

    This is the GPU-compatible version of position_next_residue. It keeps all
    computation on the same device as the input tensors.

    Note: For performance-critical code, use position_next_residue_fast() with
    pre-resolved frame column indices to avoid Python attribute lookups.

    Args:
        coords1: (n_atoms, 3) coordinates of first residue.
        coords2: (n_atoms, 3) coordinates of second residue (in canonical frame).
        rel_transform: (6,) SE(3) transform [axis-angle, translation].
        atom_to_col: Dict mapping atom type index to column index.
        residue: Residue type.

    Returns:
        (n_atoms, 3) positioned coordinates of second residue.
    """
    # Compute O3' frame from coords1
    o3p_origin, o3p_R = compute_o3p_frame_torch(coords1, atom_to_col, residue)

    # Apply transform to get target P frame
    target_p_origin, target_p_R = apply_relative_transform_torch(
        o3p_origin, o3p_R, rel_transform
    )

    # Compute current P frame from coords2
    current_p_origin, current_p_R = compute_p_frame_torch(coords2, atom_to_col, residue)

    # Compute rigid transformation to align current P frame to target P frame
    R_correction = target_p_R @ current_p_R.T
    t_correction = target_p_origin - R_correction @ current_p_origin

    # Apply transformation
    coords2_positioned = (R_correction @ coords2.T).T + t_correction

    return coords2_positioned


def position_next_residue_fast(
    coords1: torch.Tensor,
    coords2: torch.Tensor,
    rel_transform: torch.Tensor,
    prev_frame_cols: tuple[int, int, int | None],
    prev_z_toward_origin: bool,
    next_frame_cols: tuple[int, int, int | None],
    next_z_toward_origin: bool,
) -> torch.Tensor:
    """
    Position residue 2 relative to residue 1 using pre-resolved frame indices.

    This is the fast path for residue positioning. Uses pre-resolved column
    indices to compute frames with pure tensor math (no Python attribute lookups).
    The frame indices should be computed once at model initialization.

    Args:
        coords1: (n_atoms, 3) coordinates of first residue.
        coords2: (n_atoms, 3) coordinates of second residue (in canonical frame).
        rel_transform: (6,) SE(3) transform [axis-angle, translation].
        prev_frame_cols: Pre-resolved (origin, z_ref, perp_ref) column indices for
            outgoing frame of coords1 (e.g., O3' frame for RNA).
        prev_z_toward_origin: Z-axis direction for prev frame.
        next_frame_cols: Pre-resolved column indices for incoming frame of coords2
            (e.g., P frame for RNA).
        next_z_toward_origin: Z-axis direction for next frame.

    Returns:
        (n_atoms, 3) positioned coordinates of second residue.

    Example:
        >>> # Pre-resolve indices at model init
        >>> link_def = LINKING_BY_TYPE[residue.molecule_type]
        >>> prev_cols = link_def.prev_frame.resolve(residue, atom_to_col)
        >>> next_cols = link_def.next_frame.resolve(residue, atom_to_col)
        >>>
        >>> # Fast positioning at runtime
        >>> positioned = position_next_residue_fast(
        ...     coords1, coords2, transform,
        ...     prev_cols, link_def.prev_frame.z_toward_origin,
        ...     next_cols, link_def.next_frame.z_toward_origin,
        ... )
    """
    # Compute outgoing frame from coords1 using pre-resolved indices
    prev_origin, prev_R = _compute_frame_from_indices_geometry(
        coords1, prev_frame_cols, prev_z_toward_origin
    )

    # Apply transform to get target incoming frame
    target_origin, target_R = apply_relative_transform_torch(
        prev_origin, prev_R, rel_transform
    )

    # Compute current incoming frame from coords2 using pre-resolved indices
    current_origin, current_R = _compute_frame_from_indices_geometry(
        coords2, next_frame_cols, next_z_toward_origin
    )

    # Compute rigid transformation to align current frame to target frame
    R_correction = target_R @ current_R.T
    t_correction = target_origin - R_correction @ current_origin

    # Apply transformation
    coords2_positioned = (R_correction @ coords2.T).T + t_correction

    return coords2_positioned


# =============================================================================
# Single Residue Extraction
# =============================================================================


def extract_residues(
    cif_paths: list[Path],
    residue_type: "Residue",
    min_coverage: float = 0.9,
    verbose: bool = True,
) -> tuple[np.ndarray, list[int]]:
    """
    Extract all instances of a residue type from multiple structures.

    Args:
        cif_paths: List of paths to CIF files.
        residue_type: Residue enum (e.g., Residue.A for adenosine).
        min_coverage: Minimum fraction of instances an atom must appear in
            to be included. Default 0.9 excludes rare terminal atoms.
        verbose: Print progress information.

    Returns:
        coords: (n_instances, n_atoms, 3) coordinate array.
        atoms: List of atom type indices in column order.

    Example:
        >>> from ciffy.biochemistry import Residue
        >>> coords, atoms = extract_residues(cif_paths, Residue.A)
        >>> print(f"Found {len(coords)} adenosines with {len(atoms)} atoms each")
    """
    all_instances = []

    for path in cif_paths:
        if verbose:
            print(f"Processing {path.name}...", end=" ")

        try:
            poly = ciffy.load(str(path)).poly()
            seq = to_numpy(poly.sequence)
            indices = [i for i in range(len(seq)) if seq[i] == residue_type.value]

            if not indices:
                if verbose:
                    print("no matches")
                continue

            per_res_atoms = poly.reduce(poly.atoms, Scale.RESIDUE, Reduction.COLLATE)
            per_res_coords = poly.reduce(poly.coordinates, Scale.RESIDUE, Reduction.COLLATE)

            for idx in indices:
                atoms = to_numpy(per_res_atoms[idx]).tolist()
                coords = to_numpy(per_res_coords[idx])
                all_instances.append((coords, atoms))

            if verbose:
                print(f"{len(indices)} residues")

        except Exception as e:
            if verbose:
                print(f"error: {e}")

    if not all_instances:
        raise ValueError(f"No {residue_type.name} residues found")

    if verbose:
        print(f"\nCollected {len(all_instances)} raw instances")

    # Find atoms present in most instances
    atom_counts = Counter()
    for _, atoms in all_instances:
        atom_counts.update(atoms)

    min_count = int(len(all_instances) * min_coverage)
    common_atoms = sorted([a for a, c in atom_counts.items() if c >= min_count])

    if verbose:
        print(f"Atoms with >={min_coverage*100:.0f}% coverage: {len(common_atoms)}")

    # Filter and build dense array
    common_set = set(common_atoms)
    filtered = [inst for inst in all_instances if common_set.issubset(set(inst[1]))]

    if verbose:
        print(f"Instances with all common atoms: {len(filtered)}")

    coords_out = np.zeros((len(filtered), len(common_atoms), 3), dtype=np.float32)
    atom_to_col = {a: c for c, a in enumerate(common_atoms)}

    for i, (coords, atoms) in enumerate(filtered):
        for atom_idx, coord in zip(atoms, coords):
            if atom_idx in atom_to_col:
                coords_out[i, atom_to_col[atom_idx]] = coord

    return coords_out, common_atoms


def align_to_frame(
    coords: np.ndarray,
    atoms: list[int],
    residue: "Residue",
) -> np.ndarray:
    """
    Align each residue to a canonical local frame (glycosidic frame).

    Args:
        coords: (n_instances, n_atoms, 3) coordinate array.
        atoms: List of atom type indices.
        residue: Residue type for looking up atom indices.

    Returns:
        Aligned coordinates with same shape as input.
    """
    n_instances = coords.shape[0]
    aligned = np.zeros_like(coords)

    for i in range(n_instances):
        origin, R = compute_glycosidic_frame(coords[i], atoms, residue)
        aligned[i] = (coords[i] - origin) @ R

    return aligned


def compute_pca(
    coords: np.ndarray,
    n_components: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PCA on flattened coordinates.

    Args:
        coords: (n_instances, n_atoms, 3) coordinate array.
        n_components: Number of components to keep. If None, keep all.

    Returns:
        V: (k, d) PCA component matrix where k=n_components, d=n_atoms*3.
        mean: (d,) mean coordinates.
        singular_values: All singular values.
        var_explained: Cumulative variance explained for each component.
    """
    coords_flat = coords.reshape(len(coords), -1)
    mean = coords_flat.mean(axis=0)
    centered = coords_flat - mean

    _, s, Vt = np.linalg.svd(centered, full_matrices=False)

    var_explained = np.cumsum(s ** 2) / (s ** 2).sum()

    if n_components is not None:
        Vt = Vt[:n_components]

    return Vt.astype(np.float32), mean.astype(np.float32), s, var_explained


def check_bond_lengths(
    coords: np.ndarray,
    atoms: list[int],
    residue: "Residue",
) -> dict[str, float]:
    """
    Check C1'-N9/N1 glycosidic bond length statistics.

    Args:
        coords: (n_instances, n_atoms, 3) coordinate array.
        atoms: List of atom type indices.
        residue: Residue type.

    Returns:
        Dictionary with 'mean' and 'std' of the glycosidic bond length.
    """
    c1p_idx = atoms.index(residue.C1p.value)

    try:
        n_idx = atoms.index(residue.N9.value)
        bond_name = "C1'-N9"
    except (ValueError, AttributeError):
        n_idx = atoms.index(residue.N1.value)
        bond_name = "C1'-N1"

    dists = np.linalg.norm(coords[:, c1p_idx] - coords[:, n_idx], axis=-1)

    return {
        "bond": bond_name,
        "mean": float(dists.mean()),
        "std": float(dists.std()),
    }


# =============================================================================
# Extended Residue Extraction (with backbone link transforms)
# =============================================================================


def _remap_coords_to_common_atoms(
    raw_coords: np.ndarray,
    raw_atoms: list[int],
    common_atoms: list[int],
) -> np.ndarray:
    """
    Remap raw coordinates to common atom ordering.

    Args:
        raw_coords: (n_raw_atoms, 3) coordinates in raw ordering.
        raw_atoms: List of atom type indices for raw_coords.
        common_atoms: Target atom ordering.

    Returns:
        (n_common_atoms, 3) coordinates in common atom ordering.
    """
    atom_to_col = {a: c for c, a in enumerate(common_atoms)}
    n_atoms = len(common_atoms)
    coords = np.zeros((n_atoms, 3), dtype=np.float32)

    for atom_idx, coord in zip(raw_atoms, raw_coords):
        if atom_idx in atom_to_col:
            coords[atom_to_col[atom_idx]] = coord

    return coords


def extract_residues_with_links(
    cif_paths: list[Path],
    residue_type: "Residue",
    min_coverage: float = 0.9,
    max_bond_length: float = 2.0,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Extract residues with SE(3) transforms to next residue.

    This creates an extended representation where each residue includes
    information about how it connects to the next residue in the chain.

    Data flow:
    1. Extract adjacent residue pairs from structures
    2. Filter by O3'-P bond length to ensure true connectivity
    3. Find common atoms across all instances
    4. For each pair:
       a. Remap both residues to common atom ordering
       b. Align coords1 to canonical (glycosidic) frame
       c. Apply SAME transform to coords2 (preserving relative geometry)
       d. Compute O3'->P transform from aligned coordinates

    Args:
        cif_paths: List of paths to CIF files.
        residue_type: Residue enum (e.g., Residue.A for adenosine).
        min_coverage: Minimum fraction of instances an atom must appear in.
        max_bond_length: Maximum O3'-P distance to accept (filters chain breaks).
        verbose: Print progress information.

    Returns:
        coords: (n_instances, n_atoms, 3) aligned first-residue coordinates.
        transforms: (n_instances, 6) SE(3) transforms [axis-angle, translation].
        atoms: List of atom type indices in column order.
    """
    # Required atoms for link computation
    required_link_atoms = {
        residue_type.C4p.value, residue_type.C3p.value, residue_type.O3p.value,
        residue_type.P.value, residue_type.O5p.value, residue_type.OP1.value,
    }

    # Phase 1: Extract raw pairs with bond length filtering
    all_pairs = []  # (raw_coords1, raw_atoms1, raw_coords2, raw_atoms2)

    for path in cif_paths:
        if verbose:
            print(f"Processing {path.name}...", end=" ")

        try:
            poly = ciffy.load(str(path)).poly()
            seq = to_numpy(poly.sequence)
            residue_indices = [i for i in range(len(seq)) if seq[i] == residue_type.value]

            if len(residue_indices) < 2:
                if verbose:
                    print("< 2 residues")
                continue

            per_res_atoms = poly.reduce(poly.atoms, Scale.RESIDUE, Reduction.COLLATE)
            per_res_coords = poly.reduce(poly.coordinates, Scale.RESIDUE, Reduction.COLLATE)

            count = 0
            for i in range(len(residue_indices) - 1):
                idx1, idx2 = residue_indices[i], residue_indices[i + 1]

                # Must be truly adjacent in sequence
                if idx2 != idx1 + 1:
                    continue

                atoms1 = to_numpy(per_res_atoms[idx1]).tolist()
                atoms2 = to_numpy(per_res_atoms[idx2]).tolist()

                # Both must have required link atoms
                if not required_link_atoms.issubset(set(atoms1)):
                    continue
                if not required_link_atoms.issubset(set(atoms2)):
                    continue

                coords1 = to_numpy(per_res_coords[idx1])
                coords2 = to_numpy(per_res_coords[idx2])

                # Check O3'-P bond length (use each residue's own atom ordering)
                o3p_idx_1 = atoms1.index(residue_type.O3p.value)
                p_idx_2 = atoms2.index(residue_type.P.value)
                bond_length = np.linalg.norm(coords2[p_idx_2] - coords1[o3p_idx_1])

                if bond_length > max_bond_length:
                    continue

                all_pairs.append((coords1, atoms1, coords2, atoms2))
                count += 1

            if verbose:
                print(f"{count} pairs")

        except Exception as e:
            if verbose:
                print(f"error: {e}")

    if not all_pairs:
        raise ValueError(f"No {residue_type.name} residue pairs found")

    if verbose:
        print(f"\nCollected {len(all_pairs)} residue pairs")

    # Phase 2: Find common atoms across first residues
    atom_counts = Counter()
    for _, atoms1, _, _ in all_pairs:
        atom_counts.update(atoms1)

    min_count = int(len(all_pairs) * min_coverage)
    common_atoms = sorted([a for a, c in atom_counts.items() if c >= min_count])

    if verbose:
        print(f"Atoms with >={min_coverage*100:.0f}% coverage: {len(common_atoms)}")

    # Phase 3: Filter to pairs where both residues have all common atoms
    common_set = set(common_atoms)
    filtered_pairs = []
    for raw_coords1, atoms1, raw_coords2, atoms2 in all_pairs:
        if common_set.issubset(set(atoms1)) and common_set.issubset(set(atoms2)):
            filtered_pairs.append((raw_coords1, atoms1, raw_coords2, atoms2))

    if verbose:
        print(f"Pairs with all common atoms: {len(filtered_pairs)}")

    # Phase 4: Remap, align, and compute transforms
    n_pairs = len(filtered_pairs)
    n_atoms = len(common_atoms)

    coords_out = np.zeros((n_pairs, n_atoms, 3), dtype=np.float32)
    transforms_out = np.zeros((n_pairs, 6), dtype=np.float32)

    for i, (raw_coords1, atoms1, raw_coords2, atoms2) in enumerate(filtered_pairs):
        # Remap both residues to common atom ordering
        coords1 = _remap_coords_to_common_atoms(raw_coords1, atoms1, common_atoms)
        coords2 = _remap_coords_to_common_atoms(raw_coords2, atoms2, common_atoms)

        # Compute alignment frame from residue 1
        origin, R = compute_glycosidic_frame(coords1, common_atoms, residue_type)

        # Apply SAME transformation to both residues (preserves relative geometry)
        coords1_aligned = (coords1 - origin) @ R
        coords2_aligned = (coords2 - origin) @ R

        # Compute link transform: O3' frame of res1 -> P frame of res2
        o3p_origin, o3p_R = compute_o3p_frame(coords1_aligned, common_atoms, residue_type)
        p_origin, p_R = compute_p_frame(coords2_aligned, common_atoms, residue_type)
        transform = compute_relative_transform(o3p_origin, o3p_R, p_origin, p_R)

        coords_out[i] = coords1_aligned
        transforms_out[i] = transform

    return coords_out, transforms_out, common_atoms


def position_next_residue(
    coords1: np.ndarray,
    coords2: np.ndarray,
    rel_transform: np.ndarray,
    atoms: list[int],
    residue: "Residue",
) -> np.ndarray:
    """
    Position residue 2 relative to residue 1 using the link transform.

    This is the inverse of transform extraction: given coords1 and a transform,
    position coords2 so that its P frame matches the target derived from
    coords1's O3' frame + transform.

    Args:
        coords1: (n_atoms, 3) coordinates of first residue.
        coords2: (n_atoms, 3) coordinates of second residue (in canonical frame).
        rel_transform: (6,) SE(3) transform [axis-angle, translation].
        atoms: List of atom type indices.
        residue: Residue type.

    Returns:
        (n_atoms, 3) positioned coordinates of second residue.
    """
    # Compute O3' frame from coords1
    o3p_origin, o3p_R = compute_o3p_frame(coords1, atoms, residue)

    # Apply transform to get target P frame
    target_p_origin, target_p_R = apply_relative_transform(o3p_origin, o3p_R, rel_transform)

    # Compute current P frame from coords2
    current_p_origin, current_p_R = compute_p_frame(coords2, atoms, residue)

    # Compute rigid transformation to align current P frame to target P frame
    R_correction = target_p_R @ current_p_R.T
    t_correction = target_p_origin - R_correction @ current_p_origin

    # Apply transformation
    coords2_positioned = (R_correction @ coords2.T).T + t_correction

    return coords2_positioned.astype(np.float32)


# Legacy compatibility aliases
def compute_link_frames(
    coords1: np.ndarray,
    coords2: np.ndarray,
    atoms: list[int],
    residue: "Residue",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute frames at linking atoms (O3' of res1, P of res2).

    This is a convenience function that combines compute_o3p_frame and compute_p_frame.
    """
    o3p_origin, o3p_R = compute_o3p_frame(coords1, atoms, residue)
    p_origin, p_R = compute_p_frame(coords2, atoms, residue)
    return o3p_origin, o3p_R, p_origin, p_R
