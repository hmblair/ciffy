"""
Data extraction and preprocessing for residue conformations.

This module provides:
- NumPy functions for training-time data extraction
- PyTorch functions for GPU-accelerated inference

Frame computation and SE(3) transforms are implemented in ciffy.geometry.
This module provides wrapper functions that accept atoms arrays (instead of
atom_to_col dict) for compatibility with training code.
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from collections import Counter
from typing import TYPE_CHECKING

import ciffy
from ciffy.backend import Array, to_numpy, is_torch
from ciffy.biochemistry import Scale
from ciffy.operations.reduction import Reduction

# Import shared geometry primitives
from ciffy.geometry import (
    compute_o3p_frame as _compute_o3p_frame_geometry,
    compute_p_frame as _compute_p_frame_geometry,
    compute_relative_transform as _compute_relative_transform_geometry,
    apply_relative_transform as _apply_relative_transform_geometry,
    axis_angle_to_rotation as _axis_angle_to_rotation_geometry,
    rotation_to_axis_angle as _rotation_to_axis_angle_geometry,
    compute_frame_from_indices as _compute_frame_from_indices_geometry,
    # New unified functions
    compute_glycosidic_frame as _compute_glycosidic_frame_geometry,
    is_purine as _is_purine_geometry,
    position_residue_fast as _position_residue_fast_geometry,
    cross, normalize, clone,
)

# Import shared utilities
from ciffy.utils import atoms_to_col_map

if TYPE_CHECKING:
    from ciffy.biochemistry import Residue


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

    Note:
        This is a thin wrapper around ciffy.geometry.is_purine for
        backward compatibility.
    """
    return _is_purine_geometry(residue)


def _atoms_to_col_map(atoms: Array) -> dict[int, int]:
    """
    Build atom value to column index mapping from atoms array.

    Args:
        atoms: 1D array of atom type indices.

    Returns:
        Dict mapping atom value to column index.

    Note:
        This is a thin wrapper around ciffy.utils.atoms_to_col_map for
        backward compatibility.
    """
    return atoms_to_col_map(atoms)


# =============================================================================
# Frame Computation Helpers (single source of truth for each frame type)
# =============================================================================


def compute_glycosidic_frame(
    coords: Array,
    atoms: Array,
    residue: "Residue",
) -> tuple[Array, Array]:
    """
    Compute the glycosidic frame for a residue.

    This is a wrapper around ciffy.geometry.compute_glycosidic_frame that
    accepts an atoms array instead of atom_to_col dict for convenience.

    Frame definition:
    - Origin: C1' atom
    - X-axis: Toward N9 (purines) or N1 (pyrimidines)
    - Z-axis: Normal to the C1'-N9-C4 plane
    - Y-axis: Completes right-handed system

    Args:
        coords: (n_atoms, 3) coordinates (numpy or torch).
        atoms: 1D array of atom type indices.
        residue: Residue type.

    Returns:
        origin: (3,) C1' position.
        R: (3, 3) rotation matrix [x, y, z] as columns.
    """
    atom_to_col = _atoms_to_col_map(atoms)
    return _compute_glycosidic_frame_geometry(coords, atom_to_col, residue)


def compute_o3p_frame(
    coords: Array,
    atoms: Array,
    residue: "Residue",
) -> tuple[Array, Array]:
    """
    Compute the O3' frame for a residue (used for backbone linking).

    This is a wrapper around ciffy.geometry.compute_o3p_frame that accepts
    an atoms array instead of atom_to_col dict for convenience.

    Frame definition:
    - Origin: O3' atom
    - Z-axis: Along C3'->O3' bond
    - X-axis: Perpendicular, in the C4'-C3'-O3' plane
    - Y-axis: Completes right-handed system

    Args:
        coords: (n_atoms, 3) coordinates (numpy or torch).
        atoms: 1D array of atom type indices.
        residue: Residue type.

    Returns:
        origin: (3,) O3' position.
        R: (3, 3) rotation matrix.
    """
    atom_to_col = _atoms_to_col_map(atoms)
    return _compute_o3p_frame_geometry(coords, atom_to_col, residue)


def compute_p_frame(
    coords: Array,
    atoms: Array,
    residue: "Residue",
) -> tuple[Array, Array]:
    """
    Compute the P frame for a residue (used for backbone linking).

    This is a wrapper around ciffy.geometry.compute_p_frame that accepts
    an atoms array instead of atom_to_col dict for convenience.

    Frame definition:
    - Origin: P atom
    - Z-axis: Along O5'->P bond
    - X-axis: Perpendicular, toward OP1
    - Y-axis: Completes right-handed system

    Args:
        coords: (n_atoms, 3) coordinates (numpy or torch).
        atoms: 1D array of atom type indices.
        residue: Residue type.

    Returns:
        origin: (3,) P position.
        R: (3, 3) rotation matrix.
    """
    atom_to_col = _atoms_to_col_map(atoms)
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
# Deprecated: PyTorch-specific wrappers (no longer needed)
# =============================================================================
# The geometry module now supports both NumPy and PyTorch backends via inline dispatch.
# Use the regular functions (compute_o3p_frame, position_next_residue, etc.) directly -
# they work with both backends.


# =============================================================================
# Single Residue Extraction
# =============================================================================


def extract_residues(
    cif_paths: list[Path],
    residue_type: "Residue",
    min_coverage: float = 0.9,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
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
        atoms: 1D int64 array of atom type indices in column order.

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

    return coords_out, np.array(common_atoms, dtype=np.int64)


def align_to_frame(
    coords: Array,
    atoms: Array,
    residue: "Residue",
) -> Array:
    """
    Align each residue to a canonical local frame (glycosidic frame).

    Args:
        coords: (n_instances, n_atoms, 3) coordinate array (numpy or torch).
        atoms: 1D array of atom type indices.
        residue: Residue type for looking up atom indices.

    Returns:
        Aligned coordinates with same shape as input.
    """
    n_instances = coords.shape[0]

    if is_torch(coords):
        aligned = torch.zeros_like(coords)
    else:
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
    coords: Array,
    atoms: Array,
    residue: "Residue",
) -> dict[str, float]:
    """
    Check C1'-N9/N1 glycosidic bond length statistics.

    Args:
        coords: (n_instances, n_atoms, 3) coordinate array (numpy or torch).
        atoms: 1D array of atom type indices.
        residue: Residue type.

    Returns:
        Dictionary with 'mean' and 'std' of the glycosidic bond length.
    """
    # Convert atoms to list for indexing
    atoms_list = atoms.tolist() if hasattr(atoms, 'tolist') else list(atoms)
    c1p_idx = atoms_list.index(residue.C1p.value)

    if is_purine(residue):
        n_idx = atoms_list.index(residue.N9.value)
        bond_name = "C1'-N9"
    else:
        n_idx = atoms_list.index(residue.N1.value)
        bond_name = "C1'-N1"

    if is_torch(coords):
        dists = torch.linalg.norm(coords[:, c1p_idx] - coords[:, n_idx], dim=-1)
        return {
            "bond": bond_name,
            "mean": float(dists.mean().item()),
            "std": float(dists.std().item()),
        }
    else:
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        atoms: 1D int64 array of atom type indices in column order.
    """
    from ciffy.biochemistry import Residue

    # Helper to get required link atoms for a residue type
    def get_required_link_atoms(res_type: "Residue") -> set[int]:
        return {
            res_type.C4p.value, res_type.C3p.value, res_type.O3p.value,
            res_type.P.value, res_type.O5p.value, res_type.OP1.value,
        }

    # Required atoms for the target residue type
    required_link_atoms_1 = get_required_link_atoms(residue_type)

    # Phase 1: Extract raw pairs with bond length filtering
    # For each target residue, get the transform to the NEXT residue (any type)
    all_pairs = []  # (raw_coords1, raw_atoms1, raw_coords2, raw_atoms2)

    for path in cif_paths:
        if verbose:
            print(f"Processing {path.name}...", end=" ")

        try:
            poly = ciffy.load(str(path)).poly()
            seq = to_numpy(poly.sequence)
            n_residues = len(seq)

            per_res_atoms = poly.reduce(poly.atoms, Scale.RESIDUE, Reduction.COLLATE)
            per_res_coords = poly.reduce(poly.coordinates, Scale.RESIDUE, Reduction.COLLATE)

            count = 0
            for idx1 in range(n_residues - 1):
                # First residue must be target type
                if seq[idx1] != residue_type.value:
                    continue

                # Second residue is simply the next one (any type)
                idx2 = idx1 + 1
                res_type_2 = Residue.from_index(int(seq[idx2]))

                atoms1 = to_numpy(per_res_atoms[idx1]).tolist()
                atoms2 = to_numpy(per_res_atoms[idx2]).tolist()

                # Both must have required link atoms (using correct residue types)
                if not required_link_atoms_1.issubset(set(atoms1)):
                    continue
                required_link_atoms_2 = get_required_link_atoms(res_type_2)
                if not required_link_atoms_2.issubset(set(atoms2)):
                    continue

                coords1 = to_numpy(per_res_coords[idx1])
                coords2 = to_numpy(per_res_coords[idx2])

                # Check O3'-P bond length (use each residue's own atom indices)
                o3p_idx_1 = atoms1.index(residue_type.O3p.value)
                p_idx_2 = atoms2.index(res_type_2.P.value)
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

    return coords_out, transforms_out, np.array(common_atoms, dtype=np.int64)


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
        atoms: 1D array of atom type indices.
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

    if not is_torch(coords2_positioned):
        coords2_positioned = coords2_positioned.astype(np.float32)

    return coords2_positioned


def compute_link_frames(
    coords1: Array,
    coords2: Array,
    atoms: Array,
    residue: "Residue",
) -> tuple[Array, Array, Array, Array]:
    """
    Compute frames at linking atoms (O3' of res1, P of res2).

    This is a convenience function that combines compute_o3p_frame and compute_p_frame.

    Args:
        coords1: (n_atoms, 3) coordinates of first residue (numpy or torch).
        coords2: (n_atoms, 3) coordinates of second residue.
        atoms: 1D array of atom type indices.
        residue: Residue type.

    Returns:
        o3p_origin, o3p_R, p_origin, p_R: Frame positions and rotation matrices.
    """
    o3p_origin, o3p_R = compute_o3p_frame(coords1, atoms, residue)
    p_origin, p_R = compute_p_frame(coords2, atoms, residue)
    return o3p_origin, o3p_R, p_origin, p_R


# =============================================================================
# Dataset Classes
# =============================================================================


class ResidueDataset:
    """
    Dataset of residue conformations extracted from CIF files.

    Extracts residues of a specific type with SE(3) transforms to the next
    residue, suitable for training flow models.

    Example:
        >>> from ciffy.nn.flow.residue import ResidueDataset
        >>> from ciffy.biochemistry import Residue
        >>> dataset = ResidueDataset(cif_paths, Residue.A)
        >>> print(f"Found {len(dataset)} adenine residues")
        >>> coords, transform = dataset[0]
    """

    def __init__(
        self,
        cif_paths: list[Path],
        residue: "Residue",
        min_coverage: float = 0.9,
        max_bond_length: float = 2.0,
        verbose: bool = True,
    ):
        """
        Initialize dataset by extracting residues from CIF files.

        Args:
            cif_paths: List of paths to CIF files.
            residue: Residue type to extract (e.g., Residue.A).
            min_coverage: Minimum fraction of instances an atom must appear in.
            max_bond_length: Maximum O3'-P distance for valid linkage.
            verbose: Print extraction progress.
        """
        self.residue = residue
        self.min_coverage = min_coverage
        self.max_bond_length = max_bond_length

        # Extract data
        coords, transforms, atoms = extract_residues_with_links(
            cif_paths,
            residue,
            min_coverage=min_coverage,
            max_bond_length=max_bond_length,
            verbose=verbose,
        )

        self.coords = coords  # (n_instances, n_atoms, 3)
        self.transforms = transforms  # (n_instances, 6)
        self.atoms = atoms  # (n_atoms,) atom type indices
        self.n_atoms = len(atoms)

        # Create extended representation (flattened coords + transforms)
        n_instances = len(coords)
        coords_flat = coords.reshape(n_instances, -1)
        self._data = np.concatenate([coords_flat, transforms], axis=1)

    def __len__(self) -> int:
        """Number of residue instances."""
        return len(self.coords)

    def __getitem__(self, idx: int) -> np.ndarray:
        """
        Get extended representation for a residue.

        Args:
            idx: Instance index.

        Returns:
            1D array of [flattened_coords, transform] (n_atoms*3 + 6,).
        """
        return self._data[idx]

    @property
    def data(self) -> np.ndarray:
        """Full dataset as (n_instances, n_atoms*3 + 6) array."""
        return self._data

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the dataset (n_instances, n_dims)."""
        return self._data.shape
