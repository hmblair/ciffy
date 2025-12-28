"""
Data extraction and preprocessing for residue conformations.

This module provides:
- FrameIndices: Precomputed column indices for fast frame computation
- Data extraction from CIF files for flow model training
- Residue positioning for chain assembly

All frame computations use precomputed array indices (no dict lookups at runtime).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import torch

import ciffy
from ciffy.backend import Array, to_numpy, is_torch
from ciffy.biochemistry import Scale
from ciffy.operations.reduction import Reduction
from ciffy.geometry import (
    compute_frame_from_indices,
    compute_relative_transform,
    apply_relative_transform,
    is_purine,
    normalize,
    cross,
    clone,
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


def compute_glycosidic_frame(
    coords: Array,
    indices: FrameIndices | np.ndarray,
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
        R = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    else:
        R = np.stack([x_axis, y_axis, z_axis], axis=-1).astype(np.float32)
        origin = origin.astype(np.float32)

    return origin, R


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
    n_instances = coords.shape[0]

    if is_torch(coords):
        aligned = torch.zeros_like(coords)
    else:
        aligned = np.zeros_like(coords)

    for i in range(n_instances):
        origin, R = compute_glycosidic_frame(coords[i], indices)
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

    Data flow:
    1. Extract adjacent residue pairs from structures
    2. Filter by O3'-P bond length to ensure true connectivity
    3. Find common atoms across all instances
    4. Build FrameIndices once for the common atom ordering
    5. For each pair: align, compute link transform

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
    from ciffy.biochemistry.linking import LINKING_BY_TYPE

    # Get linking definition for required atoms
    link_def = LINKING_BY_TYPE.get(residue_type.molecule_type)
    if link_def is None:
        raise ValueError(f"No linking definition for {residue_type.molecule_type}")

    # Required atoms for linking (from both frames)
    def get_required_atoms(res_type: "Residue") -> set[int]:
        return link_def.required_atoms(res_type)

    required_atoms_1 = get_required_atoms(residue_type)

    # Phase 1: Extract raw pairs with bond length filtering
    all_pairs = []

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
                if seq[idx1] != residue_type.value:
                    continue

                idx2 = idx1 + 1
                res_type_2 = Residue.from_index(int(seq[idx2]))

                atoms1 = to_numpy(per_res_atoms[idx1]).tolist()
                atoms2 = to_numpy(per_res_atoms[idx2]).tolist()

                # Check required atoms
                if not required_atoms_1.issubset(set(atoms1)):
                    continue
                if not get_required_atoms(res_type_2).issubset(set(atoms2)):
                    continue

                coords1 = to_numpy(per_res_coords[idx1])
                coords2 = to_numpy(per_res_coords[idx2])

                # Check bond length
                o3p_idx = atoms1.index(residue_type.O3p.value)
                p_idx = atoms2.index(res_type_2.P.value)
                if np.linalg.norm(coords2[p_idx] - coords1[o3p_idx]) > max_bond_length:
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

    # Phase 2: Find common atoms
    atom_counts = Counter()
    for _, atoms1, _, _ in all_pairs:
        atom_counts.update(atoms1)

    min_count = int(len(all_pairs) * min_coverage)
    common_atoms = sorted([a for a, c in atom_counts.items() if c >= min_count])

    if verbose:
        print(f"Atoms with >={min_coverage*100:.0f}% coverage: {len(common_atoms)}")

    # Phase 3: Filter pairs with all common atoms
    common_set = set(common_atoms)
    filtered_pairs = [
        (c1, a1, c2, a2) for c1, a1, c2, a2 in all_pairs
        if common_set.issubset(set(a1)) and common_set.issubset(set(a2))
    ]

    if verbose:
        print(f"Pairs with all common atoms: {len(filtered_pairs)}")

    # Phase 4: Build FrameIndices once for the common atom ordering
    atoms_array = np.array(common_atoms, dtype=np.int64)
    indices = FrameIndices.from_atoms(atoms_array, residue_type)

    # Phase 5: Remap, align, and compute transforms
    n_pairs = len(filtered_pairs)
    n_atoms = len(common_atoms)

    coords_out = np.zeros((n_pairs, n_atoms, 3), dtype=np.float32)
    transforms_out = np.zeros((n_pairs, 6), dtype=np.float32)

    for i, (raw_coords1, atoms1, raw_coords2, atoms2) in enumerate(filtered_pairs):
        # Remap to common atom ordering
        coords1 = _remap_coords_to_common_atoms(raw_coords1, atoms1, common_atoms)
        coords2 = _remap_coords_to_common_atoms(raw_coords2, atoms2, common_atoms)

        # Align to glycosidic frame
        origin, R = compute_glycosidic_frame(coords1, indices)
        coords1_aligned = (coords1 - origin) @ R
        coords2_aligned = (coords2 - origin) @ R

        # Compute link transform using precomputed indices
        o3p_origin, o3p_R = compute_frame_from_indices(
            coords1_aligned, indices.prev_cols, indices.prev_z_toward
        )
        p_origin, p_R = compute_frame_from_indices(
            coords2_aligned, indices.next_cols, indices.next_z_toward
        )
        transform = compute_relative_transform(o3p_origin, o3p_R, p_origin, p_R)

        coords_out[i] = coords1_aligned
        transforms_out[i] = transform

    return coords_out, transforms_out, atoms_array


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


# =============================================================================
# Dataset Classes
# =============================================================================


class ResidueDataset:
    """
    Dataset of residue conformations extracted from CIF files.

    Extracts residues of a specific type with SE(3) transforms to the next
    residue, suitable for training flow models.

    Attributes:
        coords: (n_instances, n_atoms, 3) aligned coordinates.
        transforms: (n_instances, 6) SE(3) link transforms.
        atoms: (n_atoms,) atom type indices.
        indices: Precomputed FrameIndices for frame computation.

    Example:
        >>> from ciffy.nn.flow.residue import ResidueDataset
        >>> from ciffy.biochemistry import Residue
        >>> dataset = ResidueDataset(cif_paths, Residue.A)
        >>> print(f"Found {len(dataset)} adenine residues")
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

        # Precompute frame indices for later use
        self.indices = FrameIndices.from_atoms(atoms, residue)

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
