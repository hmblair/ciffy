"""
Data extraction and preprocessing for residue conformations.

This module provides:
- Data extraction from CIF files for flow model training
- PCA computation for dimensionality reduction
- Bond length checking utilities

Frame computation functions are imported from ciffy.geometry.
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

import ciffy
from ciffy.backend import Array, to_numpy, is_torch
from ciffy.backend.ops import isin
from ciffy.biochemistry import Scale
from ciffy.operations.reduction import Reduction
from ciffy.geometry import (
    compute_glycosidic_frame,
    align_to_frame,
    align_and_compute_transform,
    position_next_residue,
)

if TYPE_CHECKING:
    from ciffy.biochemistry import Residue
    from ciffy.biochemistry.linking import FrameDefinition


def filter_complete_residues(
    polymer: "ciffy.Polymer",
    frame_def: "FrameDefinition",
) -> "ciffy.Polymer":
    """
    Filter polymer to keep only residues with all atoms required for frame computation.

    Uses vectorized 2D lookup to find frame atoms, then reduces to count per residue.

    Args:
        polymer: Input polymer.
        frame_def: Frame definition specifying required atoms.

    Returns:
        Polymer with only complete residues.
    """
    # Get all possible values for each frame atom position
    all_frame_atoms = np.concatenate([
        frame_def.origin.index(),
        frame_def.axis_ref.index(),
        frame_def.plane_ref.index(),
    ])

    # Find which atoms are frame atoms (vectorized)
    is_frame_atom = isin(polymer.atoms, to_numpy(all_frame_atoms))

    # Count frame atoms per residue
    counts = polymer.reduce(is_frame_atom.astype(np.int64), Scale.RESIDUE, Reduction.SUM)

    # Keep residues with all 3 frame atoms
    complete_mask = to_numpy(counts) >= 3

    if not complete_mask.all():
        return polymer.select(complete_mask, Scale.RESIDUE)
    return polymer


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

    This function automatically filters to the molecule type of the
    requested residue (e.g., RNA for Residue.A) and to canonical residues.
    Mixed protein/RNA structures are handled correctly.

    Args:
        cif_paths: List of paths to CIF files (can contain mixed molecule types).
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

            # Filter by molecule type (e.g., RNA for adenosine)
            mol_type = residue_type.molecule_type
            if mol_type is not None:
                from ciffy.biochemistry import Molecule
                poly = poly.by_type(Molecule(mol_type)).canonical()
                if poly.size() == 0:
                    if verbose:
                        print("no matching molecule type")
                    continue

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

        except AttributeError as e:
            if verbose:
                print(f"skipped: incompatible residue type ({e})")
        except Exception as e:
            if verbose:
                print(f"skipped: {type(e).__name__}: {e}")

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

    This function automatically filters to the molecule type of the
    requested residue (e.g., RNA for Residue.A) and to canonical residues.
    Mixed protein/RNA structures are handled correctly.

    Data flow:
    1. Filter structure to matching molecule type
    2. Extract adjacent residue pairs from structures
    3. Filter by O3'-P bond length to ensure true connectivity
    4. Find common atoms across all instances
    5. For each pair: align, compute link transform

    Args:
        cif_paths: List of paths to CIF files (can contain mixed molecule types).
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

    # Get linking definition
    link_def = LINKING_BY_TYPE.get(residue_type.molecule_type)
    if link_def is None:
        raise ValueError(f"No linking definition for {residue_type.molecule_type}")

    # Get linking atom values for bond length check
    o3p_values = link_def.prev_atom.index()  # All O3' values
    p_values = link_def.next_atom.index()  # All P values

    # Phase 1: Extract raw pairs with bond length filtering
    all_pairs = []

    for path in cif_paths:
        if verbose:
            print(f"Processing {path.name}...", end=" ", flush=True)

        try:
            poly = ciffy.load(str(path)).poly()

            # Filter by molecule type (e.g., RNA for adenosine)
            mol_type = residue_type.molecule_type
            if mol_type is not None:
                from ciffy.biochemistry import Molecule
                poly = poly.by_type(Molecule(mol_type)).canonical()
                if poly.size() == 0:
                    if verbose:
                        print("no matching molecule type")
                    continue

            # Filter to residues with complete frame atoms (vectorized)
            poly = filter_complete_residues(poly, link_def.prev_frame)
            poly = filter_complete_residues(poly, link_def.next_frame)

            seq = to_numpy(poly.sequence)
            n_residues = len(seq)

            if n_residues < 2:
                if verbose:
                    print("0 pairs")
                continue

            # Use flat arrays with offsets instead of COLLATE
            atoms_flat = to_numpy(poly.atoms)
            coords_flat = to_numpy(poly.coordinates)
            counts = to_numpy(poly.counts(Scale.RESIDUE))
            offsets = np.zeros(n_residues + 1, dtype=np.int64)
            offsets[1:] = np.cumsum(counts)

            # Find all positions where residue matches target type (excluding last)
            target_mask = seq[:-1] == residue_type.value
            target_indices = np.where(target_mask)[0]

            count = 0
            for idx1 in target_indices:
                idx2 = idx1 + 1

                # Get atom slices (views, not copies)
                start1, end1 = offsets[idx1], offsets[idx1 + 1]
                start2, end2 = offsets[idx2], offsets[idx2 + 1]
                atoms1 = atoms_flat[start1:end1]
                atoms2 = atoms_flat[start2:end2]
                coords1 = coords_flat[start1:end1]
                coords2 = coords_flat[start2:end2]

                # Check bond length between linking atoms
                o3p_mask = np.isin(atoms1, o3p_values)
                p_mask = np.isin(atoms2, p_values)

                if not np.any(o3p_mask) or not np.any(p_mask):
                    continue

                o3p_idx = np.argmax(o3p_mask)
                p_idx = np.argmax(p_mask)

                if np.linalg.norm(coords2[p_idx] - coords1[o3p_idx]) > max_bond_length:
                    continue

                # Store copies since we're using views
                all_pairs.append((
                    coords1.copy(),
                    atoms1.tolist(),
                    coords2.copy(),
                    atoms2.tolist(),
                ))
                count += 1

            if verbose:
                print(f"{count} pairs")

        except AttributeError as e:
            if verbose:
                print(f"skipped: incompatible residue type ({e})")
        except Exception as e:
            if verbose:
                print(f"skipped: {type(e).__name__}: {e}")

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

    # Phase 4: Prepare atom array for the common atom ordering
    atoms_array = np.array(common_atoms, dtype=np.int64)

    # Phase 5: Remap, align, and compute transforms
    n_pairs = len(filtered_pairs)
    n_atoms = len(common_atoms)

    coords_out = np.zeros((n_pairs, n_atoms, 3), dtype=np.float32)
    transforms_out = np.zeros((n_pairs, 6), dtype=np.float32)

    for i, (raw_coords1, atoms1, raw_coords2, atoms2) in enumerate(filtered_pairs):
        # Remap to common atom ordering
        coords1 = _remap_coords_to_common_atoms(raw_coords1, atoms1, common_atoms)
        coords2 = _remap_coords_to_common_atoms(raw_coords2, atoms2, common_atoms)

        # Use shared alignment function
        coords_out[i], transforms_out[i] = align_and_compute_transform(
            coords1, coords2, atoms_array, residue_type
        )

    return coords_out, transforms_out, atoms_array


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
        residue: Residue type.

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
