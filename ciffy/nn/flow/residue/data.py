"""
Data extraction for residue flow models.

Extracts aligned residue coordinates and link transforms from polymers.
Uses Polymer API for clean, maintainable extraction.
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

import ciffy
from ciffy.backend import to_numpy
from ciffy.biochemistry import Scale, Molecule

if TYPE_CHECKING:
    from ciffy.biochemistry import Residue
    from ciffy.polymer import Polymer


def extract_residues_with_links(
    cif_paths: list[Path],
    residue_type: "Residue",
    min_coverage: float = 0.9,
    max_bond_length: float = 2.0,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract aligned residue coordinates with link transforms.

    Each sample contains a residue's coordinates and the transform that positions
    it relative to its predecessor. This convention means decode(z) returns
    (coords, transform) where transform positions THIS residue, enabling direct
    use with append(): poly.append(res, LocalCoordinates(coords, transform)).

    Args:
        cif_paths: CIF files to process.
        residue_type: Residue type to extract (e.g., Residue.A).
        min_coverage: Minimum fraction for atom inclusion.
        max_bond_length: Maximum O3'-P distance for valid links.
        verbose: Print progress.

    Returns:
        coords: (n, n_atoms, 3) aligned coordinates of residue.
        transforms: (n, 6) SE(3) transform to position this residue relative
            to predecessor [axis-angle, translation].
        atoms: (n_atoms,) atom type indices.
    """
    from ciffy.biochemistry.linking import LINKING_BY_TYPE, GLYCOSIDIC_FRAME
    from ciffy.geometry.transforms import compute_relative_transform

    link_def = LINKING_BY_TYPE[residue_type.molecule_type]

    # Collect all valid residue pairs
    all_coords = []  # List of (n_atoms,) coord arrays for residue j
    all_atoms = []   # List of atom index lists for residue j
    all_transforms = []  # List of (6,) transforms

    for path in cif_paths:
        if verbose:
            print(f"{path.name}...", end=" ", flush=True)

        try:
            count = _extract_from_polymer(
                ciffy.load(str(path)),
                residue_type,
                link_def,
                GLYCOSIDIC_FRAME,
                max_bond_length,
                all_coords,
                all_atoms,
                all_transforms,
            )
            if verbose:
                print(count)

        except Exception as e:
            if verbose:
                print(f"error: {e}")

    if not all_coords:
        raise ValueError(f"No {residue_type.name} pairs found")

    if verbose:
        print(f"\nTotal: {len(all_coords)} pairs")

    # Find common atoms across all instances
    atom_counts = Counter()
    for atoms in all_atoms:
        atom_counts.update(atoms)

    min_count = int(len(all_coords) * min_coverage)
    common_atoms = sorted([a for a, c in atom_counts.items() if c >= min_count])
    common_set = set(common_atoms)

    if verbose:
        print(f"Common atoms: {len(common_atoms)}")

    # Filter to instances with all common atoms and build output
    n_atoms = len(common_atoms)
    atom_to_idx = {a: i for i, a in enumerate(common_atoms)}

    coords_list = []
    transforms_list = []

    for coords, atoms, transform in zip(all_coords, all_atoms, all_transforms):
        if not common_set.issubset(atoms):
            continue

        # Reorder to common atom ordering
        reordered = np.zeros((n_atoms, 3), dtype=np.float32)
        for coord, atom in zip(coords, atoms):
            if atom in atom_to_idx:
                reordered[atom_to_idx[atom]] = coord

        coords_list.append(reordered)
        transforms_list.append(transform)

    if verbose:
        print(f"Valid pairs: {len(coords_list)}")

    return (
        np.stack(coords_list),
        np.stack(transforms_list),
        np.array(common_atoms, dtype=np.int64),
    )


def _extract_from_polymer(
    polymer: "Polymer",
    residue_type: "Residue",
    link_def,
    frame_def,
    max_bond_length: float,
    out_coords: list,
    out_atoms: list,
    out_transforms: list,
) -> int:
    """Extract residue pairs from a single polymer. Returns count added."""
    from ciffy.geometry.transforms import compute_relative_transform

    # Filter to target molecule type and strip incomplete residues
    mol_type = Molecule(residue_type.molecule_type)
    polymer = polymer.molecule_type(mol_type).strip()

    if polymer.size() == 0 or polymer.size(Scale.RESIDUE) < 2:
        return 0

    # Align to glycosidic frame for local coordinates
    try:
        aligned, _ = polymer.align(frame_def)
    except ValueError:
        return 0

    # Get link frame alignments for transform computation
    # prev_frame = O3' frame, next_frame = P frame
    try:
        _, prev_Rs = polymer.align(link_def.prev_frame)
        prev_origins = to_numpy(polymer.gather([link_def.prev_frame.origin])[:, 0])
        _, next_Rs = polymer.align(link_def.next_frame)
        next_origins = to_numpy(polymer.gather([link_def.next_frame.origin])[:, 0])
    except ValueError:
        return 0
    prev_Rs = to_numpy(prev_Rs)
    next_Rs = to_numpy(next_Rs)

    # Get sequence to find target residue type
    sequence = to_numpy(polymer.sequence)
    n_residues = polymer.size(Scale.RESIDUE)

    count = 0
    for j in range(1, n_residues):
        # Only extract residues of target type
        if sequence[j] != residue_type.value:
            continue

        i = j - 1  # Predecessor

        # Get aligned residue
        aligned_j = aligned.residue(j)

        # Check bond connectivity (O3' of i to P of j)
        if not _check_bond_length(polymer, i, j, link_def, max_bond_length):
            continue

        # Get aligned coordinates and atoms for residue j
        coords_j = to_numpy(aligned_j.coordinates)
        atoms_j = to_numpy(aligned_j.atoms).tolist()

        # Compute transform between link frames: O3' frame of i -> P frame of j
        transform = compute_relative_transform(
            prev_origins[i], prev_Rs[i],  # O3' frame of predecessor
            next_origins[j], next_Rs[j],  # P frame of current residue
        )

        out_coords.append(coords_j)
        out_atoms.append(atoms_j)
        out_transforms.append(to_numpy(transform))
        count += 1

    return count


def _check_bond_length(
    polymer: "Polymer",
    i: int,
    j: int,
    link_def,
    max_bond_length: float,
) -> bool:
    """Check if residues i and j are connected with valid bond length."""
    o3p_atoms = link_def.prev_atom.index()
    p_atoms = link_def.next_atom.index()

    # Get the two-residue segment and check O3'-P bond distance
    pair = polymer.residue([i, j])
    distances = pair.bonded_distances(o3p_atoms, p_atoms)

    if len(distances) == 0:
        return False

    return float(to_numpy(distances).min()) <= max_bond_length


# =============================================================================
# Utilities
# =============================================================================


def compute_pca(
    coords: np.ndarray,
    n_components: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PCA on flattened coordinates.

    Returns:
        V: (k, d) PCA components.
        mean: (d,) mean.
        singular_values: All singular values.
        var_explained: Cumulative variance explained.
    """
    flat = coords.reshape(len(coords), -1)
    mean = flat.mean(axis=0)
    _, s, Vt = np.linalg.svd(flat - mean, full_matrices=False)
    var_explained = np.cumsum(s ** 2) / (s ** 2).sum()

    if n_components:
        Vt = Vt[:n_components]

    return Vt.astype(np.float32), mean.astype(np.float32), s, var_explained


# =============================================================================
# Dataset
# =============================================================================


class ResidueDataset:
    """Dataset of residue conformations with link transforms."""

    def __init__(
        self,
        cif_paths: list[Path],
        residue: "Residue",
        min_coverage: float = 0.9,
        verbose: bool = True,
    ):
        self.residue = residue
        coords, transforms, atoms = extract_residues_with_links(
            cif_paths, residue, min_coverage=min_coverage, verbose=verbose
        )
        self.coords = coords
        self.transforms = transforms
        self.atoms = atoms
        self.n_atoms = len(atoms)
        self._data = np.concatenate([
            coords.reshape(len(coords), -1),
            transforms
        ], axis=1)

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self._data[idx]

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def shape(self) -> tuple[int, int]:
        return self._data.shape
