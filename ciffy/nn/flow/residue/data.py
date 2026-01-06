"""
Data extraction for residue flow models.

Extracts aligned residue coordinates and link transforms from CIF files.
Uses Polymer.align() for consistent frame computation.
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

import ciffy
from ciffy.backend import to_numpy
from ciffy.backend.ops import isin
from ciffy.biochemistry import Scale, Molecule
from ciffy.operations.reduction import Reduction
from ciffy.geometry.transforms import is_purine

if TYPE_CHECKING:
    from ciffy.biochemistry import Residue
    from ciffy.biochemistry.linking import FrameDefinition


# =============================================================================
# Filtering
# =============================================================================


def filter_complete_residues(
    polymer: "ciffy.Polymer",
    frame_def: "FrameDefinition",
) -> "ciffy.Polymer":
    """Filter to residues with all frame atoms present."""
    all_frame_atoms = np.concatenate([
        frame_def.origin.index(),
        frame_def.axis_ref.index(),
        frame_def.plane_ref.index(),
    ])
    is_frame_atom = isin(polymer.atoms, to_numpy(all_frame_atoms))
    counts = polymer.reduce(is_frame_atom.astype(np.int64), Scale.RESIDUE, Reduction.SUM)
    complete_mask = to_numpy(counts) >= 3

    if not complete_mask.all():
        return polymer.select(complete_mask, Scale.RESIDUE)
    return polymer


# =============================================================================
# Core Extraction
# =============================================================================


def _remap_to_common(coords: np.ndarray, atoms: list[int], common_atoms: list[int]) -> np.ndarray:
    """Remap coords to common atom ordering, taking first occurrence of duplicates."""
    out = np.zeros((len(common_atoms), 3), dtype=np.float32)
    atom_to_idx = {a: i for i, a in enumerate(common_atoms)}
    seen = set()
    for coord, atom in zip(coords, atoms):
        if atom in atom_to_idx and atom not in seen:
            out[atom_to_idx[atom]] = coord
            seen.add(atom)
    return out


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
    use with extend(): poly.extend(res, coords, transform, ...).

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
    from ciffy.geometry.transforms import (
        extract_frame_positions,
        frame_from_positions,
        compute_relative_transform,
    )

    link_def = LINKING_BY_TYPE[residue_type.molecule_type]
    o3p_values = link_def.prev_atom.index()
    p_values = link_def.next_atom.index()

    # Phase 1: Collect all residue instances
    all_instances = []  # (aligned_coords, atoms, origin_i, R_i, origin_j, R_j)

    for path in cif_paths:
        if verbose:
            print(f"{path.name}...", end=" ", flush=True)

        try:
            poly = ciffy.load(str(path))
            poly = poly.by_type(Molecule(residue_type.molecule_type)).canonical()
            if poly.size() == 0:
                if verbose:
                    print("skip")
                continue

            # Filter to complete residues
            poly = filter_complete_residues(poly, GLYCOSIDIC_FRAME)
            poly = filter_complete_residues(poly, link_def.prev_frame)
            poly = filter_complete_residues(poly, link_def.next_frame)

            n_res = poly.size(Scale.RESIDUE)
            if n_res < 2:
                if verbose:
                    print("0")
                continue

            # Align to glycosidic frame
            aligned, Rs = poly.align(GLYCOSIDIC_FRAME)
            Rs = to_numpy(Rs)
            origins = to_numpy(poly.gather([GLYCOSIDIC_FRAME.origin])[:, 0])

            # Get per-residue data
            seq = to_numpy(poly.sequence)
            counts = to_numpy(aligned.counts(Scale.RESIDUE))
            offsets = np.concatenate([[0], np.cumsum(counts)])
            coords = to_numpy(aligned.coordinates)
            atoms = to_numpy(aligned.atoms)
            orig_coords = to_numpy(poly.coordinates)

            count = 0
            for i in range(n_res - 1):
                j = i + 1
                # Filter for residue_type at position j (the residue being positioned)
                if seq[j] != residue_type.value:
                    continue
                s1, e1 = offsets[i], offsets[i + 1]
                s2, e2 = offsets[j], offsets[j + 1]

                # Check bond length
                atoms_i, atoms_j = atoms[s1:e1], atoms[s2:e2]
                o3p_mask = np.isin(atoms_i, o3p_values)
                p_mask = np.isin(atoms_j, p_values)
                if not (o3p_mask.any() and p_mask.any()):
                    continue

                dist = np.linalg.norm(
                    orig_coords[s2:e2][p_mask.argmax()] -
                    orig_coords[s1:e1][o3p_mask.argmax()]
                )
                if dist > max_bond_length:
                    continue

                all_instances.append((
                    coords[s1:e1].copy(),
                    atoms_i.tolist(),
                    coords[s2:e2].copy(),
                    atoms[s2:e2].tolist(),
                    origins[i], Rs[i],
                    origins[j], Rs[j],
                ))
                count += 1

            if verbose:
                print(count)

        except Exception as e:
            if verbose:
                print(f"error: {e}")

    if not all_instances:
        raise ValueError(f"No {residue_type.name} pairs found")

    if verbose:
        print(f"\nTotal: {len(all_instances)} pairs")

    # Phase 2: Find common atoms (from residue j, which is residue_type)
    atom_counts = Counter()
    for c_i, a_i, c_j, a_j, *_ in all_instances:
        atom_counts.update(a_j)

    min_count = int(len(all_instances) * min_coverage)
    common_atoms = sorted([a for a, c in atom_counts.items() if c >= min_count])
    common_set = set(common_atoms)

    if verbose:
        print(f"Common atoms: {len(common_atoms)}")

    # Phase 3: Build output arrays
    # Filter to instances where j (residue_type) has all common atoms
    valid = [(c_i, a_i, c_j, a_j, o_i, R_i, o_j, R_j)
             for c_i, a_i, c_j, a_j, o_i, R_i, o_j, R_j in all_instances
             if common_set.issubset(a_j)]

    if verbose:
        print(f"Valid pairs: {len(valid)}")

    n = len(valid)
    n_atoms = len(common_atoms)
    atoms_arr = np.array(common_atoms, dtype=np.int64)

    coords_out = np.zeros((n, n_atoms, 3), dtype=np.float32)
    transforms_out = np.zeros((n, 6), dtype=np.float32)

    for idx, (c_i, a_i, c_j, a_j, o_i, R_i, o_j, R_j) in enumerate(valid):
        # Remap to common atom ordering (take first occurrence of each atom type)
        coords_i = _remap_to_common(c_i, a_i, common_atoms)
        coords_j = _remap_to_common(c_j, a_j, common_atoms)

        # Output coords_j (the residue being positioned, which is residue_type)
        coords_out[idx] = coords_j

        # Transform j's coords to i's frame for link transform computation
        R_j_to_i = R_j.T @ R_i
        t_j_to_i = (o_j - o_i) @ R_i
        coords_j_in_i = coords_j @ R_j_to_i + t_j_to_i

        # Compute link transform (O3' frame of i -> P frame of j)
        # This transform positions residue j relative to residue i
        prev_pos = extract_frame_positions(coords_i, atoms_arr, link_def.prev_frame)
        o3p_origin, o3p_R = frame_from_positions(prev_pos)

        next_pos = extract_frame_positions(coords_j_in_i, atoms_arr, link_def.next_frame)
        p_origin, p_R = frame_from_positions(next_pos)

        transforms_out[idx] = compute_relative_transform(o3p_origin, o3p_R, p_origin, p_R)

    return coords_out, transforms_out, atoms_arr


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


def check_bond_lengths(
    coords: np.ndarray,
    atoms: np.ndarray,
    residue: "Residue",
) -> dict[str, float]:
    """Check glycosidic bond length statistics."""
    atoms_list = atoms.tolist() if hasattr(atoms, 'tolist') else list(atoms)
    c1p_idx = atoms_list.index(residue.C1p.value)

    if is_purine(residue):
        n_idx = atoms_list.index(residue.N9.value)
        bond_name = "C1'-N9"
    else:
        n_idx = atoms_list.index(residue.N1.value)
        bond_name = "C1'-N1"

    dists = np.linalg.norm(coords[:, c1p_idx] - coords[:, n_idx], axis=-1)
    return {"bond": bond_name, "mean": float(dists.mean()), "std": float(dists.std())}


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
