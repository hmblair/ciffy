"""
Data extraction and preprocessing for residue conformations.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from collections import Counter
from typing import TYPE_CHECKING

import ciffy
from ciffy.backend import to_numpy
from ciffy.types import Scale
from ciffy.operations.reduction import Reduction

if TYPE_CHECKING:
    from ciffy.biochemistry import Residue


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
    Align each residue to a canonical local frame.

    The frame is defined by:
    - Origin: C1' atom
    - X-axis: Toward N9 (purines) or N1 (pyrimidines)
    - Z-axis: Normal to the plane defined by C1', N9/N1, C4

    Args:
        coords: (n_instances, n_atoms, 3) coordinate array.
        atoms: List of atom type indices.
        residue: Residue type for looking up atom indices.

    Returns:
        Aligned coordinates with same shape as input.
    """
    n_instances = coords.shape[0]

    # Get frame atom indices
    c1p_idx = atoms.index(residue.C1p.value)
    c4_idx = atoms.index(residue.C4.value)

    # N9 for purines (A, G), N1 for pyrimidines (C, U)
    try:
        n_idx = atoms.index(residue.N9.value)
    except (ValueError, AttributeError):
        n_idx = atoms.index(residue.N1.value)

    aligned = np.zeros_like(coords)

    for i in range(n_instances):
        origin = coords[i, c1p_idx]
        n_pos = coords[i, n_idx]
        c4_pos = coords[i, c4_idx]

        # Build orthonormal frame
        x_axis = n_pos - origin
        x_axis /= np.linalg.norm(x_axis)

        y_temp = c4_pos - origin
        z_axis = np.cross(x_axis, y_temp)
        z_axis /= np.linalg.norm(z_axis)

        y_axis = np.cross(z_axis, x_axis)

        R = np.column_stack([x_axis, y_axis, z_axis])
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
