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


# =============================================================================
# Extended Residue Representation (with link transforms)
# =============================================================================


def _rotation_matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to axis-angle representation."""
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-6:
        return np.zeros(3, dtype=np.float32)
    if np.pi - angle < 1e-6:
        M = R + np.eye(3)
        col_norms = np.linalg.norm(M, axis=0)
        k = np.argmax(col_norms)
        axis = M[:, k] / col_norms[k]
        return (axis * angle).astype(np.float32)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis = axis / (2 * np.sin(angle) + 1e-8)
    return (axis * angle).astype(np.float32)


def _axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle to rotation matrix (Rodrigues' formula)."""
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = axis_angle / angle
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], dtype=np.float32)
    return np.eye(3, dtype=np.float32) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def compute_link_frames(
    coords1: np.ndarray,
    coords2: np.ndarray,
    atoms: list[int],
    residue: "Residue",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute frames at linking atoms (O3' of res1, P of res2).

    Each frame is defined using only atoms from its own residue to ensure
    the frames are intrinsic to each residue's conformation.

    Args:
        coords1: (n_atoms, 3) coordinates of first residue.
        coords2: (n_atoms, 3) coordinates of second residue.
        atoms: List of atom type indices.
        residue: Residue type.

    Returns:
        origin1: (3,) O3' position.
        R1: (3, 3) O3' frame rotation matrix.
        origin2: (3,) P position.
        R2: (3, 3) P frame rotation matrix.
    """
    atom_to_col = {a: i for i, a in enumerate(atoms)}

    # O3' frame from residue 1 (C4'-C3'-O3' defines the frame)
    c4p_1 = coords1[atom_to_col[residue.C4p.value]]
    c3p_1 = coords1[atom_to_col[residue.C3p.value]]
    o3p_1 = coords1[atom_to_col[residue.O3p.value]]

    origin1 = o3p_1
    z1 = o3p_1 - c3p_1
    z1 = z1 / (np.linalg.norm(z1) + 1e-8)
    y_temp1 = c4p_1 - c3p_1
    x1 = np.cross(y_temp1, z1)
    x1 = x1 / (np.linalg.norm(x1) + 1e-8)
    y1 = np.cross(z1, x1)
    R1 = np.column_stack([x1, y1, z1]).astype(np.float32)

    # P frame from residue 2 (OP1-P-O5' defines the frame)
    p_2 = coords2[atom_to_col[residue.P.value]]
    o5p_2 = coords2[atom_to_col[residue.O5p.value]]
    op1_2 = coords2[atom_to_col[residue.OP1.value]]

    origin2 = p_2
    z2 = p_2 - o5p_2
    z2 = z2 / (np.linalg.norm(z2) + 1e-8)
    y_temp2 = op1_2 - p_2
    x2 = np.cross(y_temp2, z2)
    x2 = x2 / (np.linalg.norm(x2) + 1e-8)
    y2 = np.cross(z2, x2)
    R2 = np.column_stack([x2, y2, z2]).astype(np.float32)

    return origin1, R1, origin2, R2


def compute_relative_transform(
    origin1: np.ndarray,
    R1: np.ndarray,
    origin2: np.ndarray,
    R2: np.ndarray,
) -> np.ndarray:
    """
    Compute relative SE(3) transform from frame 1 to frame 2.

    Args:
        origin1, R1: First frame (position and rotation).
        origin2, R2: Second frame (position and rotation).

    Returns:
        6D vector: [axis-angle (3), translation in frame1 coords (3)].
    """
    R_rel = R1.T @ R2
    axis_angle = _rotation_matrix_to_axis_angle(R_rel)
    t_world = origin2 - origin1
    t_local = R1.T @ t_world
    return np.concatenate([axis_angle, t_local]).astype(np.float32)


def apply_relative_transform(
    origin1: np.ndarray,
    R1: np.ndarray,
    rel_transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply relative transform to get frame 2 from frame 1.

    Args:
        origin1, R1: Source frame.
        rel_transform: 6D vector [axis-angle (3), translation (3)].

    Returns:
        origin2, R2: Target frame.
    """
    axis_angle = rel_transform[:3]
    t_local = rel_transform[3:]
    R_rel = _axis_angle_to_rotation_matrix(axis_angle)
    R2 = R1 @ R_rel
    t_world = R1 @ t_local
    origin2 = origin1 + t_world
    return origin2, R2


def extract_residues_with_links(
    cif_paths: list[Path],
    residue_type: "Residue",
    min_coverage: float = 0.9,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Extract residues that have a next neighbor, with SE(3) link transforms.

    This creates an extended representation where each residue includes
    information about how it connects to the next residue in the chain.
    This allows models to learn the coupling between residue conformation
    and backbone geometry.

    Args:
        cif_paths: List of paths to CIF files.
        residue_type: Residue enum (e.g., Residue.A for adenosine).
        min_coverage: Minimum fraction of instances an atom must appear in.
        verbose: Print progress information.

    Returns:
        coords: (n_instances, n_atoms, 3) coordinate array (frame-aligned).
        transforms: (n_instances, 6) SE(3) transforms [axis-angle, translation].
        atoms: List of atom type indices in column order.

    Example:
        >>> coords, transforms, atoms = extract_residues_with_links(paths, Residue.A)
        >>> # Extended representation: concatenate flattened coords with transform
        >>> extended = np.concatenate([coords.reshape(len(coords), -1), transforms], axis=1)
    """
    # Required atoms for link computation
    required_link_atoms = {
        residue_type.C4p.value, residue_type.C3p.value, residue_type.O3p.value,
        residue_type.P.value, residue_type.O5p.value, residue_type.OP1.value,
    }

    all_instances = []  # (coords1, coords2, atoms1)

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

                all_instances.append((coords1, coords2, atoms1))
                count += 1

            if verbose:
                print(f"{count} pairs")

        except Exception as e:
            if verbose:
                print(f"error: {e}")

    if not all_instances:
        raise ValueError(f"No {residue_type.name} residue pairs found")

    if verbose:
        print(f"\nCollected {len(all_instances)} residue pairs")

    # Find common atoms across all first residues
    atom_counts = Counter()
    for _, _, atoms1 in all_instances:
        atom_counts.update(atoms1)

    min_count = int(len(all_instances) * min_coverage)
    common_atoms = sorted([a for a, c in atom_counts.items() if c >= min_count])

    if verbose:
        print(f"Atoms with >={min_coverage*100:.0f}% coverage: {len(common_atoms)}")

    # Filter instances that have all common atoms in both residues
    common_set = set(common_atoms)
    filtered = []
    for coords1, coords2, atoms1 in all_instances:
        # Check first residue has all common atoms
        if not common_set.issubset(set(atoms1)):
            continue
        filtered.append((coords1, coords2, atoms1))

    if verbose:
        print(f"Instances with all common atoms: {len(filtered)}")

    # Build output arrays
    n_atoms = len(common_atoms)
    atom_to_col = {a: c for c, a in enumerate(common_atoms)}

    coords_out = np.zeros((len(filtered), n_atoms, 3), dtype=np.float32)
    transforms_out = np.zeros((len(filtered), 6), dtype=np.float32)

    for i, (raw_coords1, raw_coords2, atoms1) in enumerate(filtered):
        # Map coords1 to common atom ordering
        coords1 = np.zeros((n_atoms, 3), dtype=np.float32)
        for atom_idx, coord in zip(atoms1, raw_coords1):
            if atom_idx in atom_to_col:
                coords1[atom_to_col[atom_idx]] = coord

        # For coords2, we only need the link atoms for transform computation
        # Map using atoms1 ordering (same residue type, so same atom indices)
        atoms2 = atoms1  # Same residue type
        coords2 = np.zeros((n_atoms, 3), dtype=np.float32)
        for atom_idx, coord in zip(atoms2, raw_coords2):
            if atom_idx in atom_to_col:
                coords2[atom_to_col[atom_idx]] = coord

        # Compute link transform
        o1, R1, o2, R2 = compute_link_frames(coords1, coords2, common_atoms, residue_type)
        transform = compute_relative_transform(o1, R1, o2, R2)

        coords_out[i] = coords1
        transforms_out[i] = transform

    # Align coordinates to canonical frame
    coords_aligned = align_to_frame(coords_out, common_atoms, residue_type)

    return coords_aligned, transforms_out, common_atoms


def position_next_residue(
    coords1: np.ndarray,
    coords2: np.ndarray,
    rel_transform: np.ndarray,
    atoms: list[int],
    residue: "Residue",
) -> np.ndarray:
    """
    Position residue 2 relative to residue 1 using the link transform.

    Args:
        coords1: (n_atoms, 3) coordinates of first residue.
        coords2: (n_atoms, 3) coordinates of second residue (in canonical frame).
        rel_transform: (6,) SE(3) transform [axis-angle, translation].
        atoms: List of atom type indices.
        residue: Residue type.

    Returns:
        (n_atoms, 3) positioned coordinates of second residue.
    """
    atom_to_col = {a: i for i, a in enumerate(atoms)}

    # Compute O3' frame from coords1
    c4p_1 = coords1[atom_to_col[residue.C4p.value]]
    c3p_1 = coords1[atom_to_col[residue.C3p.value]]
    o3p_1 = coords1[atom_to_col[residue.O3p.value]]

    o3p_origin = o3p_1
    z1 = o3p_1 - c3p_1
    z1 = z1 / (np.linalg.norm(z1) + 1e-8)
    y_temp1 = c4p_1 - c3p_1
    x1 = np.cross(y_temp1, z1)
    x1 = x1 / (np.linalg.norm(x1) + 1e-8)
    y1 = np.cross(z1, x1)
    o3p_R = np.column_stack([x1, y1, z1]).astype(np.float32)

    # Apply transform to get target P frame
    target_p_origin, target_p_R = apply_relative_transform(o3p_origin, o3p_R, rel_transform)

    # Compute current P frame from coords2
    p_2 = coords2[atom_to_col[residue.P.value]]
    o5p_2 = coords2[atom_to_col[residue.O5p.value]]
    op1_2 = coords2[atom_to_col[residue.OP1.value]]

    current_p_origin = p_2
    z2 = p_2 - o5p_2
    z2 = z2 / (np.linalg.norm(z2) + 1e-8)
    y_temp2 = op1_2 - p_2
    x2 = np.cross(y_temp2, z2)
    x2 = x2 / (np.linalg.norm(x2) + 1e-8)
    y2 = np.cross(z2, x2)
    current_p_R = np.column_stack([x2, y2, z2]).astype(np.float32)

    # Compute and apply correction
    R_correction = target_p_R @ current_p_R.T
    t_correction = target_p_origin - current_p_origin

    coords2_centered = coords2 - current_p_origin
    coords2_rotated = (R_correction @ coords2_centered.T).T
    coords2_final = coords2_rotated + current_p_origin + t_correction

    return coords2_final.astype(np.float32)
