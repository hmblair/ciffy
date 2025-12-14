"""
Named dihedral angle definitions for proteins and nucleic acids.

Provides definitions for standard backbone dihedrals (phi, psi, omega for
proteins; alpha-zeta and chi for nucleic acids) and functions to identify
which Z-matrix entries correspond to these dihedrals.

Note: The primary dihedral lookup mechanism now uses the `dihedral_types`
array built by `annotate_dihedral_types()` in graph.py, which uses
precomputed data from the codegen system. The functions in this module
are kept for reference and backward compatibility.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict
import warnings

import numpy as np

from ..backend import Array, is_torch
from ..types import Molecule, Scale, DihedralType
from ..types.dihedral import DIHEDRAL_TYPE_TO_INDEX, INDEX_TO_DIHEDRAL_TYPE

if TYPE_CHECKING:
    from ..polymer import Polymer


# =============================================================================
# DIHEDRAL DEFINITIONS
# =============================================================================

# Protein backbone dihedrals: (atom1, atom2, atom3, atom4)
# The dihedral angle is the rotation around the atom2-atom3 bond
PROTEIN_DIHEDRALS = {
    # phi: C(i-1) - N(i) - CA(i) - C(i)
    'phi': ('C', 'N', 'CA', 'C'),
    # psi: N(i) - CA(i) - C(i) - N(i+1)
    'psi': ('N', 'CA', 'C', 'N'),
    # omega: CA(i) - C(i) - N(i+1) - CA(i+1)
    'omega': ('CA', 'C', 'N', 'CA'),
}

# Nucleic acid backbone dihedrals
# Atom names use Python convention (p for ')
NUCLEIC_ACID_DIHEDRALS = {
    # alpha: O3'(i-1) - P(i) - O5'(i) - C5'(i)
    'alpha': ('O3p', 'P', 'O5p', 'C5p'),
    # beta: P(i) - O5'(i) - C5'(i) - C4'(i)
    'beta': ('P', 'O5p', 'C5p', 'C4p'),
    # gamma: O5'(i) - C5'(i) - C4'(i) - C3'(i)
    'gamma': ('O5p', 'C5p', 'C4p', 'C3p'),
    # delta: C5'(i) - C4'(i) - C3'(i) - O3'(i)
    'delta': ('C5p', 'C4p', 'C3p', 'O3p'),
    # epsilon: C4'(i) - C3'(i) - O3'(i) - P(i+1)
    'epsilon': ('C4p', 'C3p', 'O3p', 'P'),
    # zeta: C3'(i) - O3'(i) - P(i+1) - O5'(i+1)
    'zeta': ('C3p', 'O3p', 'P', 'O5p'),
    # chi for purines (A, G): O4' - C1' - N9 - C4
    'chi_purine': ('O4p', 'C1p', 'N9', 'C4'),
    # chi for pyrimidines (C, U, T): O4' - C1' - N1 - C2
    'chi_pyrimidine': ('O4p', 'C1p', 'N1', 'C2'),
}


def compute_dihedral_indices(
    polymer: "Polymer",
    zmatrix_indices: Array,
) -> Dict[int, Array]:
    """
    Compute dihedral indices using precomputed residue patterns.

    .. deprecated::
        This function is deprecated. The primary mechanism now uses the
        `dihedral_types` array built by `annotate_dihedral_types()` in graph.py,
        which is faster and more reliable.

    Uses the precomputed dihedral_patterns from Residue definitions instead of
    searching through all Z-matrix entries. Returns integer-keyed dict for
    CSR conversion.

    Args:
        polymer: Source polymer with sequence information.
        zmatrix_indices: (M, 4) int64 array [atom_idx, dist_ref, ang_ref, dih_ref]

    Returns:
        Dict mapping DihedralType.value (int) → array of Z-matrix indices.
    """
    warnings.warn(
        "compute_dihedral_indices is deprecated. Use the dihedral_types array "
        "from ZMatrix.dihedral_types instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from ..biochemistry import Residue
    from ..types import DihedralType

    result: Dict[int, list] = {}

    # Build mapping: global_atom_idx → zmatrix_position
    atom_to_z: Dict[int, int] = {}
    for i in range(len(zmatrix_indices)):
        atom_idx = int(zmatrix_indices[i, 0])
        atom_to_z[atom_idx] = i

    # Get residue sizes and build cumulative atom offsets
    res_sizes = polymer.sizes(Scale.RESIDUE)
    n_residues = len(res_sizes)

    # Build cumulative atom offsets: atom_offsets[i] = first atom index of residue i
    atom_offsets = [0]
    for i in range(n_residues):
        atom_offsets.append(atom_offsets[-1] + int(res_sizes[i]))

    # Process each residue
    for res_idx in range(n_residues):
        res_type_idx = int(polymer.sequence[res_idx])

        try:
            residue = Residue(res_type_idx)
            patterns = residue.dihedral_patterns  # dict[int, Array(4,2)]

            for type_val, pattern in patterns.items():
                # Pattern is (4, 2) array: [[offset, local_idx], ...]
                # Convert to global atom indices
                global_atoms = []
                valid = True

                for i in range(4):
                    offset = int(pattern[i, 0])
                    local_idx = int(pattern[i, 1])

                    # Compute target residue index
                    target_res = res_idx + offset

                    # Check bounds
                    if target_res < 0 or target_res >= n_residues:
                        valid = False
                        break

                    # Compute global atom index
                    global_idx = atom_offsets[target_res] + local_idx
                    global_atoms.append(global_idx)

                if valid and len(global_atoms) == 4:
                    # Search Z-matrix for this dihedral pattern
                    # The dihedral a0-a1-a2-a3 could be stored in forward or reverse order
                    # depending on how the Z-matrix was built from bond connectivity.
                    #
                    # Forward: Z-matrix entry [a0, a1, a2, a3] stores dihedral a0-a1-a2-a3
                    # Reverse: Z-matrix entry [a3, a2, a1, a0] stores dihedral a3-a2-a1-a0
                    #          (same angle, opposite sign convention)
                    #
                    # Strategy: Check both forward and reverse patterns
                    found = False

                    # Try forward pattern: entry for global_atoms[0]
                    z_idx = atom_to_z.get(global_atoms[0], -1)
                    if z_idx >= 0:
                        entry_dist = int(zmatrix_indices[z_idx, 1])
                        entry_ang = int(zmatrix_indices[z_idx, 2])
                        entry_dih = int(zmatrix_indices[z_idx, 3])

                        if (entry_dist == global_atoms[1] and
                            entry_ang == global_atoms[2] and
                            entry_dih == global_atoms[3]):
                            if type_val not in result:
                                result[type_val] = []
                            result[type_val].append(z_idx)
                            found = True

                    # Try reverse pattern: entry for global_atoms[3]
                    if not found:
                        z_idx = atom_to_z.get(global_atoms[3], -1)
                        if z_idx >= 0:
                            entry_dist = int(zmatrix_indices[z_idx, 1])
                            entry_ang = int(zmatrix_indices[z_idx, 2])
                            entry_dih = int(zmatrix_indices[z_idx, 3])

                            if (entry_dist == global_atoms[2] and
                                entry_ang == global_atoms[1] and
                                entry_dih == global_atoms[0]):
                                if type_val not in result:
                                    result[type_val] = []
                                result[type_val].append(z_idx)

        except ValueError:
            pass

    # Convert to arrays
    coords = polymer.coordinates
    if is_torch(coords):
        import torch
        return {
            k: torch.tensor(v, dtype=torch.int64, device=coords.device)
            for k, v in result.items() if v
        }
    else:
        return {
            k: np.array(v, dtype=np.int64)
            for k, v in result.items() if v
        }


def compute_named_dihedral(
    polymer: "Polymer",
    dtype: "DihedralType",
) -> Array:
    """
    Compute named dihedral angles directly from Cartesian coordinates.

    Returns one value per residue, with NaN for residues where the dihedral
    cannot be computed (terminal residues, missing atoms, chain boundaries).

    This is more reliable than Z-matrix lookup because it handles:
    - Chain boundaries (different connected components)
    - Missing residues (0-atom residues)
    - Any 4-atom pattern regardless of Z-matrix structure

    Args:
        polymer: Source polymer with coordinates and sequence.
        dtype: Type of dihedral to compute.

    Returns:
        (N_residues,) array of dihedral angles in radians, with NaN for invalid.
    """
    from ..biochemistry import Residue
    from ..types.dihedral import DIHEDRAL_TYPE_TO_INDEX

    coords = polymer.coordinates
    res_sizes = polymer.sizes(Scale.RESIDUE)
    n_residues = len(res_sizes)

    # Get dihedral type index
    type_idx = DIHEDRAL_TYPE_TO_INDEX.get(dtype)
    if type_idx is None:
        # Unknown dihedral type
        if is_torch(coords):
            import torch
            return torch.full((n_residues,), float('nan'), dtype=coords.dtype, device=coords.device)
        return np.full(n_residues, np.nan, dtype=coords.dtype)

    # Build cumulative atom offsets
    atom_offsets = [0]
    for i in range(n_residues):
        atom_offsets.append(atom_offsets[-1] + int(res_sizes[i]))

    # Initialize result with NaN
    if is_torch(coords):
        import torch
        result = torch.full((n_residues,), float('nan'), dtype=coords.dtype, device=coords.device)
    else:
        result = np.full(n_residues, np.nan, dtype=coords.dtype)

    # Compute dihedral for each residue
    for res_idx in range(n_residues):
        res_type_idx = int(polymer.sequence[res_idx])

        try:
            residue = Residue(res_type_idx)
            patterns = residue.dihedral_patterns

            if type_idx not in patterns:
                continue

            pattern = patterns[type_idx]  # (4, 2) array: [[offset, local_idx], ...]

            # Resolve global atom indices
            global_atoms = []
            valid = True

            for i in range(4):
                offset = int(pattern[i, 0])
                local_idx = int(pattern[i, 1])

                # Compute target residue index
                target_res = res_idx + offset

                # Check bounds
                if target_res < 0 or target_res >= n_residues:
                    valid = False
                    break

                # Check that target residue has atoms
                target_size = int(res_sizes[target_res])
                if target_size == 0 or local_idx >= target_size:
                    valid = False
                    break

                # Compute global atom index
                global_idx = atom_offsets[target_res] + local_idx
                global_atoms.append(global_idx)

            if not valid or len(global_atoms) != 4:
                continue

            # Get coordinates of the 4 atoms
            p0 = coords[global_atoms[0]]
            p1 = coords[global_atoms[1]]
            p2 = coords[global_atoms[2]]
            p3 = coords[global_atoms[3]]

            # Compute dihedral angle
            # Vectors along the bonds
            b1 = p1 - p0
            b2 = p2 - p1
            b3 = p3 - p2

            # Normal vectors to planes
            # Standard dihedral: looking down b2 (the central bond),
            # positive angle = clockwise rotation from b1 to b3
            if is_torch(coords):
                import torch
                n1 = torch.cross(b1, b2)
                n2 = torch.cross(b2, b3)
                m1 = torch.cross(n1, b2 / torch.norm(b2))
                x = torch.dot(n1, n2)
                y = torch.dot(m1, n2)
                # Negate to match IUPAC convention for backbone dihedrals
                result[res_idx] = -torch.atan2(y, x)
            else:
                n1 = np.cross(b1, b2)
                n2 = np.cross(b2, b3)
                m1 = np.cross(n1, b2 / np.linalg.norm(b2))
                x = np.dot(n1, n2)
                y = np.dot(m1, n2)
                # Negate to match IUPAC convention for backbone dihedrals
                result[res_idx] = -np.arctan2(y, x)

        except (ValueError, IndexError):
            continue

    return result


def get_residue_dihedral_atoms(
    residue_name: str,
    dihedral_name: str,
) -> tuple[str, str, str, str] | None:
    """
    Get the atom names that define a specific dihedral for a residue type.

    Args:
        residue_name: Residue name (e.g., 'A', 'ALA').
        dihedral_name: Dihedral name (e.g., 'phi', 'chi').

    Returns:
        Tuple of (atom1, atom2, atom3, atom4) names, or None if not defined.
    """
    from ..biochemistry import Residue

    try:
        residue = Residue[residue_name]
        mol_type = residue.molecule_type
    except (KeyError, ValueError):
        return None

    if mol_type in (Molecule.PROTEIN, Molecule.PROTEIN_D, Molecule.CYCLIC_PEPTIDE):
        return PROTEIN_DIHEDRALS.get(dihedral_name)
    elif mol_type in (Molecule.RNA, Molecule.DNA, Molecule.HYBRID):
        return NUCLEIC_ACID_DIHEDRALS.get(dihedral_name)

    return None
