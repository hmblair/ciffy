"""
Named dihedral angle definitions for proteins and nucleic acids.

Provides definitions for standard backbone dihedrals (phi, psi, omega for
proteins; alpha-zeta and chi for nucleic acids) and functions to identify
which Z-matrix entries correspond to these dihedrals.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict

import numpy as np

from ..backend import Array, is_torch
from ..types import Molecule, Scale, DihedralType

if TYPE_CHECKING:
    from ..polymer import Polymer


# Mapping from DihedralType to integer index for array-based storage
# This matches the order used in code generation (generate.py)
DIHEDRAL_TYPE_TO_INDEX = {
    DihedralType.PHI: 0,
    DihedralType.PSI: 1,
    DihedralType.OMEGA: 2,
    DihedralType.ALPHA: 3,
    DihedralType.BETA: 4,
    DihedralType.GAMMA: 5,
    DihedralType.DELTA: 6,
    DihedralType.EPSILON: 7,
    DihedralType.ZETA: 8,
    DihedralType.CHI_PURINE: 9,
    DihedralType.CHI_PYRIMIDINE: 10,
}

INDEX_TO_DIHEDRAL_TYPE = {v: k for k, v in DIHEDRAL_TYPE_TO_INDEX.items()}


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

    Uses the precomputed dihedral_patterns from Residue definitions instead of
    searching through all Z-matrix entries. Returns integer-keyed dict for
    CSR conversion.

    Args:
        polymer: Source polymer with sequence information.
        zmatrix_indices: (M, 4) int64 array [atom_idx, dist_ref, ang_ref, dih_ref]

    Returns:
        Dict mapping DihedralType.value (int) → array of Z-matrix indices.
    """
    from ..biochemistry import Residue
    from ..types import DihedralType

    result: Dict[int, list] = {}

    # Build mapping: global_atom_idx → zmatrix_position
    atom_to_z: Dict[int, int] = {}
    for i in range(len(zmatrix_indices)):
        atom_idx = int(zmatrix_indices[i, 0])
        atom_to_z[atom_idx] = i

    # Get residue sizes
    res_sizes = polymer.sizes(Scale.RESIDUE)

    atom_offset = 0
    for res_idx in range(len(res_sizes)):
        res_size = int(res_sizes[res_idx])
        res_type_idx = int(polymer.sequence[res_idx])

        try:
            residue = Residue(res_type_idx)
            patterns = residue.dihedral_patterns  # dict[int, Array]

            for type_val, local_pattern in patterns.items():
                # Convert local indices to global atom indices
                global_atoms = []
                valid = True

                for local_idx in local_pattern:
                    local_idx_int = int(local_idx)
                    if local_idx_int == -1:
                        # Inter-residue atom - not supported yet in fast path
                        # (would need to look at adjacent residues)
                        valid = False
                        break
                    global_idx = atom_offset + local_idx_int
                    global_atoms.append(global_idx)

                if valid and len(global_atoms) == 4:
                    # Check if these atoms form a dihedral in Z-matrix
                    # The dihedral should be: a0 -> a1 -> a2 -> a3
                    # In Z-matrix: atom=a0, dist_ref=a1, ang_ref=a2, dih_ref=a3
                    z_idx = atom_to_z.get(global_atoms[0], -1)
                    if z_idx >= 0:
                        # Verify this Z-matrix entry matches the pattern
                        entry_dist = int(zmatrix_indices[z_idx, 1])
                        entry_ang = int(zmatrix_indices[z_idx, 2])
                        entry_dih = int(zmatrix_indices[z_idx, 3])

                        if (entry_dist == global_atoms[1] and
                            entry_ang == global_atoms[2] and
                            entry_dih == global_atoms[3]):
                            # Match found!
                            if type_val not in result:
                                result[type_val] = []
                            result[type_val].append(z_idx)

        except ValueError:
            pass

        atom_offset += res_size

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
