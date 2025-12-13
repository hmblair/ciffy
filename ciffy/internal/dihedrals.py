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
from ..types import Molecule, Scale

if TYPE_CHECKING:
    from ..polymer import Polymer
    from .graph import ZMatrixEntry


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
    zmatrix: list["ZMatrixEntry"],
) -> Dict[str, Array]:
    """
    Compute indices into the dihedral array for named dihedrals.

    Identifies which Z-matrix entries correspond to standard backbone
    dihedrals based on the atom types and bond topology.

    Args:
        polymer: Source polymer with atom type information.
        zmatrix: Z-matrix defining coordinate references.

    Returns:
        Dict mapping dihedral names to arrays of indices into the
        dihedral array. Missing dihedrals (e.g., phi for first residue)
        are excluded from the arrays.
    """
    from ..biochemistry import Residue, ATOM_NAMES

    # Determine molecule type from first residue
    if polymer.size(Scale.RESIDUE) == 0:
        return {}

    first_res_type = int(polymer.sequence[0])
    try:
        mol_type = Residue(first_res_type).molecule_type
    except ValueError:
        return {}

    # Select appropriate dihedral definitions
    if mol_type in (Molecule.PROTEIN, Molecule.PROTEIN_D, Molecule.CYCLIC_PEPTIDE):
        dihedral_defs = PROTEIN_DIHEDRALS
    elif mol_type in (Molecule.RNA, Molecule.DNA, Molecule.HYBRID):
        dihedral_defs = NUCLEIC_ACID_DIHEDRALS
    else:
        return {}

    # Build atom index -> name mapping (using Python naming convention)
    idx_to_name: Dict[int, str] = {}
    for i in range(polymer.size()):
        atom_value = int(polymer.atoms[i])
        name = ATOM_NAMES.get(atom_value, "")
        # Convert to Python naming: O3' -> O3p, H5'' -> H5pp
        py_name = name.replace("'", "p").replace('"', "pp")
        idx_to_name[i] = py_name

    # Find matching dihedrals
    result: Dict[str, list] = {name: [] for name in dihedral_defs}

    for z_idx, entry in enumerate(zmatrix):
        if entry.dihedral_ref < 0:
            continue

        # Get atom names for this Z-matrix entry's dihedral definition
        # The dihedral is: atom -> distance_ref -> angle_ref -> dihedral_ref
        a1_name = idx_to_name.get(entry.atom_idx, "")
        a2_name = idx_to_name.get(entry.distance_ref, "")
        a3_name = idx_to_name.get(entry.angle_ref, "")
        a4_name = idx_to_name.get(entry.dihedral_ref, "")

        # Check against each dihedral definition
        for dihedral_name, (d1, d2, d3, d4) in dihedral_defs.items():
            if (a1_name == d1 and a2_name == d2 and
                a3_name == d3 and a4_name == d4):
                result[dihedral_name].append(z_idx)
                break

    # Convert to arrays
    coords = polymer.coordinates
    if is_torch(coords):
        import torch
        return {
            k: torch.tensor(v, dtype=torch.long, device=coords.device)
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
