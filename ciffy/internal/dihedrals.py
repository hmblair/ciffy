"""
Named dihedral angle definitions for proteins and nucleic acids.

Provides atom name definitions for standard backbone dihedrals:
- Proteins: phi, psi, omega
- Nucleic acids: alpha, beta, gamma, delta, epsilon, zeta, chi

Note: The primary dihedral lookup mechanism uses the `dihedral_types`
array returned by the C extension during Z-matrix construction. The C code
uses precomputed lookup tables from the codegen system for optimal performance.
"""

from __future__ import annotations

from ..types.dihedral import INDEX_TO_DIHEDRAL_TYPE


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


