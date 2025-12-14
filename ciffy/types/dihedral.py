"""
Dihedral angle type enumeration for biomolecules.
"""

from enum import Enum


class DihedralType(Enum):
    """
    Named dihedral angle types for proteins and nucleic acids.

    Dihedral angles describe rotations around bonds and are fundamental
    to describing the conformation of biomolecules.

    Protein backbone dihedrals:
        PHI: C(i-1) - N(i) - CA(i) - C(i)
        PSI: N(i) - CA(i) - C(i) - N(i+1)
        OMEGA: CA(i) - C(i) - N(i+1) - CA(i+1)

    Nucleic acid backbone dihedrals:
        ALPHA: O3'(i-1) - P(i) - O5'(i) - C5'(i)
        BETA: P(i) - O5'(i) - C5'(i) - C4'(i)
        GAMMA: O5'(i) - C5'(i) - C4'(i) - C3'(i)
        DELTA: C5'(i) - C4'(i) - C3'(i) - O3'(i)
        EPSILON: C4'(i) - C3'(i) - O3'(i) - P(i+1)
        ZETA: C3'(i) - O3'(i) - P(i+1) - O5'(i+1)

    Glycosidic dihedrals:
        CHI_PURINE: O4' - C1' - N9 - C4 (adenine, guanine)
        CHI_PYRIMIDINE: O4' - C1' - N1 - C2 (cytosine, uracil, thymine)
    """

    # Protein backbone
    PHI = "phi"
    PSI = "psi"
    OMEGA = "omega"

    # Nucleic acid backbone
    ALPHA = "alpha"
    BETA = "beta"
    GAMMA = "gamma"
    DELTA = "delta"
    EPSILON = "epsilon"
    ZETA = "zeta"

    # Glycosidic dihedrals
    CHI_PURINE = "chi_purine"
    CHI_PYRIMIDINE = "chi_pyrimidine"
