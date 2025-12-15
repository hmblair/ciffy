"""
Dihedral angle type enumeration for biomolecules.
"""

from enum import IntEnum


class DihedralType(IntEnum):
    """
    Named dihedral angle types for proteins and nucleic acids.

    Dihedral angles describe rotations around bonds and are fundamental
    to describing the conformation of biomolecules. Values are integers
    that can be used directly as array indices.

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

    Example:
        >>> DihedralType.PHI.value  # Returns 0, usable as array index
        0
        >>> DihedralType.ALPHA.value  # Returns 3
        3
    """

    # Protein backbone
    PHI = 0
    PSI = 1
    OMEGA = 2

    # Nucleic acid backbone
    ALPHA = 3
    BETA = 4
    GAMMA = 5
    DELTA = 6
    EPSILON = 7
    ZETA = 8

    # Glycosidic dihedrals
    CHI_PURINE = 9
    CHI_PYRIMIDINE = 10


# Reverse mapping from integer index to DihedralType
INDEX_TO_DIHEDRAL_TYPE: dict[int, DihedralType] = {dt.value: dt for dt in DihedralType}

# Convenience tuples for common dihedral groups
PROTEIN_BACKBONE: tuple[DihedralType, ...] = (
    DihedralType.PHI,
    DihedralType.PSI,
    DihedralType.OMEGA,
)

RNA_BACKBONE: tuple[DihedralType, ...] = (
    DihedralType.ALPHA,
    DihedralType.BETA,
    DihedralType.GAMMA,
    DihedralType.DELTA,
    DihedralType.EPSILON,
    DihedralType.ZETA,
    DihedralType.CHI_PURINE,
    DihedralType.CHI_PYRIMIDINE,
)
