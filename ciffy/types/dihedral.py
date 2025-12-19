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

    Protein sidechain dihedrals:
        CHI1: N - CA - CB - XG (varies by residue)
        CHI2: CA - CB - CG - XD (varies by residue)
        CHI3: CB - CG - CD - XE (varies by residue)
        CHI4: CG - CD - XE - XZ (LYS, ARG only)

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

    # Protein sidechain dihedrals
    CHI1 = 11
    CHI2 = 12
    CHI3 = 13
    CHI4 = 14


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
)

RNA_GLYCOSIDIC: tuple[DihedralType, ...] = (
    DihedralType.CHI_PURINE,
    DihedralType.CHI_PYRIMIDINE,
)

# Extended backbone definitions (includes glycosidic bond for full conformational description)
RNA_BACKBONE_EXTENDED: tuple[DihedralType, ...] = RNA_BACKBONE + (DihedralType.CHI_PYRIMIDINE,)

# Dihedral counts (derived from tuples for single source of truth)
NUM_PROTEIN_BACKBONE_DIHEDRALS: int = len(PROTEIN_BACKBONE)
NUM_RNA_BACKBONE_DIHEDRALS: int = len(RNA_BACKBONE_EXTENDED)
MAX_DIHEDRALS_PER_RESIDUE: int = max(NUM_PROTEIN_BACKBONE_DIHEDRALS, NUM_RNA_BACKBONE_DIHEDRALS)
