"""
Residue definitions for nucleotides and amino acids.
"""

from ..utils import IndexEnum
from ..types import Molecule


class Residue(IndexEnum):
    """
    Residue types with unique integer indices.

    Includes nucleotides (RNA and DNA), amino acids, water, and ions.
    DNA nucleotides share indices with their RNA counterparts.
    """

    # RNA nucleotides
    A = 0
    C = 1
    G = 2
    U = 3

    # DNA nucleotides (share indices with RNA)
    DA = 0
    DC = 1
    DG = 2
    DU = 3
    T = 4
    DT = 4

    # Amino acids
    ALA = 5
    CYS = 6
    ASP = 7
    GLU = 8
    PHE = 9
    GLY = 10
    HIS = 11
    ILE = 12
    LYS = 13
    LEU = 14
    MET = 15
    ASN = 16
    PRO = 17
    GLN = 18
    ARG = 19
    SER = 20
    THR = 21
    VAL = 22
    TRP = 23
    TYR = 24

    # Water
    HOH = 25

    # Ions
    MG = 26
    CS = 27

    # Modified nucleotides
    GTP = 28  # Guanosine-5'-triphosphate
    CCC = 29  # Cytidine-5'-triphosphate (3' terminal)
    GNG = 30  # 2'-deoxyguanosine


# Mapping from residue index to molecule type
# Indices correspond to Residue enum values
RESIDUE_MOLECULE_TYPE: dict[int, Molecule] = {
    # RNA nucleotides (0-3)
    0: Molecule.RNA,   # A, DA
    1: Molecule.RNA,   # C, DC
    2: Molecule.RNA,   # G, DG
    3: Molecule.RNA,   # U, DU
    4: Molecule.DNA,   # T, DT

    # Amino acids (5-24)
    5: Molecule.PROTEIN,   # ALA
    6: Molecule.PROTEIN,   # CYS
    7: Molecule.PROTEIN,   # ASP
    8: Molecule.PROTEIN,   # GLU
    9: Molecule.PROTEIN,   # PHE
    10: Molecule.PROTEIN,  # GLY
    11: Molecule.PROTEIN,  # HIS
    12: Molecule.PROTEIN,  # ILE
    13: Molecule.PROTEIN,  # LYS
    14: Molecule.PROTEIN,  # LEU
    15: Molecule.PROTEIN,  # MET
    16: Molecule.PROTEIN,  # ASN
    17: Molecule.PROTEIN,  # PRO
    18: Molecule.PROTEIN,  # GLN
    19: Molecule.PROTEIN,  # ARG
    20: Molecule.PROTEIN,  # SER
    21: Molecule.PROTEIN,  # THR
    22: Molecule.PROTEIN,  # VAL
    23: Molecule.PROTEIN,  # TRP
    24: Molecule.PROTEIN,  # TYR

    # Non-polymer
    25: Molecule.WATER,    # HOH
    26: Molecule.ION,      # MG
    27: Molecule.ION,      # CS

    # Modified nucleotides (treated as RNA for now)
    28: Molecule.RNA,      # GTP
    29: Molecule.RNA,      # CCC
    30: Molecule.RNA,      # GNG
}


def residue_to_molecule(residue_idx: int) -> Molecule:
    """
    Get the molecule type for a residue index.

    Args:
        residue_idx: Integer residue index from Residue enum.

    Returns:
        Molecule type for this residue.
    """
    return RESIDUE_MOLECULE_TYPE.get(residue_idx, Molecule.UNKNOWN)


RES_ABBREV: dict[str, str] = {
    # Nucleotides (lowercase)
    'A': 'a',
    'C': 'c',
    'G': 'g',
    'U': 'u',
    'T': 't',
    'N': 'n',
    # Amino acids (single letter)
    'ALA': 'A',
    'CYS': 'C',
    'ASP': 'D',
    'GLU': 'E',
    'PHE': 'F',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LYS': 'K',
    'LEU': 'L',
    'MET': 'M',
    'ASN': 'N',
    'PRO': 'P',
    'GLN': 'Q',
    'ARG': 'R',
    'SER': 'S',
    'THR': 'T',
    'VAL': 'V',
    'TRP': 'W',
    'TYR': 'Y',
}
