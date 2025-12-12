"""
Residue definitions for nucleotides and amino acids.
"""

from ..utils import IndexEnum
from ..types import Molecule


class Residue(IndexEnum):
    """
    Residue types with unique integer indices.

    Includes nucleotides (RNA and DNA), amino acids, water, and ions.
    Uses 3-letter codes consistently to avoid ambiguity (e.g., ADE for
    adenosine, ALA for alanine).
    """

    # RNA nucleotides (3-letter codes)
    ADE = 0   # Adenosine
    CYT = 1   # Cytidine
    GUA = 2   # Guanosine
    URA = 3   # Uridine

    # DNA nucleotides
    DA = 0    # Deoxyadenosine (same index as ADE)
    DC = 1    # Deoxycytidine (same index as CYT)
    DG = 2    # Deoxyguanosine (same index as GUA)
    DT = 4    # Deoxythymidine
    THY = 4   # Thymidine (alias for DT)

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
    0: Molecule.RNA,   # ADE, DA
    1: Molecule.RNA,   # CYT, DC
    2: Molecule.RNA,   # GUA, DG
    3: Molecule.RNA,   # URA
    4: Molecule.DNA,   # THY, DT

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


# CIF residue names -> Residue index
# Maps the names used in mmCIF files to our internal indices
CIF_RESIDUE_NAMES: dict[str, int] = {
    # RNA nucleotides (CIF uses single letters)
    'A': Residue.ADE.value,
    'C': Residue.CYT.value,
    'G': Residue.GUA.value,
    'U': Residue.URA.value,
    # DNA nucleotides
    'DA': Residue.ADE.value,  # Same base index as RNA
    'DC': Residue.CYT.value,
    'DG': Residue.GUA.value,
    'DT': Residue.DT.value,
    'T': Residue.DT.value,  # Some PDB files use single T
}

# Residue index -> CIF output name
# Used by the writer to output CIF-compatible residue names
# Must match the residue prefixes used in atom definitions (atoms.py)
RESIDUE_CIF_NAMES: dict[int, str] = {
    Residue.ADE.value: 'A',   # RNA adenosine
    Residue.CYT.value: 'C',   # RNA cytidine
    Residue.GUA.value: 'G',   # RNA guanosine
    Residue.URA.value: 'U',   # RNA uridine
    Residue.DT.value: 'DT',   # DNA thymidine (no single-letter atom prefix)
    # Amino acids use their enum names (3-letter codes)
    Residue.ALA.value: 'ALA',
    Residue.CYS.value: 'CYS',
    Residue.ASP.value: 'ASP',
    Residue.GLU.value: 'GLU',
    Residue.PHE.value: 'PHE',
    Residue.GLY.value: 'GLY',
    Residue.HIS.value: 'HIS',
    Residue.ILE.value: 'ILE',
    Residue.LYS.value: 'LYS',
    Residue.LEU.value: 'LEU',
    Residue.MET.value: 'MET',
    Residue.ASN.value: 'ASN',
    Residue.PRO.value: 'PRO',
    Residue.GLN.value: 'GLN',
    Residue.ARG.value: 'ARG',
    Residue.SER.value: 'SER',
    Residue.THR.value: 'THR',
    Residue.VAL.value: 'VAL',
    Residue.TRP.value: 'TRP',
    Residue.TYR.value: 'TYR',
    # Non-polymer
    Residue.HOH.value: 'HOH',
    Residue.MG.value: 'MG',
    Residue.CS.value: 'CS',
    # Modified nucleotides
    Residue.GTP.value: 'GTP',
    Residue.CCC.value: 'CCC',
    Residue.GNG.value: 'GNG',
}


RES_ABBREV: dict[str, str] = {
    # Nucleotides (lowercase single-letter)
    'ADE': 'a',
    'CYT': 'c',
    'GUA': 'g',
    'URA': 'u',
    'THY': 't',
    'DA': 'a',
    'DC': 'c',
    'DG': 'g',
    'DT': 't',
    'N': 'n',
    # Amino acids (uppercase single-letter)
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
