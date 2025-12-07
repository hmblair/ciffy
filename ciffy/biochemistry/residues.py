"""
Residue definitions for nucleotides and amino acids.
"""

from ..utils import IndexEnum


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
