"""
Biochemistry constants for structure analysis.

Defines atom groupings for backbone, nucleobase, phosphate, and sidechain atoms.
"""

from typing import Callable
from ..utils import IndexEnum
from ._generated_atoms import (
    # RNA
    A, C, G, U,
    # DNA
    DA, DC, DG, DT,
    # Amino acids
    ALA, ARG, ASN, ASP, CYS,
    GLN, GLU, GLY, HIS, ILE,
    LEU, LYS, MET, PHE, PRO,
    SER, THR, TRP, TYR, VAL,
)

# Residue groupings
_RNA_NUCLEOTIDES = [
    ("A_", A),
    ("C_", C),
    ("G_", G),
    ("U_", U),
]

_DNA_NUCLEOTIDES = [
    ("DA_", DA),
    ("DC_", DC),
    ("DG_", DG),
    ("DT_", DT),
]

_AMINO_ACIDS = [
    ("GLY_", GLY), ("ALA_", ALA), ("VAL_", VAL), ("LEU_", LEU),
    ("ILE_", ILE), ("PRO_", PRO), ("PHE_", PHE),
    ("TRP_", TRP), ("MET_", MET), ("CYS_", CYS),
    ("SER_", SER), ("THR_", THR), ("ASN_", ASN),
    ("GLN_", GLN), ("ASP_", ASP), ("GLU_", GLU),
    ("LYS_", LYS), ("ARG_", ARG), ("HIS_", HIS), ("TYR_", TYR),
]

# Protein backbone atom names
_PROTEIN_BACKBONE_NAMES = {'N', 'CA', 'C', 'O'}


def _filter_atoms(
    residues: list[tuple[str, type]],
    predicate: Callable[[str], bool],
) -> dict[str, int]:
    """
    Filter atoms across residues using a predicate.

    Args:
        residues: List of (prefix, enum_class) tuples.
        predicate: Function that takes an atom name and returns True to include.

    Returns:
        Dictionary mapping prefixed atom names to their indices.
    """
    result = {}
    for prefix, residue in residues:
        for name, value in residue.dict().items():
            if predicate(name):
                result[prefix + name] = value
    return result


# Nucleic acid backbone: sugar-phosphate atoms (contain 'p' or 'P')
_nucleic_backbone = lambda n: 'p' in n or 'P' in n

# Nucleobase atoms: neither 'p' nor 'P'
_nucleobase = lambda n: 'p' not in n and 'P' not in n

# Phosphate atoms: contain uppercase 'P'
_phosphate = lambda n: 'P' in n

# Protein backbone atoms
_protein_backbone = lambda n: n in _PROTEIN_BACKBONE_NAMES

# Sidechain atoms: not backbone
_sidechain = lambda n: n not in _PROTEIN_BACKBONE_NAMES and n not in {'OXT', 'H', 'H2', 'H3', 'HA', 'HA2', 'HA3'}


# Combined Backbone: RNA + DNA + Protein
Backbone = IndexEnum(
    "Backbone",
    _filter_atoms(_RNA_NUCLEOTIDES, _nucleic_backbone) |
    _filter_atoms(_DNA_NUCLEOTIDES, _nucleic_backbone) |
    _filter_atoms(_AMINO_ACIDS, _protein_backbone)
)

# Nucleobase atoms (RNA only for now)
Nucleobase = IndexEnum(
    "Nucleobase",
    _filter_atoms(_RNA_NUCLEOTIDES, _nucleobase)
)

# Phosphate atoms (RNA + DNA)
Phosphate = IndexEnum(
    "Phosphate",
    _filter_atoms(_RNA_NUCLEOTIDES, _phosphate) |
    _filter_atoms(_DNA_NUCLEOTIDES, _phosphate)
)

# Sidechain atoms (protein only)
Sidechain = IndexEnum(
    "Sidechain",
    _filter_atoms(_AMINO_ACIDS, _sidechain)
)
