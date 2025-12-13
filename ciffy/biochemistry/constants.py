"""
Biochemistry constants for structure analysis.

Defines atom groupings for backbone, nucleobase, phosphate, and sidechain atoms.
"""

from typing import Callable
from ..utils import IndexEnum
from ._generated_residues import Residue

# Residue groupings (prefix, residue) - atoms accessed via residue.atoms
_RNA_NUCLEOTIDES = [
    ("A_", Residue.A),
    ("C_", Residue.C),
    ("G_", Residue.G),
    ("U_", Residue.U),
]

_DNA_NUCLEOTIDES = [
    ("DA_", Residue.DA),
    ("DC_", Residue.DC),
    ("DG_", Residue.DG),
    ("DT_", Residue.DT),
]

_AMINO_ACIDS = [
    ("GLY_", Residue.GLY), ("ALA_", Residue.ALA), ("VAL_", Residue.VAL), ("LEU_", Residue.LEU),
    ("ILE_", Residue.ILE), ("PRO_", Residue.PRO), ("PHE_", Residue.PHE),
    ("TRP_", Residue.TRP), ("MET_", Residue.MET), ("CYS_", Residue.CYS),
    ("SER_", Residue.SER), ("THR_", Residue.THR), ("ASN_", Residue.ASN),
    ("GLN_", Residue.GLN), ("ASP_", Residue.ASP), ("GLU_", Residue.GLU),
    ("LYS_", Residue.LYS), ("ARG_", Residue.ARG), ("HIS_", Residue.HIS), ("TYR_", Residue.TYR),
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
        residues: List of (prefix, ResidueType) tuples.
        predicate: Function that takes an atom name and returns True to include.

    Returns:
        Dictionary mapping prefixed atom names to their indices.
    """
    result = {}
    for prefix, residue in residues:
        for name, value in residue.atoms.dict().items():
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
