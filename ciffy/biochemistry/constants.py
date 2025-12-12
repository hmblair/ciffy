"""
Biochemistry constants for structure analysis.

Defines atom groupings for backbone, nucleobase, phosphate, and sidechain atoms.
"""

from typing import Callable
from ..utils import IndexEnum
from .nucleotides import (
    # RNA
    Adenosine, Cytosine, Guanosine, Uridine,
    # DNA
    Deoxyadenosine, Deoxycytidine, Deoxyguanosine, Thymidine,
    # Amino acids
    Glycine, Alanine, Valine, Leucine, Isoleucine, Proline,
    Phenylalanine, Tryptophan, Methionine, Cysteine, Serine, Threonine,
    Asparagine, Glutamine, AsparticAcid, GlutamicAcid, Lysine, Arginine,
    Histidine, Tyrosine,
)

# Residue groupings
_RNA_NUCLEOTIDES = [
    ("A_", Adenosine),
    ("C_", Cytosine),
    ("G_", Guanosine),
    ("U_", Uridine),
]

_DNA_NUCLEOTIDES = [
    ("DA_", Deoxyadenosine),
    ("DC_", Deoxycytidine),
    ("DG_", Deoxyguanosine),
    ("DT_", Thymidine),
]

_AMINO_ACIDS = [
    ("G_", Glycine), ("A_", Alanine), ("V_", Valine), ("L_", Leucine),
    ("I_", Isoleucine), ("P_", Proline), ("F_", Phenylalanine),
    ("W_", Tryptophan), ("M_", Methionine), ("C_", Cysteine),
    ("S_", Serine), ("T_", Threonine), ("N_", Asparagine),
    ("Q_", Glutamine), ("D_", AsparticAcid), ("E_", GlutamicAcid),
    ("K_", Lysine), ("R_", Arginine), ("H_", Histidine), ("Y_", Tyrosine),
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
